"""EXP-006: Standalone End-to-End Metric Learning & Pedestrian Attribute Recognition (ABPR).

Trains a unified Swin-Transformer Tiny multi-task architecture directly from raw dataset:
  1. Multi-Attribute Classification (Task 1 supervision via ASL + Grouping Loss)
  2. Identity Metric Space Alignment (Task 2 supervision via ArcFace Loss over 7,204 identities)
  3. Query-to-Image Cross-Modal Alignment (Task 2 supervision via InfoNCE Contrastive Loss)

Features & Design:
  - Supports Dual-GPU DDP (torchrun --nproc_per_node=2) with NCCL backend
  - Domain-balanced sampling (Market1501, PA-100k, PETA) via DomainBalancedBatchSampler
  - High-throughput parallel RAM byte caching with chunked progress reporting
  - ZERO external submission zip dependencies: 100% self-contained
  - Vectorized 2D matrix matching during validation (memory-safe, zero 18.5GB RAM leaks)
  - Strictly zero-tqdm continuous stdout logging for Kaggle background executions
  - Automatic packaging of standalone Codabench submission zip
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gc
import io
import math
import os
import shutil
import sys
import time
import warnings
import zipfile
from typing import Any, Optional

# Filter benign framework stride and unused parameter warnings
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")
warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T

from src.dataset import (
    DomainBalancedBatchSampler,
    Task2RetrievalEvalData,
    resolve_data_root,
)
from src.evaluate import get_device
from src.losses import AsymmetricLoss, CategorySoftmaxLoss
from src.metric_learning import (
    ArcFaceLoss,
    CrossModalContrastiveLoss,
    QueryEncoder,
    VisualProjector,
    compute_hybrid_distances,
    evaluate_distance_matrix,
)
from src.stage2_submission_builder import generate_stage2_run_py

# 40 canonical UPAR attributes
CANONICAL_ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun", "Accessory-Hat",
]


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialize DistributedDataParallel (DDP) if launched via torchrun on CUDA."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and torch.cuda.is_available():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed() -> None:
    """Clean up DDP process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_memory_info(device: torch.device) -> str:
    """Return current memory usage string."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        return f"VRAM: {allocated:.0f}MB / {reserved:.0f}MB"
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
        return f"UnifiedMem: {allocated:.0f}MB"
    return "CPU"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    if seconds is None or np.isnan(seconds) or np.isinf(seconds) or seconds < 0:
        return "--m --s"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def print_codabench_table(
    epoch: int,
    epochs: int,
    metrics: dict[str, float],
    exp_id: str = "EXP-006",
    participant: str = "Jyothiradithya Lingam",
    is_best: bool = False,
) -> None:
    """Print validation results in the exact official Codabench competition leaderboard table format."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print("\n" + "=" * 115, flush=True)
    print("Results", flush=True)
    print(f"Task:        Attribute-based Person Retrieval - validation set (Epoch {epoch:02d}/{epochs:02d})", flush=True)
    print(f"{'#':<4} {'Participant':<26} {'Date':<18} {'ID':<10} {'mADM':<9} {'mAP':<9} {'Rank-1':<9} {'Rank-5':<9} {'Rank-10':<9} {'mINP':<9}", flush=True)
    print("-" * 115, flush=True)
    print(
        f"{epoch:<4} {participant:<26} {now_str:<18} {exp_id:<10} "
        f"{metrics['mADM']:<9.4f} {metrics['mAP']:<9.4f} "
        f"{metrics['Rank-1']:<9.4f} {metrics['Rank-5']:<9.4f} {metrics['Rank-10']:<9.4f} "
        f"{metrics['mINP']:<9.4f}",
        flush=True,
    )
    if "PAR_mA" in metrics:
        print(f"Pedestrian Attribute Recognition (PAR) Mean Accuracy: {metrics['PAR_mA'] * 100:.2f}%", flush=True)
    if is_best:
        print(f"*** NEW BEST VALIDATION SCORE: mADM = {metrics['mADM']:.4f} ***", flush=True)
    print("=" * 115 + "\n", flush=True)


class UPAREndToEndDataset(Dataset):
    """End-to-End multi-task training dataset loading images, 40 attributes, identity IDs, domain IDs, and query vectors."""

    def __init__(
        self,
        data_root: str,
        transform: T.Compose | None = None,
        preload_to_ram: bool = True,
        debug_samples: int | None = None,
        is_master: bool = True,
    ) -> None:
        self.data_root = data_root
        self.preload_to_ram = preload_to_ram
        self.transform = transform or T.Compose([
            T.Resize((256, 128), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 1. Load canonical queries: (7204, 40)
        queries_path = os.path.join(data_root, "annotations", "task2", "train", "queries.csv")
        canonical_queries = []
        with open(queries_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)
            for r in reader:
                if r:
                    canonical_queries.append([float(x.strip()) for x in r])
        self.canonical_queries = np.array(canonical_queries, dtype=np.float32)

        # 2. Load image-to-identity mapping
        ids_path = os.path.join(data_root, "annotations", "task2", "train", "ids.csv")
        img_to_id = {}
        with open(ids_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for r in reader:
                if len(r) >= 2:
                    img_to_id[r[0].strip()] = int(r[1].strip())

        # 3. Load train.csv images, 40 attributes, and domain IDs
        train_csv = os.path.join(data_root, "annotations", "train.csv")
        self.image_paths: list[str] = []
        self.labels: list[np.ndarray] = []
        self.identity_ids: list[int] = []
        self.domain_ids: list[int] = []

        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                if not row or len(row) < 41:
                    continue
                rel_path = row[0].strip()
                attrs = np.array([float(x.strip()) for x in row[1:41]], dtype=np.float32)
                ident = img_to_id.get(rel_path, 0)

                # Domain IDs: 0 = Market1501, 1 = PA100k, 2 = PETA
                if rel_path.startswith("Market1501"):
                    did = 0
                elif rel_path.startswith("PA100k"):
                    did = 1
                elif rel_path.startswith("PETA"):
                    did = 2
                else:
                    did = 0

                self.image_paths.append(rel_path)
                self.labels.append(attrs)
                self.identity_ids.append(ident)
                self.domain_ids.append(did)

        if debug_samples is not None and debug_samples > 0:
            self.image_paths = self.image_paths[:debug_samples]
            self.labels = self.labels[:debug_samples]
            self.identity_ids = self.identity_ids[:debug_samples]
            self.domain_ids = self.domain_ids[:debug_samples]

        self.labels = np.array(self.labels, dtype=np.float32)
        self.identity_ids = np.array(self.identity_ids, dtype=np.int64)
        self.domain_ids = np.array(self.domain_ids, dtype=np.int64)

        # Preload image bytes if requested
        self.image_bytes: list[bytes] | None = None
        if self.preload_to_ram and len(self.image_paths) > 0:
            self._preload_images(is_master=is_master)

    def _preload_images(self, is_master: bool = True) -> None:
        """Preload all raw image bytes into system RAM in parallel with real-time chunk logging."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        t0 = time.time()
        n_total = len(self.image_paths)
        if is_master:
            print(f"[RAM Cache] Preloading {n_total:,} images into system RAM (16 concurrent I/O threads)...", flush=True)

        def read_file(p: str) -> bytes:
            fp = os.path.join(self.data_root, p)
            try:
                with open(fp, "rb") as bf:
                    return bf.read()
            except Exception:
                return b""

        results = [b""] * n_total
        chunk_size = 20000
        with ThreadPoolExecutor(max_workers=16) as pool:
            for start_i in range(0, n_total, chunk_size):
                end_i = min(start_i + chunk_size, n_total)
                chunk_paths = self.image_paths[start_i:end_i]
                chunk_results = list(pool.map(read_file, chunk_paths))
                results[start_i:end_i] = chunk_results
                if is_master and end_i < n_total:
                    pct = (end_i / n_total) * 100
                    elapsed_so_far = time.time() - t0
                    speed = end_i / max(elapsed_so_far, 0.1)
                    print(f"[RAM Cache] {end_i:,} / {n_total:,} images ({pct:.1f}%) loaded in {elapsed_so_far:.1f}s ({speed:.0f} img/s)...", flush=True)

        self.image_bytes = results
        total_mb = sum(len(b) for b in self.image_bytes) / (1024 * 1024)
        elapsed = time.time() - t0
        if is_master:
            print(f"[RAM Cache] Preloaded {n_total:,} images ({total_mb:.1f} MB) in {elapsed:.1f}s! All disk I/O eliminated.", flush=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        if self.image_bytes is not None and idx < len(self.image_bytes) and len(self.image_bytes[idx]) > 0:
            img = Image.open(io.BytesIO(self.image_bytes[idx])).convert("RGB")
        else:
            full_p = os.path.join(self.data_root, self.image_paths[idx])
            try:
                img = Image.open(full_p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (128, 256), color=(128, 128, 128))

        img_tensor = self.transform(img)
        label = torch.from_numpy(self.labels[idx])
        ident = int(self.identity_ids[idx])
        query_vec = torch.from_numpy(self.canonical_queries[ident])

        return img_tensor, label, ident, query_vec


class SwinPARMetric(nn.Module):
    """Unified Swin-Transformer Tiny Model with Multi-Attribute Head, Projector Heads, and ArcFace Head."""

    def __init__(
        self,
        num_attributes: int = 40,
        embedding_dim: int = 512,
        num_identities: int = 7204,
        scale: float = 30.0,
        margin: float = 0.20,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = models.swin_t(weights=weights)
        in_features = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Identity()

        # Attribute classifier head (Task 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_attributes),
        )

        # Visual metric projector (Task 2: 768 -> 512 L2-normalized)
        self.vis_projector = VisualProjector(
            in_features=in_features,
            embedding_dim=embedding_dim,
            dropout=0.1,
        )

        # Query metric encoder (Task 2: 40 -> 512 L2-normalized)
        self.query_encoder = QueryEncoder(
            num_attributes=num_attributes,
            embedding_dim=embedding_dim,
            dropout=0.1,
        )

        # ArcFace loss / classification head (7,204 identities)
        self.arcface = ArcFaceLoss(
            num_classes=num_identities,
            embedding_dim=embedding_dim,
            scale=scale,
            margin=margin,
        )

    def forward_image(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for images.
        Returns:
            feats: (B, 768)
            logits: (B, 40)
            vis_emb: (B, 512) L2-normalized
        """
        feats = self.backbone(x)
        logits = self.classifier(feats)
        vis_emb = self.vis_projector(feats)
        return feats, logits, vis_emb

    def forward_query(self, q: torch.Tensor) -> torch.Tensor:
        """Forward pass for queries.
        Returns:
            q_emb: (B, 512) L2-normalized
        """
        return self.query_encoder(q)

    def forward(
        self,
        images: torch.Tensor,
        query_vecs: torch.Tensor,
        ident_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Unified multi-task forward pass utilizing all parameters in DDP."""
        feats = self.backbone(images)
        logits = self.classifier(feats)
        vis_emb = self.vis_projector(feats)
        query_emb = self.query_encoder(query_vecs)

        arc_loss = None
        if ident_labels is not None:
            arc_loss = self.arcface(vis_emb, ident_labels)

        return logits, vis_emb, query_emb, arc_loss

    def get_backbone_state_dict(self) -> dict[str, torch.Tensor]:
        """Extracts backbone and classifier weights matching standalone SwinPAR."""
        sd = {}
        for k, v in self.backbone.state_dict().items():
            sd[f"backbone.{k}"] = v
        for k, v in self.classifier.state_dict().items():
            sd[f"classifier.{k}"] = v
        return sd

    def get_metric_heads_state_dict(self) -> dict[str, Any]:
        """Extracts metric projection weights matching Stage 2 packaging."""
        return {
            "vis_projector": self.vis_projector.state_dict(),
            "query_encoder": self.query_encoder.state_dict(),
        }


class SimplePathDataset(Dataset):
    """Dataset for gallery images during validation."""

    def __init__(self, root: str, paths: list[str], transform):
        self.root = root
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = os.path.join(self.root, self.paths[idx])
        try:
            with Image.open(p) as img:
                return self.transform(img.convert("RGB"))
        except Exception:
            return torch.zeros(3, 256, 128, dtype=torch.float32)


def run_validation(
    model: SwinPARMetric,
    retrieval_data: Task2RetrievalEvalData,
    device: torch.device,
    platt_t: np.ndarray,
    platt_b: np.ndarray,
    lambda_metric: float = 0.25,
    val_subset: int | None = None,
    batch_size: int = 128,
) -> dict[str, float]:
    """Memory-safe validation run over Task 2 gallery and queries."""
    model.eval()
    val_transform = T.Compose([
        T.Resize((256, 128), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    gallery_paths = retrieval_data.gallery_paths
    gallery_gt = retrieval_data.gallery_gt
    queries = retrieval_data.queries

    if val_subset is not None and val_subset > 0:
        gallery_paths = gallery_paths[:val_subset]
        gallery_gt = gallery_gt[:val_subset]
        queries = queries[:min(val_subset, len(queries))]

    # 1. Compute query embeddings
    with torch.no_grad():
        q_tensor = torch.from_numpy(queries).to(device)
        q_embs = model.forward_query(q_tensor).cpu().numpy()
        del q_tensor

    # 2. Extract gallery features & attribute logits in batches
    all_logits = []
    all_v_embs = []

    val_dataset = SimplePathDataset(retrieval_data.data_root, gallery_paths, val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=(device.type == "cuda"),
    )

    with torch.no_grad():
        for batch_tensor in val_loader:
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                _, logits, v_emb = model.forward_image(batch_tensor)
            all_logits.append(logits.float().cpu().numpy())
            all_v_embs.append(v_emb.float().cpu().numpy())
            del batch_tensor, logits, v_emb

    val_logits = np.vstack(all_logits)
    val_v_embs = np.vstack(all_v_embs)

    # Calibrate attribute logits: p = sigmoid((logits - b) / T)
    calib_logits = np.clip((val_logits - platt_b[None, :]) / platt_t[None, :], -20.0, 20.0)
    calib_probs = 1.0 / (1.0 + np.exp(-calib_logits))

    # Hybrid distance
    hybrid_dist = compute_hybrid_distances(
        queries,
        calib_probs,
        q_embs,
        val_v_embs,
        lambda_metric=lambda_metric,
    )

    metrics = evaluate_distance_matrix(hybrid_dist, gallery_gt, queries)

    # Compute PAR mean accuracy on gallery
    gt_binary = (gallery_gt > 0.5).astype(np.float32)
    pred_binary = (calib_probs > 0.5).astype(np.float32)
    acc_per_attr = (gt_binary == pred_binary).mean(axis=0)
    metrics["PAR_mA"] = float(acc_per_attr.mean())

    # Free evaluation memory immediately
    del val_logits, val_v_embs, calib_logits, calib_probs, hybrid_dist, q_embs, all_logits, all_v_embs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def package_submission(
    model: SwinPARMetric,
    output_zip: str,
    lambda_metric: float,
    platt_t: list[float],
    platt_b: list[float],
) -> str:
    """Builds completely self-contained Codabench submission zip from model weights."""
    print("=" * 80)
    print(f"PACKAGING EXP-006 SUBMISSION ZIP: {output_zip}")
    print("=" * 80)

    staging_dir = "/tmp/exp006_submission_staging"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    assets_dir = os.path.join(staging_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Save backbone weights
    bb_path = os.path.join(assets_dir, "model_backbone.pth")
    torch.save(model.get_backbone_state_dict(), bb_path)
    print(f"Saved model_backbone.pth ({os.path.getsize(bb_path)/(1024*1024):.1f} MB)")

    # 2. Save metric heads
    mh_path = os.path.join(assets_dir, "metric_heads.pth")
    torch.save(model.get_metric_heads_state_dict(), mh_path)
    print(f"Saved metric_heads.pth ({os.path.getsize(mh_path)/(1024*1024):.1f} MB)")

    # 3. Generate run.py
    run_py = generate_stage2_run_py(
        model_type="swin_t",
        lambda_metric=lambda_metric,
        height=256,
        width=128,
        T_params=platt_t,
        b_params=platt_b,
    )
    with open(os.path.join(staging_dir, "run.py"), "w", encoding="utf-8") as f:
        f.write(run_py)
    print("Generated self-contained run.py with embedded Platt parameters.")

    # 4. Generate metadata.yaml
    metadata_yaml = 'command: python3 -c "print(\'EXP-006 Submission package ready. Ingestion program imports run.py\')"\n'
    with open(os.path.join(staging_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        f.write(metadata_yaml)

    # 5. Zip archive
    os.makedirs(os.path.dirname(os.path.abspath(output_zip)), exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging_dir):
            for file in files:
                abs_f = os.path.join(root, file)
                rel_f = os.path.relpath(abs_f, staging_dir)
                zf.write(abs_f, rel_f)

    zip_mb = os.path.getsize(output_zip) / (1024 * 1024)
    shutil.rmtree(staging_dir)
    print(f">> Submission package ready: {output_zip} ({zip_mb:.1f} MB)\n")
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-006 Standalone End-to-End Metric Learning")
    parser.add_argument("--data_root", type=str, default="UPAR-Challenge-2027/data")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--val_batch_size", type=int, default=128, help="Validation batch size")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_heads", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=30.0, help="ArcFace scale s")
    parser.add_argument("--margin", type=float, default=0.20, help="ArcFace margin m")
    parser.add_argument("--lambda_arc", type=float, default=0.5, help="Weight for ArcFace loss")
    parser.add_argument("--lambda_cm", type=float, default=0.25, help="Weight for InfoNCE contrastive loss")
    parser.add_argument("--lambda_metric", type=float, default=0.25, help="Retrieval score fusion lambda")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--log_interval", type=int, default=25, help="Print progress every N steps")
    parser.add_argument("--debug_samples", type=int, default=None, help="Debug: limit training samples")
    parser.add_argument("--val_subset", type=int, default=None, help="Debug: limit validation samples")
    parser.add_argument("--no_preload", action="store_true", help="Disable RAM caching of image bytes")
    parser.add_argument("--no_pretrained", action="store_true", help="Initialize backbone with random weights (for offline testing)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--output_zip", type=str, default="submissions/EXP-006_submission.zip")
    parser.add_argument("--exp_id", type=str, default="EXP-006")
    args = parser.parse_args()

    # Initialize DistributedDataParallel if launched via torchrun
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_master = (rank == 0)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if is_master:
        print("=" * 80)
        print(f"[{args.exp_id}] STANDALONE END-TO-END METRIC LEARNING & ATTRIBUTE RECOGNITION")
        if is_distributed:
            print(f"[{args.exp_id}] Distributed Engine Activated: {world_size}x GPUs (DDP with NCCL backend)")
            print(f"[{args.exp_id}] Primary GPU: {torch.cuda.get_device_name(local_rank)} (Rank {rank}/{world_size})")
        else:
            print(f"[{args.exp_id}] Single-Device Engine Activated: {device}")
        print(f"[{args.exp_id}] Effective Global Batch Size: {args.batch_size * world_size} | Epochs: {args.epochs}")
        print("=" * 80, flush=True)

    # 1. Resolve dataset root (NO FUSE staging / NO cp -r)
    resolved_root = resolve_data_root(args.data_root)
    if is_master:
        print(f"[{args.exp_id}] Dataset Root resolved to: {resolved_root}")

    # 2. Platt calibration parameters
    calib_candidates = [
        "assets/stage1_calibration_params.npz",
        "checkpoints/stage1_calibration_params.npz",
        os.path.join(resolved_root, "stage1_calibration_params.npz"),
    ]
    calib_file = next((cp for cp in calib_candidates if os.path.exists(cp)), None)
    if calib_file is not None:
        calib = np.load(calib_file)
        platt_t = calib["T_005"]
        platt_b = calib["b_005"]
        if is_master:
            print(f"[{args.exp_id}] Loaded Platt calibration parameters from: {calib_file}")
    else:
        platt_t = np.array([1.5287, 1.9120, 1.4530, 1.0267, 1.7007, 1.4921, 1.5512, 1.3312, 1.2643, 1.3411, 1.4912, 1.5231, 1.3482, 1.6021, 1.6432, 1.6082, 1.4451, 1.3452, 1.6071, 1.5291, 1.1451, 1.1952, 1.2341, 1.4481, 1.5781, 1.3352, 1.6441, 1.6571, 1.6352, 1.5221, 1.4601, 1.6131, 1.5081, 1.0362, 1.2691, 1.2381, 1.3621, 1.4771, 1.4691, 1.3241], dtype=np.float32)
        platt_b = np.array([2.7781, 0.9276, 1.4545, 1.4719, 1.1888, 1.1121, 1.6231, 1.1541, 1.1921, 1.2191, 1.4131, 1.4721, 1.2481, 1.5431, 1.5841, 1.5491, 1.3711, 1.2441, 1.5481, 1.4641, 1.1741, 1.1631, 1.2041, 1.3831, 1.5081, 1.2441, 1.5861, 1.5991, 1.5781, 1.4611, 1.3971, 1.5551, 1.4441, 0.9711, 1.1831, 1.1671, 1.2981, 1.4131, 1.4041, 1.2631], dtype=np.float32)
        if is_master:
            print(f"[{args.exp_id}] Using hardcoded default Platt calibration parameters.")

    # 3. Load dataset
    if is_master:
        print(f"\n[Step 1/4] Initializing UPAR End-to-End Dataset...", flush=True)

    train_dataset = UPAREndToEndDataset(
        data_root=resolved_root,
        preload_to_ram=not args.no_preload,
        debug_samples=args.debug_samples,
        is_master=is_master,
    )

    if is_master:
        print(f"[{args.exp_id}] Loaded {len(train_dataset)} training samples.")

    # Sampler: use DomainBalancedBatchSampler if full dataset, fallback to standard sampler if small debug subset
    if args.debug_samples is not None and len(train_dataset) < args.batch_size * world_size:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        batch_sampler = None
    else:
        batch_sampler = DomainBalancedBatchSampler(
            domain_ids=train_dataset.domain_ids,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        )
        nw = args.num_workers if args.num_workers is not None else (2 if is_distributed else 4)
        if os.name == "nt" or sys.platform == "darwin":
            nw = 0

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=nw,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(nw > 0),
            prefetch_factor=2 if nw > 0 else None,
        )

    if is_master:
        print(f"[{args.exp_id}] Batches per Epoch per Rank: {len(train_loader)} | DataLoader Workers: {train_loader.num_workers}", flush=True)

    # 4. Load validation data
    if is_master:
        print(f"\n[Step 2/4] Loading Task 2 Retrieval Validation Data...", flush=True)
        retrieval_data = Task2RetrievalEvalData(data_root=resolved_root)
        print(f"[{args.exp_id}] Loaded {len(retrieval_data.gallery_paths)} gallery images and {len(retrieval_data.queries)} queries.", flush=True)
    else:
        retrieval_data = None

    # 5. Build Model & Losses
    if is_master:
        print(f"\n[Step 3/4] Building SwinPARMetric Architecture & Multi-Task Losses...", flush=True)

    raw_model = SwinPARMetric(
        num_attributes=40,
        embedding_dim=512,
        num_identities=7204,
        scale=args.scale,
        margin=args.margin,
        pretrained=not args.no_pretrained,
        dropout=0.3,
    ).to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        model = raw_model

    # Losses
    asl_loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05).to(device)
    group_loss_fn = CategorySoftmaxLoss(group_weight=0.5).to(device)
    contrastive_loss_fn = CrossModalContrastiveLoss(temperature=0.07).to(device)

    # Optimizer with differential learning rates
    opt_model = model.module if is_distributed else model
    optimizer = AdamW(
        [
            {"params": opt_model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": opt_model.classifier.parameters(), "lr": args.lr_heads},
            {"params": opt_model.vis_projector.parameters(), "lr": args.lr_heads},
            {"params": opt_model.query_encoder.parameters(), "lr": args.lr_heads},
            {"params": opt_model.arcface.parameters(), "lr": args.lr_heads},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 6. Training Loop
    if is_master:
        print(f"\n[Step 4/4] Starting End-to-End Multi-Task Training ({args.epochs} epochs)...", flush=True)

    best_madm = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    total_batches = len(train_loader)
    last_epoch_sec: Optional[float] = None
    last_val_sec: Optional[float] = None

    for epoch in range(1, args.epochs + 1):
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        running_asl = 0.0
        running_grp = 0.0
        running_arc = 0.0
        running_cm = 0.0
        total_samples = 0
        t0 = time.time()

        for batch_idx, (imgs, labels, idents, query_vecs) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            idents = idents.to(device, non_blocking=True)
            query_vecs = query_vecs.to(device, non_blocking=True)
            bs = imgs.size(0)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, vis_embs, q_embs, arc_loss = model(imgs, query_vecs, ident_labels=idents)

                loss_asl = asl_loss_fn(logits, labels)
                loss_grp = group_loss_fn(logits, labels)
                loss_cm = contrastive_loss_fn(q_embs, vis_embs)
                loss = loss_asl + 0.5 * loss_grp + args.lambda_arc * arc_loss + args.lambda_cm * loss_cm

            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            running_loss += loss.item() * bs
            running_asl += loss_asl.item() * bs
            running_grp += loss_grp.item() * bs
            running_arc += arc_loss.item() * bs
            running_cm += loss_cm.item() * bs
            total_samples += bs

            # Real-time stdout progress logging
            if is_master and ((batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == total_batches):
                elapsed_b = time.time() - t0
                speed = total_samples * world_size / max(elapsed_b, 0.1)
                pct = ((batch_idx + 1) / total_batches) * 100
                batches_left = total_batches - (batch_idx + 1)
                sec_per_batch = elapsed_b / (batch_idx + 1)
                step_eta_sec = batches_left * sec_per_batch

                epochs_remaining = args.epochs - epoch
                est_epoch_sec = last_epoch_sec if last_epoch_sec is not None else (sec_per_batch * total_batches)
                total_eta_sec = step_eta_sec + epochs_remaining * est_epoch_sec
                if last_val_sec is not None:
                    total_eta_sec += (epochs_remaining // max(args.eval_interval, 1)) * last_val_sec

                vram_str = get_memory_info(device)
                print(
                    f"[{args.exp_id}] Epoch [{epoch:02d}/{args.epochs:02d}] | "
                    f"Batch [{batch_idx+1:04d}/{total_batches:04d}] ({pct:5.1f}%) | "
                    f"Loss: {loss.item():.4f} (ASL: {loss_asl.item():.4f}, Grp: {loss_grp.item():.4f}, Arc: {arc_loss.item():.4f}, CM: {loss_cm.item():.4f}) | "
                    f"Speed: {speed:6.1f} img/s | Step ETA: {format_duration(step_eta_sec)} | Total ETA: {format_duration(total_eta_sec)} | {vram_str}",
                    flush=True,
                )

        scheduler.step()
        epoch_sec = time.time() - t0
        last_epoch_sec = epoch_sec
        curr_lr = optimizer.param_groups[0]["lr"]

        if is_master:
            mean_loss = running_loss / max(total_samples, 1)
            mean_asl = running_asl / max(total_samples, 1)
            mean_grp = running_grp / max(total_samples, 1)
            mean_arc = running_arc / max(total_samples, 1)
            mean_cm = running_cm / max(total_samples, 1)
            print(
                f"[{args.exp_id}] Epoch {epoch:02d}/{args.epochs:02d} Complete | "
                f"Loss: {mean_loss:.4f} (ASL: {mean_asl:.4f}, Grp: {mean_grp:.4f}, Arc: {mean_arc:.4f}, CM: {mean_cm:.4f}) | "
                f"LR: {curr_lr:.2e} | Duration: {format_duration(epoch_sec)}",
                flush=True,
            )

        # Validation check
        should_val = (epoch % args.eval_interval == 0) or (epoch == args.epochs) or (args.debug_samples is not None)
        if should_val and is_master:
            print(f"\n[{args.exp_id}] Running Task 2 retrieval validation (33,407 gallery, 3,462 queries)...", flush=True)
            t_val_start = time.time()
            eval_model = model.module if is_distributed else model

            metrics = run_validation(
                model=eval_model,
                retrieval_data=retrieval_data,
                device=device,
                platt_t=platt_t,
                platt_b=platt_b,
                lambda_metric=args.lambda_metric,
                val_subset=args.val_subset,
                batch_size=args.val_batch_size,
            )
            val_sec = time.time() - t_val_start
            last_val_sec = val_sec
            current_madm = metrics["mADM"]
            is_best = current_madm > best_madm

            # Print official Codabench results table
            print_codabench_table(
                epoch=epoch,
                epochs=args.epochs,
                metrics=metrics,
                exp_id=args.exp_id,
                participant="Jyothiradithya Lingam",
                is_best=is_best,
            )

            # Checkpoint saving
            if is_best or best_madm == 0.0:
                best_madm = current_madm
                best_bb_path = os.path.join(args.save_dir, f"{args.exp_id}_best_backbone.pth")
                best_mh_path = os.path.join(args.save_dir, f"{args.exp_id}_best_metric_heads.pth")

                bb_sd = eval_model.get_backbone_state_dict()
                mh_sd = eval_model.get_metric_heads_state_dict()

                torch.save(bb_sd, best_bb_path)
                torch.save(mh_sd, best_mh_path)
                print(f"[{args.exp_id}] *** Saved best checkpoints to {args.save_dir} (mADM: {best_madm:.4f}) ***")

                # Package submission zip automatically
                package_submission(
                    model=eval_model,
                    output_zip=args.output_zip,
                    lambda_metric=args.lambda_metric,
                    platt_t=platt_t.tolist(),
                    platt_b=platt_b.tolist(),
                )

        # Synchronize DDP processes after validation
        if is_distributed:
            torch.distributed.barrier()

    if is_master:
        if not os.path.exists(args.output_zip):
            eval_model = model.module if is_distributed else model
            package_submission(
                model=eval_model,
                output_zip=args.output_zip,
                lambda_metric=args.lambda_metric,
                platt_t=platt_t.tolist(),
                platt_b=platt_b.tolist(),
            )

        print("\n" + "=" * 80)
        print(f"[{args.exp_id}] Master End-to-End Pipeline Completed Successfully!")
        print(f"[{args.exp_id}] Peak Validation mADM: {best_madm:.4f}")
        print(f"[{args.exp_id}] Final Standalone Codabench Zip: {args.output_zip}")
        print("=" * 80 + "\n")

    cleanup_distributed()


if __name__ == "__main__":
    main()
