"""EXP-006: Standalone End-to-End Metric Learning & Pedestrian Attribute Recognition (ABPR).

Trains a unified Swin-Transformer Tiny multi-task architecture directly from raw dataset:
  1. Multi-Attribute Classification (Task 1 supervision via ASL + Grouping Loss)
  2. Identity Metric Space Alignment (Task 2 supervision via ArcFace Loss over 7,204 identities)
  3. Query-to-Image Cross-Modal Alignment (Task 2 supervision via InfoNCE Contrastive Loss)

Requires ZERO previous submission zips. 100% self-contained training and packaging.
Memory-safe: Vectorized 2D matrix matching, zero 18.5 GB temporary memory leaks.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import shutil
import time
import zipfile
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T

from src.dataset import Task2RetrievalEvalData, resolve_data_root
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


class UPAREndToEndDataset(Dataset):
    """End-to-End multi-task training dataset loading images, 40 attributes, identity IDs, and query vectors."""

    def __init__(
        self,
        data_root: str,
        transform: T.Compose | None = None,
        debug_samples: int | None = None,
        preload_to_ram: bool = True,
    ) -> None:
        self.data_root = data_root
        self.transform = transform or T.Compose([
            T.Resize((256, 128), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.preload_to_ram = preload_to_ram

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

        # 3. Load train.csv images and 40 attributes
        train_csv = os.path.join(data_root, "annotations", "train.csv")
        self.image_paths: list[str] = []
        self.labels: list[np.ndarray] = []
        self.identity_ids: list[int] = []

        with open(train_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)
            for row in reader:
                if not row or len(row) < 41:
                    continue
                rel_path = row[0].strip()
                attrs = np.array([float(x.strip()) for x in row[1:41]], dtype=np.float32)
                ident = img_to_id.get(rel_path, 0)

                self.image_paths.append(rel_path)
                self.labels.append(attrs)
                self.identity_ids.append(ident)

        if debug_samples is not None and debug_samples > 0:
            self.image_paths = self.image_paths[:debug_samples]
            self.labels = self.labels[:debug_samples]
            self.identity_ids = self.identity_ids[:debug_samples]

        self.labels = np.array(self.labels, dtype=np.float32)
        self.identity_ids = np.array(self.identity_ids, dtype=np.int64)

        # Preload image bytes if requested
        self.image_bytes: list[bytes] | None = None
        if self.preload_to_ram and len(self.image_paths) > 0:
            from concurrent.futures import ThreadPoolExecutor
            t0 = time.time()
            print(f"[RAM Cache] Preloading {len(self.image_paths)} images into RAM...", flush=True)

            def read_file(p: str) -> bytes:
                fp = os.path.join(self.data_root, p)
                try:
                    with open(fp, "rb") as bf:
                        return bf.read()
                except Exception:
                    return b""

            with ThreadPoolExecutor(max_workers=16) as pool:
                self.image_bytes = list(pool.map(read_file, self.image_paths))
            mb = sum(len(b) for b in self.image_bytes) / (1024 * 1024)
            print(f"[RAM Cache] Preloaded {len(self.image_paths)} images ({mb:.1f} MB) in {time.time()-t0:.1f}s.", flush=True)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        if self.image_bytes is not None and idx < len(self.image_bytes) and len(self.image_bytes[idx]) > 0:
            import io
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
    """Unified Swin-Transformer Tiny Model with Multi-Attribute Head + Metric Projector Heads."""

    def __init__(
        self,
        num_attributes: int = 40,
        embedding_dim: int = 512,
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


def run_validation(
    model: SwinPARMetric,
    retrieval_data: Task2RetrievalEvalData,
    device: torch.device,
    platt_t: np.ndarray,
    platt_b: np.ndarray,
    lambda_metric: float = 0.25,
    val_subset: int | None = None,
    batch_size: int = 64,
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

    num_gallery = len(gallery_paths)
    all_logits = []
    all_v_embs = []

    use_autocast = device.type == "cuda"

    with torch.no_grad():
        # Encode queries
        q_tensor = torch.from_numpy(queries).float().to(device)
        q_embs = model.forward_query(q_tensor).cpu().numpy()

        # Encode gallery
        for i in range(0, num_gallery, batch_size):
            batch_paths = gallery_paths[i : i + batch_size]
            tensors = []
            for rel_p in batch_paths:
                fp = os.path.join(retrieval_data.data_root, rel_p)
                try:
                    with Image.open(fp) as img:
                        tensors.append(val_transform(img.convert("RGB")))
                except Exception:
                    tensors.append(torch.zeros(3, 256, 128, dtype=torch.float32))

            batch_tensor = torch.stack(tensors, dim=0).to(device)
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, logits, v_emb = model.forward_image(batch_tensor)
            else:
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

    # Free evaluation memory immediately
    del val_logits, val_v_embs, calib_logits, calib_probs, hybrid_dist, q_embs
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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_heads", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=30.0, help="ArcFace scale s")
    parser.add_argument("--margin", type=float, default=0.20, help="ArcFace margin m")
    parser.add_argument("--lambda_arc", type=float, default=0.5, help="Weight for ArcFace loss")
    parser.add_argument("--lambda_cm", type=float, default=0.25, help="Weight for InfoNCE contrastive loss")
    parser.add_argument("--lambda_metric", type=float, default=0.25, help="Retrieval score fusion lambda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--debug_samples", type=int, default=None, help="Debug: limit training samples")
    parser.add_argument("--val_subset", type=int, default=None, help="Debug: limit validation samples")
    parser.add_argument("--no_preload", action="store_true", help="Disable RAM caching of image bytes")
    parser.add_argument("--no_pretrained", action="store_true", help="Initialize backbone with random weights (for offline testing)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--output_zip", type=str, default="submissions/EXP-006_submission.zip")
    args = parser.parse_args()

    device = get_device()
    print("=" * 80)
    print("EXP-006: STANDALONE END-TO-END METRIC LEARNING & ATTRIBUTE RECOGNITION")
    print(f"Device: {device} | Total Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # 1. Resolve dataset
    resolved_root = resolve_data_root(args.data_root)
    print(f"Data Root resolved to: {resolved_root}")

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
        print(f"Loaded Platt calibration parameters from: {calib_file}")
    else:
        platt_t = np.array([1.5287, 1.9120, 1.4530, 1.0267, 1.7007, 1.4921, 1.5512, 1.3312, 1.2643, 1.3411, 1.4912, 1.5231, 1.3482, 1.6021, 1.6432, 1.6082, 1.4451, 1.3452, 1.6071, 1.5291, 1.1451, 1.1952, 1.2341, 1.4481, 1.5781, 1.3352, 1.6441, 1.6571, 1.6352, 1.5221, 1.4601, 1.6131, 1.5081, 1.0362, 1.2691, 1.2381, 1.3621, 1.4771, 1.4691, 1.3241], dtype=np.float32)
        platt_b = np.array([2.7781, 0.9276, 1.4545, 1.4719, 1.1888, 1.1121, 1.6231, 1.1541, 1.1921, 1.2191, 1.4131, 1.4721, 1.2481, 1.5431, 1.5841, 1.5491, 1.3711, 1.2441, 1.5481, 1.4641, 1.1741, 1.1631, 1.2041, 1.3831, 1.5081, 1.2441, 1.5861, 1.5991, 1.5781, 1.4611, 1.3971, 1.5551, 1.4441, 0.9711, 1.1831, 1.1671, 1.2981, 1.4131, 1.4041, 1.2631], dtype=np.float32)
        print("Using hardcoded default Platt calibration parameters.")

    # 3. Load dataset
    print("\n[Step 1/4] Initializing UPAR End-to-End Dataset...")
    train_dataset = UPAREndToEndDataset(
        data_root=resolved_root,
        debug_samples=args.debug_samples,
        preload_to_ram=not args.no_preload,
    )
    print(f"Loaded {len(train_dataset)} training samples.")

    nw = args.num_workers if os.name != "nt" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) > args.batch_size),
    )

    # 4. Load validation data
    print("\n[Step 2/4] Loading Task 2 Retrieval Validation Data...")
    retrieval_data = Task2RetrievalEvalData(data_root=resolved_root)
    print(f"Loaded {len(retrieval_data.gallery_paths)} gallery images and {len(retrieval_data.queries)} queries.")

    # 5. Build Model & Losses
    print("\n[Step 3/4] Building SwinPARMetric Architecture & Multi-Task Losses...")
    model = SwinPARMetric(
        num_attributes=40,
        embedding_dim=512,
        pretrained=not args.no_pretrained,
        dropout=0.3,
    ).to(device)

    # Losses
    asl_loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05).to(device)
    group_loss_fn = CategorySoftmaxLoss(group_weight=0.5).to(device)
    arcface_loss_fn = ArcFaceLoss(
        num_classes=7204,
        embedding_dim=512,
        scale=args.scale,
        margin=args.margin,
    ).to(device)
    contrastive_loss_fn = CrossModalContrastiveLoss(temperature=0.07).to(device)

    # Multi-GPU support if available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and device.type == "cuda":
        print(f"Enabling DataParallel across {num_gpus} GPUs!")
        # Notice: Keep ArcFace on primary cuda:0 device for single class-center matrix
        model.backbone = nn.DataParallel(model.backbone)

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
            {"params": model.classifier.parameters(), "lr": args.lr_heads},
            {"params": model.vis_projector.parameters(), "lr": args.lr_heads},
            {"params": model.query_encoder.parameters(), "lr": args.lr_heads},
            {"params": arcface_loss_fn.parameters(), "lr": args.lr_heads},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 6. Training Loop
    print("\n[Step 4/4] Starting End-to-End Multi-Task Training...")
    best_madm = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        arcface_loss_fn.train()

        total_loss = 0.0
        total_asl = 0.0
        total_arc = 0.0
        total_cm = 0.0
        t0 = time.time()

        for batch_idx, (imgs, labels, idents, query_vecs) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            idents = idents.to(device, non_blocking=True)
            query_vecs = query_vecs.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                feats, logits, vis_embs = model.forward_image(imgs)
                q_embs = model.forward_query(query_vecs)

                loss_asl = asl_loss_fn(logits, labels)
                loss_grp = group_loss_fn(logits, labels)
                loss_arc = arcface_loss_fn(vis_embs, idents)
                loss_cm = contrastive_loss_fn(q_embs, vis_embs)

                loss = loss_asl + 0.5 * loss_grp + args.lambda_arc * loss_arc + args.lambda_cm * loss_cm

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

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_asl += loss_asl.item() * bs
            total_arc += loss_arc.item() * bs
            total_cm += loss_cm.item() * bs

        scheduler.step()
        n_samples = len(train_dataset)
        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Loss: {total_loss/n_samples:.4f} (ASL: {total_asl/n_samples:.4f}, Arc: {total_arc/n_samples:.4f}, CM: {total_cm/n_samples:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s",
            flush=True,
        )

        # Validation check
        should_val = (epoch % args.eval_interval == 0) or (epoch == args.epochs) or (args.debug_samples is not None)
        if should_val:
            print(f"--- Running Validation for Epoch {epoch} ---", flush=True)
            # Unwrap DataParallel if needed for validation
            eval_model = model
            if isinstance(model.backbone, nn.DataParallel):
                eval_model = SwinPARMetric(num_attributes=40, embedding_dim=512, pretrained=False).to(device)
                eval_model.backbone.load_state_dict(model.backbone.module.state_dict())
                eval_model.classifier.load_state_dict(model.classifier.state_dict())
                eval_model.vis_projector.load_state_dict(model.vis_projector.state_dict())
                eval_model.query_encoder.load_state_dict(model.query_encoder.state_dict())

            metrics = run_validation(
                model=eval_model,
                retrieval_data=retrieval_data,
                device=device,
                platt_t=platt_t,
                platt_b=platt_b,
                lambda_metric=args.lambda_metric,
                val_subset=args.val_subset,
                batch_size=64,
            )
            print(
                f"   [Validation λ={args.lambda_metric:.2f}] -> "
                f"mADM: {metrics['mADM']:.4f} | mAP: {metrics['mAP']:.4f} | "
                f"Rank-1: {metrics['Rank-1']*100:.2f}% | Rank-5: {metrics['Rank-5']*100:.2f}% | mINP: {metrics['mINP']:.4f}",
                flush=True,
            )

            # Checkpoint saving
            if metrics["mADM"] > best_madm or args.debug_samples is not None:
                best_madm = metrics["mADM"]
                best_bb_path = os.path.join(args.save_dir, "EXP-006_best_backbone.pth")
                best_mh_path = os.path.join(args.save_dir, "EXP-006_best_metric_heads.pth")

                bb_sd = eval_model.get_backbone_state_dict()
                mh_sd = eval_model.get_metric_heads_state_dict()

                torch.save(bb_sd, best_bb_path)
                torch.save(mh_sd, best_mh_path)
                print(f"   [Checkpoint] New peak mADM: {best_madm:.4f}! Saved checkpoints to {args.save_dir}")

                # Automatically package submission
                package_submission(
                    model=eval_model,
                    output_zip=args.output_zip,
                    lambda_metric=args.lambda_metric,
                    platt_t=platt_t.tolist(),
                    platt_b=platt_b.tolist(),
                )

    print(f"\n================================================================================")
    print(f"EXP-006 End-to-End Training Completed! Peak Validation mADM: {best_madm:.4f}")
    print(f"Final Submission Zip: {args.output_zip}")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
