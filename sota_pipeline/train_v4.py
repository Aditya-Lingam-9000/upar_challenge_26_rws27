"""SOTA Championship V4: CLIP ViT-L/14 Cross-Modal Resumable Distributed Training Pipeline.

Architecture & Framework:
1. Visual Backbone: timm `vit_large_patch14_clip_224` (304M params, 1024-dim features).
2. Shared Metric Manifold: 768-dim shared semantic projection space.
3. Multi-label Attribute Classification Head: 40-dim unnormalized logits.
4. Continuous Attribute Prompter: MLP mapping 40 binary attributes -> 768-dim embedding.
5. Cross-Modal Text Alignment: Frozen OpenCLIP ViT-L/14 text encoder with trainable text projection.
6. Multi-Objective Championship Loss: ASL + SDM + ITC + Prompter Distillation.
7. Multi-Session Resumable Training: Multi-GPU DDP via HuggingFace Accelerate with FP16 autocast.
8. Evaluation Engine: Exact mADM, Rank-1, and Attribute Accuracy with alpha=0.70 / 0.30 fusion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import gc
import io
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

# HuggingFace Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# Optional OpenCLIP import with descriptive error handling
try:
    import open_clip
except ImportError:
    open_clip = None

# Ensure repo root is in sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# =====================================================================
# Canonical UPAR Attributes & Empirical Priors
# =====================================================================
CANONICAL_ATTRIBUTES: List[str] = [
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

UPAR_PRIORS: np.ndarray = np.array([
    0.0110, 0.9650, 0.0237, 0.4292, 0.6546, 0.3114, 0.0126, 0.6288, 0.2556, 0.0821,
    0.0199, 0.0435, 0.1621, 0.0058, 0.0105, 0.0253, 0.0694, 0.2250, 0.0497, 0.0753,
    0.3751, 0.4877, 0.1788, 0.0448, 0.0150, 0.1719, 0.0021, 0.0155, 0.0008, 0.0039,
    0.0447, 0.0017, 0.0372, 0.9038, 0.0958, 0.2085, 0.2931, 0.1394, 0.0218, 0.0425,
], dtype=np.float32)


def attrs_to_text(attrs: Union[np.ndarray, torch.Tensor, List[Union[int, float]]]) -> str:
    """Converts a 40-dimensional binary attribute vector into a natural language description.

    This template structure generates naturalistic text descriptions aligned with CLIP
    pre-training semantics for cross-modal text-image alignment.

    Args:
        attrs: 40-dimensional attribute vector (binary or continuous in [0, 1]).

    Returns:
        A natural language descriptive string.
    """
    if isinstance(attrs, torch.Tensor):
        attrs = attrs.detach().cpu().numpy()
    attrs = np.asarray(attrs, dtype=np.float32).flatten()

    parts: List[str] = []

    # 1. Age (indices 0, 1, 2)
    if attrs[0] > 0.5:
        parts.append("young")
    elif attrs[2] > 0.5:
        parts.append("elderly")
    elif attrs[1] > 0.5:
        parts.append("adult")

    # 2. Gender (index 3)
    gender = "female" if attrs[3] > 0.5 else "male"
    parts.append(gender)

    # 3. Hair (indices 4, 5, 6)
    if attrs[6] > 0.5:
        parts.append("with bald hair")
    elif attrs[5] > 0.5:
        parts.append("with long hair")
    elif attrs[4] > 0.5:
        parts.append("with short hair")

    # 4. Upper body (7: Short-length, 8-19: colors)
    upper_len = "short-sleeved" if attrs[7] > 0.5 else "long-sleeved"
    color_names = [
        "black", "blue", "brown", "green", "grey",
        "orange", "pink", "purple", "red", "white", "yellow", "other-colored"
    ]
    upper_colors = [color_names[i] for i in range(12) if attrs[8 + i] > 0.5]
    upper_color_str = "/".join(upper_colors[:2]) if upper_colors else ""
    if upper_color_str:
        parts.append(f"wearing a {upper_color_str} {upper_len} shirt")
    else:
        parts.append(f"wearing a {upper_len} shirt")

    # 5. Lower body (20: Short-length, 21-32: colors, 33: Trousers&Shorts, 34: Skirt&Dress)
    lower_len = "short" if attrs[20] > 0.5 else "long"
    lower_colors = [color_names[i] for i in range(12) if attrs[21 + i] > 0.5]
    lower_color_str = "/".join(lower_colors[:2]) if lower_colors else ""

    if attrs[34] > 0.5:
        lower_garment = "skirt or dress"
    elif attrs[33] > 0.5:
        lower_garment = "trousers" if lower_len == "long" else "shorts"
    else:
        lower_garment = "pants"

    if lower_color_str:
        parts.append(f"and {lower_color_str} {lower_len} {lower_garment}")
    else:
        parts.append(f"and {lower_len} {lower_garment}")

    # 6. Accessories (35: Backpack, 36: Bag, 37: Glasses, 38: Sunglasses, 39: Hat)
    accs: List[str] = []
    if attrs[35] > 0.5:
        accs.append("a backpack")
    if attrs[36] > 0.5:
        accs.append("a bag")
    if attrs[38] > 0.5:
        accs.append("sunglasses")
    elif attrs[37] > 0.5:
        accs.append("glasses")
    if attrs[39] > 0.5:
        accs.append("a hat")

    if accs:
        parts.append(f"carrying {', '.join(accs)}")

    return f"A surveillance photo of a {' '.join(parts)}."


# =====================================================================
# Dynamic Package Imports with High-Fidelity Standalone Fallbacks
# =====================================================================
try:
    from sota_pipeline.models_v4 import CLIPAttributeRetrievalModel, ContinuousAttributePrompter
except ImportError:
    try:
        from models_v4 import CLIPAttributeRetrievalModel, ContinuousAttributePrompter
    except ImportError:
        CLIPAttributeRetrievalModel = None
        ContinuousAttributePrompter = None

try:
    from sota_pipeline.losses_v4 import V4ChampionshipLoss
except ImportError:
    try:
        from losses_v4 import V4ChampionshipLoss
    except ImportError:
        V4ChampionshipLoss = None

try:
    from sota_pipeline.dataset_v4 import (
        UPARDataset,
        UPARDatasetV4,
        upar_collate_fn,
        build_dataloaders,
    )
except ImportError:
    try:
        from dataset_v4 import (
            UPARDataset,
            UPARDatasetV4,
            upar_collate_fn,
            build_dataloaders,
        )
    except ImportError:
        UPARDatasetV4 = None
        upar_collate_fn = None
        build_dataloaders = None

if upar_collate_fn is None:
    def upar_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([item["image"] for item in batch], dim=0)
        labels = torch.stack([item.get("labels", item.get("attributes")) for item in batch], dim=0)
        texts = [str(item.get("text", item.get("texts", ""))) for item in batch]
        indices = torch.tensor([int(item.get("index", item.get("indices", 0))) for item in batch], dtype=torch.long)
        return {
            "image": images,
            "images": images,
            "labels": labels,
            "attributes": labels,
            "text": texts,
            "texts": texts,
            "index": indices,
            "indices": indices,
        }



# Fallback class definitions if modules are not yet standalone files
if CLIPAttributeRetrievalModel is None:
    import timm

    class CLIPAttributeRetrievalModel(nn.Module):
        """Visual backbone: timm vit_large_patch14_clip_224 with 768-dim metric & 40-dim attr heads."""

        def __init__(
            self,
            pretrained: bool = True,
            embed_dim: int = 768,
            num_attrs: int = 40,
            backbone_name: str = "vit_large_patch14_clip_224.openai",
        ) -> None:
            super().__init__()
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            feat_dim = self.backbone.num_features if hasattr(self.backbone, "num_features") else 1024

            self.metric_head = nn.Sequential(
                nn.Linear(feat_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            self.attr_head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_attrs),
            )

        def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            features = self.backbone(images)
            embeds = F.normalize(self.metric_head(features), p=2, dim=-1)
            logits = self.attr_head(features)
            return embeds, logits


if ContinuousAttributePrompter is None:
    class ContinuousAttributePrompter(nn.Module):
        """MLP mapping 40 binary attributes into the 768-dim shared metric manifold."""

        def __init__(
            self,
            num_attrs: int = 40,
            embed_dim: int = 768,
            hidden_dim: int = 512,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(num_attrs, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

        def forward(self, attributes: torch.Tensor) -> torch.Tensor:
            out = self.net(attributes.float())
            return F.normalize(out, p=2, dim=-1)


if V4ChampionshipLoss is None:
    class V4ChampionshipLoss(nn.Module):
        """Unified championship loss combining ASL, SDM, ITC, and Prompter Distillation."""

        def __init__(
            self,
            gamma_neg: float = 4.0,
            gamma_pos: float = 0.0,
            clip: float = 0.05,
            temperature: float = 0.05,
            w_asl: float = 1.0,
            w_sdm: float = 0.5,
            w_itc: float = 0.3,
            w_distill: float = 0.2,
        ) -> None:
            super().__init__()
            self.gamma_neg = gamma_neg
            self.gamma_pos = gamma_pos
            self.clip = clip
            self.temperature = temperature
            self.w_asl = w_asl
            self.w_sdm = w_sdm
            self.w_itc = w_itc
            self.w_distill = w_distill

        def forward(
            self,
            img_embeds: torch.Tensor,
            attr_logits: torch.Tensor,
            smooth_labels: torch.Tensor,
            text_embeds: torch.Tensor,
            prompter_embeds: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            # 1. Multi-Label Asymmetric Loss (ASL)
            xs_pos = torch.sigmoid(attr_logits)
            xs_neg = 1.0 - xs_pos
            if self.clip > 0:
                xs_neg = (xs_neg + self.clip).clamp(max=1.0)

            los_pos = smooth_labels * torch.log(xs_pos.clamp(min=1e-8))
            los_neg = (1.0 - smooth_labels) * torch.log(xs_neg.clamp(min=1e-8))

            pt = xs_pos * smooth_labels + xs_neg * (1.0 - smooth_labels)
            one_sided_gamma = self.gamma_pos * smooth_labels + self.gamma_neg * (1.0 - smooth_labels)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss_asl = - (one_sided_w * (los_pos + los_neg)).mean()

            # 2. Similarity Distribution Matching (SDM)
            sim_i2t = (img_embeds @ text_embeds.T) / self.temperature
            sim_t2i = (text_embeds @ img_embeds.T) / self.temperature
            targets = F.softmax(sim_i2t.detach(), dim=1)
            loss_sdm_i2t = F.kl_div(F.log_softmax(sim_i2t, dim=1), targets, reduction="batchmean")
            loss_sdm_t2i = F.kl_div(F.log_softmax(sim_t2i, dim=1), targets.T, reduction="batchmean")
            loss_sdm = 0.5 * (loss_sdm_i2t + loss_sdm_t2i)

            # 3. Image-Text Contrastive (ITC / InfoNCE)
            labels_itc = torch.arange(img_embeds.size(0), device=img_embeds.device)
            loss_itc_i2t = F.cross_entropy(sim_i2t, labels_itc)
            loss_itc_t2i = F.cross_entropy(sim_t2i, labels_itc)
            loss_itc = 0.5 * (loss_itc_i2t + loss_itc_t2i)

            # 4. Prompter Distillation Loss (Cosine alignment with text and visual anchors)
            loss_distill_text = (1.0 - (prompter_embeds * text_embeds.detach()).sum(dim=-1)).mean()
            loss_distill_img = (1.0 - (prompter_embeds * img_embeds.detach()).sum(dim=-1)).mean()
            loss_distill = 0.5 * (loss_distill_text + loss_distill_img)

            # Total Weighted Loss
            total_loss = (
                self.w_asl * loss_asl
                + self.w_sdm * loss_sdm
                + self.w_itc * loss_itc
                + self.w_distill * loss_distill
            )

            loss_dict = {
                "total_loss": total_loss.item(),
                "loss_asl": loss_asl.item(),
                "loss_sdm": loss_sdm.item(),
                "loss_itc": loss_itc.item(),
                "loss_distill": loss_distill.item(),
            }
            return total_loss, loss_dict


if UPARDatasetV4 is None:
    class SurveillanceDegradation:
        """Applies realistic surveillance degradation (random downsample + JPEG compression)."""

        def __init__(
            self,
            p_downsample: float = 0.5,
            p_jpeg: float = 0.5,
            downsample_range: Tuple[int, int] = (64, 160),
            quality_range: Tuple[int, int] = (30, 75),
        ) -> None:
            self.p_downsample = p_downsample
            self.p_jpeg = p_jpeg
            self.downsample_range = downsample_range
            self.quality_range = quality_range

        def __call__(self, img: Image.Image) -> Image.Image:
            # 1. Random downsampling and upsampling
            if random.random() < self.p_downsample:
                w, h = img.size
                scale_size = random.randint(*self.downsample_range)
                down_img = img.resize((scale_size, scale_size), Image.Resampling.BILINEAR)
                img = down_img.resize((w, h), Image.Resampling.BICUBIC)

            # 2. Random JPEG compression artifacts
            if random.random() < self.p_jpeg:
                quality = random.randint(*self.quality_range)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                img = Image.open(buffer).convert("RGB")

            return img

    class UPARDatasetV4(Dataset):
        """HDF5 Dataset yielding images, 40-dim binary labels, and natural language text descriptions."""

        def __init__(
            self,
            h5_path: str,
            split: str = "train",
            target_size: Tuple[int, int] = (224, 224),
            preload_to_ram: bool = False,
        ) -> None:
            super().__init__()
            self.h5_path = h5_path
            self.split = split
            self.target_size = target_size
            self._h5_file = None

            import h5py
            with h5py.File(h5_path, "r") as f:
                if split not in f:
                    raise KeyError(f"Split '{split}' not found in HDF5 container: {h5_path}")
                grp = f[split]
                lbl_key = "labels" if "labels" in grp else "attributes"
                self.img_key = "images" if "images" in grp else "image_bytes"

                self.labels = np.array(grp[lbl_key][:], dtype=np.float32)
                self.length = len(self.labels)

                self.ram_cache: Optional[List[bytes]] = None
                if preload_to_ram:
                    img_ds = grp[self.img_key]
                    self.ram_cache = [bytes(img_ds[i]) for i in range(self.length)]

            # Transforms
            norm_mean = [0.48145466, 0.4578275, 0.40821073]
            norm_std = [0.26862954, 0.26130258, 0.27577711]

            if split == "train":
                self.transform = T.Compose([
                    T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
                    T.RandomHorizontalFlip(p=0.5),
                    SurveillanceDegradation(p_downsample=0.5, p_jpeg=0.5),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    T.ToTensor(),
                    T.Normalize(mean=norm_mean, std=norm_std),
                    T.RandomErasing(p=0.5, scale=(0.02, 0.2), value="random"),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=norm_mean, std=norm_std),
                ])

        def _get_h5(self):
            if self._h5_file is None:
                import h5py
                self._h5_file = h5py.File(self.h5_path, "r", swmr=True)
            return self._h5_file

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            if self.ram_cache is not None:
                raw_bytes = self.ram_cache[idx]
            else:
                h5_f = self._get_h5()
                raw_bytes = bytes(h5_f[self.split][self.img_key][idx])

            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            img_tensor = self.transform(img)
            labels = torch.from_numpy(self.labels[idx]).float()
            text = attrs_to_text(self.labels[idx])

            return {
                "image": img_tensor,
                "labels": labels,
                "attributes": labels,
                "text": text,
                "index": idx,
            }


# =====================================================================
# Utilities
# =====================================================================
def clear_gpu_vram() -> None:
    """Flushes PyTorch caches and forces garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def format_time(seconds: float) -> str:
    """Formats duration in seconds to a human-readable HH:MM:SS string."""
    if seconds < 0 or math.isnan(seconds):
        return "00:00:00"
    return str(timedelta(seconds=int(seconds)))


def find_dataset_h5(user_path: Optional[str] = None) -> str:
    """Auto-detects the UPAR HDF5 dataset location across Kaggle and local paths."""
    if user_path and os.path.isfile(user_path):
        return user_path

    candidate_paths = [
        "/kaggle/input/datasets/hinewk/upar-binary-dataset/upar_binary_dataset.h5",
        "/kaggle/input/upar-binary-dataset/upar_binary_dataset.h5",
        "/kaggle/input/upar-challenge-2024/upar_binary_dataset.h5",
        "/kaggle/input/upar_binary_dataset.h5",
        "/kaggle/working/upar_binary_dataset.h5",
        os.path.join(os.getcwd(), "upar_binary_dataset.h5"),
        os.path.join(os.getcwd(), "data", "upar_binary_dataset.h5"),
        os.path.expanduser("~/D/RWS2027/upar_binary_dataset.h5"),
        os.path.expanduser("~/D/RWS2027/data/upar_binary_dataset.h5"),
    ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    if os.path.isdir("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith((".h5", ".hdf5")) and "upar" in f.lower():
                    return os.path.join(root, f)

    return user_path or candidate_paths[0]


def print_hardware_info(accelerator: Accelerator) -> None:
    """Prints diagnostic system, GPU, CPU, and RAM metrics."""
    if not accelerator.is_main_process:
        return

    accelerator.print("=" * 80)
    accelerator.print("      SOTA V4 CLIP ViT-L/14 CHAMPIONSHIP RESUMABLE TRAINING ENGINE")
    accelerator.print("=" * 80)
    accelerator.print(f"PyTorch Version   : {torch.__version__}")
    accelerator.print(f"CUDA Available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        accelerator.print(f"CUDA Device Count : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            accelerator.print(f"  GPU [{i}]          : {props.name} ({vram_gb:.2f} GB VRAM, sm_{props.major}{props.minor})")
    else:
        accelerator.print("  Device           : CPU Execution")

    num_cpus = os.cpu_count() or 1
    accelerator.print(f"CPU Cores         : {num_cpus}")

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        ram_avail = psutil.virtual_memory().available / (1024 ** 3)
        accelerator.print(f"Host System RAM   : {ram_avail:.2f} GB available / {ram_gb:.2f} GB total")
    except ImportError:
        pass

    accelerator.print(f"Accelerate Device : {accelerator.device}")
    accelerator.print(f"Num Processes     : {accelerator.num_processes}")
    accelerator.print(f"Mixed Precision   : {accelerator.mixed_precision}")
    accelerator.print("=" * 80)


def clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strips 'module.' and '_orig_mod.' prefixes from DDP/compile state dicts."""
    cleaned = {}
    for k, v in state_dict.items():
        name = k
        while name.startswith("module.") or name.startswith("_orig_mod."):
            if name.startswith("module."):
                name = name[7:]
            elif name.startswith("_orig_mod."):
                name = name[10:]
        cleaned[name] = v
    return cleaned


# =====================================================================
# Validation Engine (mADM, Rank-1, Attribute Accuracy)
# =====================================================================
def evaluate(
    visual_model: nn.Module,
    prompter: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    accelerator: Optional[Accelerator] = None,
    alpha: float = 0.70,
    max_queries: int = 500,
) -> Dict[str, float]:
    """Computes exact attribute prediction accuracy, normalized mADM, and Rank-1 retrieval on validation set.

    Evaluation steps:
    1. Extract visual embeddings and 40-D attribute probabilities for validation gallery images.
    2. Sample unique ground-truth attribute queries.
    3. Generate query embeddings using `prompter(query_attrs)`.
    4. Compute attribute similarity: S_attr = 1.0 - (L1_diff / 40.0).
    5. Compute metric cosine similarity: S_metric = (cos_sim + 1.0) / 2.0.
    6. Fuse similarities: S_final = alpha * S_attr + (1 - alpha) * S_metric.
    7. Compute mADM and Rank-1.

    Args:
        visual_model: The visual model predicting (img_embeds, attr_logits).
        prompter: The continuous attribute prompter mapping 40-D query -> 768-D.
        val_loader: DataLoader for validation split.
        device: Torch device to evaluate on.
        accelerator: Optional HuggingFace Accelerator instance.
        alpha: Blending coefficient (default: 0.70 attribute weight, 0.30 metric weight).
        max_queries: Maximum validation query attributes to sample.

    Returns:
        Dict containing 'attr_acc', 'madm', and 'rank1'.
    """
    was_vm_train = visual_model.training
    was_p_train = prompter.training
    visual_model.eval()
    prompter.eval()

    clear_gpu_vram()

    all_embeds_list: List[torch.Tensor] = []
    all_logits_list: List[torch.Tensor] = []
    all_labels_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch.get("labels", batch.get("attributes"))
            if labels is not None:
                labels = labels.to(device)

            img_embeds, attr_logits = visual_model(images)

            if accelerator is not None and accelerator.num_processes > 1:
                img_embeds = accelerator.gather_for_metrics(img_embeds)
                attr_logits = accelerator.gather_for_metrics(attr_logits)
                if labels is not None:
                    labels = accelerator.gather_for_metrics(labels)

            all_embeds_list.append(img_embeds.cpu())
            all_logits_list.append(attr_logits.cpu())
            if labels is not None:
                all_labels_list.append(labels.cpu())

    is_main = True if accelerator is None else accelerator.is_main_process

    if not is_main:
        visual_model.train(was_vm_train)
        prompter.train(was_p_train)
        clear_gpu_vram()
        return {"attr_acc": 0.0, "madm": 0.0, "rank1": 0.0}

    gal_embeds = torch.cat(all_embeds_list, dim=0).float().numpy()
    gal_logits = torch.cat(all_logits_list, dim=0).float().numpy()
    gal_labels = torch.cat(all_labels_list, dim=0).float().numpy() if all_labels_list else None

    gal_probs = 1.0 / (1.0 + np.exp(-gal_logits))
    del all_embeds_list, all_logits_list, all_labels_list
    clear_gpu_vram()

    # 1. Attribute Prediction Accuracy
    if gal_labels is not None:
        binary_preds = (gal_probs > 0.5).astype(np.float32)
        attr_acc = float((binary_preds == gal_labels).mean() * 100.0)
    else:
        attr_acc = 0.0

    if gal_labels is None or len(gal_labels) == 0:
        visual_model.train(was_vm_train)
        prompter.train(was_p_train)
        return {"attr_acc": attr_acc, "madm": 0.0, "rank1": 0.0}

    # 2. Sample Queries from Unique Validation Labels
    unique_queries = np.unique(gal_labels, axis=0)
    num_queries = min(len(unique_queries), max_queries)
    if len(unique_queries) > num_queries:
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(unique_queries), size=num_queries, replace=False)
        query_attrs = unique_queries[sample_indices]
    else:
        query_attrs = unique_queries

    # 3. Query Embeddings via Prompter
    q_tensor = torch.from_numpy(query_attrs).float().to(device)
    with torch.no_grad():
        q_embeds_t = prompter(q_tensor)
        q_embeds = F.normalize(q_embeds_t, p=2, dim=-1).cpu().float().numpy()

    # 4. Pairwise Similarities
    # Metric Cosine Similarity in [0, 1]
    cos_sim = np.dot(q_embeds, gal_embeds.T)  # [Q, G]
    s_metric = (cos_sim + 1.0) / 2.0

    # Calibrated Attribute Similarity: 1.0 - (L1 / 40.0)
    l1_attr = np.abs(query_attrs[:, None, :] - gal_probs[None, :, :]).sum(axis=2)  # [Q, G]
    s_attr = np.maximum(0.0, 1.0 - (l1_attr / 40.0))

    # Fused Retrieval Score (alpha=0.70 attribute + 0.30 metric)
    s_final = alpha * s_attr + (1.0 - alpha) * s_metric
    rankings = np.argsort(-s_final, axis=1)  # [Q, G]

    # 5. Ground Truth Degree of Match (DoM) & Exact Matches
    l1_gt = np.abs(query_attrs[:, None, :] - gal_labels[None, :, :]).sum(axis=2)  # [Q, G]
    exact_matches = (l1_gt == 0)
    dom = np.maximum(0.0, 1.0 - (l1_gt / 40.0))

    # Rank-1 Exact Match Score
    top1_indices = rankings[:, 0]
    rank1_matches = [exact_matches[i, top1_indices[i]] for i in range(len(query_attrs))]
    rank1 = float(np.mean(rank1_matches) * 100.0)

    # Normalized Degree of Match (nDoM)
    mean_dom = dom.mean(axis=1, keepdims=True)
    max_dom = dom.max(axis=1, keepdims=True)
    denom = np.maximum(max_dom - mean_dom, 1e-8)
    ndom = np.maximum(0.0, (dom - mean_dom) / denom)

    adm_scores: List[float] = []
    for i in range(len(query_attrs)):
        q_order = rankings[i]
        q_exact = exact_matches[i, q_order]
        ranks = np.where(q_exact)[0] + 1
        if len(ranks) > 0:
            cum_ndom = np.cumsum(ndom[i, q_order], dtype=np.float64)
            adm_scores.append(float(np.mean(cum_ndom[ranks - 1] / ranks)))
        else:
            # Fallback for queries with no exact zero-error match: top 10 average DoM
            top_k = min(10, len(q_order))
            adm_scores.append(float(np.mean(dom[i, q_order[:top_k]])))

    madm = float(np.mean(adm_scores)) if adm_scores else 0.0

    visual_model.train(was_vm_train)
    prompter.train(was_p_train)
    clear_gpu_vram()

    return {
        "attr_acc": attr_acc,
        "madm": madm,
        "rank1": rank1,
    }


# =====================================================================
# Checkpoint Saving & Loading
# =====================================================================
def save_checkpoint(
    output_dir: str,
    epoch: int,
    visual_model: nn.Module,
    prompter: nn.Module,
    text_projection: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accelerator: Accelerator,
    best_madm: float,
    global_step: int,
    is_best: bool = False,
    is_emergency: bool = False,
) -> None:
    """Saves complete resumable training checkpoint containing all states."""
    if not accelerator.is_main_process:
        return

    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "visual_model": accelerator.unwrap_model(visual_model).state_dict(),
        "prompter": accelerator.unwrap_model(prompter).state_dict(),
        "text_projection": accelerator.unwrap_model(text_projection).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_madm": best_madm,
        "global_step": global_step,
        "scaler": accelerator.scaler.state_dict() if (hasattr(accelerator, "scaler") and accelerator.scaler) else None,
    }

    if is_emergency:
        path = os.path.join(output_dir, "checkpoint_emergency.pt")
        torch.save(checkpoint, path)
        accelerator.print(f"[EMERGENCY] Saved emergency checkpoint to: {path}")
        return

    # 1. Periodic checkpoint
    path_epoch = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, path_epoch)

    # 2. Latest checkpoint (for seamless multi-session resume)
    path_latest = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, path_latest)

    # 3. Best checkpoint
    if is_best:
        path_best = os.path.join(output_dir, "checkpoint_best.pt")
        torch.save(checkpoint, path_best)
        accelerator.print(f"[BEST] New best validation checkpoint saved: {path_best} (mADM: {best_madm:.4f})")


def load_resumable_checkpoint(
    checkpoint_path: str,
    visual_model: nn.Module,
    prompter: nn.Module,
    text_projection: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    accelerator: Accelerator,
) -> Tuple[int, float, int]:
    """Loads a full multi-session checkpoint, restoring all weights, optimizer, and scheduler states."""
    accelerator.print(f"[RESUME] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    unwrapped_visual = accelerator.unwrap_model(visual_model)
    v_missing, v_unexp = unwrapped_visual.load_state_dict(clean_state_dict(checkpoint["visual_model"]), strict=False)
    accelerator.print(f"[RESUME] Visual Model loaded (Missing: {len(v_missing)}, Unexpected: {len(v_unexp)})")

    unwrapped_prompter = accelerator.unwrap_model(prompter)
    p_missing, p_unexp = unwrapped_prompter.load_state_dict(clean_state_dict(checkpoint["prompter"]), strict=False)
    accelerator.print(f"[RESUME] Prompter loaded (Missing: {len(p_missing)}, Unexpected: {len(p_unexp)})")

    if "text_projection" in checkpoint:
        unwrapped_tp = accelerator.unwrap_model(text_projection)
        t_missing, t_unexp = unwrapped_tp.load_state_dict(clean_state_dict(checkpoint["text_projection"]), strict=False)
        accelerator.print(f"[RESUME] Text Projection loaded (Missing: {len(t_missing)}, Unexpected: {len(t_unexp)})")

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        accelerator.print("[RESUME] Optimizer state successfully restored.")

    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        accelerator.print("[RESUME] LR Scheduler state successfully restored.")

    if hasattr(accelerator, "scaler") and accelerator.scaler is not None and checkpoint.get("scaler") is not None:
        accelerator.scaler.load_state_dict(checkpoint["scaler"])
        accelerator.print("[RESUME] GradScaler state successfully restored.")

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_madm = float(checkpoint.get("best_madm", 0.0))
    global_step = int(checkpoint.get("global_step", 0))

    accelerator.print(
        f"[RESUME] Resumed from epoch {start_epoch - 1}. Next epoch: {start_epoch} | "
        f"Best mADM: {best_madm:.4f} | Global Step: {global_step}"
    )
    return start_epoch, best_madm, global_step


# =====================================================================
# Main Training Function
# =====================================================================
def train(args: Optional[argparse.Namespace] = None, **kwargs) -> None:
    """Master training entrypoint for V4 CLIP ViT-L/14 Championship Pipeline.

    Args:
        args: Parsed command-line arguments.
        **kwargs: Programmatic overrides for arguments.
    """
    if args is None:
        args = parse_args([])
    for k, v in kwargs.items():
        if hasattr(args, k):
            setattr(args, k, v)

    # 1. Setup Accelerator
    set_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=args.grad_accum,
    )

    print_hardware_info(accelerator)

    # 2. Check OpenCLIP Availability
    if open_clip is None:
        raise ImportError(
            "CRITICAL: `open_clip` package is not installed.\n"
            "OpenCLIP ViT-L/14 text encoder is required for cross-modal training.\n"
            "Install it on Kaggle using: `pip install open_clip_torch`"
        )

    # 3. Dataset Path Auto-Detection
    h5_path = find_dataset_h5(args.h5_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"Could not locate UPAR HDF5 dataset at '{h5_path}'.\n"
            f"Please verify Kaggle dataset path or pass `--h5_path /path/to/upar_binary_dataset.h5`."
        )
    accelerator.print(f"[DATASET] Using HDF5 container: {h5_path}")

    # 4. Instantiate Datasets and DataLoaders
    train_dataset = UPARDatasetV4(h5_path=h5_path, split="train", target_size=(224, 224))
    val_dataset = UPARDatasetV4(h5_path=h5_path, split="val", target_size=(224, 224))

    accelerator.print(f"[DATASET] Train Samples: {len(train_dataset):,} | Val Samples: {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=upar_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=upar_collate_fn,
    )

    # 5. Initialize Models
    accelerator.print("[MODEL] Initializing CLIPAttributeRetrievalModel (vit_large_patch14_clip_224)...")
    visual_model = CLIPAttributeRetrievalModel(pretrained=True, embed_dim=768, num_attrs=40)

    accelerator.print("[MODEL] Initializing ContinuousAttributePrompter (40-D -> 768-D MLP)...")
    prompter = ContinuousAttributePrompter(num_attrs=40, embed_dim=768, hidden_dim=512)

    accelerator.print("[MODEL] Loading OpenCLIP ViT-L-14 frozen text encoder ('openai')...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # Freeze OpenCLIP text encoder completely
    clip_text_encoder = clip_model
    for param in clip_text_encoder.parameters():
        param.requires_grad = False
    clip_text_encoder.eval()
    clip_text_encoder = clip_text_encoder.to(accelerator.device)

    # Trainable Text Projection Head: 768 -> 768 shared metric space
    text_projection = nn.Sequential(
        nn.Linear(768, 768),
        nn.LayerNorm(768),
    )

    # 6. Parameter Groups & AdamW Optimizer
    backbone_params = []
    attr_head_params = []
    metric_head_params = []

    if hasattr(visual_model, "backbone"):
        backbone_params = [p for p in visual_model.backbone.parameters() if p.requires_grad]
    if hasattr(visual_model, "attr_head"):
        attr_head_params = [p for p in visual_model.attr_head.parameters() if p.requires_grad]
    if hasattr(visual_model, "metric_head"):
        metric_head_params = [p for p in visual_model.metric_head.parameters() if p.requires_grad]

    # Catch any remaining parameters
    claimed = set(backbone_params) | set(attr_head_params) | set(metric_head_params)
    for n, p in visual_model.named_parameters():
        if p.requires_grad and p not in claimed:
            metric_head_params.append(p)
            claimed.add(p)

    prompter_params = [p for p in prompter.parameters() if p.requires_grad]
    text_proj_params = [p for p in text_projection.parameters() if p.requires_grad]

    param_groups = [
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay, "name": "backbone"},
        {"params": attr_head_params, "lr": args.head_lr, "weight_decay": 0.01, "name": "attr_head"},
        {"params": metric_head_params, "lr": args.head_lr, "weight_decay": 0.01, "name": "metric_head"},
        {"params": prompter_params, "lr": args.head_lr, "weight_decay": 0.01, "name": "prompter"},
        {"params": text_proj_params, "lr": args.head_lr, "weight_decay": 0.01, "name": "text_projection"},
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # 7. Learning Rate Scheduler (Linear Warmup + Cosine Annealing)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * update_steps_per_epoch
    warmup_steps = args.warmup_epochs * update_steps_per_epoch

    def build_lr_lambdas() -> List[Any]:
        lambdas = []
        for group in optimizer.param_groups:
            init_lr = group["lr"]
            min_factor = args.eta_min / max(init_lr, 1e-9)

            def make_lambda(m_fac: float):
                return lambda step: (
                    float(step) / float(max(1, warmup_steps))
                    if step < warmup_steps
                    else max(m_fac, 0.5 * (1.0 + math.cos(math.pi * float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))))
                )

            lambdas.append(make_lambda(min_factor))
        return lambdas

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambdas())

    # 8. Loss Criterion
    criterion = V4ChampionshipLoss()

    # 9. Accelerate Prepare
    visual_model, prompter, text_projection, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        visual_model, prompter, text_projection, optimizer, train_loader, val_loader, scheduler
    )

    # 10. Resumption Handling
    start_epoch = 1
    best_madm = 0.0
    global_step = 0

    resume_candidate = args.resume_from
    if not resume_candidate:
        default_latest = os.path.join(args.output_dir, "checkpoint_latest.pt")
        if os.path.isfile(default_latest):
            resume_candidate = default_latest

    if resume_candidate and os.path.isfile(resume_candidate):
        start_epoch, best_madm, global_step = load_resumable_checkpoint(
            checkpoint_path=resume_candidate,
            visual_model=visual_model,
            prompter=prompter,
            text_projection=text_projection,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
        )

    all_trainable_params = (
        [p for p in visual_model.parameters() if p.requires_grad]
        + [p for p in prompter.parameters() if p.requires_grad]
        + [p for p in text_projection.parameters() if p.requires_grad]
    )

    # 11. Multi-Epoch Training Loop with KeyboardInterrupt Handling
    accelerator.print(f"[TRAIN] Commencing training from epoch {start_epoch} to {args.epochs}...")
    training_start_time = time.time()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            visual_model.train()
            prompter.train()
            text_projection.train()

            epoch_start_time = time.time()
            running_loss = 0.0
            running_loss_breakdown: Dict[str, float] = {}
            steps_in_epoch = len(train_loader)

            for step, batch in enumerate(train_loader):
                images = batch["image"]
                labels = batch.get("labels", batch.get("attributes"))
                texts = batch.get("text", batch.get("texts"))

                if texts is None or (isinstance(texts, (list, tuple)) and len(texts) == 0):
                    texts = [attrs_to_text(lbl) for lbl in labels]

                # Label smoothing for attribute targets
                smooth_labels = labels * (1.0 - args.label_smoothing) + (args.label_smoothing / 2.0)

                with accelerator.accumulate(visual_model, prompter, text_projection):
                    # Visual forward pass
                    img_embeds, attr_logits = visual_model(images)

                    # OpenCLIP text encoder (frozen, no gradients)
                    with torch.no_grad():
                        text_tokens = tokenizer(texts).to(accelerator.device)
                        text_features = clip_text_encoder.encode_text(text_tokens).float()

                    # Trainable text projection into 768-D shared metric space
                    text_embeds = F.normalize(text_projection(text_features), p=2, dim=-1)

                    # Prompter forward pass (using original binary labels)
                    prompter_embeds = prompter(labels)

                    # Compute unified championship loss
                    loss_out = criterion(
                        img_embeds, attr_logits, smooth_labels, text_embeds, prompter_embeds
                    )
                    if isinstance(loss_out, tuple):
                        total_loss, loss_dict = loss_out
                    else:
                        total_loss = loss_out
                        loss_dict = {"total_loss": total_loss.item()}

                    accelerator.backward(total_loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(all_trainable_params, max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                # Update Running Loss Metrics
                running_loss += total_loss.item()
                for k, v in loss_dict.items():
                    running_loss_breakdown[k] = running_loss_breakdown.get(k, 0.0) + v

                # Logging every 50 steps
                if (step + 1) % 50 == 0 or (step + 1) == steps_in_epoch:
                    elapsed = time.time() - epoch_start_time
                    speed = (step + 1) / max(1e-5, elapsed)
                    effective_imgs = speed * args.batch_size * accelerator.num_processes

                    cur_step = (step + 1)
                    avg_loss = running_loss / cur_step
                    loss_sub = " | ".join(
                        f"{k.replace('loss_', '').upper()}: {running_loss_breakdown[k] / cur_step:.3f}"
                        for k in sorted(running_loss_breakdown.keys())
                        if k != "total_loss"
                    )

                    backbone_lr = optimizer.param_groups[0]["lr"]
                    head_lr = optimizer.param_groups[1]["lr"]

                    remaining_steps = (args.epochs - epoch) * steps_in_epoch + (steps_in_epoch - cur_step)
                    eta_sec = remaining_steps / max(1e-5, speed)

                    accelerator.print(
                        f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
                        f"Step [{cur_step:04d}/{steps_in_epoch:04d}] | "
                        f"Loss: {avg_loss:.4f} ({loss_sub}) | "
                        f"BB-LR: {backbone_lr:.2e} | Head-LR: {head_lr:.2e} | "
                        f"{effective_imgs:.1f} img/s | ETA: {format_time(eta_sec)}"
                    )

            epoch_duration = time.time() - epoch_start_time
            accelerator.print(
                f"[EPOCH COMPLETED] Epoch {epoch}/{args.epochs} finished in {format_time(epoch_duration)}."
            )

            # 12. Periodic Validation
            is_best = False
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                accelerator.print(f"[EVAL] Running validation evaluation for Epoch {epoch}...")
                metrics = evaluate(
                    visual_model=accelerator.unwrap_model(visual_model),
                    prompter=accelerator.unwrap_model(prompter),
                    val_loader=val_loader,
                    device=accelerator.device,
                    accelerator=accelerator,
                    alpha=0.70,
                    max_queries=500,
                )

                if accelerator.is_main_process:
                    madm = metrics.get("madm", 0.0)
                    attr_acc = metrics.get("attr_acc", 0.0)
                    rank1 = metrics.get("rank1", 0.0)
                    accelerator.print(
                        f"--> [VAL METRICS] Epoch {epoch:03d}: mADM: {madm:.4f} | "
                        f"Attr Acc: {attr_acc:.2f}% | Rank-1: {rank1:.2f}% (Best mADM: {best_madm:.4f})"
                    )

                    if madm > best_madm:
                        best_madm = madm
                        is_best = True

            # 13. Periodic Checkpoint Saving
            if epoch % args.save_every == 0 or epoch == args.epochs or is_best:
                save_checkpoint(
                    output_dir=args.output_dir,
                    epoch=epoch,
                    visual_model=visual_model,
                    prompter=prompter,
                    text_projection=text_projection,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    accelerator=accelerator,
                    best_madm=best_madm,
                    global_step=global_step,
                    is_best=is_best,
                )

            # Synchronize distributed ranks at epoch boundary
            accelerator.wait_for_everyone()

        # 14. Save Final Best Submission Model
        if accelerator.is_main_process:
            final_path = os.path.join(args.output_dir, "best_model_v4.pt")
            unwrapped_v = accelerator.unwrap_model(visual_model)
            unwrapped_p = accelerator.unwrap_model(prompter)
            torch.save(
                {
                    "visual_model": unwrapped_v.state_dict(),
                    "prompter": unwrapped_p.state_dict(),
                    "best_madm": best_madm,
                    "epoch": args.epochs,
                },
                final_path,
            )
            total_duration = time.time() - training_start_time
            accelerator.print("=" * 80)
            accelerator.print(f"[COMPLETE] Training completed across {args.epochs} epochs in {format_time(total_duration)}!")
            accelerator.print(f"[COMPLETE] Final submission weights written to: {final_path}")
            accelerator.print(f"[COMPLETE] Best Validation mADM: {best_madm:.4f}")
            accelerator.print("=" * 80)

    except KeyboardInterrupt:
        accelerator.print("\n" + "!" * 80)
        accelerator.print("[INTERRUPT] Caught KeyboardInterrupt / SIGINT signal.")
        accelerator.print("[INTERRUPT] Saving emergency resumable checkpoint before termination...")
        accelerator.print("!" * 80)
        save_checkpoint(
            output_dir=args.output_dir,
            epoch=epoch,
            visual_model=visual_model,
            prompter=prompter,
            text_projection=text_projection,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            best_madm=best_madm,
            global_step=global_step,
            is_emergency=True,
        )
        accelerator.print("[INTERRUPT] Emergency state safely saved. Ready to resume next session.")
        sys.exit(0)


# =====================================================================
# CLI Argument Parser
# =====================================================================
def parse_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for V4 training."""
    parser = argparse.ArgumentParser(
        description="SOTA Championship V4: CLIP ViT-L/14 Resumable Distributed Training Loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset & I/O
    parser.add_argument(
        "--h5_path",
        type=str,
        default=None,
        help="Path to HDF5 dataset container (default: auto-detect on Kaggle/local)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./v4_outputs",
        help="Directory to persist checkpoints and final export models",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to existing checkpoint to resume training from (default: None)",
    )

    # Optimization & Schedule
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Total number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Per-GPU batch size",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=5e-6,
        help="Learning rate for pretrained ViT-L/14 backbone",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=5e-4,
        help="Learning rate for heads (attribute head, metric head, prompter, text_projection)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for backbone AdamW optimization",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Linear warmup duration in epochs",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-7,
        help="Minimum learning rate for cosine scheduler",
    )

    # Evaluation & Checkpointing Cadence
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Evaluate on validation set every N epochs",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor applied to multi-label targets",
    )

    # General Configuration
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker subprocesses",
    )

    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
