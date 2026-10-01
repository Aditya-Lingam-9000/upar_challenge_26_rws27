"""Submission Packaging Engine for Codabench (Competition #18196, Track 2).

Generates a standalone, air-gapped, zero-internet-dependent zip submission
conforming strictly to the Codabench ingestion program specifications:
  - run.py with rank_gallery(sample) and load_model() entrypoints
  - metadata.yaml
  - assets/model.pth (cleaned, stripped, minimal inference weights)
  - Pure PyTorch ResNet-50 architecture vendored inside run.py
  - Bulletproof against PyTorch 2.4/2.6 weights_only=True security checks
  - Progress logging with flush=True per Codabench manual
  - NaN/Inf distance sanitization
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
import torch


RUN_PY_TEMPLATE = '''"""Codabench Track 2 Submission Entrypoint.

Self-contained pure PyTorch inference module.
Strictly compatible with pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime.
Zero external package dependencies beyond torch, torchvision, numpy, and PIL.
Conforms to Codabench Ingestion Contract:
  - load_model()
  - rank_gallery(sample: dict) -> {"distances": np.ndarray}
"""
from __future__ import annotations

import os
from typing import Any
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# Canonical list of 40 UPAR attributes in challenge training order
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

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WEIGHTS_PATH = os.path.join(ASSET_DIR, "model.pth")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL: nn.Module | None = None
_TRANSFORM = T.Compose([
    T.Resize((256, 128), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ResNet50PAR(nn.Module):
    def __init__(self, num_attributes: int = 40):
        super().__init__()
        base = models.resnet50(weights=None)
        self.conv_trunk = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        in_channels = 2048
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_attributes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.conv_trunk(x)
        feat_vec = self.global_pool(feat_map).flatten(1)
        return self.classifier(feat_vec)


def load_model() -> None:
    """Load model weights into global _MODEL instance. Safe for PyTorch 2.4+."""
    global _MODEL
    if _MODEL is not None:
        return

    print(f"Loading weights from {WEIGHTS_PATH} onto device {_DEVICE}...", flush=True)
    model = ResNet50PAR(num_attributes=40)

    # weights_only=True ensures secure, safe unpickling on PyTorch 2.4/2.6
    try:
        ckpt = torch.load(WEIGHTS_PATH, map_location=_DEVICE, weights_only=True)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only argument
        ckpt = torch.load(WEIGHTS_PATH, map_location=_DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Clean state dict keys if needed
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        elif k.startswith("embedding_head."):
            continue
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    print("Model successfully loaded and initialized in eval mode.", flush=True)


@torch.no_grad()
def predict_attributes(gallery: list[dict[str, Any]], attribute_names: list[str]) -> np.ndarray:
    """Predict continuous attribute probabilities for all gallery images."""
    if _MODEL is None:
        load_model()

    num_samples = len(gallery)
    if num_samples == 0:
        return np.empty((0, len(attribute_names)), dtype=np.float32)

    batch_size = 64
    all_probs = []

    print(f"Starting attribute inference on {num_samples} gallery images (batch_size={batch_size})...", flush=True)

    for i in range(0, num_samples, batch_size):
        batch_items = gallery[i : i + batch_size]
        batch_tensors = []
        for item in batch_items:
            img_path = item["image_path"]
            try:
                with Image.open(img_path) as img:
                    tensor = _TRANSFORM(img.convert("RGB"))
            except Exception:
                # Robust fallback for unreadable/corrupted images: zero tensor
                tensor = torch.zeros(3, 256, 128, dtype=torch.float32)
            batch_tensors.append(tensor)

        batch_tensor = torch.stack(batch_tensors, dim=0).to(_DEVICE)
        logits = _MODEL(batch_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

        processed = i + len(batch_items)
        if (i // batch_size) % 50 == 0 or processed >= num_samples:
            print(f"[{processed}/{num_samples}] Gallery images processed ({processed / num_samples * 100:.1f}%)", flush=True)

    raw_probs = np.vstack(all_probs).astype(np.float32)

    # Align columns to match requested attribute_names if ordering differs
    if attribute_names != CANONICAL_ATTRIBUTES:
        print("Realigning output probabilities to match sample attribute_names...", flush=True)
        attr_to_idx = {name: idx for idx, name in enumerate(CANONICAL_ATTRIBUTES)}
        col_indices = [attr_to_idx.get(name, -1) for name in attribute_names]
        aligned_probs = np.zeros((raw_probs.shape[0], len(attribute_names)), dtype=np.float32)
        for target_col, src_col in enumerate(col_indices):
            if src_col >= 0:
                aligned_probs[:, target_col] = raw_probs[:, src_col]
            else:
                aligned_probs[:, target_col] = 0.5  # Neutral fallback for unrecognized attribute
        return aligned_probs

    return raw_probs


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """Official Codabench entrypoint.

    Args:
      sample: dict with:
        - "gallery": list of dicts with "image_path"
        - "attribute_names": list of 40 attribute names
        - "queries": (Q, 40) binary query array/list

    Returns:
      {"distances": distances}: (Q, G) float32 numpy array.
    """
    gallery = sample["gallery"]
    attribute_names = sample["attribute_names"]
    queries = np.asarray(sample["queries"], dtype=np.float32)

    probs = predict_attributes(gallery, attribute_names)

    # Vectorized continuous L1 distance between binary query q and probabilities p:
    # sum_a |q_a - p_a| = sum(p) + q @ (1.0 - 2.0 * p).T
    # Shape: (Q, G)
    distances = probs.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * probs).T

    # Sanitize distances: no NaN or Inf can ever fail ingestion
    distances = np.nan_to_num(distances, nan=20.0, posinf=40.0, neginf=0.0).astype(np.float32)

    print(f"Rank gallery complete. Computed distance matrix shape: {distances.shape}", flush=True)
    return {"distances": distances}
'''

METADATA_YAML = """command: python3 -c "print('Submission package ready. The ingestion program imports run.py')"
"""


def package_submission(
    checkpoint_path: str,
    output_zip_path: str = "submissions/EXP-001_submission.zip",
    staging_dir: str = "submissions/staging",
) -> str:
    """Create Codabench zip package."""
    os.makedirs(staging_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_zip_path), exist_ok=True)

    assets_dir = os.path.join(staging_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Write run.py
    run_py_path = os.path.join(staging_dir, "run.py")
    with open(run_py_path, "w", encoding="utf-8") as f:
        f.write(RUN_PY_TEMPLATE)

    # 2. Write metadata.yaml
    metadata_path = os.path.join(staging_dir, "metadata.yaml")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(METADATA_YAML)

    # 3. Clean and save optimized model weights (strip optimizer state dict)
    print("Extracting and cleaning model weights for minimal deployment package...")
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
    elif isinstance(raw_ckpt, dict):
        state_dict = raw_ckpt
    else:
        raise ValueError(f"Invalid checkpoint format: {type(raw_ckpt)}")

    clean_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state[k[7:]] = v.cpu()
        elif k.startswith("embedding_head."):
            continue
        else:
            clean_state[k] = v.cpu()

    dest_weights = os.path.join(assets_dir, "model.pth")
    torch.save(clean_state, dest_weights)
    print(f"Saved optimized deployment weights ({os.path.getsize(dest_weights) / (1024*1024):.2f} MB)")

    # 4. Zip the archive (without folder prefix, files strictly at root of zip)
    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(run_py_path, arcname="run.py")
        zf.write(metadata_path, arcname="metadata.yaml")
        zf.write(dest_weights, arcname="assets/model.pth")

    print(f"Successfully packaged submission to: {output_zip_path}")
    print(f"Archive size: {os.path.getsize(output_zip_path) / (1024 * 1024):.2f} MB")

    # Clean staging
    shutil.rmtree(staging_dir, ignore_errors=True)

    return output_zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Codabench Submission")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="submissions/EXP-001_submission.zip", help="Output zip")
    args = parser.parse_args()

    package_submission(args.checkpoint, args.output)
