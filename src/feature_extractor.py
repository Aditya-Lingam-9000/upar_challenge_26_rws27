"""High-throughput Feature and Logit Extraction Engine.

Extracts both 768-D un-normalized backbone features and 40-D attribute logits
from pre-trained ConvNeXt and Swin-Transformer checkpoints.
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_model_for_extraction(
    zip_path: str,
    model_type: str = "convnext_tiny",
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Loads model directly from Codabench submission zip archive."""
    print(f"Loading {model_type} from {zip_path} onto {device}...", flush=True)
    temp_dir = f"/tmp/extract_{model_type}_{os.getpid()}"
    os.makedirs(temp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)

        weights_path = os.path.join(temp_dir, "assets", "model.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"model.pth not found in {zip_path}")

        if "swin" in model_type.lower():
            import torchvision.models as models

            class SwinWrapper(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = models.swin_t(weights=None)
                    in_features = self.backbone.head.in_features
                    self.backbone.head = nn.Identity()
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(in_features, eps=1e-6),
                        nn.Linear(in_features, 512),
                        nn.GELU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(512, 40),
                    )

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    feat = self.backbone(x)
                    logits = self.classifier(feat)
                    return feat, logits

            model = SwinWrapper()

        elif "convnext" in model_type.lower():
            import torchvision.models as models

            class ConvNeXtWrapper(nn.Module):
                def __init__(self):
                    super().__init__()
                    base = models.convnext_tiny(weights=None)
                    in_features = 768
                    self.features = base.features
                    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                    self.classifier = nn.Sequential(
                        nn.LayerNorm(in_features, eps=1e-6),
                        nn.Linear(in_features, 512),
                        nn.GELU(),
                        nn.Dropout(p=0.3),
                        nn.Linear(512, 40),
                    )

                def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    feat_map = self.features(x)
                    feat = self.avgpool(feat_map).flatten(1)
                    logits = self.classifier(feat)
                    return feat, logits

            model = ConvNeXtWrapper()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Strip potential DDP module. prefix
        cleaned_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_sd, strict=False)
        model = model.to(device).eval()
        return model
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@torch.no_grad()
def extract_dataset_features_and_logits(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts features (N, 768) and logits (N, 40) across a DataLoader."""
    model.eval()
    all_feats = []
    all_logits = []
    total = len(dataloader.dataset)
    processed = 0
    t0 = time.time()

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        elif isinstance(batch, dict):
            images = batch["image"]
        else:
            images = batch

        images = images.to(device, non_blocking=True)
        feats, logits = model(images)

        all_feats.append(feats.cpu().numpy().astype(np.float32))
        all_logits.append(logits.cpu().numpy().astype(np.float32))

        processed += images.size(0)
        if processed % 6400 == 0 or processed >= total:
            speed = processed / max(1e-5, time.time() - t0)
            print(f"  Extracted [{processed:5d}/{total}] ({processed/total*100:5.1f}%) | Speed: {speed:5.1f} img/s", flush=True)

    features = np.vstack(all_feats)
    logits = np.vstack(all_logits)
    return features, logits
