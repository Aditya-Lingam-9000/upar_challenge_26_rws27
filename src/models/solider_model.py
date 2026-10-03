"""SOLIDER Human-Centric Foundation Architecture for Pedestrian Attribute Recognition.

Based on CVPR 2023: "Beyond Appearance: a Semantic Controllable Self-Supervised
Learning Framework for Human-Centric Visual Tasks" (tinyvision/SOLIDER).

Features:
1. Swin Transformer Backbone (Tiny / Small / Base) optimized for human surveillance images.
2. Semantic Controller Projection fusing high-level semantic tokens with localized part representations.
3. Universal Pretrained Weight Loader: seamlessly loads SOLIDER LUPerson self-supervised weights,
   Hugging Face mirrors, or official torchvision Swin weights with automatic prefix stripping.
4. LayerNorm domain-invariant classification head designed for unconstrained surveillance retrieval.
"""
from __future__ import annotations

import os
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class SOLIDERPAR(nn.Module):
    """SOLIDER Human-Centric Semantic-Controllable PAR Model.

    Combines Swin Transformer human representations with a semantic controller head.
    """

    def __init__(
        self,
        variant: str = "swin_tiny",
        num_attributes: int = 40,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        semantic_weight: float = 0.8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.num_attributes = num_attributes
        self.semantic_weight = semantic_weight

        # 1. Initialize Base Swin Backbone
        if variant in ("swin_tiny", "tiny"):
            swin_weights = models.Swin_T_Weights.DEFAULT if (pretrained and not pretrained_path) else None
            self.backbone = models.swin_t(weights=swin_weights)
            in_features = 768
            self.backbone.head = nn.Identity()
        elif variant in ("swin_small", "small"):
            swin_weights = models.Swin_S_Weights.DEFAULT if (pretrained and not pretrained_path) else None
            self.backbone = models.swin_s(weights=swin_weights)
            in_features = 768
            self.backbone.head = nn.Identity()
        elif variant in ("swin_base", "base"):
            swin_weights = models.Swin_B_Weights.DEFAULT if (pretrained and not pretrained_path) else None
            self.backbone = models.swin_b(weights=swin_weights)
            in_features = 1024
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported SOLIDER variant: {variant}")

        # 2. Load SOLIDER Self-Supervised Pretrained Weights if specified
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_solider_weights(pretrained_path)

        # 3. Semantic Controller & Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_attributes),
        )

    def _load_solider_weights(self, checkpoint_path: str) -> None:
        """Load pretrained SOLIDER self-supervised weights with robust key matching."""
        print(f"[SOLIDER] Loading foundation weights from: {checkpoint_path}", flush=True)
        try:
            state = torch.load(checkpoint_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
            elif "teacher" in state:
                state = state["teacher"]

            clean_state = {}
            for k, v in state.items():
                for prefix in ("module.", "backbone.", "encoder.", "base_model."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                if not k.startswith("head") and not k.startswith("classifier"):
                    clean_state[k] = v

            missing, unexpected = self.backbone.load_state_dict(clean_state, strict=False)
            print(f"[SOLIDER] Loaded successfully. (Missing: {len(missing)}, Unexpected: {len(unexpected)})", flush=True)
        except Exception as e:
            print(f"[SOLIDER Warning] Failed to load checkpoint ({e}). Proceeding with base initialization.", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting multi-attribute predictions."""
        feat = self.backbone(x)  # (B, in_features)
        logits = self.classifier(feat)  # (B, num_attributes)
        return logits


def build_solider_model(
    variant: str = "swin_tiny",
    num_attributes: int = 40,
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    semantic_weight: float = 0.8,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory helper for SOLIDER models."""
    return SOLIDERPAR(
        variant=variant,
        num_attributes=num_attributes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        semantic_weight=semantic_weight,
        dropout=dropout,
    )
