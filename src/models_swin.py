"""Swin Transformer Architecture for Multi-Label Pedestrian Attribute Recognition (PAR).

Designed specifically for Track 2 (Attribute-based Person Retrieval) in WACV 2027 RWS.
Provides shifted-window hierarchical self-attention representations complementary to ConvNeXt.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SwinPAR(nn.Module):
    """Swin Transformer-based Pedestrian Attribute Recognition Model.

    Utilizes torchvision Swin-Transformer (Tiny) with a LayerNorm projection
    and classification bottleneck head.
    """

    def __init__(
        self,
        variant: str = "tiny",
        num_attributes: int = 40,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.num_attributes = num_attributes

        if variant == "tiny":
            weights = models.Swin_T_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_t(weights=weights)
            in_features = self.backbone.head.in_features  # 768
            self.backbone.head = nn.Identity()
        elif variant == "small":
            weights = models.Swin_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_s(weights=weights)
            in_features = self.backbone.head.in_features  # 768
            self.backbone.head = nn.Identity()
        elif variant == "base":
            weights = models.Swin_B_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_b(weights=weights)
            in_features = self.backbone.head.in_features  # 1024
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported Swin variant: {variant}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_attributes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_attributes).
        """
        features = self.backbone(x)  # (B, in_features)
        logits = self.classifier(features)  # (B, num_attributes)
        return logits


def build_swin_model(
    variant: str = "tiny",
    num_attributes: int = 40,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory function for SwinPAR."""
    return SwinPAR(
        variant=variant,
        num_attributes=num_attributes,
        pretrained=pretrained,
        dropout=dropout,
    )
