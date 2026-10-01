"""ConvNeXt Model for Pedestrian Attribute Recognition & Retrieval."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvNeXtPAR(nn.Module):
    """ConvNeXt-based Multi-Label Attribute Classifier.

    Supports 'tiny' (768-dim) and 'base' (1024-dim) variants.
    """
    def __init__(
        self,
        variant: str = "tiny",
        num_attributes: int = 40,
        pretrained: bool = True,
        dropout_p: float = 0.3,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.variant = variant.lower()

        if self.variant == "tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base = models.convnext_tiny(weights=weights)
            in_features = 768
        elif self.variant == "base":
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            base = models.convnext_base(weights=weights)
            in_features = 1024
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Attribute classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_attributes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.features(x)
        feat_vec = self.avgpool(feat_map).flatten(1)
        return F.normalize(feat_vec, p=2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.features(x)
        feat_vec = self.avgpool(feat_map).flatten(1)

        logits = self.classifier(feat_vec)

        if return_embedding:
            emb = F.normalize(feat_vec, p=2, dim=-1)
            return logits, emb

        return logits
