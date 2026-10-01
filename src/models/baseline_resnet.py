"""ResNet-50 Baseline Model for Pedestrian Attribute Recognition & Retrieval."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50PAR(nn.Module):
    """ResNet-50 Multi-Label Attribute Classifier and Semantic Feature Extractor.

    Features:
      - ImageNet-1K pretrained backbone
      - Multi-layer bottleneck attribute classification head with Dropout
      - Dual output: 40-dim attribute logits AND 512-dim L2-normalized semantic feature embedding
    """
    def __init__(
        self,
        num_attributes: int = 40,
        pretrained: bool = True,
        dropout_p: float = 0.3,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)

        # Retain convolutional trunk (stem + layer1..layer4)
        self.conv_trunk = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        in_channels = 2048  # ResNet-50 layer4 output channels
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Attribute classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_attributes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized feature vector."""
        feat_map = self.conv_trunk(x)
        feat_vec = self.global_pool(feat_map).flatten(1)
        return F.normalize(feat_vec, p=2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.conv_trunk(x)
        feat_vec = self.global_pool(feat_map).flatten(1)

        logits = self.classifier(feat_vec)

        if return_embedding:
            emb = F.normalize(feat_vec, p=2, dim=-1)
            return logits, emb

        return logits
