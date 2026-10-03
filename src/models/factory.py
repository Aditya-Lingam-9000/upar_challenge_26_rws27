"""Model Factory for instantiating model architectures."""
from __future__ import annotations

import torch.nn as nn
from src.models.baseline_resnet import ResNet50PAR
from src.models.convnext_model import ConvNeXtPAR
from src.models_swin import SwinPAR
from src.models.solider_model import SOLIDERPAR


def build_model(
    model_name: str = "resnet50",
    pretrained: bool = True,
    num_attributes: int = 40,
    dropout_p: float = 0.3,
    embedding_dim: int = 512,
    pretrained_path: str | None = None,
    semantic_weight: float = 0.8,
) -> nn.Module:
    """Build model by name."""
    name = model_name.lower().strip()
    if name == "resnet50":
        return ResNet50PAR(
            num_attributes=num_attributes,
            pretrained=pretrained,
            dropout_p=dropout_p,
            embedding_dim=embedding_dim,
        )
    elif name in ["convnext_tiny", "convnext-tiny"]:
        return ConvNeXtPAR(
            variant="tiny",
            num_attributes=num_attributes,
            pretrained=pretrained,
            dropout_p=dropout_p,
            embedding_dim=embedding_dim,
        )
    elif name in ["convnext_base", "convnext-base"]:
        return ConvNeXtPAR(
            variant="base",
            num_attributes=num_attributes,
            pretrained=pretrained,
            dropout_p=dropout_p,
            embedding_dim=embedding_dim,
        )
    elif name in ["swin_t", "swin-t", "swin_tiny", "swin-tiny", "swin"]:
        return SwinPAR(
            variant="tiny",
            num_attributes=num_attributes,
            pretrained=pretrained,
            dropout=dropout_p,
        )
    elif name in ["solider_swin_tiny", "solider_tiny", "solider-swin-tiny", "solider"]:
        return SOLIDERPAR(
            variant="swin_tiny",
            num_attributes=num_attributes,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            semantic_weight=semantic_weight,
            dropout=dropout_p,
        )
    elif name in ["solider_swin_base", "solider_base", "solider-swin-base"]:
        return SOLIDERPAR(
            variant="swin_base",
            num_attributes=num_attributes,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            semantic_weight=semantic_weight,
            dropout=dropout_p,
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


