"""Championship Cross-Modal Retrieval Architecture.

Unifies Swin Transformer Visual Stream with Dual-Head Query Stream
(Hard Binary Attribute Adapter + Semantic Text Transformer Adapter).
Outputs continuous 1024-dimensional hyperspherical metric embeddings.
Strictly self-contained in pure PyTorch / Torchvision. Zero external NLP libraries.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import sys

# Ensure repository root is in sys.path for direct execution or torchrun
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from championship_pipeline.pseudo_descriptions import VOCAB_SIZE, MAX_TOKEN_LENGTH


class VisualStream(nn.Module):
    """Visual feature extraction stream with Swin-T backbone and 1024-D adapter."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.swin_t(weights=weights)
        in_features = self.backbone.head.in_features  # 768
        self.backbone.head = nn.Identity()

        # Multi-layer Visual Metric Adapter (768 -> 1024)
        self.visual_adapter = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )

        # Auxiliary PAR Multi-label Classification Head for Stage 1 warmup
        self.aux_classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 40),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze Swin backbone parameters during Stage 2 metric margin learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        """Unfreeze Swin backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.backbone.train()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            norm_embedding: (B, 1024) L2-normalized hyperspherical visual embedding.
            raw_embedding: (B, 1024) unnormalized embedding for margin scaling.
            aux_logits: (B, 40) attribute logits for classification loss.
        """
        features = self.backbone(x)  # (B, 768)
        aux_logits = self.aux_classifier(features)  # (B, 40)
        raw_embedding = self.visual_adapter(features)  # (B, 1024)
        norm_embedding = F.normalize(raw_embedding, p=2, dim=-1)
        return norm_embedding, raw_embedding, aux_logits


class QueryStream(nn.Module):
    """Dual-head query stream combining hard binary attributes and soft text tokens."""

    def __init__(
        self,
        num_attributes: int = 40,
        vocab_size: int = VOCAB_SIZE,
        max_length: int = MAX_TOKEN_LENGTH,
        text_dim: int = 256,
        query_out_dim: int = 512,
    ):
        super().__init__()
        self.max_length = max_length

        # 1. Hard Attribute Adapter (40 -> 512)
        self.hard_adapter = nn.Sequential(
            nn.Linear(num_attributes, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, query_out_dim),
            nn.ReLU(),
            nn.Linear(query_out_dim, query_out_dim),
        )

        # 2. Soft Semantic Text Transformer Adapter (Tokens -> 512)
        self.token_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, text_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.text_adapter = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, query_out_dim),
            nn.ReLU(),
            nn.Linear(query_out_dim, query_out_dim),
        )

    def forward(
        self,
        hard_attr: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hard_attr: (B, 40) float tensor of binary attributes.
            text_tokens: (B, L) long tensor of token IDs.
            text_mask: (B, L) bool tensor where True = valid token, False = padding.

        Returns:
            norm_query: (B, 1024) concatenated, normalized query embedding [E^h ; E^s].
            norm_hard: (B, 512) normalized hard attribute embedding.
            norm_soft: (B, 512) normalized soft text embedding.
        """
        # Hard embedding stream
        raw_hard = self.hard_adapter(hard_attr)  # (B, 512)
        norm_hard = F.normalize(raw_hard, p=2, dim=-1)

        # Soft text embedding stream
        seq_len = text_tokens.size(1)
        x_text = self.token_embedding(text_tokens) + self.pos_embedding[:, :seq_len, :]

        # PyTorch TransformerEncoder src_key_padding_mask expects True for positions to IGNORE (padding)
        if text_mask is not None:
            padding_mask = ~text_mask
        else:
            padding_mask = None

        text_feat = self.text_transformer(x_text, src_key_padding_mask=padding_mask)  # (B, L, 256)

        # Masked average pooling
        if text_mask is not None:
            mask_expanded = text_mask.unsqueeze(-1).float()  # (B, L, 1)
            sum_feat = (text_feat * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled_text = sum_feat / sum_mask  # (B, 256)
        else:
            pooled_text = text_feat.mean(dim=1)

        raw_soft = self.text_adapter(pooled_text)  # (B, 512)
        norm_soft = F.normalize(raw_soft, p=2, dim=-1)

        # Concatenate hard + soft queries to form full 1024-D query
        raw_query = torch.cat([raw_hard, raw_soft], dim=-1)  # (B, 1024)
        norm_query = F.normalize(raw_query, p=2, dim=-1)

        return norm_query, norm_hard, norm_soft


class CrossModalChampionshipNet(nn.Module):
    """Complete unified Cross-Modal Championship Retrieval Network."""

    def __init__(self, pretrained: bool = True, freeze_visual_backbone: bool = False):
        super().__init__()
        self.visual_stream = VisualStream(pretrained=pretrained, freeze_backbone=freeze_visual_backbone)
        self.query_stream = QueryStream()

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Encode gallery images into normalized 1024-D visual embeddings."""
        norm_p, _, _ = self.visual_stream(images)
        return norm_p

    def encode_query(
        self,
        hard_attr: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode queries into normalized 1024-D cross-modal embeddings."""
        norm_q, _, _ = self.query_stream(hard_attr, text_tokens, text_mask)
        return norm_q

    def forward(
        self,
        images: torch.Tensor | None = None,
        hard_attr: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Unified forward pass returning all feature representations."""
        output = {}
        if images is not None:
            norm_p, raw_p, aux_logits = self.visual_stream(images)
            output["norm_visual"] = norm_p
            output["raw_visual"] = raw_p
            output["aux_logits"] = aux_logits

        if hard_attr is not None and text_tokens is not None:
            norm_q, norm_h, norm_s = self.query_stream(hard_attr, text_tokens, text_mask)
            output["norm_query"] = norm_q
            output["norm_hard"] = norm_h
            output["norm_soft"] = norm_s

        return output
