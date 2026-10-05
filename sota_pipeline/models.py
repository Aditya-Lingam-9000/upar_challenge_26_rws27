"""SOTA Championship Neural Architecture for Cross-Modal Person Retrieval.

Unconstrained dual-stream architecture featuring:
1. VisualStream: EVA-02 / ViT-L with Anatomical 3-Strip Partitioning (PCB)
   and learnable Generalized Mean (GeM) pooling.
2. TextStream: High-capacity contextual text encoder with metric projection.
3. AuxiliaryAttributeHead: Multi-label classification branch trained with ASL.
4. CrossModalMLMHead: Training-time cross-attention decoder for fine-grained
   token-to-patch alignment (completely bypassed during test time).
5. SOTAChampionshipModel: Master unconstrained system class.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """Learnable Generalized Mean (GeM) Pooling layer."""

    def __init__(self, p_init: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] Sequence of patch tokens.
        Returns:
            [B, D] Pooled vector.
        """
        # Clamp to avoid numerical underflow before pow
        x_clamped = torch.clamp(x, min=self.eps)
        p = self.p.clamp(min=1.0, max=10.0)
        return (x_clamped.pow(p).mean(dim=1)).pow(1.0 / p)


class VisualStream(nn.Module):
    """
    High-capacity visual encoder with Anatomical 3-Strip PCB Partitioning.
    Extracts global CLS token and 3 localized body strips (Head, Torso, Legs).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_attrs: int = 40,
        img_height: int = 336,
        img_width: int = 168,
        patch_size: int = 14,
        backbone_type: str = "auto",
        gem_p_init: float = 3.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attrs = num_attrs
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size

        self.num_rows = img_height // patch_size
        self.num_cols = img_width // patch_size
        self.total_patches = self.num_rows * self.num_cols

        # Initialize visual backbone
        self.backbone, self.in_dim = self._initialize_backbone(backbone_type)

        # GeM Pooling for 3 Anatomical Strips
        self.gem_head = GeMPooling(p_init=gem_p_init)
        self.gem_torso = GeMPooling(p_init=gem_p_init)
        self.gem_legs = GeMPooling(p_init=gem_p_init)

        # Projection from in_dim to embed_dim if needed
        self.patch_proj = nn.Identity() if self.in_dim == embed_dim else nn.Linear(self.in_dim, embed_dim)

        # Non-linear Metric Fusion Bottleneck: 4096-D -> 1024-D
        # (CLS token + 3 strip vectors: embed_dim * 4)
        self.metric_proj = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Auxiliary Attribute Classifier Head
        self.aux_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, num_attrs)
        )

    def _initialize_backbone(self, backbone_type: str) -> Tuple[nn.Module, int]:
        """Loads SOTA pretrained Vision Transformer with fallback."""
        # 1. Try open_clip EVA-02 / ViT-L
        if backbone_type in ("auto", "eva02", "open_clip"):
            try:
                import open_clip
                model_name = "EVA02-E-14-plus" if backbone_type == "eva02" else "ViT-L-14"
                pretrained = "laion2b_s34b_b88k"
                clip_model, _, _ = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, force_quick_gelu=False
                )
                print(f"[VisualStream] Successfully loaded open_clip {model_name}!")
                return clip_model.visual, clip_model.visual.output_dim
            except Exception:
                pass

        # 2. Try torchvision Vision Transformer (ViT-B/16 or ViT-L/16)
        try:
            import torchvision.models as tv_models
            vit = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
            # Remove classification head
            in_features = vit.heads.head.in_features
            vit.heads = nn.Identity()
            print("[VisualStream] Loaded torchvision ViT-B/16 backbone.")
            return vit, in_features
        except Exception:
            pass

        # 3. Built-in Standalone Vision Transformer (Self-Contained Fallback)
        print("[VisualStream] Initializing standalone Vision Transformer encoder.")
        fallback = StandaloneViT(
            img_height=self.img_height,
            img_width=self.img_width,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_layers=12,
            num_heads=16
        )
        return fallback, self.embed_dim

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Extracts fused metric embedding, attribute logits, spatial patch tokens,
        and individual strip embeddings.
        
        Args:
            images: [B, 3, H, W] Input image batch.
        Returns:
            metric_emb: [B, embed_dim] L2-normalized continuous metric representation.
            attr_logits: [B, num_attrs] Raw classification logits for ASL.
            spatial_patches: [B, total_patches, embed_dim] Spatial patch tokens for MLM.
            strip_embeds: Tuple of (head, torso, legs) embeddings [B, embed_dim].
        """
        B = images.shape[0]

        # Extract features through backbone
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(images)
            if isinstance(feats, dict):
                cls_token = feats.get("x_norm_clstoken", feats.get("cls_token"))
                patch_tokens = feats.get("x_norm_patchtokens", feats.get("patch_tokens"))
            else:
                cls_token = feats[:, 0, :]
                patch_tokens = feats[:, 1:, :]
        elif hasattr(self.backbone, "conv_proj"):
            # Torchvision ViT: directly project patches to support arbitrary aspect ratios (e.g. 336x168 or 224x112)
            x = self.backbone.conv_proj(images) # [B, C, H_p, W_p]
            B, C, H_p, W_p = x.shape
            x = x.flatten(2).transpose(1, 2)    # [B, H_p * W_p, C]
            
            # Prepend class token
            batch_class_token = self.backbone.class_token.expand(B, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            
            # Dynamic 2D positional embedding interpolation
            pos_emb = self.backbone.encoder.pos_embedding
            if pos_emb.shape[1] != x.shape[1]:
                pos_cls = pos_emb[:, 0:1, :]
                pos_patches = pos_emb[:, 1:, :] # [1, orig_N, C]
                # Interpolate 1D sequence to new length
                pos_patches = F.interpolate(
                    pos_patches.permute(0, 2, 1),
                    size=x.shape[1] - 1,
                    mode="linear",
                    align_corners=False
                ).permute(0, 2, 1)
                pos_emb = torch.cat([pos_cls, pos_patches], dim=1)
                
            x = x + pos_emb
            x = self.backbone.encoder.dropout(x)
            x = self.backbone.encoder.layers(x)
            x = self.backbone.encoder.ln(x)
            cls_token = x[:, 0, :]
            patch_tokens = x[:, 1:, :]
        else:
            cls_token, patch_tokens = self.backbone(images)

        # Project to target embed_dim if necessary
        cls_token = self.patch_proj(cls_token)
        patch_tokens = self.patch_proj(patch_tokens)

        # Anatomical 3-Strip Partitioning
        # Reshape to 2D grid: [B, num_rows, num_cols, embed_dim]
        total_p = patch_tokens.shape[1]
        grid_rows = self.num_rows
        grid_cols = self.num_cols

        if total_p == grid_rows * grid_cols:
            patches_2d = patch_tokens.view(B, grid_rows, grid_cols, self.embed_dim)
            r_head = max(1, int(grid_rows * 0.25))
            r_torso = max(r_head + 1, int(grid_rows * 0.65))

            strip_head = patches_2d[:, :r_head, :, :].reshape(B, -1, self.embed_dim)
            strip_torso = patches_2d[:, r_head:r_torso, :, :].reshape(B, -1, self.embed_dim)
            strip_legs = patches_2d[:, r_torso:, :, :].reshape(B, -1, self.embed_dim)
        else:
            # Fallback uniform 3-part partition along sequence length
            p1 = total_p // 3
            p2 = 2 * (total_p // 3)
            strip_head = patch_tokens[:, :p1, :]
            strip_torso = patch_tokens[:, p1:p2, :]
            strip_legs = patch_tokens[:, p2:, :]

        v_head = self.gem_head(strip_head)
        v_torso = self.gem_torso(strip_torso)
        v_legs = self.gem_legs(strip_legs)

        # Concatenate CLS token with 3 anatomical strips: [B, embed_dim * 4]
        fused_raw = torch.cat([cls_token, v_head, v_torso, v_legs], dim=-1)
        metric_emb = F.normalize(self.metric_proj(fused_raw), p=2, dim=-1)
        attr_logits = self.aux_head(cls_token)

        return metric_emb, attr_logits, patch_tokens, (v_head, v_torso, v_legs)


class TextStream(nn.Module):
    """Contextual text encoder with projected continuous metric output."""

    def __init__(self, embed_dim: int = 1024, max_length: int = 77, vocab_size: int = 49408):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=16,
            dim_feedforward=embed_dim * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.ln_final = nn.LayerNorm(embed_dim)

        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, text_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_tokens: [B, SeqLen] Integer token IDs.
        Returns:
            text_metric: [B, embed_dim] L2-normalized metric text embedding.
            text_seq: [B, SeqLen, embed_dim] Full contextualized token sequence for MLM.
        """
        B, L = text_tokens.shape
        x = self.token_embedding(text_tokens) + self.pos_embedding[:, :L, :]
        x = self.transformer(x)
        x = self.ln_final(x)

        # Global representation from first token or mean
        global_feat = x[:, 0, :]
        text_metric = F.normalize(self.text_proj(global_feat), p=2, dim=-1)
        return text_metric, x


class CrossModalMLMHead(nn.Module):
    """
    2-Layer Cross-Attention Decoder for Masked Language Modeling.
    Active strictly during training; completely bypassed during inference.
    """

    def __init__(self, embed_dim: int = 1024, vocab_size: int = 49408, num_layers: int = 2):
        super().__init__()
        self.decoder_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim=embed_dim, nhead=16)
            for _ in range(num_layers)
        ])
        self.vocab_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, masked_text_seq: torch.Tensor, visual_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            masked_text_seq: [B, SeqLen, embed_dim] Text sequence with masked tokens.
            visual_patches: [B, NumPatches, embed_dim] Visual spatial patches.
        Returns:
            [B, SeqLen, vocab_size] Predicted vocabulary logits.
        """
        x = masked_text_seq
        for layer in self.decoder_layers:
            x = layer(x, visual_patches)
        return self.vocab_head(x)


class CrossAttentionBlock(nn.Module):
    """Transformer block with multi-head self-attention and cross-attention."""

    def __init__(self, embed_dim: int = 1024, nhead: int = 16):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, q_seq: torch.Tensor, kv_seq: torch.Tensor) -> torch.Tensor:
        # Self-attention
        q_norm = self.ln1(q_seq)
        sa_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        x = q_seq + sa_out

        # Cross-attention (Queries: text, Keys/Values: visual patches)
        x_norm = self.ln2(x)
        ca_out, _ = self.cross_attn(query=x_norm, key=kv_seq, value=kv_seq)
        x = x + ca_out

        # Feed-forward
        x = x + self.mlp(self.ln3(x))
        return x


class StandaloneViT(nn.Module):
    """Self-contained Vision Transformer for 100% zero-dependency execution."""

    def __init__(
        self,
        img_height: int = 336,
        img_width: int = 168,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_height // patch_size) * (img_width // patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1) + self.pos_embed
        tokens = self.ln(self.blocks(tokens))
        return tokens[:, 0, :], tokens[:, 1:, :]


class SOTAChampionshipModel(nn.Module):
    """
    Master Unconstrained SOTA Architecture for >= 0.9000 Local mADM.
    Unites VisualStream (with 3-Strip PCB), TextStream, Auxiliary Head, and MLM Head.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_attrs: int = 40,
        img_height: int = 336,
        img_width: int = 168,
        patch_size: int = 14,
        backbone_type: str = "auto",
        use_mlm: bool = True,
        vocab_size: int = 49408,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_mlm = use_mlm

        self.visual_stream = VisualStream(
            embed_dim=embed_dim,
            num_attrs=num_attrs,
            img_height=img_height,
            img_width=img_width,
            patch_size=patch_size,
            backbone_type=backbone_type
        )
        self.text_stream = TextStream(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        )
        self.mlm_head = CrossModalMLMHead(
            embed_dim=embed_dim,
            vocab_size=vocab_size
        ) if use_mlm else None

    def extract_visual(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast inference path: returns L2-normalized metric embedding and attribute logits."""
        metric_emb, attr_logits, _, _ = self.visual_stream(images)
        return metric_emb, attr_logits

    def extract_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Fast inference path: returns L2-normalized text metric embedding."""
        text_metric, _ = self.text_stream(text_tokens)
        return text_metric

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        masked_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """Full training forward pass."""
        img_metric, attr_logits, spatial_patches, strip_embeds = self.visual_stream(images)
        txt_metric, text_seq = self.text_stream(text_tokens)

        outputs = {
            "img_metric": img_metric,
            "txt_metric": txt_metric,
            "attr_logits": attr_logits,
            "strip_embeds": strip_embeds,
        }

        # Training-time Masked Language Modeling
        if self.use_mlm and masked_tokens is not None and self.mlm_head is not None:
            _, masked_seq = self.text_stream(masked_tokens)
            vocab_logits = self.mlm_head(masked_seq, spatial_patches)
            outputs["vocab_logits"] = vocab_logits

        return outputs
