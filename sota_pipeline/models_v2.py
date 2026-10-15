"""SOTA Championship V2: Cross-Modal Margin Learning Neural Architecture.

Unified architecture featuring:
1. VisualStream: OpenCLIP EVA-02-B/16 with Anatomical 3-Strip Partitioning (PCB)
   and learnable Generalized Mean (GeM, p=3) pooling.
2. HardQueryAdapter: 3-layer non-linear projection mapping 40-D binary bitvectors into 512-D latent space.
3. SoftQueryAdapter: 12-layer Contextual Text Transformer + projection mapping linguistic pseudo-descriptions into 512-D space.
4. Fused Query Space: Unified 1024-D representation Eq = [Eh || Es] / sqrt(2).
5. Visual Retrieval Adapter: 4096-D fused visual features -> 1024-D person representation Ep.
6. Auxiliary Attribute Head: Multi-label classification branch trained with ASL.
7. Support for Decoupled Two-Stage Training (Stage 1 representation learning, Stage 2 frozen-backbone ArcFace margin learning).
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
        x_clamped = torch.clamp(x, min=self.eps)
        p = self.p.clamp(min=1.0, max=10.0)
        return (x_clamped.pow(p).mean(dim=1)).pow(1.0 / p)


class VisualStreamV2(nn.Module):
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
        backbone: Optional[nn.Module] = None,
        in_dim: Optional[int] = None,
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
        if backbone is not None and in_dim is not None:
            self.backbone = backbone
            self.in_dim = in_dim
        else:
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
        """Initializes authentic OpenCLIP EVA-02-B/16 backbone."""
        candidates = [
            ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", "momeva02"),
            ("EVA02-B-16", "merged2b_s8b_b131k"),
            ("EVA02-CLIP-B-16", "merged2b_s8b_b131k"),
            ("ViT-B-16", "laion2b_s34b_b88k"),
        ]

        try:
            import timm
            for m_name in ["eva02_base_patch16_clip_224", "eva02_base_patch14_448"]:
                try:
                    trunk = timm.create_model(m_name, pretrained=True)
                    in_dim = getattr(trunk, "num_features", getattr(trunk, "embed_dim", 768))
                    trunk.head = nn.Identity()
                    print(f"[VisualStreamV2] Successfully loaded authentic timm backbone '{m_name}' (in_dim={in_dim})!")
                    return trunk, in_dim
                except Exception:
                    continue
        except ImportError:
            pass

        try:
            import open_clip
            for model_name, pretrained in candidates:
                try:
                    clip_model, _, _ = open_clip.create_model_and_transforms(
                        model_name, pretrained=pretrained, force_quick_gelu=False
                    )
                    visual = clip_model.visual
                    if hasattr(visual, "set_grad_checkpointing"):
                        visual.set_grad_checkpointing(True)
                    in_dim = getattr(visual, "output_dim", getattr(visual, "num_features", self.embed_dim))
                    print(f"[VisualStreamV2] Successfully loaded authentic open_clip {model_name} (in_dim={in_dim})!")
                    return visual, in_dim
                except Exception:
                    continue
        except ImportError:
            pass

        raise RuntimeError("[VisualStreamV2] CRITICAL: Failed to load authentic EVA-02 backbone.")

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Extracts fused metric embedding, attribute logits, and individual strip embeddings.
        """
        B, C, H, W = images.shape

        # Extract features through backbone
        if hasattr(self.backbone, "forward_features") or hasattr(self.backbone, "trunk"):
            trunk_obj = getattr(self.backbone, "trunk", self.backbone)
            if hasattr(trunk_obj, "patch_embed") and hasattr(trunk_obj.patch_embed, "patch_size"):
                p_size = trunk_obj.patch_embed.patch_size
                actual_p = p_size if isinstance(p_size, int) else p_size[0]
                exp_size = getattr(trunk_obj.patch_embed, "img_size", (224, 224))
                expected_size = exp_size if isinstance(exp_size, (tuple, list)) else (exp_size, exp_size)
            else:
                actual_p = self.patch_size
                expected_size = (224, 224)

            # Pad horizontally to square
            if H > W:
                pad_w = (H - W) // 2
                images_pad = F.pad(images, (pad_w, pad_w, 0, 0))
            else:
                pad_w = 0
                images_pad = images

            if images_pad.shape[-2:] != expected_size:
                images_in = F.interpolate(images_pad, size=expected_size, mode="bicubic", align_corners=False)
            else:
                images_in = images_pad

            feats = trunk_obj.forward_features(images_in)
            if isinstance(feats, dict):
                cls_token = feats.get("x_norm_clstoken", feats.get("cls_token"))
                patch_tokens_all = feats.get("x_norm_patchtokens", feats.get("patch_tokens"))
            else:
                cls_token = feats[:, 0, :]
                patch_tokens_all = feats[:, 1:, :]

            grid_h = expected_size[0] // actual_p
            grid_w = expected_size[1] // actual_p
            grid = patch_tokens_all.view(B, grid_h, grid_w, -1)

            if pad_w > 0:
                col_start = int(grid_w * (pad_w / images_pad.shape[-1]))
                col_span = max(1, int(grid_w * (W / images_pad.shape[-1])))
                col_end = col_start + col_span
                ped_grid = grid[:, :, col_start:col_end, :]
                ped_rows = ped_grid.shape[1]
                ped_cols = ped_grid.shape[2]
                patch_tokens = ped_grid.reshape(B, -1, ped_grid.shape[-1])
            else:
                ped_grid = grid
                ped_rows = grid_h
                ped_cols = grid_w
                patch_tokens = patch_tokens_all
        else:
            feats = self.backbone(images)
            cls_token = feats[:, 0, :]
            patch_tokens = feats[:, 1:, :]
            ped_rows = self.num_rows
            ped_cols = self.num_cols

        # Project to target embed_dim if necessary
        cls_token = self.patch_proj(cls_token)
        patch_tokens = self.patch_proj(patch_tokens)

        # Anatomical 3-Strip Partitioning
        total_p = patch_tokens.shape[1]
        grid_rows = ped_rows if total_p == ped_rows * ped_cols else self.num_rows
        grid_cols = ped_cols if total_p == ped_rows * ped_cols else self.num_cols

        if total_p == grid_rows * grid_cols:
            patches_2d = patch_tokens.view(B, grid_rows, grid_cols, self.embed_dim)
            r_head = max(1, int(grid_rows * 0.25))
            r_torso = max(r_head + 1, int(grid_rows * 0.65))

            strip_head = patches_2d[:, :r_head, :, :].reshape(B, -1, self.embed_dim)
            strip_torso = patches_2d[:, r_head:r_torso, :, :].reshape(B, -1, self.embed_dim)
            strip_legs = patches_2d[:, r_torso:, :, :].reshape(B, -1, self.embed_dim)
        else:
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

        return metric_emb, attr_logits, (v_head, v_torso, v_legs)


class HardQueryAdapter(nn.Module):
    """
    Maps 40-D discrete binary attribute query vectors into a 512-D continuous latent space.
    Follows CLEAR's non-linear projection design.
    """

    def __init__(self, in_dim: int = 40, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, hard_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hard_queries: [B, 40] Binary attribute queries (in float32).
        Returns:
            [B, out_dim] L2-normalized continuous query representation Eh.
        """
        z = self.net(hard_queries)
        return F.normalize(z, p=2, dim=-1)


class SoftQueryAdapter(nn.Module):
    """
    Projects contextual language embeddings from the Text Transformer into a 512-D latent space.
    """

    def __init__(self, text_embed_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_features: [B, text_embed_dim] Text transformer pooled output.
        Returns:
            [B, out_dim] L2-normalized continuous query representation Es.
        """
        z = self.net(text_features)
        return F.normalize(z, p=2, dim=-1)


class SOTAChampionshipModelV2(nn.Module):
    """
    Master SOTA Championship V2 Architecture.
    Combines EVA-02 VisualStream, Contextual TextStream, and Hard/Soft Query Adapters
    with support for Decoupled Two-Stage Training (Stage 1 pretraining, Stage 2 ArcFace).
    """

    def __init__(
        self,
        visual_stream: VisualStreamV2,
        text_stream: nn.Module,
        embed_dim: int = 1024,
        query_dim: int = 512,
        num_attrs: int = 40,
        text_embed_dim: int = 512,
    ):
        super().__init__()
        self.visual_stream = visual_stream
        self.text_stream = text_stream
        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.num_attrs = num_attrs
        self.text_embed_dim = text_embed_dim

        # Dual-Query Adapters
        self.hard_query_adapter = HardQueryAdapter(in_dim=num_attrs, out_dim=query_dim)
        self.soft_query_adapter = SoftQueryAdapter(text_embed_dim=text_embed_dim, out_dim=query_dim)

    def freeze_backbone(self) -> None:
        """Freezes the visual backbone and text stream for Stage 2 ArcFace margin learning."""
        print("[SOTAChampionshipModelV2] Freezing visual backbone and text stream for Stage 2...")
        for param in self.visual_stream.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.text_stream, "parameters"):
            for param in self.text_stream.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all parameters for Stage 1 fine-tuning."""
        print("[SOTAChampionshipModelV2] Unfreezing all parameters for Stage 1 fine-tuning...")
        for param in self.parameters():
            param.requires_grad = True

    def forward_visual(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts visual embedding and attribute classification logits.
        Returns:
            metric_emb: [B, 1024] L2-normalized visual person representation Ep.
            attr_logits: [B, 40] Raw logits for attribute recognition.
        """
        metric_emb, attr_logits, _ = self.visual_stream(images)
        return metric_emb, attr_logits

    def forward_query(
        self,
        hard_queries: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts Hard, Soft, and Unified Concatenated Query Representations.
        Returns:
            E_q: [B, 1024] Unified L2-normalized query vector [Eh || Es] / sqrt(2).
            E_h: [B, 512] Hard binary query embedding.
            E_s: [B, 512] Soft language query embedding.
        """
        # 1. Hard query embedding from binary attribute vector
        E_h = self.hard_query_adapter(hard_queries)  # [B, 512]

        # 2. Soft query embedding from text tokens
        text_feats = self.text_stream(text_tokens)    # [B, 1024]
        E_s = self.soft_query_adapter(text_feats)     # [B, 512]

        # 3. Concatenate and normalize unified query
        fused = torch.cat([E_h, E_s], dim=-1)         # [B, 1024]
        E_q = F.normalize(fused, p=2, dim=-1)         # [B, 1024]

        return E_q, E_h, E_s

    def forward(
        self,
        images: torch.Tensor,
        hard_queries: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full training forward pass."""
        E_p, attr_logits = self.forward_visual(images)
        E_q, E_h, E_s = self.forward_query(hard_queries, text_tokens)

        return {
            "E_p": E_p,
            "E_q": E_q,
            "E_h": E_h,
            "E_s": E_s,
            "attr_logits": attr_logits,
        }
