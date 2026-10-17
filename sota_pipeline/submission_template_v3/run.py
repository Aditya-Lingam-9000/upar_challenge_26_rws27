"""Official CodaBench Track 2 SOTA V3 Championship Ingestion Entrypoint.

Self-contained pure PyTorch inference module.
Strictly compatible with pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime, NVIDIA A10 GPU (24GB VRAM, 20 Cores, 40GB RAM),
Apple Silicon (MPS), and CPU environments.
100% air-gapped execution: zero internet access, zero open_clip, zero ftfy.

Target Architecture & Test-Time Compute Suite:
1. Foundation Vision Backbone: EVA-02-B/16 with internal surgical modifications:
   - ConvPass Depthwise Separable local spatial bypasses.
   - Deformable Multi-Head Attention (DMHA) for non-canonical drone viewpoints.
   - Fourier Phase-Content Normalization (PCNorm).
   - Anatomical Gaussian Cross-Attention (6 learnable Part Tokens with spatial Gaussian priors).
2. Continuous Multi-Attribute Prompter with Multi-Temperature Ensembling (K=4).
3. Test-Time Compute Championship Engine:
   - Dual-Pass Horizontal Flip Gallery TTA (viewpoint and shadow invariance).
   - Prior-Matching Test-Time Attribute Calibration (Newton-Raphson convergence to UPAR class priors).
   - Alpha-QE Query Expansion with cubic sharpening (alpha=3.0, k=5).
   - Cascaded Metric-First Filtering (Top-300 candidates).
   - Asymmetric Directional Penalty (missed requested traits: 1.0, extra traits: 0.15 with dead-zone margin 0.70).
   - Information-Theoretic Surprisal Dynamic Weighting (w_attr in [0.05, 0.20]).
   - Monotonic Rank Space Interpolation (zero distance ties, deterministic rank order).
"""
from __future__ import annotations

import concurrent.futures
import gc
import gzip
import html
import math
import os
import re
import sys
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# Configure local vendored packages
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(CURRENT_DIR, "vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import timm


# =====================================================================
# 1. Intelligent Dynamic Hardware Auto-Sensing Engine
# =====================================================================
class HardwareEngine:
    """Discovers host hardware resources and determines optimal runtime parameters."""

    def __init__(self):
        self.logical_cpus = os.cpu_count() or 2
        try:
            self.affinity_cpus = len(os.sched_getaffinity(0))
        except Exception:
            self.affinity_cpus = self.logical_cpus
        self.usable_cpus = max(1, min(self.logical_cpus, self.affinity_cpus))

        self.total_ram_gb = self._detect_total_ram_gb()
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        self.cuda_vram_gb = 0.0
        self.cuda_device_name = ""
        if self.has_cuda:
            try:
                self.cuda_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                self.cuda_device_name = torch.cuda.get_device_name(0)
            except Exception:
                pass

        self._configure_strategy()

    def _detect_total_ram_gb(self) -> float:
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) / (1024 * 1024)
            except Exception:
                pass
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / (1024 ** 3)
        except Exception:
            pass
        return 8.0

    def _configure_strategy(self):
        if self.has_cuda:
            self.device_type = "cuda"
            self.torch_threads = min(8, self.usable_cpus)
            self.use_half = True

            if self.cuda_vram_gb >= 20.0:  # NVIDIA A10 (24GB VRAM on CodaBench)
                self.batch_size = 256
                self.num_io_workers = min(18, max(4, self.usable_cpus - 2))
            elif self.cuda_vram_gb >= 12.0:  # Tesla T4 / RTX 3060
                self.batch_size = 128
                self.num_io_workers = min(8, max(2, self.usable_cpus - 1))
            else:
                self.batch_size = 64
                self.num_io_workers = min(4, max(1, self.usable_cpus - 1))
        elif self.has_mps:
            self.device_type = "mps"
            self.batch_size = 64
            self.num_io_workers = min(6, self.usable_cpus)
            self.torch_threads = self.usable_cpus
            self.use_half = False
        else:
            self.device_type = "cpu"
            self.use_half = False
            self.torch_threads = self.usable_cpus
            self.batch_size = 32
            self.num_io_workers = min(4, self.usable_cpus)

    def apply(self):
        torch.set_num_threads(self.torch_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, min(4, self.usable_cpus // 2)))
            except Exception:
                pass
        if self.has_cuda:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


HW = HardwareEngine()
_DEVICE = torch.device(HW.device_type)
_IMG_HEIGHT = 336
_IMG_WIDTH = 168

ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")
WEIGHTS_PATH = os.path.join(ASSETS_DIR, "model_sota_fp16.pt")
if not os.path.isfile(WEIGHTS_PATH):
    WEIGHTS_PATH = os.path.join(ASSETS_DIR, "best_model_sota_v3.pt")
if not os.path.isfile(WEIGHTS_PATH):
    WEIGHTS_PATH = os.path.join(ASSETS_DIR, "model_sota_v3_swad_fp16.pt")

_TRANSFORM = T.Compose([
    T.Resize((_IMG_HEIGHT, _IMG_WIDTH), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_CANONICAL_ATTRIBUTES: List[str] = [
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
    "Accessory-Glasses-Sun", "Accessory-Hat"
]

UPAR_ATTR_PRIORS: np.ndarray = np.array([
    0.0110, 0.9650, 0.0237, 0.4292, 0.6546, 0.3114, 0.0126, 0.6288, 0.2556, 0.0821,
    0.0199, 0.0435, 0.1621, 0.0058, 0.0105, 0.0253, 0.0694, 0.2250, 0.0497, 0.0753,
    0.3751, 0.4877, 0.1788, 0.0448, 0.0150, 0.1719, 0.0021, 0.0155, 0.0008, 0.0039,
    0.0447, 0.0017, 0.0372, 0.9038, 0.0958, 0.2085, 0.2931, 0.1394, 0.0218, 0.0425
], dtype=np.float32)


# =====================================================================
# 2. Surgical Neural Network Components
# =====================================================================
class ConvPass(nn.Module):
    def __init__(self, in_dim: int = 768, mid_dim: int = 64, kernel_size: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim
        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.act1 = nn.GELU()
        self.dw_conv = nn.Conv2d(mid_dim, mid_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=mid_dim)
        self.bn = nn.BatchNorm2d(mid_dim)
        self.act2 = nn.GELU()
        self.pw_conv = nn.Conv2d(mid_dim, mid_dim, kernel_size=1)
        self.up_proj = nn.Linear(mid_dim, in_dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int, has_cls_token: bool = False) -> torch.Tensor:
        if has_cls_token:
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
        else:
            cls_token = None
            patches = x

        B, N, D = patches.shape
        h = self.act1(self.down_proj(patches))
        h_2d = h.transpose(1, 2).reshape(B, self.mid_dim, grid_h, grid_w)
        h_conv = self.pw_conv(self.act2(self.bn(self.dw_conv(h_2d))))
        h_flat = h_conv.flatten(2).transpose(1, 2)
        out_patches = self.up_proj(h_flat) * self.gamma

        if has_cls_token and cls_token is not None:
            return torch.cat([torch.zeros_like(cls_token), out_patches], dim=1)
        return out_patches


class DeformableMultiHeadAttention(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 8, offset_scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.offset_scale = offset_scale
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.offset_net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, num_heads * 2, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def _generate_ref_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int, has_cls_token: bool = False) -> torch.Tensor:
        if has_cls_token:
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
        else:
            cls_token = None
            patches = x

        B, N, D = patches.shape
        device = patches.device
        feat_2d = patches.transpose(1, 2).reshape(B, D, grid_h, grid_w)

        offsets = torch.tanh(self.offset_net(feat_2d)) * self.offset_scale
        offsets = offsets.view(B, self.num_heads, 2, grid_h, grid_w).permute(0, 1, 3, 4, 2)

        ref_grid = self._generate_ref_grid(grid_h, grid_w, device)
        sample_grid = (ref_grid.unsqueeze(0).unsqueeze(0) + offsets).clamp(-1.0, 1.0)

        q = self.q_proj(patches).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(patches).view(B, grid_h, grid_w, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2).reshape(B * self.num_heads, self.head_dim, grid_h, grid_w)
        v = self.v_proj(patches).view(B, grid_h, grid_w, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2).reshape(B * self.num_heads, self.head_dim, grid_h, grid_w)

        sample_grid_flat = sample_grid.reshape(B * self.num_heads, grid_h, grid_w, 2).to(dtype=k.dtype)
        k_sampled = F.grid_sample(k, sample_grid_flat, mode="bilinear", padding_mode="border", align_corners=False).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v_sampled = F.grid_sample(v, sample_grid_flat, mode="bilinear", padding_mode="border", align_corners=False).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        attn_scores = torch.matmul(q, k_sampled.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v_sampled).permute(0, 2, 1, 3).contiguous().view(B, N, D)

        out_patches = patches + self.alpha * self.out_proj(attn_out)
        if has_cls_token and cls_token is not None:
            return torch.cat([cls_token, out_patches], dim=1)
        return out_patches


class FourierTransformerDGLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = nn.InstanceNorm2d(dim, affine=False)
        self.proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int, has_cls_token: bool = False) -> torch.Tensor:
        if has_cls_token:
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
        else:
            cls_token = None
            patches = x

        B, N, D = patches.shape
        feat_2d = patches.transpose(1, 2).contiguous().view(B, D, grid_h, grid_w)

        norm_feat = self.norm(feat_2d)
        fft_orig = torch.fft.rfft2(feat_2d.float(), norm="ortho")
        fft_norm = torch.fft.rfft2(norm_feat.float(), norm="ortho")

        amp_norm = torch.abs(fft_norm)
        phase_orig = torch.angle(fft_orig)

        fft_reconstructed = torch.polar(amp_norm, phase_orig)
        out_2d = torch.fft.irfft2(fft_reconstructed, s=(grid_h, grid_w), norm="ortho").to(x.dtype)

        out_tokens = self.proj(out_2d.flatten(2).transpose(1, 2))
        out_patches = patches + self.gamma * out_tokens

        if has_cls_token and cls_token is not None:
            return torch.cat([cls_token, out_patches], dim=1)
        return out_patches


class AnatomicalGaussianCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_parts: int = 6, num_heads: int = 8, sigma: float = 0.15, lambda_prior: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_parts = num_parts
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sigma = sigma
        self.lambda_prior = lambda_prior
        self.scale = self.head_dim ** -0.5

        self.part_tokens = nn.Parameter(torch.randn(num_parts, embed_dim) * 0.02)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        mu = [(i + 0.5) / num_parts for i in range(num_parts)]
        self.register_buffer("mu_priors", torch.tensor(mu, dtype=torch.float32))

    def _compute_gaussian_bias(self, grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        y_coords = torch.linspace(0.0, 1.0, grid_h, device=device, dtype=dtype).unsqueeze(1).repeat(1, grid_w).flatten()
        mu = self.mu_priors.unsqueeze(1).to(device=device, dtype=dtype)
        y = y_coords.unsqueeze(0)
        gauss = torch.exp(-((y - mu) ** 2) / (2.0 * (self.sigma ** 2)))
        return torch.log(gauss + 1e-6).unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    def forward(self, patch_tokens: torch.Tensor, grid_h: int, grid_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = patch_tokens.shape
        device = patch_tokens.device

        part_q = self.part_tokens.unsqueeze(0).expand(B, -1, -1)
        q = self.q_proj(part_q).view(B, self.num_parts, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(patch_tokens).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(patch_tokens).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if N == grid_h * grid_w:
            bias = self._compute_gaussian_bias(grid_h, grid_w, device, dtype=attn_logits.dtype)
            attn_logits = attn_logits + bias.to(dtype=attn_logits.dtype) * self.lambda_prior

        attn_weights = F.softmax(attn_logits, dim=-1).to(dtype=v.dtype)
        part_out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous().view(B, self.num_parts, D)
        fused_parts = self.norm(self.out_proj(part_out) + part_q)
        return fused_parts, attn_weights.mean(dim=1)


class SurgicalPedestrianViT(nn.Module):
    def __init__(self, embed_dim: int = 1024, num_attrs: int = 40, in_dim: int = 768, backbone: Optional[nn.Module] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attrs = num_attrs
        self.in_dim = in_dim
        self.trunk = backbone

        self.convpass_mid = ConvPass(in_dim=in_dim, mid_dim=64)
        self.convpass_high = ConvPass(in_dim=in_dim, mid_dim=64)
        self.dmha = DeformableMultiHeadAttention(dim=in_dim, num_heads=8)
        self.pcnorm = FourierTransformerDGLayer(dim=in_dim)
        self.anatomical_cross_attn = AnatomicalGaussianCrossAttention(embed_dim=in_dim, num_parts=6)

        self.metric_head = nn.Sequential(
            nn.Linear(in_dim * 7, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.aux_attr_head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, num_attrs),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = images.shape
        trunk_obj = getattr(self.trunk, "trunk", self.trunk)

        # Pad horizontally to square if 2:1 aspect ratio
        if H > W:
            pad_w = (H - W) // 2
            images_pad = F.pad(images, (pad_w, pad_w, 0, 0))
        else:
            pad_w = 0
            images_pad = images

        exp_size = (224, 224)
        if hasattr(trunk_obj, "patch_embed") and hasattr(trunk_obj.patch_embed, "img_size"):
            p_img = trunk_obj.patch_embed.img_size
            exp_size = p_img if isinstance(p_img, (tuple, list)) else (p_img, p_img)

        if images_pad.shape[-2:] != exp_size:
            images_in = F.interpolate(images_pad, size=exp_size, mode="bicubic", align_corners=False)
        else:
            images_in = images_pad

        feats = trunk_obj.forward_features(images_in)
        if isinstance(feats, dict):
            cls_token = feats.get("x_norm_clstoken", feats.get("cls_token"))
            patch_tokens_all = feats.get("x_norm_patchtokens", feats.get("patch_tokens"))
        elif feats.ndim == 3:
            cls_token = feats[:, 0, :]
            patch_tokens_all = feats[:, 1:, :]
        else:
            cls_token = feats
            patch_tokens_all = feats.unsqueeze(1).repeat(1, 196, 1)

        # Unpad pedestrian tokens to preserve true 2:1 pedestrian geometry
        actual_p = getattr(trunk_obj.patch_embed, "patch_size", (16, 16)) if hasattr(trunk_obj, "patch_embed") else (16, 16)
        actual_p = actual_p[0] if isinstance(actual_p, (tuple, list)) else actual_p
        grid_h = exp_size[0] // actual_p
        grid_w = exp_size[1] // actual_p

        if patch_tokens_all.shape[1] == grid_h * grid_w:
            grid = patch_tokens_all.view(B, grid_h, grid_w, -1)
            if pad_w > 0:
                col_start = int(grid_w * (pad_w / images_pad.shape[-1]))
                col_span = max(1, int(grid_w * (W / images_pad.shape[-1])))
                ped_grid = grid[:, :, col_start : col_start + col_span, :]
                ped_h = ped_grid.shape[1]
                ped_w = ped_grid.shape[2]
                patch_tokens = ped_grid.reshape(B, -1, ped_grid.shape[-1])
            else:
                ped_h = grid_h
                ped_w = grid_w
                patch_tokens = patch_tokens_all
        else:
            ped_h = int(math.sqrt(patch_tokens_all.shape[1]))
            ped_w = patch_tokens_all.shape[1] // ped_h
            patch_tokens = patch_tokens_all

        # Surgical pipeline on pedestrian tokens
        patch_tokens = patch_tokens + self.convpass_mid(patch_tokens, ped_h, ped_w, has_cls_token=False)
        patch_tokens = self.pcnorm(patch_tokens, ped_h, ped_w, has_cls_token=False)
        patch_tokens = self.dmha(patch_tokens, ped_h, ped_w, has_cls_token=False)
        patch_tokens = patch_tokens + self.convpass_high(patch_tokens, ped_h, ped_w, has_cls_token=False)

        # AGCA 6-part tokens
        fused_parts, _ = self.anatomical_cross_attn(patch_tokens, ped_h, ped_w)
        parts_flat = fused_parts.reshape(B, -1)
        fused_all = torch.cat([cls_token, parts_flat], dim=-1)

        metric_emb = F.normalize(self.metric_head(fused_all), p=2, dim=-1)
        attr_logits = self.aux_attr_head(cls_token)
        return metric_emb, attr_logits


class ContinuousAttributePrompter(nn.Module):
    def __init__(self, num_attrs: int = 40, embed_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.num_attrs = num_attrs
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.attr_axes = nn.Parameter(torch.randn(num_attrs, hidden_dim) * 0.02)
        self.register_buffer("attr_weights", torch.ones(num_attrs, dtype=torch.float32))

        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.direct_proj = nn.Sequential(
            nn.Linear(num_attrs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def set_empirical_weights(self, attribute_frequencies: np.ndarray):
        p = np.clip(attribute_frequencies, 1e-4, 1.0 - 1e-4)
        surprisal = -np.log2(p)
        w = np.sqrt(surprisal)
        w = w / np.mean(w)
        self.attr_weights.copy_(torch.from_numpy(w).to(dtype=self.attr_weights.dtype, device=self.attr_weights.device))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        bipolar_q = 2.0 * queries - 1.0
        weighted_q = bipolar_q * self.attr_weights.unsqueeze(0)
        cont_embed = torch.matmul(weighted_q, self.attr_axes)
        refined = self.refiner(cont_embed)
        direct = self.direct_proj(queries)
        return F.normalize(refined + direct, p=2, dim=-1)

    def forward_multi_temp(
        self,
        queries: torch.Tensor,
        temps: List[float] = [1.25, 1.80, 2.50, 4.50],
        weights: List[float] = [0.25, 0.35, 0.25, 0.15],
    ) -> torch.Tensor:
        orig_weights = self.attr_weights.clone()
        embeds = []
        for t, w in zip(temps, weights):
            scaled_w = torch.pow(orig_weights.abs() + 1e-8, 2.0 / t)
            scaled_w = scaled_w / scaled_w.mean().clamp(min=1e-6)
            self.attr_weights.copy_(scaled_w)
            embeds.append(w * self.forward(queries))
        self.attr_weights.copy_(orig_weights)
        fused = torch.stack(embeds, dim=0).sum(dim=0)
        return F.normalize(fused, p=2, dim=-1)


# =====================================================================
# 3. Test-Time Compute Engine
# =====================================================================
class TestTimeComputeEngine:
    """
    SOTA Championship Test-Time Compute Suite:
    1. Prior-Matching Attribute Calibration: Newton-Raphson calibrates gallery
       attribute logits so that predicted sigmoid probabilities match true dataset priors.
    2. Alpha-Query Expansion (Alpha-QE): Cubic power-sharpened query expansion against
       the gallery manifold (alpha=3.0, k=5).
    3. Cascaded Metric-First Filtering + Asymmetric Directional Distance:
       Isolates the top-300 candidates by continuous metric distance, then applies
       asymmetric directional penalty (missing traits penalized 1.0, extra traits 0.15 with
       dead-zone margin 0.70) with surprisal dynamic weighting strictly within the candidate pool.
       Preserves global retrieval geometry while eliminating attribute noise.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.alpha_qe = 3.0
        self.topk_qe = 5
        self.cascade_k = 300
        self.attr_priors = UPAR_ATTR_PRIORS.copy()

    def prior_matching_calibration(
        self,
        gal_logits_np: np.ndarray,
        target_priors: np.ndarray,
        max_iters: int = 5,
    ) -> np.ndarray:
        calib_logits = gal_logits_np.copy()
        for a in range(gal_logits_np.shape[1]):
            pi_a = target_priors[a]
            delta = 0.0
            for _ in range(max_iters):
                z = np.clip(calib_logits[:, a] + delta, -50.0, 50.0)
                p = 1.0 / (1.0 + np.exp(-z))
                diff = p.mean() - pi_a
                grad = (p * (1.0 - p)).mean() + 1e-8
                step = diff / grad
                delta -= np.clip(step, -10.0, 10.0)
                if abs(step) < 1e-4:
                    break
            calib_logits[:, a] += delta
        return 1.0 / (1.0 + np.exp(-np.clip(calib_logits, -50.0, 50.0)))

    def apply_alpha_qe(self, query_embeds: torch.Tensor, gal_embeds: torch.Tensor) -> torch.Tensor:
        sims = torch.matmul(query_embeds, gal_embeds.T)
        k_val = min(self.topk_qe, gal_embeds.shape[0])
        top_vals, top_inds = torch.topk(sims, k=k_val, dim=-1, largest=True)
        weights = top_vals.clamp(min=0.0).pow(self.alpha_qe).unsqueeze(-1)
        retrieved_feats = gal_embeds[top_inds]
        expanded = query_embeds + (weights * retrieved_feats).sum(dim=1)
        return F.normalize(expanded, p=2, dim=-1)

    def cascaded_asymmetric_reranking(
        self,
        dist_metric_np: np.ndarray,
        queries_raw_np: np.ndarray,
        gal_probs_calib_np: np.ndarray,
        attr_priors_np: np.ndarray,
    ) -> np.ndarray:
        Q, G = dist_metric_np.shape
        K = min(self.cascade_k, G)

        # 1. Asymmetric Directional Attribute Distance
        pos_miss = queries_raw_np @ (1.0 - gal_probs_calib_np).T  # [Q, G]
        neg_extra = np.maximum(0.0, gal_probs_calib_np - 0.70) ** 2
        neg_penalty = (1.0 - queries_raw_np) @ neg_extra.T  # [Q, G]
        dist_attr_np = pos_miss + 0.15 * neg_penalty

        # 2. Query Surprisal / Information-Theoretic Content Weighting
        surprisal = -np.log2(np.clip(attr_priors_np, 1e-4, 1.0))
        info_q = queries_raw_np @ surprisal
        mu_I = float(np.mean(info_q))
        std_I = float(np.std(info_q)) + 1e-6
        z_I = (info_q - mu_I) / std_I
        w_attr = np.clip(0.08 + 0.12 / (1.0 + np.exp(-z_I)), 0.05, 0.20)  # Attribute weight strictly in [5%, 20%]
        w_metric = 1.0 - w_attr

        final_dist = np.copy(dist_metric_np)
        metric_ranks = np.argsort(dist_metric_np, axis=1)

        for q in range(Q):
            topk_cands = metric_ranks[q, :K]
            sub_d_m = dist_metric_np[q, topk_cands]
            sub_d_a = dist_attr_np[q, topk_cands]

            # Z-score normalize within top-K candidate pool
            std_m = np.std(sub_d_m)
            std_a = np.std(sub_d_a)
            norm_m = (sub_d_m - np.mean(sub_d_m)) / (std_m if std_m > 1e-6 else 1.0)
            norm_a = (sub_d_a - np.mean(sub_d_a)) / (std_a if std_a > 1e-6 else 1.0)

            fused_sub = w_metric[q] * norm_m + w_attr[q] * norm_a
            reranked_order = topk_cands[np.argsort(fused_sub)]

            # Monotonic rank assignment: place top-K reranked candidates at the front
            remaining = metric_ranks[q, K:]
            ordered_indices = np.concatenate([reranked_order, remaining])
            final_dist[q, ordered_indices] = np.linspace(0.0, 1.0, G)

        return final_dist

    def run_inference_pipeline(
        self,
        gal_embeds_np: np.ndarray,
        gal_logits_np: np.ndarray,
        query_embeds_np: np.ndarray,
        queries_raw_np: np.ndarray,
    ) -> np.ndarray:
        t0 = time.time()
        Q, D = query_embeds_np.shape
        G, _ = gal_embeds_np.shape

        print(f"[TTCE] Running SOTA Championship Inference Suite on {_DEVICE} ({Q:,} queries, {G:,} gallery)...", flush=True)

        # 1. Prior-Matching Attribute Calibration
        t1 = time.time()
        gal_probs_calib = self.prior_matching_calibration(gal_logits_np, self.attr_priors)
        print(f"[TTCE] Prior-matching attribute calibration completed in {time.time() - t1:.3f}s", flush=True)

        # 2. Alpha-Query Expansion (Alpha-QE on GPU)
        t2 = time.time()
        with torch.no_grad():
            query_t = torch.from_numpy(query_embeds_np).float().to(self.device)
            gal_t = torch.from_numpy(gal_embeds_np).float().to(self.device)
            query_qe_t = self.apply_alpha_qe(query_t, gal_t)
            # Continuous metric distance: d_metric = 1.0 - (Q_qe @ G_tta.T)
            cos_sim = torch.matmul(query_qe_t, gal_t.T)
            dist_metric_t = 1.0 - cos_sim
            dist_metric_np = dist_metric_t.cpu().numpy()
        print(f"[TTCE] Alpha-QE (alpha={self.alpha_qe}, k={self.topk_qe}) completed in {time.time() - t2:.3f}s", flush=True)

        # 3. Cascaded Metric-First Filtering (Top-300) + Asymmetric Directional Penalty
        t3 = time.time()
        final_distances = self.cascaded_asymmetric_reranking(
            dist_metric_np=dist_metric_np,
            queries_raw_np=queries_raw_np,
            gal_probs_calib_np=gal_probs_calib,
            attr_priors_np=self.attr_priors,
        )
        print(f"[TTCE] Cascaded Asymmetric Reranking (Top-{min(self.cascade_k, G)}) completed in {time.time() - t3:.3f}s", flush=True)
        print(f"[TTCE] Total Test-Time Compute Suite finished in {time.time() - t0:.2f}s!", flush=True)
        return final_distances


# Global instances
_VISUAL_MODEL: Optional[SurgicalPedestrianViT] = None
_ATTR_PROMPTER: Optional[ContinuousAttributePrompter] = None
_TT_ENGINE: Optional[TestTimeComputeEngine] = None


def load_model():
    """Initializes and loads SOTA Championship V3 model weights."""
    global _VISUAL_MODEL, _ATTR_PROMPTER, _TT_ENGINE

    print("=" * 70, flush=True)
    print(f"[INIT] Initializing SOTA Championship V3 System on {_DEVICE}...", flush=True)
    HW.apply()

    trunk = timm.create_model("eva02_base_patch16_clip_224", pretrained=False)
    trunk.head = nn.Identity()

    visual_model = SurgicalPedestrianViT(
        embed_dim=1024,
        num_attrs=40,
        in_dim=768,
        backbone=trunk,
    )
    attr_prompter = ContinuousAttributePrompter(
        num_attrs=40,
        embed_dim=1024,
        hidden_dim=512,
    )
    attr_prompter.set_empirical_weights(UPAR_ATTR_PRIORS)

    print(f"[INIT] Loading model checkpoint from {WEIGHTS_PATH}...", flush=True)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    vis_dict = {k.replace("visual_stream.", ""): v for k, v in state_dict.items() if "visual_stream." in k}
    if not vis_dict:
        vis_dict = {k: v for k, v in state_dict.items() if not k.startswith("attr_prompter.") and not k.startswith("hyperbolic_head.")}
    visual_model.load_state_dict(vis_dict, strict=False)

    prompt_dict = {k.replace("attr_prompter.", ""): v for k, v in state_dict.items() if "attr_prompter." in k}
    if prompt_dict:
        attr_prompter.load_state_dict(prompt_dict, strict=False)

    visual_model = visual_model.to(_DEVICE).eval()
    attr_prompter = attr_prompter.to(_DEVICE).eval()

    if HW.use_half:
        visual_model = visual_model.half()
        attr_prompter = attr_prompter.half()

    _VISUAL_MODEL = visual_model
    _ATTR_PROMPTER = attr_prompter
    _TT_ENGINE = TestTimeComputeEngine(device=_DEVICE)
    print("[INIT] SOTA Championship V3 System successfully loaded & ready!", flush=True)
    print("=" * 70, flush=True)


# =====================================================================
# 4. Multi-Threaded Double-Buffered Image Prefetching Pipeline
# =====================================================================
def _load_single_image(img_path: str) -> torch.Tensor:
    try:
        with Image.open(img_path) as img:
            return _TRANSFORM(img.convert("RGB"))
    except Exception:
        return torch.zeros(3, _IMG_HEIGHT, _IMG_WIDTH, dtype=torch.float32)


def prefetch_batches(
    gallery: List[Dict[str, Any]],
    batch_size: int,
    num_workers: int,
) -> Generator[Tuple[int, int, torch.Tensor], None, None]:
    total_images = len(gallery)
    if total_images == 0:
        return

    num_batches = (total_images + batch_size - 1) // batch_size
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_workers))

    def _extract_path(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return item.get("image_path", item.get("path", item.get("image_name", item.get("file_name", ""))))
        return ""

    def _submit(b_idx: int):
        s = b_idx * batch_size
        e = min(s + batch_size, total_images)
        paths = [_extract_path(gallery[i]) for i in range(s, e)]
        futures = [executor.submit(_load_single_image, p) for p in paths]
        return s, e, futures

    try:
        next_batch_data = _submit(0)
        for b_idx in range(num_batches):
            start_idx, end_idx, futures = next_batch_data
            if b_idx + 1 < num_batches:
                next_batch_data = _submit(b_idx + 1)
            tensors = [f.result() for f in futures]
            batch_tensor = torch.stack(tensors)
            if torch.cuda.is_available():
                try:
                    batch_tensor = batch_tensor.pin_memory()
                except Exception:
                    pass
            yield start_idx, end_idx, batch_tensor
    finally:
        executor.shutdown(wait=False)


# =====================================================================
# 5. Official CodaBench Retrieval Entrypoint
# =====================================================================
@torch.inference_mode()
def rank_gallery(
    sample_or_queries: Union[Dict[str, Any], Any],
    gallery_arg: Optional[Any] = None,
) -> Union[Dict[str, Any], np.ndarray]:
    t0 = time.time()

    if _VISUAL_MODEL is None or _ATTR_PROMPTER is None or _TT_ENGINE is None:
        load_model()

    is_sample_dict = isinstance(sample_or_queries, dict) and gallery_arg is None
    if is_sample_dict:
        sample = sample_or_queries
        gallery = sample.get("gallery", sample.get("gallery_images", []))
        q_input = sample.get("queries", sample.get("query_attributes", []))
    else:
        q_input = sample_or_queries
        gallery = gallery_arg if gallery_arg is not None else []

    if isinstance(q_input, list) and len(q_input) > 0 and isinstance(q_input[0], dict):
        q_list = []
        for q in q_input:
            vec = q.get("attribute_vector", q.get("attributes", q.get("query_attributes", None)))
            if vec is not None:
                q_list.append(vec)
            else:
                q_list.append(q)
        queries_raw = np.asarray(q_list, dtype=np.float32)
    else:
        queries_raw = np.asarray(q_input, dtype=np.float32)

    if queries_raw.ndim == 1:
        queries_raw = queries_raw.reshape(1, -1)

    G = len(gallery)
    Q = len(queries_raw)

    print(f"[RETRIEVAL] Processing {Q:,} queries across {G:,} gallery candidates on {_DEVICE}...", flush=True)
    if G == 0 or Q == 0:
        empty_res = np.zeros((Q, G), dtype=np.float32)
        return {"distances": empty_res} if is_sample_dict else empty_res

    # 1. Batched Gallery Feature Extraction with Dual-Pass Horizontal Flip TTA
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    batch_size = HW.batch_size
    num_workers = HW.num_io_workers
    gal_embeds_list = []
    gal_logits_list = []

    processed_count = 0
    for start_idx, end_idx, batch_x in prefetch_batches(gallery, batch_size=batch_size, num_workers=num_workers):
        batch_x = batch_x.to(_DEVICE, non_blocking=True)
        if HW.use_half:
            batch_x = batch_x.half()

        # Pass 1: Original orientation
        metric_emb_orig, attr_logits_orig = _VISUAL_MODEL(batch_x)

        # Pass 2: Horizontal Flip TTA
        batch_x_flip = torch.flip(batch_x, dims=[-1])
        metric_emb_flip, attr_logits_flip = _VISUAL_MODEL(batch_x_flip)

        # TTA Fusion: normalized sum for embeddings, mean for logits
        metric_emb = F.normalize(metric_emb_orig + metric_emb_flip, p=2, dim=-1)
        attr_logits = 0.5 * (attr_logits_orig + attr_logits_flip)

        gal_embeds_list.append(metric_emb.cpu().float())
        gal_logits_list.append(attr_logits.cpu().float())

        processed_count = end_idx
        if (processed_count // batch_size) % 25 == 0 or processed_count == G:
            elapsed_now = time.time() - t0
            fps_now = max(1.0, processed_count) / max(0.01, elapsed_now)
            eta_sec = (G - processed_count) / fps_now
            print(f"[RETRIEVAL] Processed {processed_count:,}/{G:,} images ({fps_now:.1f} img/s | Elapsed: {elapsed_now:.1f}s | ETA: {eta_sec:.1f}s)...", flush=True)

    gal_embeds = torch.cat(gal_embeds_list, dim=0).numpy()
    gal_logits = torch.cat(gal_logits_list, dim=0).numpy()

    del gal_embeds_list, gal_logits_list
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Multi-Temperature Continuous Query Prompt Ensembling
    print(f"[RETRIEVAL] Synthesizing multi-temperature continuous query representations for {Q:,} queries...", flush=True)
    q_raw_t = torch.from_numpy(queries_raw).float().to(_DEVICE)
    if HW.use_half:
        q_raw_t = q_raw_t.half()

    E_q_t = _ATTR_PROMPTER.forward_multi_temp(q_raw_t)
    query_embeds = E_q_t.cpu().float().numpy()

    del q_raw_t, E_q_t
    gc.collect()

    # 3. Test-Time Compute Suite (Prior-Matching Calibration + Alpha-QE + Cascaded Asymmetric Reranking)
    final_distances = _TT_ENGINE.run_inference_pipeline(
        gal_embeds_np=gal_embeds,
        gal_logits_np=gal_logits,
        query_embeds_np=query_embeds,
        queries_raw_np=queries_raw,
    )

    elapsed = time.time() - t0
    print(f"[RETRIEVAL] SOTA Championship V3 retrieval completed ({final_distances.shape}) in {elapsed:.2f}s!", flush=True)
    if is_sample_dict:
        return {"distances": final_distances}
    return final_distances


if __name__ == "__main__":
    print("Testing SOTA Championship V3 rank_gallery entrypoint...")
    dummy_sample = {
        "queries": np.random.binomial(1, 0.2, (5, 40)).astype(np.float32),
        "gallery": [{"image_path": "UPAR-Challenge-2027/data/Market1501/query/0001_c1s1_001051_00.jpg"}] * 10,
    }
    res = rank_gallery(dummy_sample)
    print("Self-test result shape:", res["distances"].shape)
    print("Self-test min:", res["distances"].min(), "max:", res["distances"].max())
    assert not np.isnan(res["distances"]).any()
    print("Self-test SUCCESS!")
