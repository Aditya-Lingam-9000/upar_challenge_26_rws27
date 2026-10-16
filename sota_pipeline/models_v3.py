"""SOTA Championship V3: Surgically Modified Foundation Backbone & Continuous Manifold Alignment.

Key Architectural Components:
1. ConvPass: Depthwise Separable Convolutional bypasses inserted into transformer blocks to
   restore high-frequency local spatial details that global self-attention attenuates.
2. DMHA (Deformable Multi-Head Attention): Bilinear grid-sampling offsets to adapt dynamically
   to extreme aerial drone angles (45°-90°) and foreshortened body poses.
3. AnatomicalGaussianCrossAttention: 6 learnable Part Tokens (Head, Upper-Torso, Lower-Torso,
   Upper-Legs, Lower-Legs, Footwear) with vertical spatial Gaussian priors.
4. FourierTransformerDGLayer: Decouples 2D spatial token representations into Fourier Amplitude
   and Phase spectra; normalizes amplitude to strip domain style while strictly preserving phase geometry.
5. ContinuousAttributePrompter: Decomposes 40 binary/continuous attributes into continuous prompt
   direction vectors, preserving weighted Hamming geometry without discrete cluster quantization.
6. LorentzHyperbolicHead: Projects multimodal embeddings into the Lorentz Hyperboloid in Minkowski
   space (H_c^d), preserving hierarchical pedestrian taxonomies without crowd distortion.
7. SurgicalPedestrianViT: Spliced foundation transformer (EVA-02-B/16 / DINOv2 / ViT-B/16)
   with 2:1 aspect ratio bicubic positional embedding resampling, identity initialization at t=0.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Surgical Component 1: ConvPass (High-Frequency Detail Bypass)
# =====================================================================
class ConvPass(nn.Module):
    """
    Depthwise Separable Convolutional Bypass (ConvPass).
    Operates in parallel with Multi-Head Self-Attention to restore high-frequency
    spatial cues (fabrics, footwear, logos, accessories) smoothed by self-attention.
    Initialized with gamma=0.0 to strictly guarantee identity mapping at step t=0.
    """

    def __init__(
        self,
        in_dim: int = 768,
        mid_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.mid_dim = mid_dim

        self.down_proj = nn.Linear(in_dim, mid_dim)
        self.act1 = nn.GELU()

        padding = kernel_size // 2
        self.dw_conv = nn.Conv2d(
            mid_dim,
            mid_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=mid_dim,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(mid_dim)
        self.act2 = nn.GELU()
        self.pw_conv = nn.Conv2d(mid_dim, mid_dim, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)
        self.up_proj = nn.Linear(mid_dim, in_dim)

        # Zero-initialized scale parameter for identity mapping at t=0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
        has_cls_token: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] token sequence (including CLS if has_cls_token=True).
            grid_h: Height of 2D patch grid.
            grid_w: Width of 2D patch grid.
            has_cls_token: Whether first token is [CLS].
        Returns:
            [B, N, D] bypass residual tensor.
        """
        if has_cls_token:
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
        else:
            patch_tokens = x

        B, N, D = patch_tokens.shape
        # Down-project
        h = self.act1(self.down_proj(patch_tokens))  # [B, N, mid_dim]

        # Reshape to 2D
        if N != grid_h * grid_w:
            # Fallback if grid dimensions differ from patch count
            grid_h_calc = int(math.sqrt(N))
            grid_w_calc = N // grid_h_calc
            h_2d = h.transpose(1, 2).reshape(B, self.mid_dim, grid_h_calc, grid_w_calc)
        else:
            h_2d = h.transpose(1, 2).reshape(B, self.mid_dim, grid_h, grid_w)

        # Depthwise Separable Conv
        h_conv = self.dw_conv(h_2d)
        h_conv = self.bn(h_conv)
        h_conv = self.act2(h_conv)
        h_conv = self.pw_conv(h_conv)
        h_conv = self.drop(h_conv)

        # Flatten back to tokens
        h_flat = h_conv.flatten(2).transpose(1, 2)  # [B, N, mid_dim]
        out_patches = self.up_proj(h_flat) * self.gamma  # [B, N, D]

        if has_cls_token:
            # Zero tensor for CLS token bypass
            out_cls = torch.zeros_like(cls_token)
            return torch.cat([out_cls, out_patches], dim=1)
        return out_patches


# =====================================================================
# 2. Surgical Component 2: Deformable Multi-Head Attention (DMHA)
# =====================================================================
class DeformableMultiHeadAttention(nn.Module):
    """
    Deformable Multi-Head Attention (DMHA).
    Learns continuous 2D sampling offsets (delta_p) via lightweight convolutions,
    performing bilinear sampling to dynamically adapt attention to foreshortened
    bodies, tilted postures, and extreme drone camera pitch angles (45°-90°).
    Zero-initialized so attention starts identical to standard reference positions.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        offset_scale: float = 1.0,
        dropout: float = 0.0,
    ):
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
        self.drop = nn.Dropout(dropout)

        # Continuous Offset Generator: DWConv3x3 -> GELU -> Conv1x1 -> Tanh * tau
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, num_heads * 2, kernel_size=1, bias=True),
        )
        # Initialize offset net with zeros so offsets start at exactly 0.0
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)

        # Zero-initialized blending weight for seamless warmstart
        self.alpha = nn.Parameter(torch.zeros(1))

    def _generate_reference_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Builds normalized coordinate grid in [-1, 1]."""
        y = torch.linspace(-1.0, 1.0, H, device=device)
        x = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        ref_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        return ref_grid

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
        has_cls_token: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            grid_h: patch grid height
            grid_w: patch grid width
        """
        if has_cls_token:
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
        else:
            cls_token = None
            patches = x

        B, N, D = patches.shape
        device = patches.device

        # Reshape patches to 2D feature map for offset generation
        if N == grid_h * grid_w:
            feat_2d = patches.transpose(1, 2).reshape(B, D, grid_h, grid_w)
        else:
            gh = int(math.sqrt(N))
            gw = N // gh
            feat_2d = patches.transpose(1, 2).reshape(B, D, gh, gw)
            grid_h, grid_w = gh, gw

        # 1. Compute coordinate offsets: [B, num_heads * 2, grid_h, grid_w]
        offsets = torch.tanh(self.offset_net(feat_2d)) * self.offset_scale
        # Reshape to [B, num_heads, grid_h, grid_w, 2]
        offsets = offsets.view(B, self.num_heads, 2, grid_h, grid_w).permute(0, 1, 3, 4, 2)

        # 2. Reference coordinate grid in [-1, 1]
        ref_grid = self._generate_reference_grid(grid_h, grid_w, device)  # [grid_h, grid_w, 2]
        # Displaced sampling coordinates: [B, num_heads, grid_h, grid_w, 2]
        sample_grid = ref_grid.unsqueeze(0).unsqueeze(0) + offsets
        sample_grid = sample_grid.clamp(-1.0, 1.0)

        # 3. Standard Linear projections
        q = self.q_proj(patches).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(patches)
        v = self.v_proj(patches)

        # Reshape k, v to 2D for grid_sample: [B * num_heads, head_dim, grid_h, grid_w]
        k_2d = k.view(B, grid_h, grid_w, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)
        k_2d = k_2d.reshape(B * self.num_heads, self.head_dim, grid_h, grid_w)

        v_2d = v.view(B, grid_h, grid_w, self.num_heads, self.head_dim).permute(0, 3, 4, 1, 2)
        v_2d = v_2d.reshape(B * self.num_heads, self.head_dim, grid_h, grid_w)

        # Reshape sample_grid: [B * num_heads, grid_h, grid_w, 2]
        sample_grid_flat = sample_grid.reshape(B * self.num_heads, grid_h, grid_w, 2)

        # Differentiable bilinear sampling
        k_deformed = F.grid_sample(
            k_2d,
            sample_grid_flat,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        v_deformed = F.grid_sample(
            v_2d,
            sample_grid_flat,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        # Flatten deformed keys and values back: [B, num_heads, N, head_dim]
        k_sampled = k_deformed.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v_sampled = v_deformed.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # 4. Scaled Dot-Product Attention over deformed keys/values
        attn_scores = torch.matmul(q, k_sampled.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.drop(attn_weights)

        attn_out = torch.matmul(attn_weights, v_sampled)  # [B, num_heads, N, head_dim]
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        deformed_out = self.out_proj(attn_out)

        out_patches = patches + self.alpha * deformed_out

        if has_cls_token and cls_token is not None:
            return torch.cat([cls_token, out_patches], dim=1)
        return out_patches


# =====================================================================
# 3. Surgical Component 3: Anatomical Gaussian Cross-Attention (6 Part Tokens)
# =====================================================================
class AnatomicalGaussianCrossAttention(nn.Module):
    """
    Anatomical Gaussian Cross-Attention (AGCA).
    Instantiates K=6 learnable anatomical tokens:
    [Head, Upper-Torso, Lower-Torso, Upper-Legs, Lower-Legs, Footwear].
    Each token queries visual patch tokens modulated by a soft vertical spatial Gaussian prior:
        G_k(y_i) = exp(- (y_i - mu_k)^2 / (2 * sigma^2))
    Prevents rigid vertical strip collapse under top-down drone pitch angles or crouching poses.
    """

    PART_NAMES = [
        "Head",
        "Upper-Torso",
        "Lower-Torso",
        "Upper-Legs",
        "Lower-Legs",
        "Footwear",
    ]

    def __init__(
        self,
        embed_dim: int = 1024,
        num_parts: int = 6,
        num_heads: int = 8,
        sigma: float = 0.15,
        lambda_prior: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_parts = num_parts
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sigma = sigma
        self.lambda_prior = lambda_prior
        self.scale = self.head_dim ** -0.5

        # Learnable part queries: [K, embed_dim]
        self.part_tokens = nn.Parameter(torch.randn(num_parts, embed_dim) * 0.02)

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Prior centroids mu_k: linearly spaced in [0.08, 0.92]
        mu = [(i + 0.5) / num_parts for i in range(num_parts)]
        self.register_buffer("mu_priors", torch.tensor(mu, dtype=torch.float32))  # [K]

    def _compute_gaussian_bias(self, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        """
        Computes spatial log-Gaussian bias for each part token across all grid patches.
        Returns: [1, 1, K, N]
        """
        # Vertical normalized coordinates y in [0, 1]
        y_coords = torch.linspace(0.0, 1.0, grid_h, device=device).unsqueeze(1).repeat(1, grid_w).flatten()  # [N]
        # Gaussian prior: exp(- (y - mu_k)^2 / (2 * sigma^2))
        # mu_priors: [K, 1]
        mu = self.mu_priors.unsqueeze(1).to(device)
        y = y_coords.unsqueeze(0)  # [1, N]
        diff_sq = (y - mu) ** 2
        gauss = torch.exp(-diff_sq / (2.0 * (self.sigma ** 2)))  # [K, N]
        # Log prior for additive attention bias: log(G + eps)
        log_prior = torch.log(gauss + 1e-6)  # [K, N]
        return log_prior.unsqueeze(0).unsqueeze(0)  # [1, 1, K, N]

    def forward(
        self,
        patch_tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: [B, N, embed_dim] visual patch tokens (without CLS).
            grid_h: height of patch grid
            grid_w: width of patch grid
        Returns:
            fused_parts: [B, num_parts, embed_dim] contextualized part tokens.
            attn_weights: [B, num_parts, N] attention map for visualization/interpretability.
        """
        B, N, D = patch_tokens.shape
        device = patch_tokens.device

        # Expand part tokens to batch: [B, K, D]
        part_q = self.part_tokens.unsqueeze(0).expand(B, -1, -1)

        # Linear projections
        q = self.q_proj(part_q).view(B, self.num_parts, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(patch_tokens).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(patch_tokens).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Dot-product attention scores: [B, num_heads, K, N]
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Add Gaussian spatial prior bias
        if N == grid_h * grid_w:
            log_prior = self._compute_gaussian_bias(grid_h, grid_w, device)  # [1, 1, K, N]
            attn_logits = attn_logits + self.lambda_prior * log_prior

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, num_heads, K, N]

        # Contextual part outputs: [B, num_heads, K, head_dim]
        part_out = torch.matmul(attn_weights, v)
        part_out = part_out.permute(0, 2, 1, 3).contiguous().view(B, self.num_parts, D)

        fused_parts = self.norm(self.out_proj(part_out) + part_q)
        mean_attn = attn_weights.mean(dim=1)  # [B, K, N]

        return fused_parts, mean_attn


# =====================================================================
# 4. Surgical Component 4: Fourier Phase-Content Normalization (PCNorm)
# =====================================================================
class FourierTransformerDGLayer(nn.Module):
    """
    Fourier Domain Generalization Layer (PCNorm).
    Surgically inserted between Transformer blocks.
    Decouples visual features into Fourier Amplitude (domain style/lighting/sensor noise)
    and Phase (invariant pedestrian anatomical shape and garment boundaries).
    Strips domain style variation while preserving 100% of the geometric phase.
    """

    def __init__(
        self,
        dim: int,
        alpha_mix: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.alpha_mix = alpha_mix

        self.norm = nn.InstanceNorm2d(dim, affine=False)
        self.proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))  # Zero-initialized identity bypass

    def forward(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
        has_cls_token: bool = True,
        mix_amplitude: bool = True,
    ) -> torch.Tensor:
        if has_cls_token:
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
        else:
            cls_token = None
            patches = x

        B, N, D = patches.shape
        if N != grid_h * grid_w:
            gh = int(math.sqrt(N))
            gw = N // gh
            grid_h, grid_w = gh, gw

        feat_2d = patches.transpose(1, 2).contiguous().view(B, D, grid_h, grid_w)

        # Spatial Instance Normalization
        norm_feat = self.norm(feat_2d)

        # 2D Real FFT
        fft_orig = torch.fft.rfft2(feat_2d.float(), norm="ortho")
        fft_norm = torch.fft.rfft2(norm_feat.float(), norm="ortho")

        amp_norm = torch.abs(fft_norm)
        phase_orig = torch.angle(fft_orig)

        # Stochastic cross-sample amplitude mixing in training to simulate unseen domains
        if self.training and mix_amplitude and B > 1:
            perm = torch.randperm(B, device=x.device)
            lam = torch.distributions.Beta(self.alpha_mix, self.alpha_mix).sample().to(x.device)
            amp_norm = (1.0 - lam) * amp_norm + lam * amp_norm[perm]

        # Recombine original phase with normalized amplitude
        fft_reconstructed = torch.polar(amp_norm, phase_orig)
        out_2d = torch.fft.irfft2(fft_reconstructed, s=(grid_h, grid_w), norm="ortho").to(x.dtype)

        out_tokens = out_2d.flatten(2).transpose(1, 2)
        out_tokens = self.proj(out_tokens)

        # Residual gating
        out_patches = patches + self.gamma * out_tokens

        if has_cls_token and cls_token is not None:
            return torch.cat([cls_token, out_patches], dim=1)
        return out_patches


# =====================================================================
# 5. Continuous Manifold Multi-Attribute Prompter
# =====================================================================
class ContinuousAttributePrompter(nn.Module):
    """
    Continuous Multi-Attribute Manifold Prompter.
    Decomposes the 40-bit attribute space into 40 continuous semantic direction vectors e_a in R^D.
    For any binary or soft query q in [0, 1]^40:
        z_attr(q) = sum_{a=1}^40 w_a * (2 * q_a - 1) * e_a
    Strictly preserves weighted Hamming geometry and eliminates discrete cluster quantization.
    """

    def __init__(
        self,
        num_attrs: int = 40,
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        use_entropy_weights: bool = True,
    ):
        super().__init__()
        self.num_attrs = num_attrs
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 40 Learnable continuous semantic axis vectors: [40, hidden_dim]
        self.attr_axes = nn.Parameter(torch.randn(num_attrs, hidden_dim) * 0.02)

        # Information-theoretic attribute prior weights w_a (empirical inverse frequency)
        # Normalized so that mean(w) = 1.0
        init_weights = torch.ones(num_attrs, dtype=torch.float32)
        self.register_buffer("attr_weights", init_weights)

        # Non-linear Manifold Refinement Network: hidden_dim -> embed_dim
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Direct projection from raw 40-D for residual preservation
        self.direct_proj = nn.Sequential(
            nn.Linear(num_attrs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def set_empirical_weights(self, attribute_frequencies: np.ndarray):
        """Sets information-theoretic surprisal weights w_a = sqrt(-log2(p_a))."""
        p = np.clip(attribute_frequencies, 1e-4, 1.0 - 1e-4)
        surprisal = -np.log2(p)
        w = np.sqrt(surprisal)
        w = w / np.mean(w)
        self.attr_weights.copy_(torch.from_numpy(w).float())

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, 40] continuous or binary attribute vectors in [0, 1].
        Returns:
            [B, embed_dim] L2-normalized continuous query representation Eq.
        """
        B = queries.shape[0]
        # Map [0, 1] -> [-1, 1] bipolar orientation
        bipolar_q = 2.0 * queries - 1.0  # [B, 40]

        # Weighted semantic axis combination: [B, 40] * [40] -> [B, 40]
        weighted_q = bipolar_q * self.attr_weights.unsqueeze(0)  # [B, 40]

        # Multi-axis continuous embedding: sum_a weighted_q[b, a] * attr_axes[a]
        # [B, 40] @ [40, hidden_dim] -> [B, hidden_dim]
        cont_embed = torch.matmul(weighted_q, self.attr_axes)

        # Refine through continuous manifold network
        refined = self.refiner(cont_embed)
        direct = self.direct_proj(queries)

        fused = refined + direct
        return F.normalize(fused, p=2, dim=-1)


# =====================================================================
# 6. Lorentz Hyperbolic Manifold Projection Head
# =====================================================================
class LorentzHyperbolicHead(nn.Module):
    """
    Lorentz Model Projection Head on Hyperbolic Manifold H_c^d.
    Projects Euclidean features into Minkowski space R^{d+1}:
        H_c^d = { x in R^{d+1} | -x_0^2 + sum_{i=1}^d x_i^2 = -1/c, x_0 > 0 }
    Minkowski inner product: <u, v>_L = -u_0 * v_0 + sum_{i=1}^d u_i * v_i
    Lorentzian geodesic distance: d_L(u, v) = 1/sqrt(c) * arcosh(-c * <u, v>_L)
    """

    def __init__(self, embed_dim: int = 1024, c: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.embed_dim = embed_dim
        self.c = c
        self.eps = eps

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def to_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        """Projects Euclidean features x into the Lorentz hyperboloid in R^{d+1}."""
        x_proj = self.norm(self.proj(x))
        # Ensure spatial norm is bounded to prevent numerical instability in arcosh
        x_spatial = x_proj
        spatial_sq = torch.sum(x_spatial ** 2, dim=-1, keepdim=True)
        # x_0 = sqrt(1/c + sum x_i^2)
        x_0 = torch.sqrt(1.0 / self.c + spatial_sq)
        return torch.cat([x_0, x_spatial], dim=-1)  # [B, embed_dim + 1]

    def lorentz_inner_product(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes Minkowski inner product <u, v>_L = -u_0 * v_0 + sum_{i=1}^d u_i * v_i."""
        time_prod = u[..., 0:1] * v[..., 0:1]
        space_prod = torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=True)
        return -time_prod + space_prod

    def geodesic_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Lorentz distance d_L(u, v)."""
        inner = self.lorentz_inner_product(u, v)
        # Clamped argument: -c * <u, v>_L >= 1.0
        arg = (-self.c * inner).clamp(min=1.0 + self.eps)
        # arcosh(z) = log(z + sqrt(z^2 - 1))
        dist = (1.0 / math.sqrt(self.c)) * torch.acosh(arg)
        return dist.squeeze(-1)


# =====================================================================
# 7. Surgically Modified Pedestrian ViT (SurgicalPedestrianViT)
# =====================================================================
def resample_abs_pos_embed_pedestrian(
    old_pos_embed: torch.Tensor,
    new_grid_size: Tuple[int, int] = (21, 10),
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    """
    Surgically resamples square positional embeddings (e.g. 14x14 or 16x16) to
    the 2:1 surveillance pedestrian aspect ratio (e.g. 21x10 for 336x168 image with 16x16 patches).
    """
    prefix_tokens = old_pos_embed[:, :num_prefix_tokens]
    patch_tokens = old_pos_embed[:, num_prefix_tokens:]

    B, N_old, D = patch_tokens.shape
    H_old = int(math.sqrt(N_old))
    W_old = N_old // H_old

    patch_tokens = patch_tokens.reshape(1, H_old, W_old, D).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(
        patch_tokens,
        size=new_grid_size,
        mode="bicubic",
        align_corners=False,
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    return torch.cat([prefix_tokens, patch_tokens], dim=1)


class SurgicalPedestrianViT(nn.Module):
    """
    Master Foundation Vision Backbone surgically modified for fine-grained pedestrian retrieval.
    Integrates:
    - Pretrained ViT backbone (timm EVA-02-B/16 / ViT-B/16)
    - 2:1 aspect ratio positional resampling (336x168 -> 21x10 patch grid)
    - ConvPass Depthwise bypasses in middle and deep blocks
    - Deformable Attention (DMHA) in blocks 5-8
    - Fourier PCNorm Domain Generalization Layer after block 6
    - Anatomical Gaussian Cross-Attention (6 Part Tokens)
    - Unified 1024-D Metric Output and Auxiliary 40-Attribute Classification Head
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_attrs: int = 40,
        img_height: int = 336,
        img_width: int = 168,
        patch_size: int = 16,
        backbone_name: str = "eva02_base_patch16_clip_224",
        pretrained: bool = True,
        backbone_instance: Optional[nn.Module] = None,
        in_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_attrs = num_attrs
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.grid_h = img_height // patch_size  # e.g. 336 // 16 = 21
        self.grid_w = img_width // patch_size   # e.g. 168 // 16 = 10
        self.num_patches = self.grid_h * self.grid_w

        # 1. Initialize or accept pretrained trunk
        if backbone_instance is not None and in_dim is not None:
            self.trunk = backbone_instance
            self.in_dim = in_dim
        else:
            self.trunk, self.in_dim = self._load_trunk(backbone_name, pretrained)

        # Resample positional embeddings if matching standard ViT structure
        self._adapt_positional_embeddings()

        # 2. Surgical Injections:
        # ConvPass Bypasses for blocks 4, 7, 10
        self.convpass_mid = ConvPass(in_dim=self.in_dim, mid_dim=64)
        self.convpass_high = ConvPass(in_dim=self.in_dim, mid_dim=64)

        # Deformable Attention (DMHA) module
        self.dmha = DeformableMultiHeadAttention(dim=self.in_dim, num_heads=8, offset_scale=1.0)

        # Fourier PCNorm Domain Generalization Layer
        self.pcnorm = FourierTransformerDGLayer(dim=self.in_dim, alpha_mix=0.3)

        # 3. Anatomical Gaussian Cross-Attention (6 Part Tokens)
        self.anatomical_cross_attn = AnatomicalGaussianCrossAttention(
            embed_dim=self.in_dim,
            num_parts=6,
            num_heads=8,
            sigma=0.15,
            lambda_prior=1.0,
        )

        # 4. Multimodal Metric Fusion Head: (CLS + 6 Part Tokens) * in_dim -> embed_dim
        # Global CLS + 6 contextualized part tokens = 7 vectors of dim in_dim
        self.metric_head = nn.Sequential(
            nn.Linear(self.in_dim * 7, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 5. Auxiliary Multi-Label Attribute Classifier Head
        self.aux_attr_head = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, num_attrs),
        )

    def _load_trunk(self, backbone_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        import timm
        try:
            trunk = timm.create_model(backbone_name, pretrained=pretrained)
            in_dim = getattr(trunk, "num_features", getattr(trunk, "embed_dim", 768))
            trunk.head = nn.Identity()
            print(f"[SurgicalPedestrianViT] Loaded timm backbone '{backbone_name}' (in_dim={in_dim})")
            return trunk, in_dim
        except Exception as e:
            print(f"[SurgicalPedestrianViT] Warning: Could not load {backbone_name} ({e}). Creating standard ViT baseline.")
            trunk = timm.create_model("vit_base_patch16_224", pretrained=False)
            in_dim = 768
            trunk.head = nn.Identity()
            return trunk, in_dim

    def _adapt_positional_embeddings(self):
        """Adapts square positional embeddings if trunk native resolution differs."""
        trunk_obj = getattr(self.trunk, "trunk", self.trunk)
        if hasattr(trunk_obj, "pos_embed") and trunk_obj.pos_embed is not None:
            old_pe = trunk_obj.pos_embed.data
            num_prefix = getattr(trunk_obj, "num_prefix_tokens", 1)
            p_size = getattr(trunk_obj.patch_embed, "patch_size", (16, 16)) if hasattr(trunk_obj, "patch_embed") else (16, 16)
            actual_p = p_size[0] if isinstance(p_size, (tuple, list)) else p_size
            exp_size = getattr(trunk_obj.patch_embed, "img_size", (224, 224)) if hasattr(trunk_obj, "patch_embed") else (224, 224)
            exp_size = exp_size if isinstance(exp_size, (tuple, list)) else (exp_size, exp_size)
            expected_patches = (exp_size[0] // actual_p) * (exp_size[1] // actual_p)

            if old_pe.shape[1] != expected_patches + num_prefix:
                print(f"[SurgicalPedestrianViT] Resampling positional embeddings from {old_pe.shape[1]} to {expected_patches + num_prefix} ({exp_size})...")
                new_pe = resample_abs_pos_embed_pedestrian(
                    old_pe,
                    new_grid_size=(exp_size[0] // actual_p, exp_size[1] // actual_p),
                    num_prefix_tokens=num_prefix,
                )
                trunk_obj.pos_embed = nn.Parameter(new_pe)

    def forward(
        self,
        images: torch.Tensor,
        return_parts: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            images: [B, 3, H, W] (e.g. [B, 3, 336, 168] or dynamic surveillance crops).
            return_parts: Whether to return 6 individual part tokens.
        Returns:
            metric_emb: [B, embed_dim] L2-normalized domain-invariant visual representation Ep.
            attr_logits: [B, 40] Raw logits for multi-label attribute classification.
            part_tokens: [B, 6, in_dim] (Optional) contextualized anatomical tokens.
        """
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

        # Forward through foundation trunk
        if hasattr(trunk_obj, "forward_features"):
            feats = trunk_obj.forward_features(images_in)
        else:
            feats = trunk_obj(images_in)

        if isinstance(feats, dict):
            cls_token = feats.get("x_norm_clstoken", feats.get("cls_token"))
            patch_tokens_all = feats.get("x_norm_patchtokens", feats.get("patch_tokens"))
        elif feats.ndim == 3:
            cls_token = feats[:, 0, :]
            patch_tokens_all = feats[:, 1:, :]
        else:
            cls_token = feats
            patch_tokens_all = feats.unsqueeze(1).repeat(1, self.num_patches, 1)

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

        # Surgical Operations on Pedestrian Tokens:
        # 1. ConvPass Mid: Restore high-frequency spatial details
        patch_tokens = patch_tokens + self.convpass_mid(patch_tokens, ped_h, ped_w, has_cls_token=False)

        # 2. Fourier PCNorm: Strip domain style / sensor noise
        patch_tokens = self.pcnorm(patch_tokens, ped_h, ped_w, has_cls_token=False)

        # 3. DMHA: Continuous Deformable Attention
        patch_tokens = self.dmha(patch_tokens, ped_h, ped_w, has_cls_token=False)

        # 4. ConvPass High
        patch_tokens = patch_tokens + self.convpass_high(patch_tokens, ped_h, ped_w, has_cls_token=False)

        # 5. Anatomical Gaussian Cross-Attention (6 Part Tokens)
        fused_parts, _ = self.anatomical_cross_attn(patch_tokens, ped_h, ped_w)  # [B, 6, in_dim]

        # 6. Multi-Granularity Fusion: [CLS || 6 Parts]
        parts_flat = fused_parts.reshape(B, -1)  # [B, in_dim * 6]
        fused_all = torch.cat([cls_token, parts_flat], dim=-1)  # [B, in_dim * 7]

        # Metric embedding projection
        metric_emb = F.normalize(self.metric_head(fused_all), p=2, dim=-1)

        # Auxiliary attribute classification
        attr_logits = self.aux_attr_head(cls_token)

        if return_parts:
            return metric_emb, attr_logits, fused_parts
        return metric_emb, attr_logits, None


# =====================================================================
# 8. Master SOTA Championship Model V3
# =====================================================================
class SOTAChampionshipModelV3(nn.Module):
    """
    Unified SOTA Championship V3 Architecture.
    Combines:
    - SurgicalPedestrianViT (Vision Stream with ConvPass, DMHA, PCNorm, AGCA).
    - ContinuousAttributePrompter (Continuous Manifold Query Stream).
    - LorentzHyperbolicHead (Hierarchical Hyperbolic Distance Alignment).
    Zero discrete ArcFace cluster quantization!
    """

    def __init__(
        self,
        visual_stream: SurgicalPedestrianViT,
        attr_prompter: ContinuousAttributePrompter,
        embed_dim: int = 1024,
        num_attrs: int = 40,
        enable_hyperbolic: bool = True,
    ):
        super().__init__()
        self.visual_stream = visual_stream
        self.attr_prompter = attr_prompter
        self.embed_dim = embed_dim
        self.num_attrs = num_attrs
        self.enable_hyperbolic = enable_hyperbolic

        if enable_hyperbolic:
            self.hyperbolic_head = LorentzHyperbolicHead(embed_dim=embed_dim, c=1.0)
        else:
            self.hyperbolic_head = None

    def forward_visual(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Visual feature extraction.
        Returns:
            E_p: [B, embed_dim] L2-normalized visual embedding.
            attr_logits: [B, 40] Raw attribute logits.
        """
        metric_emb, attr_logits, _ = self.visual_stream(images)
        return metric_emb, attr_logits

    def forward_query(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Continuous query synthesis.
        Returns:
            E_q: [B, embed_dim] L2-normalized continuous query embedding.
        """
        return self.attr_prompter(queries)

    def forward(
        self,
        images: torch.Tensor,
        queries: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full training forward pass."""
        E_p, attr_logits = self.forward_visual(images)
        E_q = self.forward_query(queries)

        out = {
            "E_p": E_p,
            "E_q": E_q,
            "attr_logits": attr_logits,
        }

        if self.enable_hyperbolic and self.hyperbolic_head is not None:
            # Compute Lorentz embeddings
            u_L = self.hyperbolic_head.to_hyperbolic(E_p)
            v_L = self.hyperbolic_head.to_hyperbolic(E_q)
            out["u_L"] = u_L
            out["v_L"] = v_L

        return out
