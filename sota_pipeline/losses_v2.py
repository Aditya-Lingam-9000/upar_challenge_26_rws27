"""SOTA Championship V2: Cross-Modal Additive Angular Margin (ArcFace) Loss Engine.

Formulates:
1. CrossModalMultiInstanceArcFaceLoss:
   - Scale s = 30.0, Angular Margin m = 0.35.
   - Multi-Instance attribute clustering: samples sharing identical 40-D attribute vectors
     are clustered as positives and masked from the negative denominator.
   - Dual-Pair Hard/Soft Query supervision: L = 0.30 * L_hard + 0.70 * L_soft.
2. AsymmetricLoss (ASL):
   - Decoupled focusing parameters (gamma- = 4.0, gamma+ = 0.0, margin clip = 0.05)
     for extreme class imbalance across rare pedestrian attributes.
3. SOTAChampionshipLossV2:
   - Unified multi-task loss combining Stage 1 and Stage 2 objectives.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalMultiInstanceArcFaceLoss(nn.Module):
    """
    Cross-Modal Multi-Instance Additive Angular Margin Loss (ArcFace).
    Optimizes unit hyperspherical alignment between visual person representations
    and dual-pathway (Hard + Soft) attribute queries.
    """

    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.35,
        beta_hard: float = 0.30,
        beta_soft: float = 0.70,
        eps: float = 1e-7
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.beta_hard = beta_hard
        self.beta_soft = beta_soft
        self.eps = eps

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def _apply_margin(self, cosine: torch.Tensor) -> torch.Tensor:
        """Applies monotonic angular margin: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)."""
        cosine_clamped = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)
        sine = torch.sqrt(1.0 - torch.pow(cosine_clamped, 2))
        phi = cosine_clamped * self.cos_m - sine * self.sin_m
        # Monotonic fallback past threshold
        phi = torch.where(cosine_clamped > self.th, phi, cosine_clamped - self.mm)
        return phi

    def _compute_pair_arcface(
        self,
        embeds_a: torch.Tensor,
        embeds_b: torch.Tensor,
        pos_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes multi-instance ArcFace loss between two sets of L2-normalized embeddings.
        Args:
            embeds_a: [B, D] Normalized visual features.
            embeds_b: [B, D] Normalized query features.
            pos_mask: [B, B] Boolean mask indicating ground-truth matches (y_i == y_j).
        """
        B = embeds_a.shape[0]
        # Cosine similarity matrix: [B, B]
        cosine = torch.matmul(embeds_a, embeds_b.T)
        phi = self._apply_margin(cosine)

        # For positive pairs, use phi (with margin); for negatives, use standard cosine
        logits = torch.where(pos_mask, phi, cosine) * self.scale

        # Multi-Instance Softmax Loss
        # Log-Sum-Exp across all candidates
        log_sum_exp = torch.logsumexp(logits, dim=-1, keepdim=True)  # [B, 1]
        log_probs = logits - log_sum_exp                            # [B, B]

        # Average log probability across true positive matches for each sample
        num_pos = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pos_log_probs = (log_probs * pos_mask.float()).sum(dim=-1, keepdim=True) / num_pos
        loss = -pos_log_probs.mean()

        return loss

    def forward(
        self,
        E_p: torch.Tensor,
        E_h: torch.Tensor,
        E_s: torch.Tensor,
        attr_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            E_p: [B, 1024] Visual person representation.
            E_h: [B, 512] Hard binary query embedding.
            E_s: [B, 512] Soft language query embedding.
            attr_targets: [B, 40] Binary ground truth attributes.
        """
        B = E_p.shape[0]

        # Multi-Instance Positive Mask: images sharing identical 40 attributes
        # [B, 1, 40] == [1, B, 40] -> [B, B]
        pos_mask = (attr_targets.unsqueeze(1) == attr_targets.unsqueeze(0)).all(dim=-1)

        # Visual sub-spaces matching Hard and Soft adapters
        E_p_hard = F.normalize(E_p[:, :512], p=2, dim=-1)
        E_p_soft = F.normalize(E_p[:, 512:], p=2, dim=-1)

        # 1. Hard Query Margin Loss
        loss_hard = self._compute_pair_arcface(E_p_hard, E_h, pos_mask)

        # 2. Soft Query Margin Loss
        loss_soft = self._compute_pair_arcface(E_p_soft, E_s, pos_mask)

        # 3. Unified Cross-Modal Loss
        loss_total = self.beta_hard * loss_hard + self.beta_soft * loss_soft

        return {
            "loss_arcface_total": loss_total,
            "loss_arcface_hard": loss_hard,
            "loss_arcface_soft": loss_soft
        }


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Pedestrian Attribute Classification.
    Dynamically downweights easy negative samples using decoupled gamma- focusing parameter.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal: bool = True
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal = disable_torch_grad_focal

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K] Raw classification logits.
            y: [B, K] Binary targets in {0, 1}.
        """
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Basic Cross Entropy
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing Weights
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1.0 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum(dim=-1).mean()


class SOTAChampionshipLossV2(nn.Module):
    """
    Unified Master Loss Engine for SOTA Championship V2.
    Supports Stage 1 (ASL + Strip Multi-Task) and Stage 2 (ArcFace Margin Learning).
    """

    def __init__(
        self,
        scale: float = 30.0,
        margin: float = 0.35,
        asl_weight: float = 1.0,
        arcface_weight: float = 2.0,
    ):
        super().__init__()
        self.arcface = CrossModalMultiInstanceArcFaceLoss(scale=scale, margin=margin)
        self.asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        self.asl_weight = asl_weight
        self.arcface_weight = arcface_weight

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        attr_targets: torch.Tensor,
        stage: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_outputs: Dictionary with E_p, E_h, E_s, attr_logits.
            attr_targets: [B, 40] Binary ground truth attributes.
            stage: 1 for representation warmup, 2 for ArcFace margin learning.
        """
        loss_dict: Dict[str, torch.Tensor] = {}

        # 1. Attribute Classification Loss (ASL)
        if "attr_logits" in model_outputs:
            loss_asl = self.asl(model_outputs["attr_logits"], attr_targets)
            loss_dict["loss_asl"] = loss_asl
        else:
            loss_asl = torch.tensor(0.0, device=attr_targets.device)

        # 2. Cross-Modal ArcFace Margin Loss
        arc_out = self.arcface(
            E_p=model_outputs["E_p"],
            E_h=model_outputs["E_h"],
            E_s=model_outputs["E_s"],
            attr_targets=attr_targets
        )
        loss_dict.update(arc_out)

        # Staged Weighting
        if stage == 1:
            # Stage 1: Focus primarily on representation learning and attribute classification
            total_loss = self.asl_weight * loss_asl + 0.50 * arc_out["loss_arcface_total"]
        else:
            # Stage 2: Heavy focus on ArcFace angular margin learning
            total_loss = 0.20 * loss_asl + self.arcface_weight * arc_out["loss_arcface_total"]

        loss_dict["loss_total"] = total_loss
        return loss_dict
