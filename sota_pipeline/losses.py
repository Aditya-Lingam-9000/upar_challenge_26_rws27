"""SOTA Loss Formulations for Unconstrained Cross-Modal Person Retrieval.

Implements:
1. MultiInstanceSDMLoss: Soft-target Similarity Distribution Matching (SDM)
   eliminating the false-negative penalty collapse for attribute-equivalent candidates.
2. AsymmetricLoss (ASL): Multi-label loss with decoupled focusing and margin clipping
   to conquer extreme attribute class imbalance (e.g. bald head, glasses, backpacks).
3. CrossModalMLMLoss: Conditioned visual-patch masked language token reconstruction.
4. StripSDMLoss: Anatomical strip consistency loss across Head, Torso, and Legs.
5. TotalChampionshipLoss: Master multi-task loss module.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInstanceSDMLoss(nn.Module):
    """
    Similarity Distribution Matching (SDM) with Multi-Instance Soft Targets.
    Eliminates the diagonal false-negative trap where attribute-identical pedestrians
    are incorrectly penalized as hard negatives.
    """

    def __init__(self, epsilon_hamming: int = 1, tau_init: float = 14.28):
        super().__init__()
        self.epsilon = epsilon_hamming
        # Learnable logit scale parameter (initialized to log(14.28) ~ 2.659)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(tau_init))

    def forward(
        self,
        img_embeds: torch.Tensor,
        txt_embeds: torch.Tensor,
        attr_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img_embeds: [B, D] L2-normalized or raw visual embeddings.
            txt_embeds: [B, D] L2-normalized or raw textual embeddings.
            attr_labels: [B, 40] Binary ground-truth attribute matrix.
        Returns:
            Scalar bidirectional soft SDM loss.
        """
        B = img_embeds.shape[0]
        v = F.normalize(img_embeds, p=2, dim=-1)
        t = F.normalize(txt_embeds, p=2, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        sim_v2t = torch.matmul(v, t.t()) * scale  # [B, B]
        sim_t2v = sim_v2t.t()                     # [B, B]

        # Pairwise Hamming distance between attribute profiles: ||a_i - a_j||_1
        diff = torch.abs(attr_labels.unsqueeze(1) - attr_labels.unsqueeze(0)).sum(dim=-1) # [B, B]

        # Multi-Instance Soft Targets:
        # Distance <= epsilon gets weight 1.0; distance == epsilon + 1 gets weight 0.5
        soft_targets = torch.zeros_like(diff, dtype=torch.float32)
        soft_targets[diff <= self.epsilon] = 1.0
        soft_targets[diff == (self.epsilon + 1)] = 0.5
        # Ensure diagonal self-match is always fully positive
        soft_targets.fill_diagonal_(1.0)

        # Normalize rows into valid probability distributions
        q_v2t = soft_targets / (soft_targets.sum(dim=-1, keepdim=True) + 1e-8)
        q_t2v = soft_targets.t() / (soft_targets.t().sum(dim=-1, keepdim=True) + 1e-8)

        # Compute cross-entropy with soft targets (equivalent to KL divergence)
        log_p_v2t = F.log_softmax(sim_v2t, dim=-1)
        log_p_t2v = F.log_softmax(sim_t2v, dim=-1)

        loss_v2t = -(q_v2t * log_p_v2t).sum(dim=-1).mean()
        loss_t2v = -(q_t2v * log_p_t2v).sum(dim=-1).mean()

        return 0.5 * (loss_v2t + loss_t2v)


class AsymmetricLoss(nn.Module):
    """
    Multi-Label Asymmetric Loss (ASL) with Margin Clipping and Gamma Decoupling.
    (Ridnik et al., ICCV 2021). Completely zeroes out gradients from easy background negatives.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = False
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K] Raw model output logits before sigmoid.
            targets: [B, K] Binary ground-truth attribute labels in {0, 1}.
        Returns:
            Scalar ASL multi-label classification loss.
        """
        targets = targets.float()
        probs = torch.sigmoid(logits)
        probs_neg = 1.0 - probs

        # Margin clipping on negative probabilities: p_m = max(p - m, 0)
        if self.clip > 0:
            probs_neg_clipped = (probs_neg + self.clip).clamp(max=1.0)
            probs_pos_from_clipped_neg = 1.0 - probs_neg_clipped
        else:
            probs_pos_from_clipped_neg = probs

        # Positive and negative cross-entropy components
        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(probs_neg_clipped.clamp(min=self.eps))

        # Asymmetric focusing weights
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1.0 - probs) ** self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * ((probs_pos_from_clipped_neg) ** self.gamma_neg)

        loss = -(loss_pos + loss_neg).sum(dim=-1).mean()
        return loss


class CrossModalMLMLoss(nn.Module):
    """
    Cross-Modal Masked Language Modeling Loss (IRRA / APTM).
    Conditioned on visual spatial patches to predict 15% masked description tokens.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, vocab_logits: torch.Tensor, masked_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vocab_logits: [B, SeqLen, VocabSize] Logits from cross-attention head.
            masked_labels: [B, SeqLen] Ground truth word indices (-100 for non-masked).
        """
        logits_flat = vocab_logits.view(-1, vocab_logits.shape[-1])
        labels_flat = masked_labels.view(-1)
        return self.criterion(logits_flat, labels_flat)


class StripSDMLoss(nn.Module):
    """
    Anatomical Strip Consistency Loss.
    Aligns Head, Torso, and Legs strip embeddings with corresponding sub-attribute text.
    """

    def __init__(self, tau_init: float = 14.28):
        super().__init__()
        self.sdm_head = MultiInstanceSDMLoss(epsilon_hamming=1, tau_init=tau_init)
        self.sdm_torso = MultiInstanceSDMLoss(epsilon_hamming=1, tau_init=tau_init)
        self.sdm_legs = MultiInstanceSDMLoss(epsilon_hamming=1, tau_init=tau_init)

    def forward(
        self,
        strip_embeds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        strip_text_embeds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attr_labels: torch.Tensor
    ) -> torch.Tensor:
        v_head, v_torso, v_legs = strip_embeds
        t_head, t_torso, t_legs = strip_text_embeds

        loss_h = self.sdm_head(v_head, t_head, attr_labels)
        loss_t = self.sdm_torso(v_torso, t_torso, attr_labels)
        loss_l = self.sdm_legs(v_legs, t_legs, attr_labels)

        return (loss_h + loss_t + loss_l) / 3.0


class TotalChampionshipLoss(nn.Module):
    """
    Master Multi-Task Loss Module combining:
    - Multi-Instance Soft-Target SDM (lambda_sdm)
    - Cross-Modal Masked Language Modeling (lambda_mlm)
    - Multi-Label Asymmetric Loss (lambda_asl)
    - Anatomical Strip SDM (lambda_strip)
    """

    def __init__(
        self,
        lambda_sdm: float = 1.0,
        lambda_mlm: float = 0.5,
        lambda_asl: float = 1.0,
        lambda_strip: float = 0.5,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        epsilon_hamming: int = 1,
    ):
        super().__init__()
        self.lambda_sdm = lambda_sdm
        self.lambda_mlm = lambda_mlm
        self.lambda_asl = lambda_asl
        self.lambda_strip = lambda_strip

        self.sdm_criterion = MultiInstanceSDMLoss(epsilon_hamming=epsilon_hamming)
        self.asl_criterion = AsymmetricLoss(gamma_neg=asl_gamma_neg, clip=asl_clip)
        self.mlm_criterion = CrossModalMLMLoss()
        self.strip_criterion = StripSDMLoss()

    def forward(
        self,
        img_metric: torch.Tensor,
        txt_metric: torch.Tensor,
        attr_logits: torch.Tensor,
        attr_labels: torch.Tensor,
        vocab_logits: Optional[torch.Tensor] = None,
        masked_labels: Optional[torch.Tensor] = None,
        strip_embeds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        strip_text_embeds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes weighted multi-task loss and returns scalar loss + detailed metric dict."""
        loss_sdm = self.sdm_criterion(img_metric, txt_metric, attr_labels)
        loss_asl = self.asl_criterion(attr_logits, attr_labels)

        total_loss = self.lambda_sdm * loss_sdm + self.lambda_asl * loss_asl
        metrics = {
            "loss_sdm": float(loss_sdm.item()),
            "loss_asl": float(loss_asl.item()),
        }

        # Optional MLM Loss
        if vocab_logits is not None and masked_labels is not None and self.lambda_mlm > 0:
            loss_mlm = self.mlm_criterion(vocab_logits, masked_labels)
            total_loss = total_loss + self.lambda_mlm * loss_mlm
            metrics["loss_mlm"] = float(loss_mlm.item())

        # Optional Strip Loss
        if strip_embeds is not None and strip_text_embeds is not None and self.lambda_strip > 0:
            loss_strip = self.strip_criterion(strip_embeds, strip_text_embeds, attr_labels)
            total_loss = total_loss + self.lambda_strip * loss_strip
            metrics["loss_strip"] = float(loss_strip.item())

        metrics["loss_total"] = float(total_loss.item())
        return total_loss, metrics
