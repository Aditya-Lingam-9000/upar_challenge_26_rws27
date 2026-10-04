"""Championship Metric & Margin Loss Formulations.

Implements:
1. Bidirectional Additive Angular Margin Loss (ArcFace) for hyperspherical alignment.
2. Similarity Distribution Matching Loss (SDM / IRRA InfoNCE) for probabilistic matching.
3. Multi-label Asymmetric Loss (ASL) for Stage 1 visual representation pretraining.
4. Unified ChampionshipLoss manager for seamless Two-Stage execution.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAngularMarginLoss(nn.Module):
    """Bidirectional ArcFace loss enforcing geodesic angular separation on hypersphere."""

    def __init__(self, s: float = 30.0, m: float = 0.35, easy_margin: bool = False):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def _arcface_logits(self, cosine: torch.Tensor) -> torch.Tensor:
        """Apply angular margin m to diagonal positive elements."""
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        batch_size = cosine.size(0)
        one_hot = torch.eye(batch_size, device=cosine.device, dtype=cosine.dtype)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.s

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Bidirectional ArcFace loss between two normalized embedding sets."""
        # feat_a: (B, D), feat_b: (B, D)
        cosine = torch.matmul(feat_a, feat_b.t())  # (B, B)
        logits_a2b = self._arcface_logits(cosine)
        logits_b2a = self._arcface_logits(cosine.t())

        labels = torch.arange(cosine.size(0), device=cosine.device)
        loss_a2b = F.cross_entropy(logits_a2b, labels)
        loss_b2a = F.cross_entropy(logits_b2a, labels)

        return 0.5 * (loss_a2b + loss_b2a)


class SimilarityDistributionMatchingLoss(nn.Module):
    """Bidirectional Similarity Distribution Matching (SDM) loss via KL divergence."""

    def __init__(self, tau: float = 0.02):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        feat_img: torch.Tensor,
        feat_query: torch.Tensor,
        identity_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute bidirectional KL divergence between similarity and true label distributions."""
        batch_size = feat_img.size(0)
        sim = torch.matmul(feat_img, feat_query.t()) / self.tau  # (B, B)

        # Ground truth matching distribution
        if identity_labels is not None:
            # Multi-instance matching when multiple crops share the same identity/attribute set
            pos_mask = (identity_labels.unsqueeze(0) == identity_labels.unsqueeze(1)).float()
            p_true = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            q_true = pos_mask / pos_mask.sum(dim=0, keepdim=True).clamp(min=1.0)
        else:
            # Standard diagonal pairing within batch
            p_true = torch.eye(batch_size, device=feat_img.device, dtype=feat_img.dtype)
            q_true = p_true

        # Log-softmax of predicted similarities
        log_p = F.log_softmax(sim, dim=1)
        log_q = F.log_softmax(sim.t(), dim=1)

        kl_i2t = F.kl_div(log_p, p_true, reduction="batchmean")
        kl_t2i = F.kl_div(log_q, q_true, reduction="batchmean")

        return 0.5 * (kl_i2t + kl_t2i)


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label attribute classification warmup."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Positive loss
        pos_probs = probs
        pos_loss = targets * torch.log(pos_probs.clamp(min=self.eps))
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1.0 - pos_probs) ** self.gamma_pos)

        # Negative loss with asymmetric margin clipping
        neg_probs = 1.0 - probs
        if self.clip > 0:
            neg_probs = (neg_probs + self.clip).clamp(max=1.0)
        neg_loss = (1.0 - targets) * torch.log(neg_probs.clamp(min=self.eps))
        if self.gamma_neg > 0:
            neg_loss = neg_loss * (probs ** self.gamma_neg)

        loss = -(pos_loss + neg_loss)
        return loss.mean()


class ChampionshipLoss(nn.Module):
    """Unified loss engine orchestrating Stage 1 PAR warmup and Stage 2 Metric Learning."""

    def __init__(
        self,
        stage: int = 2,
        s: float = 30.0,
        m: float = 0.35,
        beta_hard: float = 0.30,
        beta_soft: float = 0.70,
        beta_sdm: float = 0.20,
    ):
        super().__init__()
        self.stage = stage
        self.beta_hard = beta_hard
        self.beta_soft = beta_soft
        self.beta_sdm = beta_sdm

        self.arcface = AdditiveAngularMarginLoss(s=s, m=m)
        self.sdm = SimilarityDistributionMatchingLoss(tau=0.02)
        self.asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05)
        self.bce = nn.BCEWithLogitsLoss()

    def set_stage(self, stage: int) -> None:
        self.stage = stage

    def forward(
        self,
        model_outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        identity_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss depending on active training stage."""
        loss_dict = {}

        if self.stage == 1:
            # Stage 1: Representation warmup via multi-label classification
            aux_logits = model_outputs["aux_logits"]
            loss_asl = self.asl(aux_logits, targets)
            loss_dict["loss_asl"] = loss_asl
            loss_dict["total_loss"] = loss_asl

        else:
            # Stage 2: Cross-modal hyperspherical metric learning
            norm_v = model_outputs["norm_visual"]   # (B, 1024)
            norm_q = model_outputs["norm_query"]    # (B, 1024)
            norm_h = model_outputs["norm_hard"]     # (B, 512)
            norm_s = model_outputs["norm_soft"]     # (B, 512)

            # Visual slice corresponding to hard (first 512-D) and soft (last 512-D)
            norm_v_h = F.normalize(norm_v[:, :512], p=2, dim=-1)
            norm_v_s = F.normalize(norm_v[:, 512:], p=2, dim=-1)

            loss_arcface_hard = self.arcface(norm_v_h, norm_h)
            loss_arcface_soft = self.arcface(norm_v_s, norm_s)
            loss_sdm = self.sdm(norm_v, norm_q, identity_labels)

            total_loss = (
                self.beta_hard * loss_arcface_hard
                + self.beta_soft * loss_arcface_soft
                + self.beta_sdm * loss_sdm
            )

            loss_dict["loss_arcface_hard"] = loss_arcface_hard
            loss_dict["loss_arcface_soft"] = loss_arcface_soft
            loss_dict["loss_sdm"] = loss_sdm
            loss_dict["total_loss"] = total_loss

        return loss_dict
