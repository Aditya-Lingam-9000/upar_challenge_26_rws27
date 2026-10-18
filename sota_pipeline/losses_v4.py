"""V4 CLIP ViT-L/14 Championship Pipeline: Cross-Modal & Attribute Loss Functions.

This module provides the complete training objectives for the WACV 2027 UPAR Challenge
Track 2 Championship Pipeline:
1. `AsymmetricLoss` (ASL): Multi-label attribute classification loss with asymmetric focusing
   and probability margin clipping (Ridnik et al., ICCV 2021) to handle severe pedestrian class imbalance.
2. `SDMLoss`: Similarity Distribution Matching loss (IRRA, CVPR 2023) computing bidirectional
   Kullback-Leibler (KL) divergence with soft similarity distribution targets.
3. `ITCLoss`: Symmetric Image-Text Contrastive loss with learnable temperature logit scale.
4. `PromptDistillLoss`: Smooth L1 distillation objective transferring frozen OpenCLIP ViT-L/14
   text representations into the lightweight MLP attribute prompter.
5. `V4ChampionshipLoss`: Master unified loss combining ASL, SDM, ITC, and Prompter Distillation.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Asymmetric Loss (ASL) for Multi-Label Attribute Classification
# =====================================================================
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label pedestrian attribute classification.

    Addresses extreme positive-negative class imbalance (e.g. Bald at 1.26%, Hat at 4.25%)
    by decoupling positive and negative focusing parameters and shifting negative probabilities:
        p = sigmoid(logits)
        p_m = max(p - clip, 0)
        L_+ = - y * (1 - p)^gamma_pos * log(p)
        L_- = - (1 - y) * p_m^gamma_neg * log(1 - p_m)
        L = L_+ + L_-

    When p <= clip, p_m = 0, completely zeroing out gradients from easy background negatives.

    Ref: Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.

    Args:
        gamma_neg: Focusing parameter for negative samples (default: 4.0).
        gamma_pos: Focusing parameter for positive samples (default: 0.0).
        clip: Negative probability shift margin m (default: 0.05).
        reduction: Reduction method ('mean', 'sum', or 'none'). Default is 'mean'.
        eps: Small epsilon for numerical stability inside log (default: 1e-8).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes multi-label Asymmetric Loss.

        Args:
            logits: [B, num_attrs] Unnormalized classification logits.
            targets: [B, num_attrs] Binary ground-truth targets in {0, 1}.

        Returns:
            Scalar loss tensor (or unreduced tensor if reduction='none').
        """
        targets = targets.to(dtype=logits.dtype)
        p = torch.sigmoid(logits)

        # Positive loss component
        pos_loss = targets * torch.log(p.clamp(min=self.eps))
        if self.gamma_pos > 0.0:
            pos_loss = pos_loss * torch.pow(1.0 - p, self.gamma_pos)

        # Negative loss component with probability clipping margin
        if self.clip > 0.0:
            p_m = (p - self.clip).clamp(min=0.0)
        else:
            p_m = p

        neg_loss = (1.0 - targets) * torch.log((1.0 - p_m).clamp(min=self.eps))
        if self.gamma_neg > 0.0:
            neg_loss = neg_loss * torch.pow(p_m, self.gamma_neg)

        loss = -(pos_loss + neg_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")


# =====================================================================
# 2. Similarity Distribution Matching (SDM) Loss (IRRA, CVPR 2023)
# =====================================================================
class SDMLoss(nn.Module):
    """Similarity Distribution Matching (SDM) Loss from IRRA (CVPR 2023).

    Aligns global cross-modal representation distributions using bidirectional
    Kullback-Leibler (KL) divergence over soft similarity distributions rather than
    forcing rigid, hard 1-hot positive/negative assignments.

    Computes soft target probability distributions from batch similarity matrices,
    eliminating the false-negative penalty collapse when multiple gallery images
    or text queries share similar semantic profiles.

    Args:
        temperature: Temperature scale tau for cosine similarities (default: 0.02).
        eps: Small epsilon for numerical stabilization (default: 1e-8).
    """

    def __init__(self, temperature: float = 0.02, eps: float = 1e-8) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.eps = eps

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes bidirectional SDM loss between image and text embeddings.

        Args:
            image_embeds: [B, D] Visual embeddings. L2-normalized internally.
            text_embeds: [B, D] Textual embeddings. L2-normalized internally.
            targets: Optional [B, B] soft target similarity matrix. If None,
                     computed from normalized text semantic similarity matrix.

        Returns:
            Scalar bidirectional KL divergence loss.
        """
        # Ensure unit hypersphere normalization
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        batch_size = image_embeds.shape[0]
        if batch_size <= 1:
            # Fallback for degenerate single-item batch
            sim = (image_embeds * text_embeds).sum()
            return 1.0 - sim

        # Cross-modal cosine similarity matrix scaled by temperature: [B, B]
        sim_i2t = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        sim_t2i = sim_i2t.t()

        # Construct soft target distribution from teacher/text semantic similarities
        if targets is not None:
            # External soft targets provided (e.g. continuous Jaccard attribute similarity)
            target_mat = targets.to(dtype=image_embeds.dtype, device=image_embeds.device)
            p_target_i2t = F.softmax(target_mat / self.temperature, dim=-1)
            p_target_t2i = F.softmax(target_mat.t() / self.temperature, dim=-1)
        else:
            # Soft targets derived from semantic text-to-text similarity matrix
            sim_targets = torch.matmul(text_embeds.detach(), text_embeds.detach().t()) / self.temperature
            p_target_i2t = F.softmax(sim_targets, dim=-1)
            p_target_t2i = F.softmax(sim_targets.t(), dim=-1)

        # Predicted log-probability distributions
        log_p_i2t = F.log_softmax(sim_i2t, dim=-1)
        log_p_t2i = F.log_softmax(sim_t2i, dim=-1)

        # Bidirectional Kullback-Leibler Divergence: KL(target || predicted)
        # In PyTorch F.kl_div: input is log-probs, target is true probabilities
        loss_i2t = F.kl_div(log_p_i2t, p_target_i2t, reduction="batchmean")
        loss_t2i = F.kl_div(log_p_t2i, p_target_t2i, reduction="batchmean")

        loss_sdm = 0.5 * (loss_i2t + loss_t2i)
        return loss_sdm


# =====================================================================
# 3. Image-Text Contrastive (ITC) Loss
# =====================================================================
class ITCLoss(nn.Module):
    """Symmetric Image-Text Contrastive (ITC) Loss (InfoNCE).

    Optimizes dual cross-entropy objectives across the batch diagonal where each
    image matches its corresponding paired text description:
        L_ITC = 0.5 * ( CrossEntropy(S_i2t, y) + CrossEntropy(S_t2i, y) )
    where y = [0, 1, ..., B-1].

    Includes a learnable logit scale parameter initialized to log(1 / init_temperature)
    with gradient stabilization clipping at exp(logit_scale) <= 100.0.

    Args:
        init_temperature: Initial temperature parameter (default: 0.07).
    """

    def __init__(self, init_temperature: float = 0.07) -> None:
        super().__init__()
        self.init_temperature = float(init_temperature)
        # Learnable inverse temperature logit scale: log(1 / init_temperature)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / self.init_temperature), dtype=torch.float32)
        )

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Computes symmetric InfoNCE contrastive loss.

        Args:
            image_embeds: [B, D] Visual embeddings. L2-normalized internally.
            text_embeds: [B, D] Textual embeddings. L2-normalized internally.

        Returns:
            Scalar symmetric cross-entropy loss.
        """
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        batch_size = image_embeds.shape[0]
        # Diagonal matching labels: image i matches text i
        labels = torch.arange(batch_size, device=image_embeds.device, dtype=torch.long)

        # Scale clamped to prevent FP16 numerical explosion
        scale = self.logit_scale.exp().clamp(max=100.0)

        # Scaled cosine similarity matrices: [B, B]
        logits_per_image = scale * torch.matmul(image_embeds, text_embeds.t())
        logits_per_text = logits_per_image.t()

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        loss_itc = 0.5 * (loss_i2t + loss_t2i)
        return loss_itc


# =====================================================================
# 4. Prompter Distillation Loss
# =====================================================================
class PromptDistillLoss(nn.Module):
    """Distillation loss transferring CLIP text encoder knowledge into MLP Prompter.

    Computes Smooth L1 (Huber) regression loss between the ContinuousAttributePrompter
    embeddings and frozen OpenCLIP ViT-L/14 text embeddings. The text representations
    are strictly detached to prevent gradients from destabilizing the foundation text encoder.

    Args:
        beta: Threshold at which Smooth L1 transitions between L1 and L2 (default: 1.0).
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.criterion = nn.SmoothL1Loss(beta=self.beta)

    def forward(self, prompter_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """Computes Smooth L1 distillation loss.

        Args:
            prompter_embeds: [B, D] Predictions from ContinuousAttributePrompter.
            text_embeds: [B, D] Teacher targets from OpenCLIP text encoder.

        Returns:
            Scalar Smooth L1 distillation loss.
        """
        # Detach teacher representations to prevent backward gradients
        teacher_targets = text_embeds.detach()
        loss = self.criterion(prompter_embeds, teacher_targets)
        return loss


# =====================================================================
# 5. Master Championship V4 Combined Loss
# =====================================================================
class V4ChampionshipLoss(nn.Module):
    """Master combined loss engine for V4 ViT-L/14 Championship training.

    Unites four complementary learning objectives:
    1. Multi-label Attribute Classification (ASL, weight: w_asl=1.0)
    2. Similarity Distribution Matching (SDM, weight: w_sdm=1.0)
    3. Image-Text Symmetric Contrastive (ITC, weight: w_itc=0.5)
    4. Query Prompter Distillation (Smooth L1, weight: w_distill=2.0)

    Total Loss:
        L_total = w_asl * L_ASL + w_sdm * L_SDM + w_itc * L_ITC + w_distill * L_Distill

    Args:
        w_asl: Weight for Asymmetric multi-label attribute classification loss (default: 1.0).
        w_sdm: Weight for Similarity Distribution Matching loss (default: 1.0).
        w_itc: Weight for Image-Text Contrastive InfoNCE loss (default: 0.5).
        w_distill: Weight for Prompter distillation regression loss (default: 2.0).
        asl_kwargs: Optional dictionary of keyword arguments passed to `AsymmetricLoss`.
        sdm_kwargs: Optional dictionary of keyword arguments passed to `SDMLoss`.
        itc_kwargs: Optional dictionary of keyword arguments passed to `ITCLoss`.
    """

    def __init__(
        self,
        w_asl: float = 1.0,
        w_sdm: float = 1.0,
        w_itc: float = 0.5,
        w_distill: float = 2.0,
        asl_kwargs: Optional[Dict[str, Union[float, str]]] = None,
        sdm_kwargs: Optional[Dict[str, float]] = None,
        itc_kwargs: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.w_asl = float(w_asl)
        self.w_sdm = float(w_sdm)
        self.w_itc = float(w_itc)
        self.w_distill = float(w_distill)

        self.asl = AsymmetricLoss(**(asl_kwargs or {}))
        self.sdm = SDMLoss(**(sdm_kwargs or {}))
        self.itc = ITCLoss(**(itc_kwargs or {}))
        self.distill = PromptDistillLoss()

    def forward(
        self,
        img_embeds: torch.Tensor,
        attr_logits: torch.Tensor,
        attr_targets: torch.Tensor,
        text_embeds: torch.Tensor,
        prompter_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluates all component losses and returns the weighted total.

        Args:
            img_embeds: [B, D] Visual metric representations from CLIPAttributeRetrievalModel.
            attr_logits: [B, 40] Raw attribute logits from CLIPAttributeRetrievalModel.
            attr_targets: [B, 40] Ground truth binary attribute targets in {0, 1}.
            text_embeds: [B, D] Text representations from OpenCLIP ViT-L/14 text encoder.
            prompter_embeds: [B, D] Query representations from ContinuousAttributePrompter.

        Returns:
            Tuple of:
            - total_loss: Scalar weighted sum of all four objectives.
            - loss_dict: Dictionary containing individual detached loss values:
                         {"loss_asl", "loss_sdm", "loss_itc", "loss_distill", "loss_total"}.
        """
        loss_asl = self.asl(attr_logits, attr_targets)
        loss_sdm = self.sdm(img_embeds, text_embeds)
        loss_itc = self.itc(img_embeds, text_embeds)
        loss_distill = self.distill(prompter_embeds, text_embeds)

        total_loss = (
            self.w_asl * loss_asl
            + self.w_sdm * loss_sdm
            + self.w_itc * loss_itc
            + self.w_distill * loss_distill
        )

        loss_dict: Dict[str, torch.Tensor] = {
            "loss_asl": loss_asl.detach(),
            "loss_sdm": loss_sdm.detach(),
            "loss_itc": loss_itc.detach(),
            "loss_distill": loss_distill.detach(),
            "loss_total": total_loss.detach(),
        }

        return total_loss, loss_dict
