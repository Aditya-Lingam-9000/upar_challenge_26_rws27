"""Loss formulations for multi-attribute pedestrian recognition and retrieval.

Implements:
  1. Asymmetric Loss (ASL) for extreme multi-label positive/negative imbalance
  2. Category-Specific Softmax Loss for mutually exclusive attribute subsets
  3. Combined Multi-Task Loss
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label learning with severe class imbalance.

    Paper: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al., ICCV 2021)
    Suppresses easy negative gradients using asymmetric focusing parameters:
      gamma_neg=4.0, gamma_pos=0.0, clip=0.05
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw unnormalized logits
            targets: (B, C) binary ground-truth labels in {0, 1}
        """
        targets = targets.float()
        probs = torch.sigmoid(logits)

        # Asymmetric clipping on negatives
        probs_neg = (probs - self.clip).clamp(min=0.0)

        # Asymmetric focal weights
        pos_weight = (1.0 - probs).pow(self.gamma_pos)
        neg_weight = probs_neg.pow(self.gamma_neg)

        # Basic BCE terms with numerical stabilization
        pos_loss = targets * pos_weight * torch.log(probs.clamp(min=self.eps))
        neg_loss = (1.0 - targets) * neg_weight * torch.log((1.0 - probs_neg).clamp(min=self.eps))

        loss = -(pos_loss + neg_loss)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CategorySoftmaxLoss(nn.Module):
    """Category-Specific Softmax Loss for mutually exclusive attribute groups.

    Standard multi-label BCE treats all 40 attributes as independent binaries.
    In reality, certain attribute subsets are mutually exclusive:
      - Age (Young, Adult, Old)
      - Hair Length (Short, Long, Bald)
      - UpperBody Color (12 colors)
      - LowerBody Color (12 colors)
      - LowerBody Type (Trousers&Shorts, Skirt&Dress)

    Enforcing softmax within each group resolves cross-color bleeding and mutual exclusion violations.
    """
    # 40 UPAR Attributes and their indices:
    # 0..2: Age (Young, Adult, Old)
    # 3: Gender-Female
    # 4..6: Hair-Length (Short, Long, Bald)
    # 7: UpperBody-Length-Short
    # 8..19: UpperBody-Color (Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow, Other)
    # 20: LowerBody-Length-Short
    # 21..32: LowerBody-Color (Black, Blue, Brown, Green, Grey, Orange, Pink, Purple, Red, White, Yellow, Other)
    # 33..34: LowerBody-Type (Trousers&Shorts, Skirt&Dress)
    # 35..39: Accessories (Backpack, Bag, Glasses-Normal, Glasses-Sun, Hat)
    GROUPS: dict[str, tuple[int, int]] = {
        "age": (0, 3),             # indices 0, 1, 2
        "hair_length": (4, 7),     # indices 4, 5, 6
        "upper_color": (8, 20),    # indices 8..19 (12 colors)
        "lower_color": (21, 33),   # indices 21..32 (12 colors)
        "lower_type": (33, 35),    # indices 33, 34
    }

    def __init__(self, group_weight: float = 1.0) -> None:
        super().__init__()
        self.group_weight = group_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes categorical cross-entropy for groups where at least one label is active.
        """
        total_loss = torch.tensor(0.0, device=logits.device)
        num_valid_groups = 0

        for name, (start, end) in self.GROUPS.items():
            sub_logits = logits[:, start:end]    # (B, num_classes)
            sub_targets = targets[:, start:end]  # (B, num_classes)

            # Mask samples where at least one class in this group is active (sum >= 1)
            active_mask = (sub_targets.sum(dim=-1) > 0)
            if not active_mask.any():
                continue

            active_logits = sub_logits[active_mask]
            active_targets = sub_targets[active_mask]

            # Soft targets (in case multiple annotations were checked, normalize to distribution)
            target_dist = active_targets / active_targets.sum(dim=-1, keepdim=True).clamp(min=1.0)

            # Categorical cross entropy: - sum( p * log(softmax(x)) )
            log_probs = F.log_softmax(active_logits, dim=-1)
            group_loss = -(target_dist * log_probs).sum(dim=-1).mean()

            total_loss = total_loss + group_loss
            num_valid_groups += 1

        if num_valid_groups == 0:
            return torch.tensor(0.0, device=logits.device)

        return self.group_weight * (total_loss / num_valid_groups)


class CombinedABPRLoss(nn.Module):
    """Unified loss combining Asymmetric Multi-Label Loss and Category Group Softmax."""
    def __init__(
        self,
        asl_weight: float = 1.0,
        group_weight: float = 0.5,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.asl = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
        self.group_softmax = CategorySoftmaxLoss(group_weight=group_weight)
        self.asl_weight = asl_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss_asl = self.asl(logits, targets)
        loss_group = self.group_softmax(logits, targets)

        total_loss = self.asl_weight * loss_asl + loss_group

        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_asl": float(loss_asl.item()),
            "loss_group": float(loss_group.item()),
        }
