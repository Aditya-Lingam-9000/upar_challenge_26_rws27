"""Stage 2: Cross-Modal Metric Learning & ArcFace Alignment Engine for ABPR.

Implements Phase 2 decoupled metric learning (CLEAR paradigm):
  1. VisualProjector: Projects 768-D backbone features into 512-D L2-normalized embedding.
  2. QueryEncoder: Projects 40-D discrete/soft attribute queries into 512-D L2-normalized embedding.
  3. ArcFaceLoss: Additive Angular Margin classification loss (s=30.0, m=0.20) over 7,204 identities.
  4. CrossModalContrastiveLoss: InfoNCE symmetric contrastive loss between queries and images.
  5. HybridScorer: Combines calibrated attribute L1 distance with cross-modal metric cosine distance.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualProjector(nn.Module):
    """Projects backbone features (768-D) to unit-hypersphere metric space (512-D)."""

    def __init__(self, in_features: int = 768, embedding_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return F.normalize(feat, p=2, dim=-1)


class QueryEncoder(nn.Module):
    """Projects 40-D attribute queries into the shared 512-D unit-hypersphere metric space."""

    def __init__(self, num_attributes: int = 40, embedding_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_attributes, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-6),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        feat = self.net(q)
        return F.normalize(feat, p=2, dim=-1)


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace).

    Forces intra-class compactness and inter-class discrepancy on the unit hypersphere.
    Formula: cos(theta + m) for ground truth class, cos(theta) for other classes, scaled by s.
    """

    def __init__(
        self,
        num_classes: int = 7204,
        embedding_dim: int = 512,
        scale: float = 30.0,
        margin: float = 0.20,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin

        # Class weight centers on unit hypersphere
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) L2-normalized embeddings.
            labels: (B,) class identity indices in [0, num_classes - 1].
        """
        # Normalize weights to unit hypersphere
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: (B, C)
        cosine = F.linear(embeddings, norm_weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sine = torch.sqrt(1.0 - torch.square(cosine))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle numerical stability for large angles (theta + m > pi)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot mask for ground-truth classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply margin only to ground-truth angle
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return F.cross_entropy(output, labels)


class CrossModalContrastiveLoss(nn.Module):
    """Symmetric InfoNCE Loss aligning Query Embeddings with Visual Embeddings."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb: torch.Tensor, vis_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: (B, D) L2-normalized query embeddings.
            vis_emb: (B, D) L2-normalized image embeddings.
        """
        # Similarity matrix: (B, B)
        sim = torch.matmul(query_emb, vis_emb.t()) / self.temperature

        # Symmetric cross-entropy targets: diagonal elements (0, 1, ..., B-1)
        labels = torch.arange(query_emb.size(0), device=query_emb.device)
        loss_q2v = F.cross_entropy(sim, labels)
        loss_v2q = F.cross_entropy(sim.t(), labels)

        return 0.5 * (loss_q2v + loss_v2q)


def compute_hybrid_distances(
    query_vectors: np.ndarray,
    gallery_probs: np.ndarray,
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    lambda_metric: float = 0.25,
) -> np.ndarray:
    """Computes combined Attribute L1 + Visual-Semantic Cosine Distance matrix.

    Args:
        query_vectors: (Q, 40) binary attribute queries.
        gallery_probs: (G, 40) calibrated predicted attribute probabilities.
        query_embeddings: (Q, 512) L2-normalized query embeddings.
        gallery_embeddings: (G, 512) L2-normalized gallery embeddings.
        lambda_metric: weighting factor for metric cosine distance in [0, 1].

    Returns:
        distances: (Q, G) distance matrix.
    """
    # 1. Attribute continuous L1 distance: (Q, G)
    attr_dist = gallery_probs.sum(axis=1, keepdims=True).T + query_vectors @ (1.0 - 2.0 * gallery_probs).T
    # Scale to [0, 1] range (divide by 40 attributes)
    attr_dist = attr_dist / 40.0

    # 2. Metric cosine distance: 1 - cosine_similarity in [0, 2]
    # Scaled to [0, 1]: (1 - cos) / 2
    cos_sim = query_embeddings @ gallery_embeddings.T  # (Q, G)
    metric_dist = 0.5 * (1.0 - cos_sim)

    # 3. Hybrid fusion
    hybrid_dist = (1.0 - lambda_metric) * attr_dist + lambda_metric * metric_dist
    return hybrid_dist.astype(np.float32)


def evaluate_distance_matrix(
    distances: np.ndarray,
    gallery_gt: np.ndarray,
    queries: np.ndarray,
) -> dict[str, float]:
    """Computes official UPAR retrieval metrics (mADM, mAP, Rank-1, Rank-5, Rank-10, mINP)

    directly from a distance matrix of shape (Q, G).
    """
    Q = queries.shape[0]
    diffs = np.abs(queries[:, None, :] - gallery_gt[None, :, :]).sum(axis=-1)
    is_correct = (diffs == 0)
    ranks = np.argsort(distances, axis=1)

    adms, aps, r1_list, r5_list, r10_list, inps = [], [], [], [], [], []
    dom = 1.0 - (diffs / 40.0)
    mu_dom = np.mean(dom, axis=1, keepdims=True)
    denom = np.maximum(1.0 - mu_dom, 1e-6)
    ndom = np.maximum(0.0, (dom - mu_dom) / denom)

    for q_idx in range(Q):
        correct_mask = is_correct[q_idx]
        num_pos = np.sum(correct_mask)
        if num_pos == 0:
            continue
        q_ranks = ranks[q_idx]
        correct_positions = np.where(correct_mask[q_ranks])[0] + 1

        r1_list.append(1.0 if correct_positions[0] <= 1 else 0.0)
        r5_list.append(1.0 if correct_positions[0] <= 5 else 0.0)
        r10_list.append(1.0 if correct_positions[0] <= 10 else 0.0)

        precision_at_k = np.arange(1, len(correct_positions) + 1) / correct_positions
        aps.append(np.mean(precision_at_k))
        inps.append(len(correct_positions) / correct_positions[-1])

        ranked_ndom = ndom[q_idx][q_ranks]
        cum_ndom = np.cumsum(ranked_ndom)
        adm_k = cum_ndom[correct_positions - 1] / correct_positions
        adms.append(np.mean(adm_k))

    return {
        "mADM": float(np.mean(adms)),
        "mAP": float(np.mean(aps)),
        "Rank-1": float(np.mean(r1_list)),
        "Rank-5": float(np.mean(r5_list)),
        "Rank-10": float(np.mean(r10_list)),
        "mINP": float(np.mean(inps)),
    }

