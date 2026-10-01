"""Exact Offline Metric Engine for RWS 2027 Track 2 (ABPR).

Replicates the official Codabench evaluation suite down to exact precision.
Computes:
  - mADM (mean Attribute Distance Match) - Primary competition metric
  - mAP (mean Average Precision) - Identity retrieval metric
  - CMC Rank-1, Rank-5, Rank-10 - Cumulative Match Characteristic hit rates
  - mINP (mean Inverse Negative Penalty) - Hardest positive penalty metric
  - PAR mA - Pedestrian Attribute Recognition Mean Accuracy across 40 attributes
"""
from __future__ import annotations

from typing import Any
import numpy as np


def compute_retrieval_metrics(
    distances: np.ndarray,
    queries: np.ndarray,
    gallery_gt: np.ndarray,
) -> dict[str, float]:
    """Compute exact competition retrieval metrics for Track 2.

    Args:
        distances: (Q, G) float array of pairwise distances between queries and gallery.
                   Lower distance indicates higher confidence/better match.
        queries: (Q, 40) binary ground truth query attribute matrix.
        gallery_gt: (G, 40) binary ground truth gallery attribute matrix.

    Returns:
        Dictionary with mADM, mAP, Rank-1, Rank-5, Rank-10, mINP.
    """
    Q, G = distances.shape
    assert queries.shape == (Q, 40), f"queries shape {queries.shape} != ({Q}, 40)"
    assert gallery_gt.shape == (G, 40), f"gallery_gt shape {gallery_gt.shape} != ({G}, 40)"

    queries = np.asarray(queries, dtype=np.float32)
    gallery_gt = np.asarray(gallery_gt, dtype=np.float32)

    # 1. Degree of Match (dom) between queries and gallery ground truth:
    # ||q - y||_1 = sum(y) + q @ (1 - 2y).T
    l1_diff = gallery_gt.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * gallery_gt).T
    l1_diff = np.maximum(0.0, l1_diff)  # numerical guard
    dom = 1.0 - (l1_diff / 40.0)        # (Q, G) in [0.0, 1.0]

    # Ground truth exact matches: all 40 attributes agree
    exact_matches = (l1_diff == 0)      # (Q, G) boolean

    # 2. Normalized Degree of Match (ndom):
    mean_dom = dom.mean(axis=1, keepdims=True)  # (Q, 1)
    max_dom = dom.max(axis=1, keepdims=True)    # (Q, 1)
    denom = np.maximum(max_dom - mean_dom, 1e-8)
    ndom = np.maximum(0.0, (dom - mean_dom) / denom)

    # 3. Sort gallery candidates per query (ascending distance -> closest first)
    # Using stable sort (mergesort or standard argsort) to keep gallery order on ties
    ranking = np.argsort(distances, axis=1)  # (Q, G) indices

    ap_list: list[float] = []
    adm_list: list[float] = []
    inp_list: list[float] = []
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0
    valid_queries = 0

    for i in range(Q):
        q_order = ranking[i]
        q_matches = exact_matches[i, q_order]
        q_ndom = ndom[i, q_order]

        # 1-indexed ranks of exact matches: r_1 < r_2 < ... < r_R
        ranks = np.where(q_matches)[0] + 1
        R = len(ranks)

        if R == 0:
            # Query has no exact match in gallery (fallback if non-closed validation)
            continue

        valid_queries += 1

        # Cumulative Match Characteristic (CMC)
        min_rank = ranks[0]
        if min_rank <= 1:
            rank1_count += 1
        if min_rank <= 5:
            rank5_count += 1
        if min_rank <= 10:
            rank10_count += 1

        # Average Precision: AP = (1 / R) * sum_{k=1}^R (k / r_k)
        precisions = np.arange(1, R + 1, dtype=np.float64) / ranks
        ap_list.append(float(np.mean(precisions)))

        # Inverse Negative Penalty: INP = R / r_R
        inp_list.append(float(R / ranks[-1]))

        # Average Degree of Match: ADM = (1 / R) * sum_{k=1}^R [ (sum_{j=1}^{r_k} ndom(j)) / r_k ]
        cum_ndom = np.cumsum(q_ndom, dtype=np.float64)
        adm_terms = cum_ndom[ranks - 1] / ranks
        adm_list.append(float(np.mean(adm_terms)))

    denom_q = max(valid_queries, 1)
    return {
        "mADM": float(np.mean(adm_list)) if adm_list else 0.0,
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "Rank-1": float(rank1_count / denom_q),
        "Rank-5": float(rank5_count / denom_q),
        "Rank-10": float(rank10_count / denom_q),
        "mINP": float(np.mean(inp_list)) if inp_list else 0.0,
    }


def compute_par_metrics(
    preds_probs: np.ndarray,
    gt_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute Pedestrian Attribute Recognition (PAR) multi-label metrics.

    Args:
        preds_probs: (N, 40) float array of predicted probabilities in [0, 1].
        gt_labels: (N, 40) binary ground truth labels in {0, 1}.
        threshold: Decision threshold for binarization (default: 0.5).

    Returns:
        Dictionary with mA (mean accuracy), label_ma, precision, recall, f1, etc.
    """
    preds_probs = np.asarray(preds_probs, dtype=np.float32)
    gt_labels = np.asarray(gt_labels, dtype=np.float32)
    pred_bin = (preds_probs > threshold).astype(np.float32)

    eps = 1e-8

    # Label-level metrics (across N samples for each of the 40 attributes)
    tp = np.sum((gt_labels == 1) & (pred_bin == 1), axis=0)
    tn = np.sum((gt_labels == 0) & (pred_bin == 0), axis=0)
    fp = np.sum((gt_labels == 0) & (pred_bin == 1), axis=0)
    fn = np.sum((gt_labels == 1) & (pred_bin == 0), axis=0)

    gt_pos = tp + fn
    gt_neg = tn + fp

    label_pos_recall = tp / (gt_pos + eps)
    label_neg_recall = tn / (gt_neg + eps)
    label_ma = 0.5 * (label_pos_recall + label_neg_recall)
    mean_ma = float(np.mean(label_ma))

    label_prec = tp / (tp + fp + eps)
    label_acc = tp / (tp + fp + fn + eps)
    label_f1 = 2.0 * label_prec * label_pos_recall / (label_prec + label_pos_recall + eps)

    # Instance-level metrics (across 40 attributes for each sample)
    inst_tp = np.sum((gt_labels == 1) & (pred_bin == 1), axis=1)
    inst_fp = np.sum((gt_labels == 0) & (pred_bin == 1), axis=1)
    inst_fn = np.sum((gt_labels == 1) & (pred_bin == 0), axis=1)

    inst_prec = inst_tp / (inst_tp + inst_fp + eps)
    inst_recall = inst_tp / (inst_tp + inst_fn + eps)
    inst_acc = inst_tp / (inst_tp + inst_fp + inst_fn + eps)
    inst_f1 = 2.0 * inst_prec * inst_recall / (inst_prec + inst_recall + eps)

    return {
        "PAR_mA": mean_ma,
        "label_ma": label_ma.tolist(),
        "mean_label_prec": float(np.mean(label_prec)),
        "mean_label_recall": float(np.mean(label_pos_recall)),
        "mean_label_f1": float(np.mean(label_f1)),
        "inst_acc": float(np.mean(inst_acc)),
        "inst_prec": float(np.mean(inst_prec)),
        "inst_recall": float(np.mean(inst_recall)),
        "inst_f1": float(np.mean(inst_f1)),
    }
