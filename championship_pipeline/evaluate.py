"""Championship Evaluation Engine for Cross-Modal Person Retrieval.

Replicates the official Codabench evaluation suite down to exact precision.
Computes:
  - mADM (primary challenge ranking metric)
  - mAP (mean Average Precision)
  - CMC Rank-1, Rank-5, Rank-10
  - mINP (Inverse Negative Penalty)
Evaluates pairwise distance matrices directly in continuous 1024-D metric space.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from championship_pipeline.dataset import CrossModalUPARDataset, resolve_data_root
from championship_pipeline.pseudo_descriptions import PseudoDescriptionEngine
from championship_pipeline.rerank import fast_k_reciprocal_rerank


def compute_retrieval_metrics(
    distances: np.ndarray,
    queries: np.ndarray,
    gallery_gt: np.ndarray,
) -> dict[str, float]:
    """Compute official challenge metrics from pairwise distance matrix.

    Args:
        distances: (Q, G) float array where lower distance indicates higher match.
        queries: (Q, 40) binary ground truth query attribute matrix.
        gallery_gt: (G, 40) binary ground truth gallery attribute matrix.

    Returns:
        dict with mADM, mAP, Rank-1, Rank-5, Rank-10, mINP.
    """
    Q, G = distances.shape
    queries = np.asarray(queries, dtype=np.float32)
    gallery_gt = np.asarray(gallery_gt, dtype=np.float32)

    # 1. Degree of Match (dom) between queries and gallery ground truth:
    l1_diff = gallery_gt.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * gallery_gt).T
    l1_diff = np.maximum(0.0, l1_diff)
    dom = 1.0 - (l1_diff / 40.0)

    # Ground truth exact matches: all 40 attributes agree
    exact_matches = (l1_diff == 0)

    # 2. Normalized Degree of Match (ndom):
    mean_dom = dom.mean(axis=1, keepdims=True)
    max_dom = dom.max(axis=1, keepdims=True)
    denom = np.maximum(max_dom - mean_dom, 1e-8)
    ndom = np.maximum(0.0, (dom - mean_dom) / denom)

    # 3. Sort gallery candidates per query (ascending distance -> closest first)
    ranking = np.argsort(distances, axis=1)

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

        ranks = np.where(q_matches)[0] + 1
        R = len(ranks)

        if R == 0:
            continue

        valid_queries += 1

        # CMC
        min_rank = ranks[0]
        if min_rank <= 1:
            rank1_count += 1
        if min_rank <= 5:
            rank5_count += 1
        if min_rank <= 10:
            rank10_count += 1

        # Average Precision
        precisions = np.arange(1, R + 1, dtype=np.float64) / ranks
        ap_list.append(float(np.mean(precisions)))

        # Inverse Negative Penalty
        inp_list.append(float(R / ranks[-1]))

        # Average Degree of Match (ADM)
        cum_ndom = np.cumsum(q_ndom, dtype=np.float64)
        adm_terms = cum_ndom[ranks - 1] / ranks
        adm_list.append(float(np.mean(adm_terms)))

    if valid_queries == 0:
        return {
            "mADM": 0.0, "mAP": 0.0,
            "Rank-1": 0.0, "Rank-5": 0.0, "Rank-10": 0.0, "mINP": 0.0,
        }

    return {
        "mADM": float(np.mean(adm_list)),
        "mAP": float(np.mean(ap_list)),
        "Rank-1": float(rank1_count / valid_queries * 100.0),
        "Rank-5": float(rank5_count / valid_queries * 100.0),
        "Rank-10": float(rank10_count / valid_queries * 100.0),
        "mINP": float(np.mean(inp_list)),
    }


@torch.no_grad()
def evaluate_crossmodal_retrieval(
    model: torch.nn.Module,
    val_loader: DataLoader,
    queries: np.ndarray,
    gallery_gt: np.ndarray,
    device: torch.device,
    use_reranking: bool = False,
    engine: PseudoDescriptionEngine | None = None,
) -> dict[str, float]:
    """Execute complete validation retrieval in continuous 1024-D metric space."""
    model.eval()
    if engine is None:
        engine = PseudoDescriptionEngine()

    # 1. Extract 1024-D visual embeddings for all gallery images
    all_gallery_feats = []
    t0 = time.time()
    for batch in val_loader:
        imgs = batch["image"].to(device, non_blocking=True)
        norm_v = model.encode_visual(imgs)
        all_gallery_feats.append(norm_v.cpu().numpy())

    gallery_feats = np.vstack(all_gallery_feats).astype(np.float32)  # (G, 1024)
    gallery_feats = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    # 2. Encode all queries into 1024-D cross-modal embeddings
    q_data = [engine.process_query(queries[i]) for i in range(len(queries))]
    hard_tensor = torch.stack([d["hard_vector"] for d in q_data]).to(device)
    tokens_tensor = torch.stack([d["tokens"] for d in q_data]).to(device)
    mask_tensor = torch.stack([d["mask"] for d in q_data]).to(device)

    # Process queries in batches of 128
    all_query_feats = []
    q_batch_size = 128
    for i in range(0, len(queries), q_batch_size):
        h_b = hard_tensor[i : i + q_batch_size]
        t_b = tokens_tensor[i : i + q_batch_size]
        m_b = mask_tensor[i : i + q_batch_size]
        norm_q = model.encode_query(h_b, t_b, m_b)
        all_query_feats.append(norm_q.cpu().numpy())

    query_feats = np.vstack(all_query_feats).astype(np.float32)  # (Q, 1024)
    query_feats = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)

    # 3. Compute pairwise distances
    if use_reranking:
        distances = fast_k_reciprocal_rerank(query_feats, gallery_feats, k1=20, k2=6, lambda_value=0.3)
    else:
        # Standard hyperspherical cosine distance: d = 1.0 - (q @ g.T)
        distances = 1.0 - np.dot(query_feats, gallery_feats.T)
        distances = np.nan_to_num(distances, nan=2.0, posinf=2.0, neginf=0.0).astype(np.float32)

    metrics = compute_retrieval_metrics(distances, queries, gallery_gt)
    metrics["eval_time"] = time.time() - t0
    return metrics
