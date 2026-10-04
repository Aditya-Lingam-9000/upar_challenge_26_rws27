"""Vectorized k-Reciprocal Reranking for Cross-Modal Person Retrieval.

Optimized implementation of Zhong et al. (CVPR 2017) adapted for cross-modal
retrieval matrices. Refines initial hyperspherical cosine rankings using
Jaccard distance over mutual reciprocal neighborhood graphs.
"""
from __future__ import annotations

import numpy as np
import torch


def fast_k_reciprocal_rerank(
    prob_feat: np.ndarray,
    gal_feat: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Vectorized k-reciprocal reranking.

    Args:
        prob_feat: (Q, D) normalized query embeddings.
        gal_feat: (G, D) normalized gallery embeddings.
        k1: Number of primary nearest neighbors.
        k2: Number of secondary expansion neighbors.
        lambda_value: Weight balance between original distance and Jaccard distance.

    Returns:
        final_dist: (Q, G) reranked distance matrix.
    """
    num_query = prob_feat.shape[0]
    num_gallery = gal_feat.shape[0]

    # Combine query and gallery for full neighborhood construction
    all_feat = np.vstack([prob_feat, gal_feat]).astype(np.float32)

    # 1. Compute pairwise cosine distance matrix: D = 2 - 2 * (A @ B.T)
    # Since embeddings are L2-normalized: ||u - v||^2 = 2 - 2 * (u . v)
    sim = np.dot(all_feat, all_feat.T)
    original_dist = (2.0 - 2.0 * sim).clip(min=0.0)

    # Free memory
    del all_feat, sim

    # 2. Extract initial ranking index matrix
    initial_rank = np.argsort(original_dist, axis=1)

    nn_k1 = []
    for i in range(num_query + num_gallery):
        # Forward k1 neighbors
        forward_k1 = initial_rank[i, : k1 + 1]
        # Backward k1 neighbors of forward neighbors
        backward_k1 = initial_rank[forward_k1, : k1 + 1]
        # Reciprocal condition: i must be in backward_k1
        rows = np.where(backward_k1 == i)[0]
        reciprocal_k1 = forward_k1[rows]

        # Robust expansion: if reciprocal neighbor set is dense, expand with half k1
        if len(reciprocal_k1) > (k1 / 2):
            for candidate in reciprocal_k1[:k2]:
                cand_forward = initial_rank[candidate, : int(k1 / 2) + 1]
                cand_backward = initial_rank[cand_forward, : int(k1 / 2) + 1]
                cand_rows = np.where(cand_backward == candidate)[0]
                reciprocal_k1 = np.unique(np.concatenate([reciprocal_k1, cand_forward[cand_rows]]))

        nn_k1.append(reciprocal_k1)

    # 3. Construct sparse Jaccard representation
    V = np.zeros((num_query + num_gallery, num_query + num_gallery), dtype=np.float32)
    for i in range(num_query + num_gallery):
        neighbors = nn_k1[i]
        weight = np.exp(-original_dist[i, neighbors])
        V[i, neighbors] = weight / (np.sum(weight) + 1e-8)

    # Jaccard distance between query and gallery
    q_V = V[:num_query, :]
    g_V = V[num_query:, :]

    # Vectorized minimum and maximum intersections
    min_inter = np.dot(q_V, g_V.T)
    # Norm terms: sum(a^2) + sum(b^2) - intersection
    q_norm = np.sum(q_V ** 2, axis=1, keepdims=True)
    g_norm = np.sum(g_V ** 2, axis=1, keepdims=True)
    jaccard_dist = 1.0 - (min_inter / (q_norm + g_norm.T - min_inter + 1e-8))

    # 4. Final hybrid distance blending
    orig_q_g = original_dist[:num_query, num_query:]
    final_dist = (1.0 - lambda_value) * jaccard_dist + lambda_value * orig_q_g

    return final_dist.astype(np.float32)
