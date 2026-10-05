"""Unit tests for InformationWeightedL1, GPU Reranking, and ReciprocalRankFusion."""
import numpy as np
import torch
from sota_pipeline.reranking import (
    InformationWeightedL1,
    ReciprocalRankFusion,
    VectorizedGPUReranking,
)


def test_information_weighted_l1():
    # Simulate attribute priors (bald head=0.01, short sleeve=0.80)
    priors = np.ones(40) * 0.50
    priors[0] = 0.01 # Rare trait
    priors[1] = 0.80 # Common trait

    engine = InformationWeightedL1(prior_frequencies=priors)
    assert engine.weights is not None
    # Rare trait must have strictly higher weight than common trait
    assert engine.weights[0] > engine.weights[1]
    print(f"[PASS] Surprisal weights: Rare={engine.weights[0]:.2f}, Common={engine.weights[1]:.2f}")

    Q, G = 5, 20
    q_attrs = np.random.randint(0, 2, (Q, 40)).astype(np.float32)
    g_attrs = np.random.rand(G, 40).astype(np.float32)

    dist = engine.compute_distance(q_attrs, g_attrs)
    assert dist.shape == (Q, G)
    assert not torch.isnan(dist).any()
    print(f"[PASS] Computed Info-Weighted L1 matrix shape: {dist.shape}")


def test_reciprocal_rank_fusion():
    Q, G = 4, 10
    dist_metric = np.random.rand(Q, G)
    dist_attr = np.random.rand(Q, G)

    rrf = ReciprocalRankFusion(k0=60, alpha=0.65)
    fused_ranks = rrf.fuse_ranks(dist_metric, dist_attr)

    assert fused_ranks.shape == (Q, G)
    assert fused_ranks.dtype == np.int32
    # Ensure every query ranking is a valid permutation of gallery indices [0..G-1]
    for q in range(Q):
        assert set(fused_ranks[q]) == set(range(G))
    print(f"[PASS] RRF fused ranks verified shape: {fused_ranks.shape}")


def test_vectorized_gpu_reranking():
    Q, G, D = 3, 10, 64
    q_emb = F_norm = torch.randn(Q, D)
    q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=-1)
    g_emb = torch.randn(G, D)
    g_emb = torch.nn.functional.normalize(g_emb, p=2, dim=-1)

    reranker = VectorizedGPUReranking(k1=5, k2=2, lambda_val=0.30)
    reranked_dist = reranker.rerank(q_emb, g_emb)

    assert reranked_dist.shape == (Q, G)
    assert not torch.isnan(reranked_dist).any()
    print(f"[PASS] Vectorized GPU Reranking output shape: {reranked_dist.shape}")


if __name__ == "__main__":
    test_information_weighted_l1()
    test_reciprocal_rank_fusion()
    test_vectorized_gpu_reranking()
    print("All reranking and post-processing tests passed!")
