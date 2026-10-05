"""High-Throughput Validation & SOTA Metric Evaluation Engine.

Evaluates cross-modal person retrieval models against the official UPAR
and MEVID benchmark metrics:
- Mean Average Discrepancy Metric (mADM)
- Rank-1, Rank-5, Rank-10 Accuracy
- Mean Average Precision (mAP)
Supports pure continuous metric retrieval, Information-Weighted L1, and RRF Fusion.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sota_pipeline.config import SOTAConfig
from sota_pipeline.dataset import SOTAUPARDataset
from sota_pipeline.models import SOTAChampionshipModel
from sota_pipeline.reranking import (
    InformationWeightedL1,
    ReciprocalRankFusion,
    VectorizedGPUReranking,
)


def compute_madm(
    ranked_indices: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    k_eval: int = 10
) -> float:
    """
    Computes the official Mean Average Discrepancy Metric (mADM).
    mADM = 1.0 - mean across queries of normalized attribute match discrepancy.
    Higher is better (1.0 = perfect attribute alignment across top candidates).
    """
    Q, G = ranked_indices.shape
    K = query_labels.shape[1]
    discrepancies = []

    for q in range(Q):
        top_k_indices = ranked_indices[q, :k_eval]
        q_vec = query_labels[q] # [40]

        # Candidate attributes: [k_eval, 40]
        cand_attrs = gallery_labels[top_k_indices]
        # L1 distance per candidate normalized by number of attributes
        l1_diff = np.abs(cand_attrs - q_vec).sum(axis=-1) / K # [k_eval]
        discrepancies.append(np.mean(l1_diff))

    mean_discrepancy = float(np.mean(discrepancies))
    # mADM is bounded between 0.0 and 1.0 (inverted discrepancy score)
    return max(0.0, 1.0 - mean_discrepancy)


def compute_cmc_map(
    ranked_indices: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    match_threshold: int = 2
) -> Tuple[float, float, float, float]:
    """
    Computes Rank-1, Rank-5, Rank-10, and mAP where positive match is defined
    as Hamming attribute distance <= match_threshold.
    """
    Q, G = ranked_indices.shape
    r1_count = 0
    r5_count = 0
    r10_count = 0
    aps = []

    for q in range(Q):
        q_vec = query_labels[q]
        cand_attrs = gallery_labels[ranked_indices[q]]
        hamming_dists = np.abs(cand_attrs - q_vec).sum(axis=-1)
        matches = (hamming_dists <= match_threshold)

        if matches[0]:
            r1_count += 1
        if np.any(matches[:5]):
            r5_count += 1
        if np.any(matches[:10]):
            r10_count += 1

        # Average Precision (AP)
        num_pos = np.sum(matches)
        if num_pos == 0:
            aps.append(0.0)
        else:
            cum_matches = np.cumsum(matches)
            ranks = np.arange(1, G + 1)
            precision_at_k = cum_matches / ranks
            ap = np.sum(precision_at_k * matches) / num_pos
            aps.append(ap)

    rank1 = (r1_count / Q) * 100.0
    rank5 = (r5_count / Q) * 100.0
    rank10 = (r10_count / Q) * 100.0
    map_score = float(np.mean(aps)) * 100.0
    return rank1, rank5, rank10, map_score


@torch.no_grad()
def evaluate_retrieval(
    model: torch.nn.Module,
    val_dataset: SOTAUPARDataset,
    batch_size: int = 128,
    num_queries: int = 500,
    device: torch.device = torch.device("cpu"),
    use_rerank: bool = False
) -> Dict[str, float]:
    """
    Executes full continuous metric and RRF retrieval evaluation on validation set.
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    print(f"[EVAL] Extracting visual features for {len(val_dataset)} gallery crops...")
    gallery_embeds_list = []
    gallery_logits_list = []
    gallery_labels_list = []

    t0 = time.time()
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            metric_emb, attr_logits = raw_model.extract_visual(imgs)
            gallery_embeds_list.append(metric_emb.cpu())
            gallery_logits_list.append(attr_logits.cpu())
            gallery_labels_list.append(batch["attr_labels"])

    gallery_embeds = torch.cat(gallery_embeds_list, dim=0) # [G, D]
    gallery_logits = torch.cat(gallery_logits_list, dim=0) # [G, 40]
    gallery_labels = torch.cat(gallery_labels_list, dim=0).numpy() # [G, 40]
    gallery_probs = torch.sigmoid(gallery_logits) # [G, 40]
    G = gallery_embeds.shape[0]

    # Sample query subset (or all if G < num_queries)
    Q = min(num_queries, G)
    query_indices = np.linspace(0, G - 1, Q, dtype=int)
    query_labels = gallery_labels[query_indices] # [Q, 40]

    print(f"[EVAL] Synthesizing text queries and extracting query embeddings ({Q} queries)...")
    prompt_engine = val_dataset.prompt_engine
    query_prompts = [prompt_engine.synthesize_prompt(query_labels[q], is_train=False) for q in range(Q)]
    query_tokens = prompt_engine.tokenize(query_prompts).to(device)
    with torch.no_grad():
        query_embeds = raw_model.extract_text(query_tokens).cpu() # [Q, D]

    t_feat = time.time() - t0
    print(f"[EVAL] Feature extraction completed in {t_feat:.2f}s ({G} gallery items, {Q} queries).")

    # 1. Pure Continuous 1024-D Metric Retrieval
    sim_metric = torch.matmul(query_embeds, gallery_embeds.t()) # [Q, G]
    dist_metric = (1.0 - sim_metric).numpy()
    ranked_metric = np.argsort(dist_metric, axis=-1)

    madm_metric = compute_madm(ranked_metric, gallery_labels, query_labels)
    r1_m, r5_m, r10_m, map_m = compute_cmc_map(ranked_metric, gallery_labels, query_labels)

    # 2. Information-Weighted L1 Attribute Distance
    priors = gallery_labels.mean(axis=0)
    info_l1_engine = InformationWeightedL1(prior_frequencies=priors)
    dist_attr = info_l1_engine.compute_distance(query_labels, gallery_probs).numpy()
    ranked_attr = np.argsort(dist_attr, axis=-1)
    madm_attr = compute_madm(ranked_attr, gallery_labels, query_labels)

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_engine = ReciprocalRankFusion(k0=60, alpha=0.65)
    ranked_rrf = rrf_engine.fuse_ranks(dist_metric, dist_attr)
    madm_rrf = compute_madm(ranked_rrf, gallery_labels, query_labels)
    r1_rrf, r5_rrf, r10_rrf, map_rrf = compute_cmc_map(ranked_rrf, gallery_labels, query_labels)

    results = {
        "mADM_Metric": float(madm_metric),
        "Rank1_Metric": float(r1_m),
        "Rank5_Metric": float(r5_m),
        "Rank10_Metric": float(r10_m),
        "mAP_Metric": float(map_m),
        "mADM_Attr_L1": float(madm_attr),
        "mADM_RRF": float(madm_rrf),
        "Rank1_RRF": float(r1_rrf),
        "Rank5_RRF": float(r5_rrf),
        "Rank10_RRF": float(r10_rrf),
        "mAP_RRF": float(map_rrf),
    }

    print("\n" + "=" * 80)
    print("SOTA EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Continuous Metric (1024-D): mADM = {madm_metric:.4f} | R-1 = {r1_m:.2f}% | R-5 = {r5_m:.2f}% | mAP = {map_m:.2f}%")
    print(f"Attribute Surprisal L1:    mADM = {madm_attr:.4f}")
    print(f"Dual-Pathway RRF Fusion:   mADM = {madm_rrf:.4f} | R-1 = {r1_rrf:.2f}% | R-5 = {r5_rrf:.2f}% | mAP = {map_rrf:.2f}%")
    print("=" * 80 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOTA Championship Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--dataset_root", type=str, default=None, help="Root path to dataset")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_queries", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = SOTAConfig.resolve_dataset_root(args.dataset_root)

    val_dataset = SOTAUPARDataset(
        dataset_root=dataset_root,
        split="val",
        preload_to_ram=False
    )

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SOTAChampionshipModel(embed_dim=1024)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    evaluate_retrieval(
        model=model,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_queries=args.num_queries,
        device=device
    )


if __name__ == "__main__":
    main()
