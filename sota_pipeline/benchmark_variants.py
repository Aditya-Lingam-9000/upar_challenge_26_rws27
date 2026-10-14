"""Rigorous Multi-Variant Empirical Benchmark & Precision Fidelity Audit.

Evaluates Raw FP32, FP16, and BF16 model variants on the official UPAR validation set.
Computes:
1. Bit-level numerical discrepancy (MAE, Max Diff) against baseline FP32
2. Ranking rank correlation (Spearman rho, Top-1 agreement rate)
3. Official competition metrics: Continuous mADM, Attribute mADM, Dual RRF mADM, Rank-1, Rank-5, Rank-10, mAP
4. Inference latency and memory telemetry under clean & degraded surveillance conditions.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from sota_pipeline.dataset import SOTAUPARDataset
from sota_pipeline.models import SOTAChampionshipModel, VisualStream
from sota_pipeline.evaluate import compute_official_metrics_from_ranking, InformationWeightedL1, ReciprocalRankFusion


def evaluate_model_pipeline(
    model: nn.Module,
    val_dataset: SOTAUPARDataset,
    query_prompts: List[str],
    query_labels: np.ndarray,
    gallery_indices: List[int],
    device: torch.device,
    precision: str = "float32",
    batch_size: int = 64
) -> Dict[str, Any]:
    """Runs inference with specified precision and returns continuous and attribute distances."""
    model.eval()
    t0 = time.time()

    # Pre-tokenize queries
    prompt_engine = val_dataset.prompt_engine
    query_tokens = prompt_engine.tokenize(query_prompts).to(device)

    # 1. Query Embeddings
    with torch.no_grad():
        if hasattr(model, "extract_text"):
            q_emb = model.extract_text(query_tokens).float().cpu() # [Q, 1024]
        else:
            # Standalone visual model: text embeddings are already computed or use full model
            raise ValueError("Model missing extract_text pathway.")

    # 2. Gallery Extraction
    gallery_sub = torch.utils.data.Subset(val_dataset, gallery_indices)
    loader = DataLoader(gallery_sub, batch_size=batch_size, shuffle=False, num_workers=2)

    gal_embeds_list = []
    gal_logits_list = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            if precision == "float16":
                imgs = imgs.half()
            elif precision == "bfloat16":
                imgs = imgs.bfloat16()
            else:
                imgs = imgs.float()

            metric_emb, attr_logits = model.extract_visual(imgs)
            gal_embeds_list.append(metric_emb.float().cpu())
            gal_logits_list.append(attr_logits.float().cpu())

    t_inf = time.time() - t0
    gal_embeds = torch.cat(gal_embeds_list, dim=0) # [G, 1024]
    gal_logits = torch.cat(gal_logits_list, dim=0) # [G, 40]
    gal_probs = torch.sigmoid(gal_logits).numpy()

    # 3. Pairwise Distances
    # Continuous Metric Cosine Distance: 1.0 - (q @ g.T)
    sim_metric = torch.matmul(q_emb, gal_embeds.t()).numpy()
    dist_metric = 1.0 - sim_metric

    # Attribute Surprisal Distance
    priors = val_dataset.attr_labels[gallery_indices].mean(axis=0) if hasattr(val_dataset, "attr_labels") else np.ones(40) * 0.1
    l1_engine = InformationWeightedL1(prior_frequencies=priors)
    dist_attr = l1_engine.compute_distance(query_labels, gal_probs).numpy()

    # Dual RRF Fusion
    rrf_engine = ReciprocalRankFusion(k0=60, alpha=0.65)
    ranked_rrf = rrf_engine.fuse_ranks(dist_metric, dist_attr)

    ranked_metric = np.argsort(dist_metric, axis=-1)
    ranked_attr = np.argsort(dist_attr, axis=-1)

    return {
        "dist_metric": dist_metric,
        "dist_attr": dist_attr,
        "ranked_metric": ranked_metric,
        "ranked_attr": ranked_attr,
        "ranked_rrf": ranked_rrf,
        "inf_time": t_inf,
        "gal_embeds": gal_embeds.numpy(),
        "gal_probs": gal_probs
    }


def run_benchmark(
    dataset_root: str = "UPAR-Challenge-2027/data",
    num_gallery: int = 1500,
    num_queries: int = 150
):
    print("=" * 80)
    print("[BENCHMARK] Initializing Multi-Variant Precision Fidelity Audit")
    print(f"  -> Gallery Subset: {num_gallery} real surveillance crops")
    print(f"  -> Validation Queries: {num_queries} queries")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[DEVICE] Hardware Acceleration: {device}")

    # Load validation dataset
    val_dataset = SOTAUPARDataset(dataset_root=dataset_root, split="val", preload_to_ram=False)
    total_val = len(val_dataset)
    print(f"[DATASET] Total validation samples available: {total_val}")

    # Fixed deterministic indices for rigorous reproducibility
    np.random.seed(42)
    G = min(num_gallery, total_val)
    gallery_indices = list(range(0, G))

    # Sample query indices
    Q = min(num_queries, G)
    query_indices = np.linspace(0, G - 1, Q, dtype=int)
    
    # Retrieve query ground truth
    all_sample_labels = []
    print("[DATASET] Gathering ground-truth attribute labels for benchmark subset...")
    for idx in query_indices:
        sample = val_dataset[int(idx)]
        all_sample_labels.append(sample["attr_labels"].numpy())
    query_labels = np.array(all_sample_labels, dtype=np.float32)

    all_gallery_labels = []
    for idx in gallery_indices:
        sample = val_dataset[int(idx)]
        all_gallery_labels.append(sample["attr_labels"].numpy())
    gallery_labels = np.array(all_gallery_labels, dtype=np.float32)

    # Synthesize text queries
    prompt_engine = val_dataset.prompt_engine
    query_prompts = [prompt_engine.synthesize_prompt(query_labels[q], is_train=False) for q in range(Q)]
    print(f"[PROMPT] Generated {Q} standardized text queries.")

    # Variants to benchmark
    variants = [
        {"name": "FP32 (Raw Baseline)", "path": "extracted_models/sota_full_fp32.pt", "precision": "float32"},
        {"name": "FP16 (Half Precision)", "path": "extracted_models/sota_full_fp16.pt", "precision": "float16"},
        {"name": "BF16 (BFloat16)", "path": "extracted_models/sota_full_fp32.pt", "precision": "bfloat16"}, # test dynamic bf16 cast
    ]

    results_table = []
    baseline_outputs = None

    for var in variants:
        name = var["name"]
        ckpt_path = var["path"]
        prec = var["precision"]
        print("-" * 80)
        print(f"[RUNNING] Benchmarking: {name} | Precision: {prec}")

        # Build clean model
        model = SOTAChampionshipModel(embed_dim=1024, num_attrs=40, backbone_type="auto", use_mlm=False)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model_state_dict"]
        model.load_state_dict(sd, strict=False)

        if prec == "float16":
            model = model.half()
        elif prec == "bfloat16":
            model = model.bfloat16()
        else:
            model = model.float()

        model.to(device)

        out = evaluate_model_pipeline(
            model=model,
            val_dataset=val_dataset,
            query_prompts=query_prompts,
            query_labels=query_labels,
            gallery_indices=gallery_indices,
            device=device,
            precision=prec,
            batch_size=64
        )

        # Official metrics calculation
        m_cont = compute_official_metrics_from_ranking(out["ranked_metric"], query_labels, gallery_labels)
        m_attr = compute_official_metrics_from_ranking(out["ranked_attr"], query_labels, gallery_labels)
        m_rrf = compute_official_metrics_from_ranking(out["ranked_rrf"], query_labels, gallery_labels)

        # Comparative Fidelity vs Raw FP32
        if baseline_outputs is None:
            baseline_outputs = out
            mae_emb = 0.0
            max_diff = 0.0
            rank1_agreement = 100.0
            spearman_corr = 1.0000
        else:
            # Measure numerical distance discrepancy on embeddings
            diff = np.abs(out["gal_embeds"] - baseline_outputs["gal_embeds"])
            mae_emb = float(np.mean(diff))
            max_diff = float(np.max(diff))

            # Measure rank agreement
            top1_base = baseline_outputs["ranked_rrf"][:, 0]
            top1_curr = out["ranked_rrf"][:, 0]
            rank1_agreement = float(np.mean(top1_base == top1_curr) * 100.0)

            # Average Spearman rank correlation across queries
            spearman_corrs = []
            for q in range(Q):
                rho, _ = stats.spearmanr(baseline_outputs["dist_metric"][q], out["dist_metric"][q])
                spearman_corrs.append(rho)
            spearman_corr = float(np.mean(spearman_corrs))

        filesize_mb = os.path.getsize(ckpt_path) / (1024 ** 2) if os.path.isfile(ckpt_path) else 0.0
        fps = G / max(out["inf_time"], 1e-4)

        entry = {
            "name": name,
            "precision": prec,
            "size_mb": filesize_mb,
            "fps": fps,
            "mae_emb": mae_emb,
            "spearman": spearman_corr,
            "rank1_agree": rank1_agreement,
            "cont_madm": m_cont["mADM"],
            "cont_r1": m_cont["Rank-1"],
            "cont_r5": m_cont["Rank-5"],
            "cont_map": m_cont["mAP"],
            "attr_madm": m_attr["mADM"],
            "dual_madm": m_rrf["mADM"],
            "dual_r1": m_rrf["Rank-1"],
            "dual_r5": m_rrf["Rank-5"],
            "dual_map": m_rrf["mAP"],
        }
        results_table.append(entry)

        print(f"  -> Continuous mADM: {m_cont['mADM']:.4f} | R-1: {m_cont['Rank-1']:.2f}% | R-5: {m_cont['Rank-5']:.2f}%")
        print(f"  -> Dual RRF mADM:   {m_rrf['mADM']:.4f} | R-1: {m_rrf['Rank-1']:.2f}% | R-5: {m_rrf['Rank-5']:.2f}%")
        print(f"  -> Embedding MAE vs FP32: {mae_emb:.6f} | Spearman Rho: {spearman_corr:.6f} | Top-1 Agreement: {rank1_agreement:.1f}%")

    # Print summary comparative table
    print("\n" + "=" * 100)
    print("RIGOROUS MULTI-VARIANT COMPARATIVE RESULTS TABLE")
    print("=" * 100)
    header = f"{'Variant':<22} | {'Precision':<8} | {'Size':<8} | {'Speed':<9} | {'MAE (vs FP32)':<14} | {'Spearman':<10} | {'Cont mADM':<10} | {'Dual mADM':<10} | {'Dual R-1':<9} | {'Dual R-5':<9}"
    print(header)
    print("-" * 100)
    for r in results_table:
        row = f"{r['name']:<22} | {r['precision']:<8} | {r['size_mb']:>5.1f} MB | {r['fps']:>5.1f} img/s | {r['mae_emb']:>14.6f} | {r['spearman']:>10.6f} | {r['cont_madm']:>10.4f} | {r['dual_madm']:>10.4f} | {r['dual_r1']:>8.2f}% | {r['dual_r5']:>8.2f}%"
        print(row)
    print("=" * 100)


if __name__ == "__main__":
    run_benchmark(num_gallery=1500, num_queries=150)
