"""Stage 1 Execution Script: Run Full Forensic Calibration, IE-ADW & Ensemble Benchmarking."""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import Task2RetrievalEvalData, UPARDataset, resolve_data_root
from src.evaluate import get_device
from src.stage1_optimizer import (
    load_model_from_zip,
    extract_gallery_logits,
    evaluate_predictions,
    fit_platt_calibration,
    compute_inverse_entropy_weights,
)


def main():
    print("=" * 80)
    print("STAGE 1 OPTIMIZATION: PROBABILITY CALIBRATION, IE-ADW & ENSEMBLE BENCHMARK")
    print("=" * 80)

    device = get_device()
    print(f"Compute Device: {device}")

    data_root = resolve_data_root("UPAR-Challenge-2027/data")
    retrieval_data = Task2RetrievalEvalData(data_root=data_root)
    val_csv = os.path.join(data_root, "annotations", "val.csv")

    val_dataset = UPARDataset(
        csv_path=val_csv,
        data_root=data_root,
        height=256,
        width=128,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    os.makedirs("checkpoints", exist_ok=True)
    exp004_cache = "checkpoints/val_logits_exp004.npy"
    exp005_cache = "checkpoints/val_logits_exp005.npy"

    # 1. Extract or load EXP-004 logits
    if os.path.exists(exp004_cache):
        print(f"\n[1/4] Loading cached EXP-004 logits from {exp004_cache}...")
        logits_004 = np.load(exp004_cache)
    else:
        print("\n[1/4] Extracting EXP-004 logits from submissions/EXP-004_submission.zip...")
        model_004 = load_model_from_zip("submissions/EXP-004_submission.zip", "convnext_tiny", device)
        logits_004 = extract_gallery_logits(model_004, val_loader, device)
        np.save(exp004_cache, logits_004)
        del model_004
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 2. Extract or load EXP-005 logits
    if os.path.exists(exp005_cache):
        print(f"\n[2/4] Loading cached EXP-005 logits from {exp005_cache}...")
        logits_005 = np.load(exp005_cache)
    else:
        print("\n[2/4] Extracting EXP-005 logits from submissions/EXP-005A_submission.zip...")
        model_005 = load_model_from_zip("submissions/EXP-005A_submission.zip", "swin_t", device)
        logits_005 = extract_gallery_logits(model_005, val_loader, device)
        np.save(exp005_cache, logits_005)
        del model_005
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gallery_gt = retrieval_data.gallery_gt
    queries = retrieval_data.queries

    print(f"\nLogits loaded: EXP-004 shape {logits_004.shape}, EXP-005 shape {logits_005.shape}")

    # 3. Fit Platt Calibration
    print("\n[3/4] Fitting Platt Probability Calibration parameters...")
    T_004, b_004 = fit_platt_calibration(logits_004, gallery_gt)
    T_005, b_005 = fit_platt_calibration(logits_005, gallery_gt)

    # Compute raw vs calibrated probabilities
    raw_p_004 = 1.0 / (1.0 + np.exp(-logits_004))
    raw_p_005 = 1.0 / (1.0 + np.exp(-logits_005))

    calib_p_004 = 1.0 / (1.0 + np.exp(-np.clip((logits_004 - b_004[None, :]) / T_004[None, :], -20.0, 20.0)))
    calib_p_005 = 1.0 / (1.0 + np.exp(-np.clip((logits_005 - b_005[None, :]) / T_005[None, :], -20.0, 20.0)))

    # Compute Inverse-Entropy Weights
    ie_weights = compute_inverse_entropy_weights(gallery_gt, queries)
    print(f"Inverse-Entropy Weights computed: shape {ie_weights.shape}, mean {np.mean(ie_weights):.2f}, range [{np.min(ie_weights):.2f}, {np.max(ie_weights):.2f}]")

    # 4. Benchmarking All Configurations
    print("\n[4/4] Executing Comparative Evaluation Matrix across 33,407 Gallery Images...")
    results = {}

    configs = [
        ("EXP-004 (Raw ConvNeXt-T, Uniform L1)", raw_p_004, None),
        ("EXP-004 (Calibrated ConvNeXt-T, Uniform L1)", calib_p_004, None),
        ("EXP-004 (Calibrated + IE-ADW Weighted L1)", calib_p_004, ie_weights),
        ("EXP-005 (Raw Swin-T, Uniform L1)", raw_p_005, None),
        ("EXP-005 (Calibrated Swin-T, Uniform L1)", calib_p_005, None),
        ("EXP-005 (Calibrated + IE-ADW Weighted L1)", calib_p_005, ie_weights),
        ("Ensemble (Raw 50/50 Blend, Uniform L1)", 0.50 * raw_p_004 + 0.50 * raw_p_005, None),
        ("Ensemble (Calibrated 50/50 Blend, Uniform L1)", 0.50 * calib_p_004 + 0.50 * calib_p_005, None),
        ("Ensemble (STAGE 1 FULL: Calibrated Blend + IE-ADW)", 0.52 * calib_p_004 + 0.48 * calib_p_005, ie_weights),
    ]

    for name, p_mat, w_mat in configs:
        t0 = time.time()
        res = evaluate_predictions(p_mat, retrieval_data, w_mat)
        elapsed = time.time() - t0
        results[name] = res
        print(f"  --> {name:<50} | mADM: {res['mADM']:.4f} | mAP: {res['mAP']:.4f} | R-1: {res['Rank-1']*100:5.2f}% | R-5: {res['Rank-5']*100:5.2f}% | R-10: {res['Rank-10']*100:5.2f}% | mINP: {res['mINP']:.4f} ({elapsed:.1f}s)")

    # Save calibrated parameters
    np.savez(
        "checkpoints/stage1_calibration_params.npz",
        T_004=T_004,
        b_004=b_004,
        T_005=T_005,
        b_005=b_005,
        pos_rates=np.mean(gallery_gt, axis=0),
    )
    print("\nSaved calibration parameters to checkpoints/stage1_calibration_params.npz")

    # Print final grand table
    print("\n" + "=" * 115)
    print("STAGE 1 COMPARATIVE BENCHMARK SCORECARD: BEFORE VS. AFTER")
    print("=" * 115)
    header = f"{'Configuration':<52} | {'mADM':<7} | {'mAP':<7} | {'Rank-1':<7} | {'Rank-5':<7} | {'Rank-10':<7} | {'mINP':<7} | {'PAR mA':<7}"
    print(header)
    print("-" * 115)
    for name, res in results.items():
        row = (
            f"{name:<52} | "
            f"{res['mADM']:.4f}  | "
            f"{res['mAP']:.4f}  | "
            f"{res['Rank-1']*100:5.2f}%  | "
            f"{res['Rank-5']*100:5.2f}%  | "
            f"{res['Rank-10']*100:5.2f}%  | "
            f"{res['mINP']:.4f}  | "
            f"{res['PAR_mA']*100:5.2f}%"
        )
        print(row)
    print("=" * 115)


if __name__ == "__main__":
    main()
