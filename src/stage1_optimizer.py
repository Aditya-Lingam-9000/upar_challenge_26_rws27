"""Stage 1 Optimization Engine: Calibration, Inverse-Entropy Weighting & Heterogeneous Ensembling.

Evaluates and optimizes:
1. EXP-004 (ConvNeXt-Tiny) alone: Raw vs Calibrated vs Weighted
2. EXP-005 (Swin-T) alone: Raw vs Calibrated vs Weighted
3. Heterogeneous Ensemble (ConvNeXt-Tiny + Swin-T):
   - Linear probability blending: P_ens = alpha * P_conv + (1 - alpha) * P_swin
   - Per-attribute temperature scaling & bias calibration
   - Inverse-Entropy Attribute Distance Weighting (IE-ADW)
"""
from __future__ import annotations

import io
import os
import sys
import time
import zipfile
from typing import Any

import numpy as np
from scipy.optimize import minimize
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import Task2RetrievalEvalData, UPARDataset, resolve_data_root, UPAR_40_ATTRIBUTES
from src.evaluate import get_device, evaluate_retrieval
from src.metrics import compute_par_metrics
from src.models import build_model


def load_model_from_zip(zip_path: str, model_name: str, device: torch.device) -> torch.nn.Module:
    """Load model directly from a submission zip file."""
    print(f"Loading {model_name} from {zip_path} onto {device}...")
    model = build_model(model_name, pretrained=False)
    with zipfile.ZipFile(zip_path, "r") as zf:
        buf = io.BytesIO(zf.read("assets/model.pth"))
        sd = torch.load(buf, map_location=device)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items() if not k.startswith("embedding_head.")}
        model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_gallery_logits(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_fp16: bool = True,
    log_interval: int = 50,
) -> np.ndarray:
    """Extract raw gallery logits (pre-sigmoid) for calibration."""
    model.eval()
    all_logits = []
    total_batches = len(val_loader)
    total_samples = len(val_loader.dataset)
    processed = 0
    t0 = time.time()

    for batch_idx, (images, _, _, _) in enumerate(val_loader):
        batch_size = images.size(0)
        images = images.to(device)
        if use_fp16 and device.type in ["mps", "cuda"]:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)
        all_logits.append(logits.float().cpu().numpy())
        processed += batch_size

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t0
            speed = processed / max(elapsed, 1e-4)
            pct = (batch_idx + 1) / total_batches * 100
            print(f"  Extraction: [{processed:5d}/{total_samples:5d}] ({pct:5.1f}%) | Speed: {speed:5.1f} img/s", flush=True)

    return np.vstack(all_logits).astype(np.float32)


def compute_weighted_distance_matrix(
    probs: np.ndarray,
    queries: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute (Q, G) distance matrix with optional attribute weights.

    Formula:
      D(q, p) = sum_a w_a * |q_a - p_a|
      If q_a in {0, 1}:
        |q_a - p_a| = q_a * (1 - p_a) + (1 - q_a) * p_a = p_a + q_a * (1 - 2*p_a)
      With weights:
        sum_a w_a * |q_a - p_a| = sum(w * p) + q @ (w * (1 - 2*p)).T
    """
    Q, A = queries.shape
    G, _ = probs.shape

    if weights is None:
        # Uniform L1 distance
        distances = probs.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * probs).T
    elif weights.ndim == 1:
        # Static 1D weights per attribute: shape (A,)
        w = weights.astype(np.float32)
        wp = probs * w[None, :]  # (G, A)
        term1 = wp.sum(axis=1, keepdims=True).T  # (1, G)
        w_factor = w[None, :] * (1.0 - 2.0 * probs)  # (G, A)
        distances = term1 + queries @ w_factor.T  # (Q, G)
    elif weights.ndim == 2:
        # Query-adaptive weights: shape (Q, A)
        # distances[q, g] = sum_a W[q, a] * |q[a] - p[g, a]|
        # Vectorized implementation:
        # For each query, distances = (W * queries) @ (1 - 2*p).T + W @ p.T
        # Shape: (Q, G)
        W = weights.astype(np.float32)
        # When q_a = 0: W[q, a] * p[g, a]
        # When q_a = 1: W[q, a] * (1 - p[g, a]) = W[q, a] - W[q, a] * p[g, a]
        # |q - p| = p + q*(1 - 2p)
        # W * |q - p| = W*p + W*q*(1 - 2p)
        term1 = W @ probs.T  # (Q, G)
        term2 = (W * queries) @ (1.0 - 2.0 * probs).T  # (Q, G)
        distances = term1 + term2
    else:
        raise ValueError(f"Invalid weights shape: {weights.shape}")

    return np.nan_to_num(distances, nan=20.0, posinf=40.0, neginf=0.0).astype(np.float32)


def evaluate_predictions(
    probs: np.ndarray,
    retrieval_data: Task2RetrievalEvalData,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Run full retrieval and PAR evaluation given gallery probabilities."""
    queries = retrieval_data.queries  # (Q, 40)
    distances = compute_weighted_distance_matrix(probs, queries, weights)

    # Compute official retrieval metrics
    gallery_gt = retrieval_data.gallery_gt
    query_gt = retrieval_data.queries
    Q = queries.shape[0]

    # Precompute correct matches mask: (Q, G) without 18.5 GB 3D broadcast:
    q_sum = query_gt.sum(axis=1, keepdims=True)
    g_sum = gallery_gt.sum(axis=1, keepdims=True).T
    q_dot_g = query_gt @ gallery_gt.T
    diffs = np.clip(q_sum + g_sum - 2.0 * q_dot_g, 0.0, 40.0)
    is_correct = (diffs < 0.5)

    # Sort ranks
    ranks = np.argsort(distances, axis=1)

    adms, aps, r1_list, r5_list, r10_list, inps = [], [], [], [], [], []

    # Degree of match matrix
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

        # AP
        precision_at_k = np.arange(1, len(correct_positions) + 1) / correct_positions
        aps.append(np.mean(precision_at_k))

        # INP
        inps.append(len(correct_positions) / correct_positions[-1])

        # ADM
        ranked_ndom = ndom[q_idx][q_ranks]
        cum_ndom = np.cumsum(ranked_ndom)
        adm_k = cum_ndom[correct_positions - 1] / correct_positions
        adms.append(np.mean(adm_k))

    par_metrics = compute_par_metrics(probs, gallery_gt)

    return {
        "mADM": float(np.mean(adms)),
        "mAP": float(np.mean(aps)),
        "Rank-1": float(np.mean(r1_list)),
        "Rank-5": float(np.mean(r5_list)),
        "Rank-10": float(np.mean(r10_list)),
        "mINP": float(np.mean(inps)),
        "PAR_mA": float(par_metrics["PAR_mA"]),
    }


def fit_platt_calibration(logits: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-attribute temperature T_a and bias b_a using logistic cross-entropy."""
    print("Fitting per-attribute Platt scaling (T_a, b_a) on validation data...")
    num_attrs = logits.shape[1]
    temperatures = np.ones(num_attrs, dtype=np.float32)
    biases = np.zeros(num_attrs, dtype=np.float32)

    for a in range(num_attrs):
        z = logits[:, a]
        y = ground_truth[:, a]

        # Skip degenerate attributes with 0 positives or 0 negatives
        pos_count = np.sum(y == 1)
        if pos_count == 0 or pos_count == len(y):
            continue

        def loss_fn(params):
            log_t, b = params
            t = np.exp(log_t)
            scaled = (z - b) / t
            # Stable BCE
            p = 1.0 / (1.0 + np.exp(-np.clip(scaled, -20.0, 20.0)))
            eps = 1e-7
            bce = -np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
            return bce

        res = minimize(loss_fn, [0.0, 0.0], method="L-BFGS-B", bounds=[(-2.0, 2.0), (-5.0, 5.0)])
        if res.success:
            temperatures[a] = float(np.exp(res.x[0]))
            biases[a] = float(res.x[1])

    return temperatures, biases


def compute_inverse_entropy_weights(ground_truth: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Compute per-query attribute weights based on self-information: -log2(p(q_a))."""
    # Prior positive frequency across gallery
    pos_rates = np.clip(np.mean(ground_truth, axis=0), 1e-4, 1.0 - 1e-4)

    # For query q_a = 1: weight is -log2(pos_rate)
    # For query q_a = 0: weight is -log2(1 - pos_rate)
    w_pos = -np.log2(pos_rates)
    w_neg = -np.log2(1.0 - pos_rates)

    # Shape: (Q, 40)
    query_weights = queries * w_pos[None, :] + (1.0 - queries) * w_neg[None, :]

    # Normalize each query row to have mean 1.0 (so total sum across 40 attributes = 40.0)
    query_weights = query_weights / np.mean(query_weights, axis=1, keepdims=True)

    return query_weights.astype(np.float32)
