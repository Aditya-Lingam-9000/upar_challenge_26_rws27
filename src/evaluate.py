"""Offline Evaluation Engine for Track 2 (Attribute-based Person Retrieval).

Evaluates any checkpoint against the 33,407 validation gallery images and 3,462 queries.
Computes:
  - mADM (primary challenge metric)
  - mAP
  - CMC Rank-1, Rank-5, Rank-10
  - mINP
  - PAR mA
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import Task2RetrievalEvalData, UPARDataset, get_default_transforms, resolve_data_root
from src.metrics import compute_par_metrics, compute_retrieval_metrics
from src.models import build_model


def get_device() -> torch.device:
    """Select best available device (MPS on macOS, CUDA on Linux, CPU fallback)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def extract_gallery_probabilities(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_fp16: bool = True,
    log_interval: int = 50,
) -> np.ndarray:
    """Extract continuous attribute prediction probabilities for all gallery images."""
    model.eval()
    all_probs = []
    total_batches = len(val_loader)
    total_samples = len(val_loader.dataset)
    processed = 0
    t_start = time.time()

    for batch_idx, (images, _, _, _) in enumerate(val_loader):
        batch_size = images.size(0)
        images = images.to(device)
        if use_fp16 and device.type in ["mps", "cuda"]:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_probs.append(probs)
        processed += batch_size

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t_start
            speed = processed / max(elapsed, 1e-4)
            pct = (batch_idx + 1) / total_batches * 100
            print(f"[Evaluation] Gallery Extraction: [{processed:5d}/{total_samples:5d}] ({pct:5.1f}%) | "
                  f"Batch [{batch_idx + 1:3d}/{total_batches:3d}] | Speed: {speed:5.1f} img/s", flush=True)

    return np.vstack(all_probs)



def evaluate_retrieval(
    gallery_probs: np.ndarray,
    retrieval_data: Task2RetrievalEvalData,
) -> dict[str, float]:
    """Compute continuous distance matrix and evaluate Track 2 retrieval metrics.

    Args:
        gallery_probs: (G, 40) float array of predicted attribute probabilities
        retrieval_data: Task2RetrievalEvalData container
    """
    queries = retrieval_data.queries      # (Q, 40)
    gallery_gt = retrieval_data.gallery_gt  # (G, 40)

    # Compute continuous L1 distance without creating (Q, G, 40) tensor:
    # sum|q - p| = sum(p) + q @ (1 - 2p).T
    distances = (
        gallery_probs.sum(axis=1, keepdims=True).T
        + queries @ (1.0 - 2.0 * gallery_probs).T
    )  # (Q, G)

    metrics = compute_retrieval_metrics(
        distances=distances,
        queries=queries,
        gallery_gt=gallery_gt,
    )
    return metrics


def run_full_evaluation(
    model: torch.nn.Module,
    data_root: str,
    device: torch.device,
    batch_size: int = 128,
    height: int = 256,
    width: int = 128,
    num_workers: int = 0,
    use_fp16: bool = True,
) -> dict[str, float]:
    """Run full validation evaluation on all 33,407 gallery images and 3,462 queries."""
    data_root = resolve_data_root(data_root)
    val_csv = os.path.join(data_root, "annotations", "val.csv")
    val_dataset = UPARDataset(
        csv_path=val_csv,
        data_root=data_root,
        height=height,
        width=width,
        is_train=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    retrieval_data = Task2RetrievalEvalData(data_root=data_root)

    t0 = time.time()
    gallery_probs = extract_gallery_probabilities(model, val_loader, device, use_fp16=use_fp16)
    infer_time = time.time() - t0


    t1 = time.time()
    retrieval_metrics = evaluate_retrieval(gallery_probs, retrieval_data)
    par_metrics = compute_par_metrics(gallery_probs, retrieval_data.gallery_gt)
    eval_time = time.time() - t1

    combined_results = {
        "mADM": retrieval_metrics["mADM"],
        "mAP": retrieval_metrics["mAP"],
        "Rank-1": retrieval_metrics["Rank-1"],
        "Rank-5": retrieval_metrics["Rank-5"],
        "Rank-10": retrieval_metrics["Rank-10"],
        "mINP": retrieval_metrics["mINP"],
        "PAR_mA": par_metrics["PAR_mA"],
        "infer_time_sec": infer_time,
        "eval_time_sec": eval_time,
    }

    return combined_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Track 2 Model Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--data_root", type=str, default="UPAR-Challenge-2027/data", help="Data root")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=128, help="Image width")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model = build_model(args.model, pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    results = run_full_evaluation(
        model=model,
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
    )

    print("\n" + "=" * 60)
    print("--- EVALUATION REPORT ---")
    print(f"mADM:    {results['mADM']:.6f}  (Live SOTA: 0.627993 | Target: >= 0.6800)")
    print(f"mAP:     {results['mAP']:.6f}")
    print(f"Rank-1:  {results['Rank-1'] * 100:.2f}%")
    print(f"Rank-5:  {results['Rank-5'] * 100:.2f}%")
    print(f"Rank-10: {results['Rank-10'] * 100:.2f}%")
    print(f"mINP:    {results['mINP']:.6f}")
    print(f"PAR mA:  {results['PAR_mA'] * 100:.2f}%")
    print(f"Inference Time: {results['infer_time_sec']:.1f}s | Eval Time: {results['eval_time_sec']:.1f}s")
    print("=" * 60)
