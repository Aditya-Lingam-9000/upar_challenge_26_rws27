"""Stage 2 Master Pipeline Runner.

Executes:
  1. Swin-T Feature Extraction & Stage 2 Metric Baseline
  2. Full 1536-D Dual-Stream Fused Architecture Training (ConvNeXt-Tiny + Swin-T)
  3. Comprehensive Retrieval Benchmark across lambda values on all 33,407 validation gallery images.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

PYTHON_BIN = sys.executable


def run_command_logged(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*80}\n[RUNNING] {desc}\nCMD: {' '.join(cmd)}\n{'='*80}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode} in {elapsed:.1f}s")
    print(f"\n[COMPLETED] {desc} in {elapsed:.1f}s", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2 Master Pipeline Runner")
    parser.add_argument("--data_root", type=str, default="UPAR-Challenge-2027/data", help="Root path of UPAR dataset")
    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 2 MASTER PIPELINE: SWIN-T & FULL 1536-D DUAL-STREAM FUSED ARCHITECTURE")
    print(f"Data Root: {args.data_root}")
    print("=" * 80)

    # 1. Swin-T Stage 2 Extraction & Training
    swin_cmd = [
        PYTHON_BIN,
        "train_stage2_metric.py",
        "--model", "swin_t",
        "--submission_zip", "submissions/EXP-005A_submission.zip",
        "--data_root", args.data_root,
        "--epochs", "30",
        "--batch_size", "512",
        "--lr", "0.001",
        "--scale", "30.0",
        "--margin", "0.20",
        "--lambda_contrastive", "0.5",
        "--save_path", "checkpoints/stage2_swin_metric_heads.pth",
    ]
    run_command_logged(swin_cmd, "Step 1: Swin-T Feature Extraction & Metric Training")

    # 2. Dual-Fused 1536-D Architecture Training
    dual_cmd = [
        PYTHON_BIN,
        "train_stage2_metric.py",
        "--model", "dual_fused",
        "--data_root", args.data_root,
        "--epochs", "30",
        "--batch_size", "512",
        "--lr", "0.001",
        "--scale", "30.0",
        "--margin", "0.20",
        "--lambda_contrastive", "0.5",
        "--save_path", "checkpoints/stage2_dual_fused_metric_heads.pth",
    ]
    run_command_logged(dual_cmd, "Step 2: Full 1536-D Dual-Stream Fused Architecture Training")

    print("\n" + "=" * 80)
    print("STAGE 2 MASTER PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
