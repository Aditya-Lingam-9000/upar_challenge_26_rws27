"""Hardware Profiling & Calibration Engine for RWS 2027 Track 2 (ABPR).

Executes a 3 to 5 epoch calibration run to profile:
  1. Image throughput (images/sec)
  2. Exact time taken per training epoch and evaluation step
  3. Apple Silicon unified memory profile:
     - MPS Active Allocated Memory
     - MPS Driver Unified Memory
     - Host System RAM Footprint
  4. Loss convergence across Asymmetric Loss (ASL) and Category Softmax
  5. Validation metric trajectory (mADM, mAP, Rank-1, PAR mA)
  6. Actionable recommendations for the long production run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DomainBalancedBatchSampler, UPARDataset
from src.evaluate import get_device, run_full_evaluation
from src.losses import CombinedABPRLoss
from src.models import build_model


def get_hardware_telemetry(device: torch.device) -> dict[str, float]:
    """Capture unified memory and system RAM metrics."""
    telemetry = {
        "mps_allocated_mb": 0.0,
        "mps_driver_mb": 0.0,
        "sys_ram_used_gb": 0.0,
        "sys_ram_total_gb": 16.0,
    }

    if device.type == "mps":
        if hasattr(torch.mps, "current_allocated_memory"):
            telemetry["mps_allocated_mb"] = float(torch.mps.current_allocated_memory() / (1024 ** 2))
        if hasattr(torch.mps, "driver_allocated_memory"):
            telemetry["mps_driver_mb"] = float(torch.mps.driver_allocated_memory() / (1024 ** 2))

    # Read host memory via vm_stat / psutil or fallback
    try:
        import psutil
        mem = psutil.virtual_memory()
        telemetry["sys_ram_used_gb"] = float(mem.used / (1024 ** 3))
        telemetry["sys_ram_total_gb"] = float(mem.total / (1024 ** 3))
    except ImportError:
        pass

    return telemetry


def run_calibration(
    epochs: int = 3,
    batch_size: int = 64,
    batches_per_epoch: int = 100,  # 100 batches * 64 = 6,400 representative images per epoch
    lr: float = 1e-4,
    model_name: str = "resnet50",
    data_root: str = "UPAR-Challenge-2027/data",
    height: int = 256,
    width: int = 128,
    save_dir: str = "checkpoints/calibration",
    exp_id: str = "CALIB-001",
) -> dict[str, Any]:
    """Execute calibration profiling run."""
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print("=" * 70)
    print(f"--- HARDWARE PROFILING & CALIBRATION RUN ({exp_id}) ---")
    print(f"Platform:      {platform.system()} ({platform.machine()})")
    print(f"Compute Engine: {device} (Apple Silicon Metal Performance Shaders)")
    print(f"Model:         {model_name} (Pretrained ResNet-50)")
    print(f"Batch Size:    {batch_size}")
    print(f"Batches/Epoch: {batches_per_epoch} ({batches_per_epoch * batch_size:,} images / epoch)")
    print(f"Epochs:        {epochs}")
    print("=" * 70)

    # 1. Dataset & Loader
    train_csv = os.path.join(data_root, "annotations", "train.csv")
    train_dataset = UPARDataset(
        csv_path=train_csv,
        data_root=data_root,
        height=height,
        width=width,
        is_train=True,
    )
    sampler = DomainBalancedBatchSampler(
        domain_ids=train_dataset.domain_ids,
        batch_size=batch_size,
    )

    # Use num_workers=0 on macOS Darwin for zero IPC serialization latency
    num_workers = 0 if platform.system() == "Darwin" else 2
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )

    # 2. Model & Optimization
    model = build_model(model_name, pretrained=True, num_attributes=40)
    model.to(device)
    criterion = CombinedABPRLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    initial_telemetry = get_hardware_telemetry(device)
    print(f"Initial Memory Footprint:")
    print(f"  MPS Allocated:  {initial_telemetry['mps_allocated_mb']:.1f} MB")
    print(f"  MPS Driver Mem: {initial_telemetry['mps_driver_mb']:.1f} MB")
    print("-" * 70)

    epoch_logs = []
    total_t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_asl = 0.0
        running_group = 0.0
        total_samples = 0

        t0_train = time.time()
        batch_count = 0

        pbar = tqdm(train_loader, total=batches_per_epoch, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, labels, _, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            bs = images.size(0)

            optimizer.zero_grad()
            logits = model(images)
            loss, details = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += details["loss_total"] * bs
            running_asl += details["loss_asl"] * bs
            running_group += details["loss_group"] * bs
            total_samples += bs
            batch_count += 1

            pbar.set_postfix({
                "loss": f"{details['loss_total']:.4f}",
                "asl": f"{details['loss_asl']:.4f}",
                "grp": f"{details['loss_group']:.4f}",
            })

            if batch_count >= batches_per_epoch:
                break

        if device.type == "mps":
            torch.mps.synchronize()

        train_time = time.time() - t0_train
        train_throughput = total_samples / train_time
        scheduler.step()

        mem_after_train = get_hardware_telemetry(device)

        # Validation Step
        print(f"\n[Epoch {epoch}/{epochs}] Running validation on 33,407 gallery images...")
        t0_val = time.time()
        val_metrics = run_full_evaluation(
            model=model,
            data_root=data_root,
            device=device,
            batch_size=batch_size,
            height=height,
            width=width,
            num_workers=num_workers,
        )
        val_time = time.time() - t0_val

        avg_loss = running_loss / max(total_samples, 1)
        avg_asl = running_asl / max(total_samples, 1)
        avg_group = running_group / max(total_samples, 1)
        curr_lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "train_time_sec": round(train_time, 2),
            "val_time_sec": round(val_time, 2),
            "total_epoch_time_sec": round(train_time + val_time, 2),
            "train_throughput_imgs_per_sec": round(train_throughput, 1),
            "train_loss": round(avg_loss, 4),
            "train_asl": round(avg_asl, 4),
            "train_group": round(avg_group, 4),
            "val_mADM": round(val_metrics["mADM"], 6),
            "val_mAP": round(val_metrics["mAP"], 4),
            "val_Rank-1": round(val_metrics["Rank-1"], 4),
            "val_PAR_mA": round(val_metrics["PAR_mA"], 4),
            "mps_allocated_mb": round(mem_after_train["mps_allocated_mb"], 1),
            "mps_driver_mb": round(mem_after_train["mps_driver_mb"], 1),
            "learning_rate": curr_lr,
        }
        epoch_logs.append(log_entry)

        print(f"\n--- EPOCH {epoch} SUMMARY ---")
        print(f"  Training Time:   {train_time:.1f}s ({train_throughput:.1f} imgs/s)")
        print(f"  Validation Time: {val_time:.1f}s (Inference: {val_metrics['infer_time_sec']:.1f}s, Eval: {val_metrics['eval_time_sec']:.1f}s)")
        print(f"  Total Epoch:     {train_time + val_time:.1f}s ({(train_time + val_time) / 60:.2f} mins)")
        print(f"  Losses:          Total: {avg_loss:.4f} | ASL: {avg_asl:.4f} | Group: {avg_group:.4f}")
        print(f"  Validation mADM: {val_metrics['mADM']:.6f}  (Live SOTA: 0.627993 | Dummy: 0.156672)")
        print(f"  Validation mAP:  {val_metrics['mAP']:.4f} | Rank-1: {val_metrics['Rank-1']*100:.2f}% | PAR mA: {val_metrics['PAR_mA']*100:.2f}%")
        print(f"  MPS Allocated:   {mem_after_train['mps_allocated_mb']:.1f} MB")
        print(f"  MPS Driver Mem:  {mem_after_train['mps_driver_mb']:.1f} MB (Unified Memory)")
        print("-" * 70)

    total_elapsed = time.time() - total_t0

    # Save summary log
    summary_path = os.path.join(save_dir, "calibration_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "exp_id": exp_id,
            "total_elapsed_sec": round(total_elapsed, 2),
            "batch_size": batch_size,
            "epochs": epochs,
            "epoch_logs": epoch_logs,
        }, f, indent=2)

    return {
        "total_elapsed_sec": total_elapsed,
        "epoch_logs": epoch_logs,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABPR Hardware Calibration")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batches_per_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="resnet50")
    args = parser.parse_args()

    run_calibration(
        epochs=args.epochs,
        batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        lr=args.lr,
        model_name=args.model,
    )
