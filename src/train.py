"""Production Training Engine for RWS 2027 Track 2 (ABPR).

Features:
  - Supports ResNet-50-IBN and ConvNeXt architectures
  - Domain-balanced sampling (Market1501, PA-100k, PETA)
  - Multi-task loss: Asymmetric Loss (ASL) + Category Softmax Loss
  - Apple Silicon Metal (MPS) and CUDA acceleration
  - Cosine annealing learning rate scheduler with warmup
  - Checkpoint manager saving best model by validation mADM
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

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


from src.submission_generator import package_submission


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: CombinedABPRLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_fp16: bool = True,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    running_asl = 0.0
    running_group = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for batch_idx, (images, labels, _, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        if use_fp16 and device.type in ["mps", "cuda"]:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(images)
                loss, details = criterion(logits.float(), labels)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
        else:
            logits = model(images)
            loss, details = criterion(logits, labels)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += details["loss_total"] * batch_size
        running_asl += details["loss_asl"] * batch_size
        running_group += details["loss_group"] * batch_size
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{details['loss_total']:.4f}",
            "asl": f"{details['loss_asl']:.4f}",
            "grp": f"{details['loss_group']:.4f}",
        })

    if device.type == "mps":
        torch.mps.synchronize()

    return {
        "train_loss": running_loss / max(total_samples, 1),
        "train_asl": running_asl / max(total_samples, 1),
        "train_group": running_group / max(total_samples, 1),
    }



def train_abpr(
    model_name: str = "resnet50",
    data_root: str = "UPAR-Challenge-2027/data",
    epochs: int = 30,
    batch_size: int = 96,
    val_batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    asl_weight: float = 1.0,
    group_weight: float = 0.5,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    height: int = 256,
    width: int = 128,
    use_fp16: bool = True,
    accumulation_steps: int = 1,
    save_dir: str = "checkpoints",
    exp_id: str = "EXP-001",
    eval_interval: int = 1,
    num_workers: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
) -> dict[str, float]:
    """Execute complete training pipeline and return best evaluation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()
    print(f"[{exp_id}] Initializing training on device: {device} (FP16: {use_fp16})")

    if num_workers is None:
        import platform
        num_workers = 0 if platform.system() == "Darwin" else 2

    # 1. Dataset & Loaders
    train_csv = os.path.join(data_root, "annotations", "train.csv")
    train_dataset = UPARDataset(
        csv_path=train_csv,
        data_root=data_root,
        height=height,
        width=width,
        is_train=True,
    )

    batch_sampler = DomainBalancedBatchSampler(
        domain_ids=train_dataset.domain_ids,
        batch_size=batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"[{exp_id}] Loaded {len(train_dataset)} training images across 3 surveillance domains.")
    print(f"[{exp_id}] Domain batches per epoch: {len(batch_sampler)}")

    # 2. Model Initialization
    model = build_model(model_name, pretrained=True, num_attributes=40)
    model.to(device)

    # 3. Loss & Optimizer
    criterion = CombinedABPRLoss(
        asl_weight=asl_weight,
        group_weight=group_weight,
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=clip,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_epoch = 1
    best_madm = 0.0
    best_metrics = {}
    best_path = os.path.join(save_dir, f"{exp_id}_best.pth")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[{exp_id}] Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_madm = ckpt.get("best_madm", 0.0)
        best_metrics = ckpt.get("metrics", {})
        print(f"[{exp_id}] Resumed successfully. Starting at epoch {start_epoch}, previous best mADM: {best_madm:.6f}")

    for group in optimizer.param_groups:
        group["lr"] = lr
        group["initial_lr"] = lr

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
        last_epoch=start_epoch - 1 if start_epoch > 1 else -1,
    )

    # 4. Training Loop
    history = []
    print(f"[{exp_id}] Starting training from epoch {start_epoch} through {epochs} (total {epochs - start_epoch + 1} remaining epochs)...")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_fp16=use_fp16,
            accumulation_steps=accumulation_steps,
        )
        scheduler.step()
        epoch_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"[{exp_id}] Epoch {epoch:02d}/{epochs} - Loss: {train_stats['train_loss']:.4f} "
              f"(ASL: {train_stats['train_asl']:.4f}, Grp: {train_stats['train_group']:.4f}) "
              f"- LR: {curr_lr:.2e} - Time: {epoch_time:.1f}s")

        if device.type == "mps":
            torch.mps.empty_cache()

        # Validation evaluation
        if epoch % eval_interval == 0 or epoch == epochs:
            print(f"[{exp_id}] Running full validation evaluation (33,407 gallery, 3,462 queries)...")
            val_metrics = run_full_evaluation(
                model=model,
                data_root=data_root,
                device=device,
                batch_size=val_batch_size,
                height=height,
                width=width,
                num_workers=num_workers,
                use_fp16=use_fp16,
            )

            current_madm = val_metrics["mADM"]
            print(f"[{exp_id}] Val Evaluation -> mADM: {current_madm:.6f} | "
                  f"mAP: {val_metrics['mAP']:.4f} | Rank-1: {val_metrics['Rank-1']*100:.2f}% | "
                  f"PAR mA: {val_metrics['PAR_mA']*100:.2f}%")

            is_best = current_madm > best_madm
            if is_best:
                best_madm = current_madm
                best_metrics = val_metrics
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_madm": best_madm,
                    "metrics": val_metrics,
                    "config": {
                        "model": model_name,
                        "batch_size": batch_size,
                        "lr": lr,
                        "height": height,
                        "width": width,
                    },
                }, best_path)
                print(f"[{exp_id}] *** NEW BEST mADM: {best_madm:.6f} -> Saved to {best_path} ***")

            # Save latest checkpoint
            latest_path = os.path.join(save_dir, f"{exp_id}_latest.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "current_madm": current_madm,
                "best_madm": best_madm,
                "metrics": val_metrics,
            }, latest_path)

            record = {
                "epoch": epoch,
                "train_loss": train_stats["train_loss"],
                "train_asl": train_stats["train_asl"],
                "train_group": train_stats["train_group"],
                "val_mADM": current_madm,
                "val_mAP": val_metrics["mAP"],
                "val_Rank-1": val_metrics["Rank-1"],
                "val_PAR_mA": val_metrics["PAR_mA"],
                "lr": curr_lr,
            }
            history.append(record)

            # Write training log
            log_path = os.path.join(save_dir, f"{exp_id}_history.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    # Auto package Codabench submission if checkpoint exists
    if os.path.exists(best_path):
        os.makedirs("submissions", exist_ok=True)
        sub_zip = os.path.join("submissions", f"{exp_id}_submission.zip")
        package_submission(best_path, sub_zip)
        print(f"[{exp_id}] Auto-packaged submission ready: {sub_zip}")

    print(f"[{exp_id}] Training complete! Best validation mADM: {best_madm:.6f}")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ABPR Model for RWS 2027 Track 2")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "convnext_tiny", "convnext_base"])
    parser.add_argument("--data_root", type=str, default="UPAR-Challenge-2027/data")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--asl_weight", type=float, default=1.0)
    parser.add_argument("--group_weight", type=float, default=0.5)
    parser.add_argument("--gamma_neg", type=float, default=4.0)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--exp_id", type=str, default="EXP-001")
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_abpr(
        model_name=args.model,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        asl_weight=args.asl_weight,
        group_weight=args.group_weight,
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.clip,
        height=args.height,
        width=args.width,
        use_fp16=args.fp16,
        accumulation_steps=args.accumulation_steps,
        save_dir=args.save_dir,
        exp_id=args.exp_id,
        eval_interval=args.eval_interval,
        num_workers=args.num_workers,
        resume_checkpoint=args.resume,
    )

