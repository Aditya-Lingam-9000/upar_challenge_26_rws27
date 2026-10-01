"""Production Multi-GPU Training Engine for RWS 2027 Track 2 (ABPR).

Features:
  - Supports ResNet-50 and ConvNeXt-Base (LayerNorm domain invariant) architectures
  - Multi-GPU support (NVIDIA DataParallel across 2x Tesla T4 GPUs)
  - Mixed Precision training (FP16 Autocast + GradScaler on CUDA/MPS)
  - Domain-balanced sampling (Market1501, PA-100k, PETA)
  - Combined multi-task loss: Asymmetric Loss (ASL) + 5-Group Category Softmax
  - Strictly zero-tqdm continuous stdout logging for Kaggle background executions
  - Cosine annealing learning rate scheduler with warmup
  - Checkpoint manager saving best model by validation mADM
  - Automated Codabench submission packaging with TTA & temperature scaling
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Optional

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import DomainBalancedBatchSampler, UPARDataset, resolve_data_root
from src.evaluate import get_device, run_full_evaluation
from src.losses import CombinedABPRLoss
from src.models import build_model
from src.submission_generator import package_submission


def get_memory_info(device: torch.device) -> str:
    """Return current memory usage string."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        return f"VRAM: {allocated:.0f}MB / {reserved:.0f}MB"
    elif device.type == "mps":
        allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
        return f"UnifiedMem: {allocated:.0f}MB"
    return "CPU"


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: CombinedABPRLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    use_fp16: bool = True,
    accumulation_steps: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """Train model for one epoch with continuous stdout logging."""
    model.train()
    running_loss = 0.0
    running_asl = 0.0
    running_group = 0.0
    total_samples = 0

    total_batches = len(train_loader)
    total_images_in_epoch = len(train_loader.dataset)
    t_epoch_start = time.time()
    t_batch_start = time.time()

    for batch_idx, (images, labels, _, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Mixed precision forward pass
        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(images)
                loss, details = criterion(logits.float(), labels)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        elif use_fp16 and device.type == "mps":
            with torch.autocast(device_type="mps", dtype=torch.float16):
                logits = model(images)
                loss, details = criterion(logits.float(), labels)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
            loss.backward()
        else:
            logits = model(images)
            loss, details = criterion(logits, labels)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            loss.backward()

        # Optimizer step with optional gradient scaler
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
            if scaler is not None and device.type == "cuda":
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            optimizer.zero_grad()

        running_loss += details["loss_total"] * batch_size
        running_asl += details["loss_asl"] * batch_size
        running_group += details["loss_group"] * batch_size
        total_samples += batch_size

        # Continuous stdout logging at fixed intervals
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t_epoch_start
            speed = total_samples / max(elapsed, 1e-4)
            pct = (batch_idx + 1) / total_batches * 100
            curr_lr = optimizer.param_groups[0]["lr"]
            mem_str = get_memory_info(device)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"[{now_str}] Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Batch [{batch_idx + 1:04d}/{total_batches:04d}] ({pct:5.1f}%) | "
                f"Loss: {details['loss_total']:.4f} (ASL: {details['loss_asl']:.4f}, Grp: {details['loss_group']:.4f}) | "
                f"LR: {curr_lr:.2e} | Speed: {speed:5.1f} img/s | {mem_str}",
                flush=True,
            )

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    return {
        "train_loss": running_loss / max(total_samples, 1),
        "train_asl": running_asl / max(total_samples, 1),
        "train_group": running_group / max(total_samples, 1),
    }


def train_abpr(
    model_name: str = "convnext_base",
    data_root: str = "UPAR-Challenge-2027/data",
    epochs: int = 30,
    batch_size: int = 64,
    val_batch_size: int = 128,
    lr: float = 1e-4,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-4,
    asl_weight: float = 1.0,
    group_weight: float = 0.5,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    height: int = 384,
    width: int = 192,
    use_fp16: bool = True,
    accumulation_steps: int = 1,
    save_dir: str = "checkpoints",
    exp_id: str = "EXP-003",
    eval_interval: int = 1,
    num_workers: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """Execute complete training pipeline and return best evaluation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    device = get_device()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[{exp_id}] ==========================================================================")
    print(f"[{exp_id}] Starting RWS 2027 Production Training Run: {exp_id}")
    print(f"[{exp_id}] Architecture: {model_name} | Input Resolution: {height}x{width}")
    print(f"[{exp_id}] Device: {device} | Detected CUDA GPUs: {num_gpus} | FP16: {use_fp16}")
    if num_gpus > 1:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"[{exp_id}] Multi-GPU DataParallel Activated across {num_gpus} devices: {gpu_names}")
    print(f"[{exp_id}] Batch Size: {batch_size} (effective per step) | Epochs: {epochs}")
    print(f"[{exp_id}] ==========================================================================", flush=True)

    if num_workers is None:
        import platform
        if platform.system() == "Darwin":
            num_workers = 0
        else:
            # On Linux / Kaggle, use 4 workers for fast data pipelining
            num_workers = min(4 * max(num_gpus, 1), 8)

    # 1. Dataset & Loaders
    resolved_root = resolve_data_root(data_root)
    if resolved_root != data_root:
        print(f"[{exp_id}] Auto-detected & resolved true dataset root: '{resolved_root}' (original: '{data_root}')", flush=True)
        data_root = resolved_root

    train_csv = os.path.join(data_root, "annotations", "train.csv")
    if not os.path.exists(train_csv):
        # Print diagnostic directory contents if missing
        if os.path.exists(data_root):
            print(f"[{exp_id}] Contents of {data_root}: {os.listdir(data_root)}", flush=True)
        raise FileNotFoundError(f"Training annotations not found at: {train_csv}")

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
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"[{exp_id}] Dataset Loaded: {len(train_dataset)} training images across Market1501, PA100k, PETA.")
    print(f"[{exp_id}] Batches per Epoch: {len(batch_sampler)} | DataLoader Workers: {num_workers}", flush=True)

    # 2. Model Initialization
    raw_model = build_model(model_name, pretrained=True, num_attributes=40)
    raw_model.to(device)

    # Multi-GPU DataParallel wrapping if 2+ CUDA GPUs available
    if device.type == "cuda" and num_gpus > 1:
        model = nn.DataParallel(raw_model)
    else:
        model = raw_model

    # 3. Loss & Optimizer
    criterion = CombinedABPRLoss(
        asl_weight=asl_weight,
        group_weight=group_weight,
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        clip=clip,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Gradient Scaler for mixed precision on CUDA
    scaler = torch.cuda.amp.GradScaler() if (use_fp16 and device.type == "cuda") else None

    start_epoch = 1
    best_madm = 0.0
    best_metrics = {}
    best_path = os.path.join(save_dir, f"{exp_id}_best.pth")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[{exp_id}] Resuming from checkpoint: {resume_checkpoint}", flush=True)
        ckpt = torch.load(resume_checkpoint, map_location=device)
        raw_state = ckpt["model_state_dict"]
        # Handle DataParallel prefix compatibility
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(raw_state)
        else:
            clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in raw_state.items()}
            model.load_state_dict(clean_state)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_madm = ckpt.get("best_madm", 0.0)
        best_metrics = ckpt.get("metrics", {})
        print(f"[{exp_id}] Resumed successfully. Starting at epoch {start_epoch}, previous best mADM: {best_madm:.6f}", flush=True)

    for group in optimizer.param_groups:
        group["lr"] = lr
        group["initial_lr"] = lr

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=min_lr,
        last_epoch=start_epoch - 1 if start_epoch > 1 else -1,
    )

    # 4. Training Loop
    history = []
    print(f"[{exp_id}] Beginning execution from Epoch {start_epoch} through {epochs}...", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
            use_fp16=use_fp16,
            accumulation_steps=accumulation_steps,
            scaler=scaler,
            log_interval=log_interval,
        )
        scheduler.step()
        epoch_time = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{exp_id}] Epoch {epoch:02d}/{epochs:02d} Complete | "
            f"Train Loss: {train_stats['train_loss']:.4f} (ASL: {train_stats['train_asl']:.4f}, Grp: {train_stats['train_group']:.4f}) | "
            f"LR: {curr_lr:.2e} | Duration: {epoch_time:.1f}s ({epoch_time / 60:.1f}m)",
            flush=True,
        )

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Validation evaluation
        if epoch % eval_interval == 0 or epoch == epochs:
            print(f"[{exp_id}] Running full validation evaluation (33,407 gallery, 3,462 queries)...", flush=True)
            # Evaluate using base model (unwrapped from DataParallel)
            eval_model = model.module if isinstance(model, nn.DataParallel) else model
            val_metrics = run_full_evaluation(
                model=eval_model,
                data_root=data_root,
                device=device,
                batch_size=val_batch_size,
                height=height,
                width=width,
                num_workers=num_workers,
                use_fp16=use_fp16,
            )

            current_madm = val_metrics["mADM"]
            print(
                f"[{exp_id}] Val Evaluation (Epoch {epoch:02d}) -> "
                f"mADM: {current_madm:.6f} | "
                f"mAP: {val_metrics['mAP']:.4f} | "
                f"Rank-1: {val_metrics['Rank-1']*100:.2f}% | "
                f"Rank-5: {val_metrics['Rank-5']*100:.2f}% | "
                f"Rank-10: {val_metrics['Rank-10']*100:.2f}% | "
                f"mINP: {val_metrics['mINP']:.4f} | "
                f"PAR mA: {val_metrics['PAR_mA']*100:.2f}%",
                flush=True,
            )

            is_best = current_madm > best_madm
            raw_save_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

            if is_best:
                best_madm = current_madm
                best_metrics = val_metrics
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": raw_save_state,
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
                print(f"[{exp_id}] *** NEW BEST VALIDATION mADM: {best_madm:.6f} -> Saved to {best_path} ***", flush=True)

            # Save latest checkpoint
            latest_path = os.path.join(save_dir, f"{exp_id}_latest.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": raw_save_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "current_madm": current_madm,
                "best_madm": best_madm,
                "metrics": val_metrics,
                "config": {
                    "model": model_name,
                    "batch_size": batch_size,
                    "lr": lr,
                    "height": height,
                    "width": width,
                },
            }, latest_path)

            record = {
                "epoch": epoch,
                "train_loss": train_stats["train_loss"],
                "train_asl": train_stats["train_asl"],
                "train_group": train_stats["train_group"],
                "val_mADM": current_madm,
                "val_mAP": val_metrics["mAP"],
                "val_Rank-1": val_metrics["Rank-1"],
                "val_Rank-5": val_metrics["Rank-5"],
                "val_Rank-10": val_metrics["Rank-10"],
                "val_mINP": val_metrics["mINP"],
                "val_PAR_mA": val_metrics["PAR_mA"],
                "lr": curr_lr,
            }
            history.append(record)

            # Write training history JSON log
            log_path = os.path.join(save_dir, f"{exp_id}_history.json")
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    # Auto package Codabench submission if best checkpoint exists
    if os.path.exists(best_path):
        os.makedirs("submissions", exist_ok=True)
        sub_zip = os.path.join("submissions", f"{exp_id}_submission.zip")
        package_submission(
            checkpoint_path=best_path,
            output_zip_path=sub_zip,
            model_name=model_name,
            height=height,
            width=width,
        )
        print(f"[{exp_id}] Auto-packaged Codabench submission: {sub_zip} ({os.path.getsize(sub_zip)/(1024*1024):.2f} MB)", flush=True)

    print(f"[{exp_id}] Training execution finished! Peak Validation mADM: {best_madm:.6f}", flush=True)
    return best_metrics


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ABPR Model for RWS 2027 Track 2")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--model", type=str, default="convnext_base", choices=["resnet50", "convnext_tiny", "convnext_base"])
    parser.add_argument("--data_root", type=str, default="UPAR-Challenge-2027/data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--asl_weight", type=float, default=1.0)
    parser.add_argument("--group_weight", type=float, default=0.5)
    parser.add_argument("--gamma_neg", type=float, default=4.0)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--exp_id", type=str, default="EXP-003")
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    # If YAML config is specified, load parameters from file
    if args.config and os.path.exists(args.config):
        cfg = load_yaml_config(args.config)
        print(f"Loaded configuration from YAML: {args.config}")
        model_name = cfg.get("model", {}).get("backbone", args.model)
        data_root = cfg.get("paths", {}).get("data_dir", args.data_root)
        epochs = cfg.get("epochs", args.epochs)
        batch_size = cfg.get("batch_size", args.batch_size)
        val_batch_size = cfg.get("val_batch_size", args.val_batch_size)
        lr = float(cfg.get("learning_rate", args.lr))
        min_lr = float(cfg.get("min_lr", args.min_lr))
        weight_decay = float(cfg.get("weight_decay", args.weight_decay))
        height = cfg.get("image_height", args.height)
        width = cfg.get("image_width", args.width)
        exp_id = cfg.get("experiment_id", args.exp_id)
        save_dir = cfg.get("paths", {}).get("checkpoint_dir", args.save_dir)
        num_workers = cfg.get("num_workers", args.num_workers)
    else:
        model_name = args.model
        data_root = args.data_root
        epochs = args.epochs
        batch_size = args.batch_size
        val_batch_size = args.val_batch_size
        lr = args.lr
        min_lr = args.min_lr
        weight_decay = args.weight_decay
        height = args.height
        width = args.width
        exp_id = args.exp_id
        save_dir = args.save_dir
        num_workers = args.num_workers

    train_abpr(
        model_name=model_name,
        data_root=data_root,
        epochs=epochs,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        lr=lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        asl_weight=args.asl_weight,
        group_weight=args.group_weight,
        gamma_neg=args.gamma_neg,
        gamma_pos=args.gamma_pos,
        clip=args.clip,
        height=height,
        width=width,
        use_fp16=args.fp16,
        accumulation_steps=args.accumulation_steps,
        save_dir=save_dir,
        exp_id=exp_id,
        eval_interval=args.eval_interval,
        num_workers=num_workers,
        resume_checkpoint=args.resume,
        log_interval=args.log_interval,
    )
