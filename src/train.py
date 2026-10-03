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
import warnings
from typing import Optional

# Filter benign framework stride and unused parameter warnings
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")
warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import (
    DomainBalancedBatchSampler,
    UPARDataset,
    get_default_transforms,
    get_surveillance_mevid_transforms,
    resolve_data_root,
    stage_dataset_to_fast_local_storage,
)
from src.evaluate import get_device, run_full_evaluation
from src.losses import CombinedABPRLoss
from src.models import build_model
from src.submission_generator import package_submission


def setup_distributed() -> tuple[bool, int, int, int]:
    """Initialize DistributedDataParallel (DDP) if launched via torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
            )
        else:
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="env://",
            )
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed() -> None:
    """Clean up DDP process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


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


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string (e.g., '14m 23s' or '02h 15m 10s')."""
    if seconds is None or np.isnan(seconds) or np.isinf(seconds) or seconds < 0:
        return "--m --s"
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def print_codabench_table(
    epoch: int,
    epochs: int,
    metrics: dict[str, float],
    exp_id: str = "EXP-003",
    participant: str = "Jyothiradithya Lingam",
    is_best: bool = False,
) -> None:
    """Print validation results in the exact official Codabench competition leaderboard table format."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print("\n" + "=" * 115, flush=True)
    print("Results", flush=True)
    print(f"Task:        Attribute-based Person Retrieval - validation set (Epoch {epoch:02d}/{epochs:02d})", flush=True)
    print(f"{'#':<4} {'Participant':<26} {'Date':<18} {'ID':<10} {'mADM':<9} {'mAP':<9} {'Rank-1':<9} {'Rank-5':<9} {'Rank-10':<9} {'mINP':<9}", flush=True)
    print("-" * 115, flush=True)
    print(
        f"{epoch:<4} {participant:<26} {now_str:<18} {exp_id:<10} "
        f"{metrics['mADM']:<9.4f} {metrics['mAP']:<9.4f} "
        f"{metrics['Rank-1']:<9.4f} {metrics['Rank-5']:<9.4f} {metrics['Rank-10']:<9.4f} "
        f"{metrics['mINP']:<9.4f}",
        flush=True,
    )
    if "PAR_mA" in metrics:
        print(f"Pedestrian Attribute Recognition (PAR) Mean Accuracy: {metrics['PAR_mA'] * 100:.2f}%", flush=True)
    if is_best:
        print(f"*** NEW BEST VALIDATION SCORE: mADM = {metrics['mADM']:.4f} ***", flush=True)
    print("=" * 115 + "\n", flush=True)


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
    scaler: Optional[torch.amp.GradScaler] = None,
    log_interval: int = 50,
    is_master: bool = True,
    world_size: int = 1,
    last_epoch_sec: Optional[float] = None,
    last_val_sec: Optional[float] = None,
) -> dict[str, float]:
    """Train model for one epoch with continuous stdout logging and real-time ETAs."""
    model.train()
    running_loss = 0.0
    running_asl = 0.0
    running_group = 0.0
    total_samples = 0

    total_batches = len(train_loader)
    t_epoch_start = time.time()

    for batch_idx, (images, labels, _, _) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Mixed precision forward pass
        if use_fp16 and device.type == "cuda":
            try:
                autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16)

            with autocast_ctx:
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

        # Continuous stdout logging on master rank with real-time ETAs
        if is_master and ((batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches):
            elapsed = time.time() - t_epoch_start
            effective_samples = total_samples * world_size
            speed = effective_samples / max(elapsed, 1e-4)
            pct = (batch_idx + 1) / total_batches * 100
            curr_lr = optimizer.param_groups[0]["lr"]
            mem_str = get_memory_info(device)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Precise ETA math: per-epoch and across all remaining epochs
            done_batches = batch_idx + 1
            rem_batches_epoch = total_batches - done_batches
            sec_per_batch = elapsed / done_batches
            epoch_eta_sec = rem_batches_epoch * sec_per_batch

            est_epoch_sec = last_epoch_sec if last_epoch_sec is not None else (total_batches * sec_per_batch)
            est_val_sec = last_val_sec if last_val_sec is not None else 150.0
            remaining_epochs = epochs - epoch
            total_eta_sec = epoch_eta_sec + est_val_sec + remaining_epochs * (est_epoch_sec + est_val_sec)

            print(
                f"[{now_str}] Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Batch [{batch_idx + 1:04d}/{total_batches:04d}] ({pct:5.1f}%) | "
                f"Loss: {details['loss_total']:.4f} (ASL: {details['loss_asl']:.4f}, Grp: {details['loss_group']:.4f}) | "
                f"LR: {curr_lr:.2e} | Speed: {speed:5.1f} img/s | "
                f"ETA: [Epoch: {format_duration(epoch_eta_sec)} | Total: {format_duration(total_eta_sec)}] | "
                f"{mem_str}",
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
    eval_interval: int = 2,
    num_workers: Optional[int] = None,
    resume_checkpoint: Optional[str] = None,
    log_interval: int = 50,
    preload_ram: bool = True,
    surveillance_aug: bool = False,
    pretrained_path: Optional[str] = None,
    semantic_weight: float = 0.8,
) -> dict[str, float]:
    """Execute complete training pipeline and return best evaluation metrics."""
    os.makedirs(save_dir, exist_ok=True)
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_master = (rank == 0)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if is_master:
        print(f"[{exp_id}] ==========================================================================")
        print(f"[{exp_id}] Starting RWS 2027 Production Training Run: {exp_id}")
        print(f"[{exp_id}] Architecture: {model_name} | Input Resolution: {height}x{width}")
        print(f"[{exp_id}] Device: {device} | Detected CUDA GPUs: {num_gpus} | FP16: {use_fp16}")
        if is_distributed:
            print(f"[{exp_id}] DistributedDataParallel (DDP) Activated: {world_size} processes (NCCL backend, zero PCIe weight broadcast)")
        else:
            print(f"[{exp_id}] Single-GPU Engine Activated: direct local execution (zero DataParallel GIL overhead)")
        print(f"[{exp_id}] Effective Global Batch Size: {batch_size * world_size} | Epochs: {epochs}")
        if surveillance_aug:
            print(f"[{exp_id}] Surveillance Domain Augmentations: ENABLED (MEVID 75px ResolutionDrop + Motion Blur + Erasing)")
        if pretrained_path:
            print(f"[{exp_id}] Foundation Pretrained Weights: {pretrained_path}")
        print(f"[{exp_id}] ==========================================================================", flush=True)

    if num_workers is None:
        import platform
        if platform.system() == "Darwin":
            num_workers = 0
        else:
            num_workers = 2 if is_distributed else min(4, max(num_gpus, 1) * 2)

    # 1. Dataset & Loaders
    resolved_root = resolve_data_root(data_root)
    if resolved_root != data_root:
        if is_master:
            print(f"[{exp_id}] Auto-detected & resolved true dataset root: '{resolved_root}' (original: '{data_root}')", flush=True)
        data_root = resolved_root

    train_csv = os.path.join(data_root, "annotations", "train.csv")
    if not os.path.exists(train_csv):
        if is_master and os.path.exists(data_root):
            print(f"[{exp_id}] Contents of {data_root}: {os.listdir(data_root)}", flush=True)
        raise FileNotFoundError(f"Training annotations not found at: {train_csv}")

    if surveillance_aug:
        train_tf = get_surveillance_mevid_transforms(height=height, width=width, is_train=True)
    else:
        train_tf = get_default_transforms(height=height, width=width, is_train=True)

    train_dataset = UPARDataset(
        csv_path=train_csv,
        data_root=data_root,
        transform=train_tf,
        height=height,
        width=width,
        is_train=True,
        preload_to_ram=preload_ram,
    )

    batch_sampler = DomainBalancedBatchSampler(
        domain_ids=train_dataset.domain_ids,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    if is_master:
        print(f"[{exp_id}] Dataset Loaded: {len(train_dataset)} training images across Market1501, PA100k, PETA.")
        print(f"[{exp_id}] Batches per Epoch per Rank: {len(batch_sampler)} | DataLoader Workers: {num_workers} (Prefetch: 2)", flush=True)

    # 2. Model Initialization
    raw_model = build_model(
        model_name,
        pretrained=True,
        num_attributes=40,
        pretrained_path=pretrained_path,
        semantic_weight=semantic_weight,
    )
    raw_model.to(device)

    # Wrap with DistributedDataParallel if multi-process DDP, otherwise use raw_model directly (NEVER DataParallel)
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
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
    if use_fp16 and device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    start_epoch = 1
    best_madm = 0.0
    best_metrics = {}
    best_path = os.path.join(save_dir, f"{exp_id}_best.pth")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if is_master:
            print(f"[{exp_id}] Resuming from checkpoint: {resume_checkpoint}", flush=True)
        ckpt = torch.load(resume_checkpoint, map_location=device)
        raw_state = ckpt["model_state_dict"]
        clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in raw_state.items()}
        if is_distributed:
            model.module.load_state_dict(clean_state)
        else:
            model.load_state_dict(clean_state)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_madm = ckpt.get("best_madm", 0.0)
        best_metrics = ckpt.get("metrics", {})
        if is_master:
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
    if is_master:
        print(f"[{exp_id}] Beginning execution from Epoch {start_epoch} through {epochs}...", flush=True)

    last_epoch_sec: Optional[float] = None
    last_val_sec: Optional[float] = None

    for epoch in range(start_epoch, epochs + 1):
        batch_sampler.set_epoch(epoch)
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
            is_master=is_master,
            world_size=world_size,
            last_epoch_sec=last_epoch_sec,
            last_val_sec=last_val_sec,
        )
        scheduler.step()
        epoch_time = time.time() - t0
        last_epoch_sec = epoch_time

        curr_lr = optimizer.param_groups[0]["lr"]
        if is_master:
            print(
                f"[{exp_id}] Epoch {epoch:02d}/{epochs:02d} Complete | "
                f"Train Loss: {train_stats['train_loss']:.4f} (ASL: {train_stats['train_asl']:.4f}, Grp: {train_stats['train_group']:.4f}) | "
                f"LR: {curr_lr:.2e} | Duration: {format_duration(epoch_time)}",
                flush=True,
            )

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        # Validation evaluation on master rank
        if is_master and (epoch % eval_interval == 0 or epoch == epochs):
            print(f"[{exp_id}] Running full validation evaluation (33,407 gallery, 3,462 queries)...", flush=True)
            t_val_start = time.time()
            eval_model = model.module if hasattr(model, "module") else model
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
            val_duration = time.time() - t_val_start
            last_val_sec = val_duration

            current_madm = val_metrics["mADM"]
            is_best = current_madm > best_madm

            # Print results formatted strictly matching the official Codabench server table
            print_codabench_table(
                epoch=epoch,
                epochs=epochs,
                metrics=val_metrics,
                exp_id=exp_id,
                participant="Jyothiradithya Lingam",
                is_best=is_best,
            )

            raw_save_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

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

        if is_distributed:
            torch.distributed.barrier()

    # Auto package Codabench submission if best checkpoint exists (master rank)
    if is_master and os.path.exists(best_path):
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

    if is_master:
        print(f"[{exp_id}] Training execution finished! Peak Validation mADM: {best_madm:.6f}", flush=True)
    cleanup_distributed()
    return best_metrics


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ABPR Model for RWS 2027 Track 2")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--model", type=str, default="convnext_base", help="Model architecture (resnet50, convnext_tiny, convnext_base, swin_t, etc.)")
    parser.add_argument("--data_root", type=str, default="UPAR-Challenge-2027/data")
    parser.add_argument("--epochs", type=int, default=25)
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
    parser.add_argument("--no_preload_ram", action="store_true", default=False, help="Disable preloading dataset into RAM")
    parser.add_argument("--surveillance_aug", action="store_true", default=False, help="Activate MEVID surveillance domain augmentations")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to foundation pretrained weights")
    parser.add_argument("--semantic_weight", type=float, default=0.8, help="Semantic controller weight for SOLIDER")
    args = parser.parse_args()

    preload_ram = not args.no_preload_ram

    # If YAML config is specified, load parameters from file
    if args.config and os.path.exists(args.config):
        cfg = load_yaml_config(args.config)
        print(f"Loaded configuration from YAML: {args.config}")
        model_name = cfg.get("model", {}).get("backbone", args.model)
        pretrained_path = cfg.get("model", {}).get("pretrained_path", args.pretrained_path)
        semantic_weight = float(cfg.get("model", {}).get("semantic_weight", args.semantic_weight))
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
        surveillance_aug = cfg.get("augmentations", {}).get("surveillance_domain", args.surveillance_aug)
    else:
        model_name = args.model
        pretrained_path = args.pretrained_path
        semantic_weight = args.semantic_weight
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
        surveillance_aug = args.surveillance_aug

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
        preload_ram=preload_ram,
        surveillance_aug=surveillance_aug,
        pretrained_path=pretrained_path,
        semantic_weight=semantic_weight,
    )

