"""Production Distributed Training Engine for SOTA Championship Person Retrieval.

Supports:
1. PyTorch DistributedDataParallel (DDP) across Kaggle Dual Tesla T4 GPUs.
2. Differential Parameter Groups: Backbone LR (5e-5) vs. Heads LR (3.5e-4).
3. Automatic Mixed Precision (torch.cuda.amp) + Gradient Clipping.
4. TotalChampionshipLoss: Soft-target Multi-Instance SDM + MLM + ASL + Strip Loss.
5. Multi-Session Resumption: Automatically saves complete 7-component state and
   exits gracefully before Kaggle's 12-hour session timeout.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from sota_pipeline.config import SOTAConfig
from sota_pipeline.dataset import SOTAUPARDataset
from sota_pipeline.evaluate import evaluate_retrieval
from sota_pipeline.losses import TotalChampionshipLoss
from sota_pipeline.models import SOTAChampionshipModel
from sota_pipeline.resumable_manager import ResumableCheckpointManager


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """Initializes distributed environment for single-machine or multi-GPU DDP."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_parameter_groups(model: nn.Module, backbone_lr: float, adapter_lr: float, weight_decay: float):
    """Separates pretrained visual backbone parameters from newly initialized heads."""
    raw_model = model.module if hasattr(model, "module") else model

    backbone_params = []
    head_params = []
    no_decay = []

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "gem_p" in name:
            no_decay.append(param)
        elif "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": adapter_lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": adapter_lr, "weight_decay": 0.0},
    ]
    return param_groups


def main():
    parser = argparse.ArgumentParser(description="Train SOTA Championship Cross-Modal Architecture")
    parser.add_argument("--dataset_root", type=str, default=None, help="Root path to UPAR dataset")
    parser.add_argument("--output_dir", type=str, default="sota_outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_sota")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--adapter_lr", type=float, default=3.5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_session_hours", type=float, default=10.75)
    parser.add_argument("--eval_every", type=int, default=1, help="Validation interval in epochs")
    parser.add_argument("--val_queries", type=int, default=500, help="Number of queries for validation")
    parser.add_argument("--log_interval", type=int, default=25, help="Step-wise log interval")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Batch size for validation inference")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers per GPU process")
    parser.add_argument("--run_baseline", action="store_true", help="Run initial zero-shot validation before training")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    is_master = (rank == 0)

    cfg = SOTAConfig()
    dataset_root = cfg.resolve_dataset_root(args.dataset_root)

    if is_master:
        print("=" * 80)
        print(f"[SOTA-CHAMPIONSHIP] Dual-GPU Distributed Multi-Modal Training Pipeline")
        print(f"[SOTA-CHAMPIONSHIP] Device: {device} | World Size (DDP): {world_size}")
        print(f"[SOTA-CHAMPIONSHIP] Batch Size: {args.batch_size}/GPU | Global: {args.batch_size * world_size} | Effective: {args.batch_size * world_size * args.accumulation_steps}")
        print(f"[SOTA-CHAMPIONSHIP] Epochs: {args.epochs} | Session Timeout Budget: {args.max_session_hours} hours")
        print(f"[SOTA-CHAMPIONSHIP] Checkpoint Directory: {os.path.abspath(args.checkpoint_dir)}")
        print("=" * 80)

    # 1. Datasets and Loaders
    train_dataset = SOTAUPARDataset(
        dataset_root=dataset_root,
        split="train",
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        use_srdm=True,
        preload_to_ram=True
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_dataset = SOTAUPARDataset(
        dataset_root=dataset_root,
        split="val",
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        use_srdm=False,
        preload_to_ram=True
    )

    # 2. Model & Distributed Wrapping
    model = SOTAChampionshipModel(
        embed_dim=cfg.embed_dim,
        num_attrs=cfg.num_attrs,
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        patch_size=cfg.patch_size,
        use_mlm=cfg.use_mlm_head
    )
    model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    # 3. Optimizer & Schedulers
    param_groups = build_parameter_groups(model, args.backbone_lr, args.adapter_lr, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    warmup_epochs = cfg.warmup_epochs
    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=args.min_lr
    )

    use_amp = (not args.no_amp) and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = GradScaler(enabled=use_amp)

    # 4. Multi-Task Criterion
    criterion = TotalChampionshipLoss(
        lambda_sdm=cfg.lambda_sdm,
        lambda_mlm=cfg.lambda_mlm,
        lambda_asl=cfg.lambda_asl,
        lambda_strip=cfg.lambda_strip,
        asl_gamma_neg=cfg.asl_gamma_neg,
        asl_clip=cfg.asl_clip_margin,
        epsilon_hamming=cfg.sdm_epsilon_hamming
    ).to(device)

    # 5. Resumable Checkpoint Manager
    resumable_mgr = ResumableCheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        max_hours=args.max_session_hours
    )

    start_epoch = 1
    global_step = 0
    best_val_madm = 0.0

    # Auto-resume from argument or previous checkpoint file if present
    resume_path = args.resume
    if resume_path is None:
        candidates = [
            os.path.join(args.checkpoint_dir, "last_checkpoint.pt"),
            os.path.join(args.checkpoint_dir, "best_model.pt"),
            os.path.join("/kaggle/working", "checkpoints_sota", "last_checkpoint.pt"),
            os.path.join("/kaggle/working", "checkpoints_sota", "best_model.pt"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                resume_path = c
                break

        # If not found locally, recursively search /kaggle/input for any previous session checkpoint
        if resume_path is None and os.path.isdir("/kaggle/input"):
            print("[AUTO-RESUME] Searching /kaggle/input for previous session checkpoints...")
            for root, _, files in os.walk("/kaggle/input"):
                if "last_checkpoint.pt" in files:
                    found = os.path.join(root, "last_checkpoint.pt")
                    if os.path.isfile(found):
                        resume_path = found
                        print(f"[AUTO-RESUME] Discovered previous session checkpoint at: {resume_path}")
                        break
            if resume_path is None:
                for root, _, files in os.walk("/kaggle/input"):
                    if "best_model.pt" in files:
                        found = os.path.join(root, "best_model.pt")
                        if os.path.isfile(found):
                            resume_path = found
                            print(f"[AUTO-RESUME] Discovered previous best model at: {resume_path}")
                            break

    if resume_path is not None and os.path.isfile(resume_path):
        start_epoch, global_step, best_val_madm = resumable_mgr.load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device
        )

    # Optional Epoch 0 Baseline Zero-Shot Validation
    if start_epoch == 1 and args.run_baseline and val_dataset is not None:
        if is_master:
            print("\n" + "=" * 80)
            print("[BASELINE] Running Initial Pretrained Zero-Shot Validation (Epoch 0 Baseline)...")
            print("=" * 80)
        val_model = model.module if hasattr(model, "module") else model
        base_results = evaluate_retrieval(
            model=val_model,
            val_dataset=val_dataset,
            batch_size=args.eval_batch_size,
            num_queries=min(args.val_queries, len(val_dataset)),
            device=device
        )
        if is_master:
            best_val_madm = base_results.get("mADM_RRF", 0.0)
            print(f"[BASELINE] Pretrained Zero-Shot Established: mADM = {best_val_madm:.4f}\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. Main Optimization Loop
    for epoch in range(start_epoch, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        t_epoch_start = time.time()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            attr_labels = batch["attr_labels"].to(device, non_blocking=True)
            text_tokens = batch["text_tokens"].to(device, non_blocking=True)
            masked_tokens = batch["masked_tokens"].to(device, non_blocking=True)
            masked_labels = batch["masked_labels"].to(device, non_blocking=True)

            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outputs = model(images, text_tokens, masked_tokens=masked_tokens, masked_labels=masked_labels)
                img_metric = outputs["img_metric"]
                txt_metric = outputs["txt_metric"]
                attr_logits = outputs["attr_logits"]
                strip_embeds = outputs["strip_embeds"]
                vocab_logits = outputs.get("vocab_logits")
                mlm_labels = outputs.get("masked_labels", masked_labels)

                loss, metrics = criterion(
                    img_metric=img_metric,
                    txt_metric=txt_metric,
                    attr_logits=attr_logits,
                    attr_labels=attr_labels,
                    vocab_logits=vocab_logits,
                    masked_labels=mlm_labels,
                )
                loss_scaled = loss / args.accumulation_steps

            scaler.scale(loss_scaled).backward()

            if batch_idx % args.accumulation_steps == 0 or batch_idx == num_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            global_step += 1

            if is_master and (batch_idx % args.log_interval == 0 or batch_idx == num_batches):
                elapsed = time.time() - t_epoch_start
                speed = (batch_idx * args.batch_size * world_size) / max(elapsed, 1e-4)

                # Real-Time Precise ETAs
                steps_done = batch_idx
                steps_rem = num_batches - steps_done
                sec_per_step = elapsed / max(steps_done, 1)
                ep_eta_sec = steps_rem * sec_per_step
                ep_eta_str = f"{int(ep_eta_sec // 60):02d}m {int(ep_eta_sec % 60):02d}s"

                total_steps_rem = (total_epochs - epoch) * num_batches + steps_rem
                total_eta_sec = total_steps_rem * sec_per_step
                total_eta_str = f"{int(total_eta_sec // 3600):02d}h {int((total_eta_sec % 3600) // 60):02d}m"

                # Dual GPU VRAM Telemetry (driver-level total device memory)
                vram_str = ""
                if torch.cuda.is_available():
                    vram_parts = []
                    for d in range(torch.cuda.device_count()):
                        try:
                            free_b, total_b = torch.cuda.mem_get_info(d)
                            used_gb = (total_b - free_b) / (1024**3)
                            vram_parts.append(f"GPU{d}: {used_gb:.1f}GB")
                        except Exception:
                            vram_parts.append(f"GPU{d}: {torch.cuda.memory_reserved(d) / (1024**3):.1f}GB")
                    vram_str = " | " + " | ".join(vram_parts)

                loss_val = loss.item()
                sdm_val = metrics.get("loss_sdm", 0.0)
                asl_val = metrics.get("loss_asl", 0.0)
                mlm_val = metrics.get("loss_mlm", 0.0)

                print(
                    f"[Ep {epoch:02d}/{total_epochs:02d}] | Step [{batch_idx:04d}/{num_batches:04d}] | "
                    f"Loss: {loss_val:.4f} (SDM: {sdm_val:.2f}, ASL: {asl_val:.2f}, MLM: {mlm_val:.2f}) | "
                    f"Ep ETA: {ep_eta_str} | Run ETA: {total_eta_str} | Speed: {speed:.1f} img/s{vram_str}"
                )

        if epoch > warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        if is_master:
            print(f"\n[TRAIN] Epoch {epoch:02d}/{total_epochs:02d} Completed. Mean Multi-Task Loss: {avg_loss:.4f}\n")

        # 7. Validation Evaluation After Every Epoch
        if val_dataset is not None and (epoch % args.eval_every == 0 or epoch == total_epochs):
            if is_master:
                print(f"[VAL] Running Full Distributed Validation Retrieval for Epoch {epoch:02d}...")
            val_model = model.module if hasattr(model, "module") else model
            val_results = evaluate_retrieval(
                model=val_model,
                val_dataset=val_dataset,
                batch_size=args.eval_batch_size,
                num_queries=min(args.val_queries, len(val_dataset)),
                device=device
            )
            curr_madm = val_results.get("mADM_RRF", 0.0) if is_master else 0.0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if is_master:
                is_best = (curr_madm > best_val_madm)
                if is_best:
                    best_val_madm = curr_madm
                    print(f"[CHECKPOINT] >>> New Historical Best Validation mADM: {best_val_madm:.4f}! <<<")

                # Strictly maintain ONLY best_model.pt and last_checkpoint.pt
                resumable_mgr.save_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_metric=best_val_madm,
                    is_best=is_best,
                    filename="last_checkpoint.pt"
                )

        # 8. Multi-Session Wall-Clock Timeout Check
        if resumable_mgr.should_exit_for_timeout():
            if is_master:
                print(f"[TIMEOUT] Reached session limit ({args.max_session_hours} hours).")
                print(f"[TIMEOUT] Preserved complete 7-component state bundle. Ready for next session!")
            break

    cleanup_distributed()
    if is_master:
        print(f"\n[COMPLETE] Training session concluded. Historical Best Validation mADM: {best_val_madm:.4f}\n")


if __name__ == "__main__":
    main()
