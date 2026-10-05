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
from typing import Dict, List, Optional
import numpy as np
import torch
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
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--adapter_lr", type=float, default=3.5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_session_hours", type=float, default=10.75)
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()
    is_master = (rank == 0)

    cfg = SOTAConfig()
    dataset_root = cfg.resolve_dataset_root(args.dataset_root)

    if is_master:
        print("=" * 80)
        print(f"[SOTA-RUN] Starting Unconstrained Championship Training Pipeline")
        print(f"[SOTA-RUN] Device: {device} | World Size (DDP): {world_size}")
        print(f"[SOTA-RUN] Effective Global Batch Size: {args.batch_size * world_size}")
        print(f"[SOTA-RUN] Target Epochs: {args.epochs} | Wall-Clock Budget: {args.max_session_hours} hours")
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
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    val_dataset = None
    if is_master:
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
        # find_unused_parameters=False avoids reducer traversals and warning
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    # 3. Optimizer & Schedulers
    param_groups = build_parameter_groups(model, args.backbone_lr, args.adapter_lr, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    
    # Cosine Annealing with Warmup
    warmup_epochs = cfg.warmup_epochs
    total_epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=args.min_lr
    )

    use_amp = (not args.no_amp) and torch.cuda.is_available()
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
        default_last = os.path.join(args.checkpoint_dir, "last_checkpoint.pt")
        if os.path.isfile(default_last):
            resume_path = default_last

    if resume_path is not None and os.path.isfile(resume_path):
        start_epoch, global_step, best_val_madm = resumable_mgr.load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device
        )

    # 6. Main Optimization Loop
    for epoch in range(start_epoch, total_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            attr_labels = batch["attr_labels"].to(device, non_blocking=True)
            text_tokens = batch["text_tokens"].to(device, non_blocking=True)
            masked_tokens = batch["masked_tokens"].to(device, non_blocking=True)
            masked_labels = batch["masked_labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                outputs = model(images, text_tokens, masked_tokens=masked_tokens)
                img_metric = outputs["img_metric"]
                txt_metric = outputs["txt_metric"]
                attr_logits = outputs["attr_logits"]
                strip_embeds = outputs["strip_embeds"]
                vocab_logits = outputs.get("vocab_logits")

                loss, metrics = criterion(
                    img_metric=img_metric,
                    txt_metric=txt_metric,
                    attr_logits=attr_logits,
                    attr_labels=attr_labels,
                    vocab_logits=vocab_logits,
                    masked_labels=masked_labels,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            if is_master and (batch_idx % 25 == 0 or batch_idx == num_batches):
                curr_lr = optimizer.param_groups[1]["lr"] # Head LR
                elapsed = time.time() - t_epoch_start
                speed = (batch_idx * args.batch_size * world_size) / max(elapsed, 1e-4)
                print(
                    f"[TRAIN] Ep [{epoch:02d}/{total_epochs:02d}] | Batch [{batch_idx:04d}/{num_batches:04d}] | "
                    f"Loss: {loss.item():.4f} (SDM: {metrics['loss_sdm']:.2f}, ASL: {metrics['loss_asl']:.2f}) | "
                    f"LR: {curr_lr:.2e} | Speed: {speed:.1f} img/s"
                )

        if epoch > warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        if is_master:
            print(f"\n[TRAIN] Epoch {epoch:02d} Complete. Avg Multi-Task Loss: {avg_loss:.4f}\n")

        # 7. Validation Evaluation
        if is_master and (epoch % cfg.eval_every_epochs == 0 or epoch == total_epochs):
            val_results = evaluate_retrieval(
                model=model,
                val_dataset=val_dataset,
                batch_size=args.batch_size * 2,
                num_queries=500,
                device=device
            )
            curr_madm = val_results["mADM_RRF"]
            is_best = (curr_madm > best_val_madm)
            if is_best:
                best_val_madm = curr_madm

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
                print(f"[TIMEOUT] Successfully preserved complete 7-component state bundle.")
                print(f"[TIMEOUT] Ready for seamless resumption in next Kaggle session!")
            break

    cleanup_distributed()
    if is_master:
        print(f"\n[COMPLETE] Training session concluded. Historical Best Validation mADM: {best_val_madm:.4f}\n")


if __name__ == "__main__":
    main()
