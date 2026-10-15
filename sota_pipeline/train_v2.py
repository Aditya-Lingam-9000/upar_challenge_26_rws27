"""SOTA Championship V2: Two-Stage Decoupled ArcFace Distributed Training Pipeline.

Features:
1. Two-Stage Decoupled Training:
   - Stage 1: Representation Pre-training (ASL + Strip Multi-Task).
   - Stage 2: Frozen-Backbone Cross-Modal Additive Angular Margin Learning (ArcFace, s=30.0, m=0.35).
   - Optional warmstart from Stage 1 checkpoint to immediately begin Stage 2.
2. Dual-Query Stream Supervision:
   - Hard Query Adapter (40-D bits -> 512-D).
   - Soft Linguistic Query Adapter (Text Transformer -> 512-D).
   - Fused Unified Query Vector Eq = [Eh || Es] / sqrt(2) in R^1024.
3. Strict CodaBench Validation Engine:
   - Evaluates exact-match retrieval (l1_diff == 0) matching competition server.
4. Kaggle Dual-T4 High-Throughput Infrastructure:
   - RAM preloading of 97,669 images into host memory.
   - Accelerate DDP distributed execution with FP16 autocast.
   - Dynamic VRAM memory management with periodic cache flushing.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from sota_pipeline.config import SOTAConfig
from sota_pipeline.dataset import SOTAUPARDataset
from sota_pipeline.losses_v2 import SOTAChampionshipLossV2
from sota_pipeline.models_v2 import SOTAChampionshipModelV2, VisualStreamV2
from sota_pipeline.prompt_engine import DynamicPromptEngine
from sota_pipeline.resumable_manager import ResumableCheckpointManager


def clear_gpu_vram():
    """Flushes PyTorch tensor caches and forces garbage collection across all GPUs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def evaluate_coda_bench_exact(
    model: SOTAChampionshipModelV2,
    val_loader: DataLoader,
    val_queries_raw: np.ndarray,
    val_gallery_gt: np.ndarray,
    accelerator: Accelerator,
    prompt_engine: DynamicPromptEngine,
) -> Dict[str, float]:
    """
    Evaluates exact CodaBench Track 2 metrics (l1_diff == 0) across all validation samples.
    """
    model.eval()
    clear_gpu_vram()

    gal_embeds_list = []
    gal_logits_list = []

    with torch.inference_mode():
        for batch in val_loader:
            images = batch["image"]
            E_p, attr_logits = model.forward_visual(images)
            gal_embeds_list.append(accelerator.gather_for_metrics(E_p).cpu())
            gal_logits_list.append(accelerator.gather_for_metrics(attr_logits).cpu())

    if not accelerator.is_main_process:
        clear_gpu_vram()
        return {}

    gal_embeds = torch.cat(gal_embeds_list, dim=0).float().numpy()[:len(val_gallery_gt)]
    gal_logits = torch.cat(gal_logits_list, dim=0).float().numpy()[:len(val_gallery_gt)]
    gal_probs = 1.0 / (1.0 + np.exp(-gal_logits))

    del gal_embeds_list, gal_logits_list
    clear_gpu_vram()

    Q = len(val_queries_raw)
    G = len(gal_embeds)

    # 1. Generate Unified Query Embeddings Eq = [Eh || Es] / sqrt(2)
    device = accelerator.device
    q_hard_t = torch.from_numpy(val_queries_raw).float().to(device)

    prompts = [prompt_engine.synthesize_prompt(val_queries_raw[i], is_train=False) for i in range(Q)]
    tokens = prompt_engine.tokenize(prompts).to(device)

    with torch.inference_mode():
        E_q_t, _, _ = model.forward_query(q_hard_t, tokens)
        q_embeds = E_q_t.cpu().float().numpy()

    # Normalize embeddings
    q_embeds = q_embeds / (np.linalg.norm(q_embeds, axis=1, keepdims=True) + 1e-8)
    gal_embeds = gal_embeds / (np.linalg.norm(gal_embeds, axis=1, keepdims=True) + 1e-8)

    # 2. Pairwise Continuous Hyperspherical Cosine Distance
    cos_sim = np.dot(q_embeds, gal_embeds.T)  # [Q, G]
    distances = 1.0 - cos_sim

    # 3. Exact CodaBench Metrics Computation
    queries = np.asarray(val_queries_raw, dtype=np.float32)
    gallery_gt = np.asarray(val_gallery_gt, dtype=np.float32)

    l1_diff = gallery_gt.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * gallery_gt).T
    l1_diff = np.maximum(0.0, l1_diff)
    dom = 1.0 - (l1_diff / 40.0)

    # STRICT ZERO-ATTRIBUTE TOLERANCE (Exact match condition on CodaBench)
    exact_matches = (l1_diff == 0)

    mean_dom = dom.mean(axis=1, keepdims=True)
    max_dom = dom.max(axis=1, keepdims=True)
    denom = np.maximum(max_dom - mean_dom, 1e-8)
    ndom = np.maximum(0.0, (dom - mean_dom) / denom)

    ranking = np.argsort(distances, axis=1)

    adm_list, ap_list, inp_list = [], [], []
    r1_count, r5_count, r10_count = 0, 0, 0
    valid_q = 0

    for i in range(Q):
        q_order = ranking[i]
        q_matches = exact_matches[i, q_order]
        q_ndom = ndom[i, q_order]

        ranks = np.where(q_matches)[0] + 1
        R = len(ranks)
        if R == 0:
            continue

        valid_q += 1
        min_r = ranks[0]
        if min_r <= 1:
            r1_count += 1
        if min_r <= 5:
            r5_count += 1
        if min_r <= 10:
            r10_count += 1

        ap_list.append(float(np.mean(np.arange(1, R + 1, dtype=np.float64) / ranks)))
        inp_list.append(float(R / ranks[-1]))

        cum_ndom = np.cumsum(q_ndom, dtype=np.float64)
        adm_list.append(float(np.mean(cum_ndom[ranks - 1] / ranks)))

    denom_val = max(valid_q, 1)
    metrics = {
        "mADM": float(np.mean(adm_list)) if adm_list else 0.0,
        "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
        "Rank-1": float(r1_count / denom_val) * 100.0,
        "Rank-5": float(r5_count / denom_val) * 100.0,
        "Rank-10": float(r10_count / denom_val) * 100.0,
        "mINP": float(np.mean(inp_list)) if inp_list else 0.0,
    }

    clear_gpu_vram()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SOTA Championship V2 Two-Stage Retraining Pipeline")
    parser.add_argument("--epochs_stage1", type=int, default=10, help="Stage 1 representation fine-tuning epochs")
    parser.add_argument("--epochs_stage2", type=int, default=15, help="Stage 2 ArcFace margin learning epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size per GPU")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--backbone_lr", type=float, default=5e-5, help="Learning rate for visual backbone (Stage 1)")
    parser.add_argument("--adapter_lr", type=float, default=3.5e-4, help="Learning rate for retrieval adapters")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum cosine annealing learning rate")
    parser.add_argument("--arcface_scale", type=float, default=30.0, help="ArcFace scale parameter s")
    parser.add_argument("--arcface_margin", type=float, default=0.35, help="ArcFace angular margin parameter m")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to upar_binary_dataset.h5")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_sota_v2", help="Checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="sota_v2_outputs", help="Output summary directory")
    parser.add_argument("--warmstart_checkpoint", type=str, default=None, help="Optional checkpoint to skip Stage 1")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--val_queries", type=int, default=500, help="Number of validation queries")
    parser.add_argument("--eval_batch_size", type=int, default=384, help="Validation batch size")
    args = parser.parse_args()

    # 1. Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        mixed_precision="fp16"
    )
    set_seed(42 + accelerator.process_index)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        print("=" * 80)
        print("[SOTA-V2] Cross-Modal Additive Angular Margin Championship Pipeline")
        print(f"[SOTA-V2] Device: {accelerator.device} | World Size: {accelerator.num_processes}")
        print(f"[SOTA-V2] Batch Size per GPU: {args.batch_size} | Effective Batch Size: {args.batch_size * accelerator.num_processes * args.accumulation_steps}")
        print(f"[SOTA-V2] Stage 1 Epochs: {args.epochs_stage1} | Stage 2 Epochs: {args.epochs_stage2}")
        print(f"[SOTA-V2] ArcFace Scale: {args.arcface_scale} | Angular Margin: {args.arcface_margin}")
        print("=" * 80, flush=True)

    # 2. Datasets & DataLoaders
    prompt_engine = DynamicPromptEngine(p_dropout=0.15)
    train_dataset = SOTAUPARDataset(
        dataset_root=args.dataset_root,
        split="train",
        img_height=336,
        img_width=168,
        use_srdm=True,
        preload_to_ram=True,
        prompt_engine=prompt_engine
    )

    val_dataset = SOTAUPARDataset(
        dataset_root=args.dataset_root,
        split="val",
        img_height=336,
        img_width=168,
        use_srdm=False,
        preload_to_ram=True,
        prompt_engine=prompt_engine
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 3. Model & Loss Engine
    visual_stream = VisualStreamV2(
        embed_dim=1024,
        num_attrs=40,
        img_height=336,
        img_width=168,
        backbone_type="auto"
    )

    # Pretrained authentic CLIP text model
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("EVA02-B-16", pretrained="merged2b_s8b_b131k")
    text_model = clip_model.text

    model = SOTAChampionshipModelV2(
        visual_stream=visual_stream,
        text_stream=text_model,
        embed_dim=1024,
        query_dim=512,
        num_attrs=40
    )

    criterion = SOTAChampionshipLossV2(
        scale=args.arcface_scale,
        margin=args.arcface_margin,
        asl_weight=1.0,
        arcface_weight=2.0
    )

    # Handle Warmstart if specified
    start_stage = 1
    if args.warmstart_checkpoint and os.path.isfile(args.warmstart_checkpoint):
        if accelerator.is_main_process:
            print(f"[WARMSTART] Loading Stage 1 weights from {args.warmstart_checkpoint}...")
        ckpt = torch.load(args.warmstart_checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Load visual stream weights
        vis_sd = {}
        for k, v in state_dict.items():
            if not k.startswith("visual_stream."):
                continue
            new_k = k.replace("visual_stream.", "")
            if new_k.startswith("backbone.trunk."):
                new_k = new_k.replace("backbone.trunk.", "backbone.")
            vis_sd[new_k] = v

        if not vis_sd:
            for k, v in state_dict.items():
                if not k.startswith("text_") and not k.startswith("hard_") and not k.startswith("soft_"):
                    new_k = k
                    if new_k.startswith("backbone.trunk."):
                        new_k = new_k.replace("backbone.trunk.", "backbone.")
                    vis_sd[new_k] = v

        load_res = model.visual_stream.load_state_dict(vis_sd, strict=False)
        start_stage = 2
        if accelerator.is_main_process:
            print(f"[WARMSTART] Loaded {len(vis_sd)} visual tensors! Missing: {len(load_res.missing_keys)}, Unexpected: {len(load_res.unexpected_keys)}")
            print("[WARMSTART] Visual backbone weights successfully loaded! Skipping directly to Stage 2 ArcFace...")

    # Stage 2 Setup: Freeze backbone
    if start_stage == 2:
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.adapter_lr,
            weight_decay=1e-4
        )
        total_epochs = args.epochs_stage2
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.visual_stream.backbone.parameters(), "lr": args.backbone_lr},
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": args.adapter_lr}
        ], weight_decay=1e-4)
        total_epochs = args.epochs_stage1 + args.epochs_stage2

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs * len(train_loader),
        eta_min=args.min_lr
    )

    # Accelerate prepare
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # Validation Query & Ground Truth preparation
    if hasattr(val_dataset, "get_all_labels"):
        val_gallery_gt = val_dataset.get_all_labels()
    else:
        val_gallery_gt = np.asarray(getattr(val_dataset, "labels", getattr(val_dataset, "attr_labels")), dtype=np.float32)

    if hasattr(val_dataset, "get_query_subset"):
        val_queries_raw = val_dataset.get_query_subset(num_queries=args.val_queries)
    else:
        Q = min(args.val_queries, len(val_gallery_gt))
        indices = np.linspace(0, len(val_gallery_gt) - 1, Q, dtype=int)
        val_queries_raw = val_gallery_gt[indices]

    best_madm = 0.0

    # 4. Training Loop
    for epoch in range(1, total_epochs + 1):
        current_stage = 2 if (start_stage == 2 or epoch > args.epochs_stage1) else 1

        if current_stage == 2 and epoch == args.epochs_stage1 + 1 and start_stage == 1:
            # Transition to Stage 2: Freeze backbone
            if accelerator.is_main_process:
                print("\n" + "=" * 80)
                print("[STAGE-TRANSITION] Transitioning from Stage 1 to Stage 2 (ArcFace Margin Learning)!")
                print("[STAGE-TRANSITION] Freezing visual backbone; optimizing retrieval adapters only.")
                print("=" * 80 + "\n", flush=True)
            accelerator.unwrap_model(model).freeze_backbone()
            clear_gpu_vram()

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"]
            attr_targets = (batch["attr_labels"] if "attr_labels" in batch else batch["attributes"]).float()

            # Dynamic linguistic prompt synthesis
            if "text_tokens" in batch and batch["text_tokens"] is not None:
                text_tokens = batch["text_tokens"].to(accelerator.device)
            else:
                prompts = [prompt_engine.synthesize_prompt(attr_targets[i], is_train=True) for i in range(len(images))]
                text_tokens = prompt_engine.tokenize(prompts).to(accelerator.device)

            with accelerator.accumulate(model):
                outputs = model(
                    images=images,
                    hard_queries=attr_targets,
                    text_tokens=text_tokens
                )

                losses = criterion(
                    model_outputs=outputs,
                    attr_targets=attr_targets,
                    stage=current_stage
                )
                loss = losses["loss_total"]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

            if (step + 1) % 50 == 0 and accelerator.is_main_process:
                print(f"[TRAIN] Epoch [{epoch:2d}/{total_epochs:2d}] (Stage {current_stage}) | "
                      f"Step [{step+1:4d}/{len(train_loader):4d}] | "
                      f"Loss: {loss.item():.4f} (ArcFace: {losses.get('loss_arcface_total', 0.0):.4f}, ASL: {losses.get('loss_asl', 0.0):.4f}) | "
                      f"LR: {lr_scheduler.get_last_lr()[0]:.2e}", flush=True)

        mean_epoch_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0

        if accelerator.is_main_process:
            print(f"[EPOCH {epoch:2d}] Completed in {elapsed:.1f}s | Mean Loss: {mean_epoch_loss:.4f}", flush=True)

        # 5. Distributed Validation Pass
        if epoch % args.eval_every == 0:
            metrics = evaluate_coda_bench_exact(
                model=accelerator.unwrap_model(model),
                val_loader=val_loader,
                val_queries_raw=val_queries_raw,
                val_gallery_gt=val_gallery_gt,
                accelerator=accelerator,
                prompt_engine=prompt_engine
            )

            if accelerator.is_main_process:
                madm = metrics.get("mADM", 0.0)
                r1 = metrics.get("Rank-1", 0.0)
                map_val = metrics.get("mAP", 0.0)
                print(f"\n[EVALUATION] Epoch {epoch:2d} (Stage {current_stage}) Results:")
                print(f"  * mADM:   {madm:.4f}  (Best: {best_madm:.4f} | Live #1: 0.6783)")
                print(f"  * Rank-1: {r1:.2f}% (Live #1: 40.79%)")
                print(f"  * mAP:    {map_val:.4f}  (Live #1: 0.3952)")
                print(f"  * Rank-5: {metrics.get('Rank-5', 0.0):.2f}% | Rank-10: {metrics.get('Rank-10', 0.0):.2f}% | mINP: {metrics.get('mINP', 0.0):.4f}\n", flush=True)

                # Save Best Model Checkpoint
                if madm > best_madm:
                    best_madm = madm
                    save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                    torch.save({
                        "epoch": epoch,
                        "stage": current_stage,
                        "metrics": metrics,
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    }, save_path)
                    print(f"[CHECKPOINT] *** New Best mADM ({best_madm:.4f})! Saved to {save_path} ***", flush=True)

                # Save Last Resumable Checkpoint
                last_path = os.path.join(args.checkpoint_dir, "last_checkpoint.pt")
                torch.save({
                    "epoch": epoch,
                    "stage": current_stage,
                    "metrics": metrics,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }, last_path)

        clear_gpu_vram()

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print(f"[COMPLETE] SOTA V2 Training Complete! Best Validation mADM: {best_madm:.4f}")
        print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
