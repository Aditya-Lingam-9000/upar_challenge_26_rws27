"""Stage 2 Training Script: Cross-Modal Metric Learning with ArcFace.

Trains VisualProjector and QueryEncoder to align image features with query semantics
using ArcFace Additive Angular Margin loss and Symmetric InfoNCE Contrastive loss.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import Task2RetrievalEvalData, UPARDataset, resolve_data_root
from src.evaluate import get_device
from src.feature_extractor import extract_dataset_features_and_logits, load_model_for_extraction
from src.metric_learning import (
    ArcFaceLoss,
    CrossModalContrastiveLoss,
    QueryEncoder,
    VisualProjector,
    compute_hybrid_distances,
    evaluate_distance_matrix,
)


def load_task2_train_annotations(data_root: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads training identity IDs and canonical query vectors for all 97,669 training images."""
    task2_train_dir = os.path.join(data_root, "annotations", "task2", "train")
    queries_path = os.path.join(task2_train_dir, "queries.csv")
    ids_path = os.path.join(task2_train_dir, "ids.csv")

    # 1. Load canonical query vectors (7204, 40)
    canonical_queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for r in reader:
            if r:
                canonical_queries.append([float(x.strip()) for x in r])
    canonical_queries = np.array(canonical_queries, dtype=np.float32)

    # 2. Load per-image query ID index (maps each image in train.csv to 0..7203)
    train_csv_path = os.path.join(data_root, "annotations", "train.csv")
    img_to_query_id = {}
    with open(ids_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if len(r) >= 2:
                img_to_query_id[r[0].strip()] = int(r[1].strip())

    train_ids = []
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for r in reader:
            if r:
                p = r[0].strip()
                train_ids.append(img_to_query_id.get(p, 0))

    train_ids = np.array(train_ids, dtype=np.int64)
    # Per-sample query vectors: (97669, 40)
    sample_queries = canonical_queries[train_ids]

    return train_ids, sample_queries


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Metric Learning Training")
    parser.add_argument("--model", type=str, default="convnext_tiny", choices=["convnext_tiny", "swin_t", "dual_fused"])
    parser.add_argument("--submission_zip", type=str, default="submissions/EXP-004_submission.zip")
    parser.add_argument("--convnext_zip", type=str, default="submissions/EXP-004_submission.zip")
    parser.add_argument("--swin_zip", type=str, default="submissions/EXP-005A_submission.zip")
    parser.add_argument("--data_root", type=str, default="UPAR-Challenge-2027/data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=30.0)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--lambda_contrastive", type=float, default=0.5)
    parser.add_argument("--save_path", type=str, default="checkpoints/stage2_metric_heads.pth")
    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 2: CROSS-MODAL METRIC LEARNING & ARCFACE ALIGNMENT TRAINING")
    print("=" * 80)

    device = get_device()
    print(f"Compute Device: {device}")
    data_root = resolve_data_root(args.data_root)

    os.makedirs("checkpoints", exist_ok=True)
    train_csv = os.path.join(data_root, "annotations", "train.csv")
    val_csv = os.path.join(data_root, "annotations", "val.csv")

    calib_paths = ["assets/stage1_calibration_params.npz", "checkpoints/stage1_calibration_params.npz"]
    calib_file = None
    for cp in calib_paths:
        if os.path.exists(cp):
            calib_file = cp
            break

    if calib_file is not None:
        calib = np.load(calib_file)
        T_004, b_004 = calib["T_004"], calib["b_004"]
        T_005, b_005 = calib["T_005"], calib["b_005"]
        print(f"Loaded calibration parameters from: {calib_file}")
    else:
        print("[WARNING] stage1_calibration_params.npz not found! Using default calibration vectors.")
        T_004 = np.array([1.5416, 1.8003, 1.3536, 1.0762, 1.6780, 1.4883, 1.5458, 1.3283, 1.2592, 1.3323, 1.4851, 1.5165, 1.3414, 1.5971, 1.6384, 1.6033, 1.4398, 1.3406, 1.6017, 1.5242, 1.1394, 1.1895, 1.2291, 1.4429, 1.5724, 1.3303, 1.6394, 1.6521, 1.6309, 1.5173, 1.4552, 1.6083, 1.5034, 1.0315, 1.2646, 1.2332, 1.3571, 1.4721, 1.4649, 1.3195], dtype=np.float32)
        b_004 = np.array([2.8053, 0.9997, 1.3599, 1.4451, 1.2190, 1.1278, 1.6392, 1.1691, 1.2062, 1.2343, 1.4283, 1.4871, 1.2625, 1.5582, 1.5993, 1.5641, 1.3857, 1.2589, 1.5627, 1.4789, 1.1891, 1.1782, 1.2185, 1.3982, 1.5231, 1.2584, 1.6012, 1.6143, 1.5932, 1.4761, 1.4123, 1.5701, 1.4592, 0.9852, 1.1983, 1.1821, 1.3129, 1.4281, 1.4192, 1.2783], dtype=np.float32)
        T_005 = np.array([1.5287, 1.9120, 1.4530, 1.0267, 1.7007, 1.4921, 1.5512, 1.3312, 1.2643, 1.3411, 1.4912, 1.5231, 1.3482, 1.6021, 1.6432, 1.6082, 1.4451, 1.3452, 1.6071, 1.5291, 1.1451, 1.1952, 1.2341, 1.4481, 1.5781, 1.3352, 1.6441, 1.6571, 1.6352, 1.5221, 1.4601, 1.6131, 1.5081, 1.0362, 1.2691, 1.2381, 1.3621, 1.4771, 1.4691, 1.3241], dtype=np.float32)
        b_005 = np.array([2.7781, 0.9276, 1.4545, 1.4719, 1.1888, 1.1121, 1.6231, 1.1541, 1.1921, 1.2191, 1.4131, 1.4721, 1.2481, 1.5431, 1.5841, 1.5491, 1.3711, 1.2441, 1.5481, 1.4641, 1.1741, 1.1631, 1.2041, 1.3831, 1.5081, 1.2441, 1.5861, 1.5991, 1.5781, 1.4611, 1.3971, 1.5551, 1.4441, 0.9711, 1.1831, 1.1671, 1.2981, 1.4131, 1.4041, 1.2631], dtype=np.float32)

    if args.model == "dual_fused":
        print("\n[Mode] DUAL-STREAM FUSED BACKBONE (ConvNeXt-Tiny + Swin-T -> 1536-D)")
        train_feat_cache_c = "checkpoints/train_feats_convnext_tiny.npy"
        val_feat_cache_c = "checkpoints/val_feats_convnext_tiny.npy"
        val_logit_cache_c = "checkpoints/val_logits_convnext_tiny.npy"

        train_feat_cache_s = "checkpoints/train_feats_swin_t.npy"
        val_feat_cache_s = "checkpoints/val_feats_swin_t.npy"
        val_logit_cache_s = "checkpoints/val_logits_swin_t.npy"

        # 1. Load ConvNeXt features
        if not os.path.exists(train_feat_cache_c):
            raise FileNotFoundError(f"Missing {train_feat_cache_c}. Run extraction on ConvNeXt first.")
        train_feats_c = np.load(train_feat_cache_c)
        val_feats_c = np.load(val_feat_cache_c)
        val_logits_c = np.load(val_logit_cache_c)

        # 2. Load Swin-T features
        if not os.path.exists(train_feat_cache_s):
            raise FileNotFoundError(f"Missing {train_feat_cache_s}. Run extraction on Swin-T first.")
        train_feats_s = np.load(train_feat_cache_s)
        val_feats_s = np.load(val_feat_cache_s)
        val_logits_s = np.load(val_logit_cache_s)

        # 3. Concatenate into 1536-D representation
        train_feats = np.concatenate([train_feats_c, train_feats_s], axis=1)  # (97669, 1536)
        val_feats = np.concatenate([val_feats_c, val_feats_s], axis=1)        # (33407, 1536)
        in_features = 1536
        print(f"Fused Features: Train {train_feats.shape}, Val {val_feats.shape}")

        # Compute calibrated ensemble probabilities (50/50 blend)
        p_c = 1.0 / (1.0 + np.exp(-np.clip((val_logits_c - b_004[None, :]) / T_004[None, :], -20.0, 20.0)))
        p_s = 1.0 / (1.0 + np.exp(-np.clip((val_logits_s - b_005[None, :]) / T_005[None, :], -20.0, 20.0)))
        calib_val_probs = 0.50 * p_c + 0.50 * p_s

    else:
        train_feat_cache = f"checkpoints/train_feats_{args.model}.npy"
        val_feat_cache = f"checkpoints/val_feats_{args.model}.npy"
        val_logit_cache = f"checkpoints/val_logits_{args.model}.npy"
        in_features = 768

        if not os.path.exists(train_feat_cache):
            print(f"\n[1/4] Extracting training features using {args.model}...")
            model = load_model_for_extraction(args.submission_zip, args.model, device)
            train_ds = UPARDataset(csv_path=train_csv, data_root=data_root, height=256, width=128, is_train=False)
            train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=0)
            train_feats, _ = extract_dataset_features_and_logits(model, train_loader, device)
            np.save(train_feat_cache, train_feats)
            print(f"Saved {train_feats.shape} training features to {train_feat_cache}")
            del train_ds, train_loader
        else:
            print(f"\n[1/4] Loading cached training features from {train_feat_cache}...")
            train_feats = np.load(train_feat_cache)

        if not os.path.exists(val_feat_cache) or not os.path.exists(val_logit_cache):
            print(f"\n[2/4] Extracting validation features and logits using {args.model}...")
            model = load_model_for_extraction(args.submission_zip, args.model, device)
            val_ds = UPARDataset(csv_path=val_csv, data_root=data_root, height=256, width=128, is_train=False)
            val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
            val_feats, val_logits = extract_dataset_features_and_logits(model, val_loader, device)
            np.save(val_feat_cache, val_feats)
            np.save(val_logit_cache, val_logits)
            print(f"Saved {val_feats.shape} val features and {val_logits.shape} val logits.")
            del val_ds, val_loader
        else:
            print(f"\n[2/4] Loading cached validation features from {val_feat_cache}...")
            val_feats = np.load(val_feat_cache)
            val_logits = np.load(val_logit_cache)

        T_param = T_004 if "convnext" in args.model else T_005
        b_param = b_004 if "convnext" in args.model else b_005
        calib_val_probs = 1.0 / (1.0 + np.exp(-np.clip((val_logits - b_param[None, :]) / T_param[None, :], -20.0, 20.0)))

    # Load annotations
    print("\n[3/4] Loading Task 2 Identity and Query annotations...")
    train_ids, train_queries = load_task2_train_annotations(data_root)
    num_classes = len(np.unique(train_ids))
    print(f"Loaded {len(train_ids)} training samples across {num_classes} unique identity classes.")
    retrieval_data = Task2RetrievalEvalData(data_root=data_root)

    # 2. Build Models and Optimizers
    print("\n[4/4] Initializing Stage 2 Metric Learning Networks...")
    vis_projector = VisualProjector(in_features=in_features, embedding_dim=512, dropout=0.1).to(device)
    query_encoder = QueryEncoder(num_attributes=40, embedding_dim=512, dropout=0.1).to(device)
    arcface_loss_fn = ArcFaceLoss(
        num_classes=num_classes,
        embedding_dim=512,
        scale=args.scale,
        margin=args.margin,
    ).to(device)
    contrastive_loss_fn = CrossModalContrastiveLoss(temperature=0.07).to(device)

    train_tensor_feats = torch.from_numpy(train_feats).float()
    train_tensor_queries = torch.from_numpy(train_queries).float()
    train_tensor_ids = torch.from_numpy(train_ids).long()

    train_dataset = TensorDataset(train_tensor_feats, train_tensor_queries, train_tensor_ids)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        list(vis_projector.parameters()) + list(query_encoder.parameters()) + list(arcface_loss_fn.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"Beginning training for {args.epochs} epochs (Batch size: {args.batch_size}, Batches/epoch: {len(train_loader)})...")
    best_madm = 0.0

    for epoch in range(1, args.epochs + 1):
        vis_projector.train()
        query_encoder.train()
        arcface_loss_fn.train()

        total_loss = 0.0
        total_arc = 0.0
        total_cm = 0.0
        t0 = time.time()

        for feats, queries, labels in train_loader:
            feats = feats.to(device, non_blocking=True)
            queries = queries.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            v_emb = vis_projector(feats)
            q_emb = query_encoder(queries)

            loss_arc = arcface_loss_fn(v_emb, labels)
            loss_cm = contrastive_loss_fn(q_emb, v_emb)

            loss = loss_arc + args.lambda_contrastive * loss_cm
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            total_arc += loss_arc.item() * feats.size(0)
            total_cm += loss_cm.item() * feats.size(0)

        scheduler.step()
        elapsed = time.time() - t0
        n_samples = len(train_dataset)
        avg_loss = total_loss / n_samples
        avg_arc = total_arc / n_samples
        avg_cm = total_cm / n_samples

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {avg_loss:.4f} (Arc: {avg_arc:.4f}, CM: {avg_cm:.4f}) | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s",
            flush=True,
        )

        # Evaluate at intervals or final
        if epoch % 5 == 0 or epoch == args.epochs:
            vis_projector.eval()
            query_encoder.eval()

            with torch.no_grad():
                val_v_emb = vis_projector(torch.from_numpy(val_feats).float().to(device)).cpu().numpy()
                val_q_emb = query_encoder(torch.from_numpy(retrieval_data.queries).float().to(device)).cpu().numpy()

            for lam in [0.0, 0.15, 0.25, 0.35]:
                hybrid_dist = compute_hybrid_distances(
                    retrieval_data.queries,
                    calib_val_probs,
                    val_q_emb,
                    val_v_emb,
                    lambda_metric=lam,
                )
                metrics = evaluate_distance_matrix(hybrid_dist, retrieval_data.gallery_gt, retrieval_data.queries)
                print(
                    f"   [Eval λ={lam:.2f}] -> mADM: {metrics['mADM']:.4f} | mAP: {metrics['mAP']:.4f} | "
                    f"R-1: {metrics['Rank-1']*100:5.2f}% | R-5: {metrics['Rank-5']*100:5.2f}% | mINP: {metrics['mINP']:.4f}"
                )
                if metrics["mADM"] > best_madm:
                    best_madm = metrics["mADM"]
                    torch.save(
                        {
                            "vis_projector": vis_projector.state_dict(),
                            "query_encoder": query_encoder.state_dict(),
                            "epoch": epoch,
                            "lambda_metric": lam,
                            "metrics": metrics,
                        },
                        args.save_path,
                    )

    print(f"\nStage 2 Training Completed. Peak mADM: {best_madm:.4f}. Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
