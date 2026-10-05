#!/bin/bash
# ==============================================================================
# SOTA Championship Pipeline: Kaggle Dual Tesla T4 Launch Script
# ==============================================================================
# Executes distributed multi-GPU training with PyTorch DistributedDataParallel (DDP).
# Automatically detects previous session checkpoints for 100% fail-safe resumption
# across Kaggle's 12-hour session execution limits.
# ==============================================================================

set -e

export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO

DATASET_ROOT="/kaggle/input/datasets/hinewk/upar-challenge-2027-data/data"
if [ ! -d "$DATASET_ROOT" ]; then
    DATASET_ROOT="/kaggle/input/upar-challenge-2027-data/data"
fi

CHECKPOINT_DIR="/kaggle/working/checkpoints_sota"
mkdir -p "$CHECKPOINT_DIR"

PREVIOUS_CHECKPOINT="/kaggle/input/sota-championship-stage1/checkpoints_sota/last_checkpoint.pt"
RESUME_FLAG=""

if [ -f "$PREVIOUS_CHECKPOINT" ]; then
    echo "[LAUNCH] Detected previous session checkpoint at: $PREVIOUS_CHECKPOINT"
    echo "[LAUNCH] Activating seamless multi-session resumption..."
    RESUME_FLAG="--resume $PREVIOUS_CHECKPOINT"
elif [ -f "$CHECKPOINT_DIR/last_checkpoint.pt" ]; then
    echo "[LAUNCH] Detected local checkpoint at: $CHECKPOINT_DIR/last_checkpoint.pt"
    RESUME_FLAG="--resume $CHECKPOINT_DIR/last_checkpoint.pt"
else
    echo "[LAUNCH] No previous checkpoint detected. Initializing fresh SOTA championship run..."
fi

torchrun --nproc_per_node=2 \
         --master_port=29500 \
         sota_pipeline/train.py \
         --dataset_root "$DATASET_ROOT" \
         --checkpoint_dir "$CHECKPOINT_DIR" \
         --batch_size 64 \
         --epochs 70 \
         --backbone_lr 5e-5 \
         --adapter_lr 3.5e-4 \
         --min_lr 1e-6 \
         --max_session_hours 10.75 \
         $RESUME_FLAG
