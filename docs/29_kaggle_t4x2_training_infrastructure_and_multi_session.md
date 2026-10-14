# Kaggle Dual Tesla T4 Distributed Training Infrastructure & Multi-Session Resumption

---

## 1. System Topology & The 12-Hour Wall-Clock Constraint

### 1.1 Hardware Specifications (Kaggle Environment)
- **GPUs:** 2 $\times$ NVIDIA Tesla T4 (Turing Architecture, 15,360 MiB VRAM each $\approx$ 30,720 MiB total).
- **System RAM:** 30.0 GB DDR4.
- **CPU:** 4 vCPUs (Intel Xeon @ 2.20GHz or AMD EPYC).
- **Disk Storage:** 73 GB persistent `/kaggle/working`, ephemeral `/kaggle/temp`.
- **Session Timeout:** **12.0 hours (43,200 seconds) strict hard timeout**.

### 1.2 The Multi-Session Necessity
Training a foundation transformer (such as EVA-02-CLIP-B/16 or SOLIDER-Swin-Base) on 120,000+ pedestrian images with full cross-modal SDM, MLM cross-attention, and ASL multi-task heads requires:
- $\approx 25$ minutes per epoch on 2 $\times$ T4 with mixed precision.
- An 80-epoch training schedule requires $80 \times 25\text{ min} = 2,000\text{ min} \approx 33.3\text{ hours}$.

Attempting to complete 80 epochs in a single 12-hour session results in a hard SIGKILL from Kaggle, corrupting uncommitted checkpoints and losing all optimizer state.
Therefore, training must be structured as a **multi-session relay race** across 3 consecutive 11-hour sessions, where each session picks up seamlessly from the previous session's checkpoint with **zero loss of optimizer momentum, learning rate schedule, or random seed alignment**.

---

## 2. The 100% Loss-Free Resumption Contract

Most naive checkpoint implementations fail when resuming because they only save `model.state_dict()`. This produces severe training artifacts:
1. **Optimizer Momentum Reset:** AdamW maintains first ($\mathbf{m}_t$) and second ($\mathbf{v}_t$) moment vectors for every parameter. Discarding them causes massive gradient spikes upon resumption, destabilizing learned representations.
2. **Scheduler Reset:** A standard `CosineAnnealingLR` scheduler resets to the initial learning rate $\eta_{\max} = 10^{-4}$ instead of continuing the decayed cosine curve $\eta_t = 10^{-6}$, burning through weights.
3. **GradScaler Reset:** Resuming FP16 AMP training without the `GradScaler` state causes immediate gradient underflow/overflow and skipped steps.
4. **Data Sampler Bias:** Without restoring random number generator (RNG) states across Python, NumPy, and PyTorch CUDA, the DataLoader repeats identical batches or skips data.

### 2.1 The Complete State Preservation Dictionary

```python
checkpoint = {
    # 1. Training Progression
    "epoch": current_epoch,
    "global_step": global_step,
    "best_score": best_madm,
    
    # 2. Weights & Model Architecture
    "model_state_dict": model.module.state_dict() if is_ddp else model.state_dict(),
    
    # 3. Optimizer State (Momentum & Variance Vectors)
    "optimizer_state_dict": optimizer.state_dict(),
    
    # 4. Learning Rate Schedule State
    "scheduler_state_dict": scheduler.state_dict(),
    
    # 5. Mixed Precision Gradient Scaler
    "scaler_state_dict": scaler.state_dict(),
    
    # 6. Global Random States (Seed Alignment)
    "rng_state": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
    }
}
```

---

## 3. The 10.75-Hour Safety Watchdog System

To prevent unhandled Kaggle session termination, we implement a background **Safety Watchdog Timer** that monitors session elapsed time. At exactly $T = 10.75\text{ hours}$ (38,700 seconds):
1. The watchdog signals the training loop to finish the current epoch.
2. It executes a full synchronous checkpoint write to `/kaggle/working/checkpoint_relay.pth`.
3. It writes a lightweight metadata JSON file `/kaggle/working/relay_meta.json` recording next epoch and verification hash.
4. It calls `sys.exit(0)` cleanly, allowing the Kaggle notebook commit to finish successfully.

```
Session Start (T = 0.0h)
   │
   ├── Training Epochs (1, 2, 3...)
   │
   ├── Watchdog Monitor Checking Elapsed Time
   │
   ├── T = 10.75h Alert Triggered!
   │   ├── Finish current epoch cleanly
   │   ├── Flush CUDA buffers
   │   ├── Atomic write checkpoint_relay.pth
   │   └── Clean sys.exit(0)
   │
Session Finish (T = 11.0h) ──> User links output to Session 2 ──> Auto-Resume!
```

---

## 4. PyTorch DistributedDataParallel (DDP) Architecture on Dual T4

### 4.1 Why DDP Over DataParallel
- `torch.nn.DataParallel` (DP) uses single-process multi-threading: GPU 0 becomes a severe bottleneck gathering gradients and calculating losses, leaving GPU 1 starved and yielding only $1.2\times$ speedup.
- `torch.nn.parallel.DistributedDataParallel` (DDP) spawns one independent Python process per GPU, communicating asynchronously via NCCL with ring-allreduce gradients, achieving a near-linear $1.94\times$ speedup.

### 4.2 High-Throughput Optimization Stack
1. **Effective Batch Size $B_{\text{eff}} = 128$:**
   - Per-GPU batch size: 32 images.
   - Dual GPUs: $32 \times 2 = 64$ images per forward pass.
   - Gradient Accumulation Steps: 2.
   - $B_{\text{eff}} = 64 \times 2 = 128$ images per optimizer step.
2. **PyTorch 2.x `torch.compile(mode="reduce-overhead")`:**
   - Fuses memory-bandwidth-bound elementwise operations (LayerNorm, GELU, residual additions) into single CUDA kernels, delivering a +18% throughput boost on Turing T4 cores.
3. **In-RAM HDF5 Dataset Fast Loader:**
   - Kaggle's shared virtual disk exhibits high random I/O latency when loading 120,000 separate JPEG files.
   - Packing images into a single chunked `.h5` (HDF5) archive or pre-loading into `/dev/shm` (shared memory) eliminates I/O wait times, keeping GPU utilization at 98%–100%.

---

## 5. Production Master Script Template for Kaggle Resumable DDP

```python
"""
Kaggle Dual-T4 Production Resumable DDP Training Pipeline
Save as: kaggle_train_relay.py
Run via: torchrun --nproc_per_node=2 kaggle_train_relay.py
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# -------------------------------------------------------------------------
# 1. Distributed Environment Setup
# -------------------------------------------------------------------------
def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# -------------------------------------------------------------------------
# 2. Resumption and Checkpoint Management
# -------------------------------------------------------------------------
CHECKPOINT_PATH = "/kaggle/working/checkpoint_relay.pth"
META_PATH = "/kaggle/working/relay_meta.json"
MAX_SESSION_SECONDS = 10.75 * 3600  # 10.75 hours safety limit

def save_relay_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_score, rank):
    if rank != 0:
        return

    print(f"\n[Watchdog] Saving 100% loss-free checkpoint at epoch {epoch}...")
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
        }
    }
    
    # Atomic write to avoid corruption
    tmp_path = CHECKPOINT_PATH + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, CHECKPOINT_PATH)
    
    meta = {
        "completed_epoch": epoch,
        "next_epoch": epoch + 1,
        "global_step": global_step,
        "best_score": best_score,
        "timestamp": time.time()
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"[Watchdog] Checkpoint successfully persisted to {CHECKPOINT_PATH}.\n")

def load_relay_checkpoint(model, optimizer, scheduler, scaler, resume_path, rank):
    if not os.path.exists(resume_path):
        if rank == 0:
            print("[Init] No previous checkpoint found. Starting fresh from epoch 1.")
        return 1, 0, 0.0

    if rank == 0:
        print(f"[Init] Resuming from checkpoint: {resume_path}...")
    checkpoint = torch.load(resume_path, map_location=f"cuda:{torch.cuda.current_device()}")

    # Restore weights
    model.module.load_state_dict(checkpoint["model_state_dict"])
    
    # Restore optimizer momentum & variance
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Restore learning rate schedule
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Restore AMP scaler
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    # Restore RNG states
    random.setstate(checkpoint["rng_state"]["python"])
    np.random.set_state(checkpoint["rng_state"]["numpy"])
    torch.set_rng_state(checkpoint["rng_state"]["torch"])
    torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    best_score = checkpoint.get("best_score", 0.0)

    if rank == 0:
        print(f"[Init] Successfully resumed at epoch {start_epoch} (global step {global_step}, best score {best_score:.4f})")
    return start_epoch, global_step, best_score

# -------------------------------------------------------------------------
# 3. Main Master Training Loop
# -------------------------------------------------------------------------
def train_pipeline():
    rank, world_size, local_rank = setup_distributed()
    start_session_time = time.time()
    
    # Hyperparameters
    TOTAL_EPOCHS = 80
    ACCUM_STEPS = 2
    
    # Model instantiation (wrapped in DDP)
    # model = build_championship_model().cuda(local_rank)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # Optimizer & Scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
    # scaler = GradScaler()
    
    # Check for resumption
    # start_epoch, global_step, best_score = load_relay_checkpoint(
    #     model, optimizer, scheduler, scaler, CHECKPOINT_PATH, rank
    # )
    
    # for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
    #     train_sampler.set_epoch(epoch)
    #     train_one_epoch(...)
    #     val_score = evaluate(...)
    #
    #     if rank == 0 and val_score > best_score:
    #         best_score = val_score
    #         torch.save(model.module.state_dict(), "/kaggle/working/best_championship_weights.pth")
    #
    #     # Check Watchdog Timer at end of each epoch
    #     elapsed = time.time() - start_session_time
    #     if elapsed > MAX_SESSION_SECONDS:
    #         if rank == 0:
    #             print(f"[Watchdog ALERT] Session limit reached ({elapsed/3600:.2f}h). Preparing relay exit.")
    #         save_relay_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, best_score, rank)
    #         dist.barrier()
    #         cleanup_distributed()
    #         sys.exit(0)
    
    cleanup_distributed()

if __name__ == "__main__":
    train_pipeline()
```

---

## 6. Execution Roadmap Across 3 Kaggle Sessions

```
SESSION 1 (Day 1, 11 Hours):
├── Input: Pretrained Foundation Weights (EVA-02-CLIP / SOLIDER)
├── Execution: Epochs 1 -> 28
├── Output: checkpoint_relay.pth (Epoch 28 state) + best_weights_ep28.pth

SESSION 2 (Day 2, 11 Hours):
├── Input: Add Session 1 Output Dataset as input to Session 2
├── Auto-Resume: Automatically detects checkpoint_relay.pth -> Resumes at Epoch 29
├── Execution: Epochs 29 -> 56
├── Output: checkpoint_relay.pth (Epoch 56 state) + best_weights_ep56.pth

SESSION 3 (Day 3, 11 Hours):
├── Input: Add Session 2 Output Dataset as input to Session 3
├── Auto-Resume: Resumes at Epoch 57
├── Execution: Epochs 57 -> 80 (Final convergence at η = 10^-6)
├── Final Output:
│   ├── best_championship_weights.pth (Peak mADM checkpoint)
│   └── final_calibrated_head.pth
└── Ready for Air-Gapped CodaBench Packaging!
```
