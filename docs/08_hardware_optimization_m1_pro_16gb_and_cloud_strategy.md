# Report 08: Hardware Optimization: Apple M1 Pro (16GB) Workflow & Cloud Training Strategy

**Document:** Compute Architecture & Workflow Optimization  
**Local Hardware:** Apple M1 Pro (10-core CPU, 16-core GPU, 16GB Unified Memory)  
**Local Accelerator:** Apple Metal Performance Shaders (MPS) via `torch.device("mps")`  
**Cloud Accelerators:** NVIDIA RTX 4090, A100, T4 (Colab / Kaggle / RunPod)  
**Document Status:** Publication-Grade Hardware & Pipeline Guide  

---

## 1. Hardware Analysis: The Apple M1 Pro Profile

The local workstation is an Apple Silicon M1 Pro configured with **16GB Unified Memory Architecture (UMA)**.

### 1.1 Advantages of Apple Silicon for Development
- **Unified Memory Architecture (UMA):** CPU and GPU share the same physical memory pool. Zero CPU-to-GPU PCIe transfer overhead when preparing batches or evaluating metrics.
- **Metal Performance Shaders (MPS):** Native acceleration for convolution, matrix multiplication, and transformer attention operations in PyTorch ($\ge 2.0$).
- **Low Power & Silent Execution:** Ideal for continuous local code editing, unit testing, data preprocessing, and exploratory data analysis.

### 1.2 Limitations of 16GB RAM for Full-Scale Transformer Training
- **The Memory Ceiling:** The 16GB RAM is shared between macOS system services ($\approx 3.5\text{ GB}$), open applications/IDE ($\approx 2.0\text{ GB}$), and PyTorch tensors. PyTorch has a practical ceiling of approximately **9.5 to 10.5 GB** before triggering OS memory compression and disk swapping.
- **Swap Thrashing Danger:** If a training script allocates $>11\text{ GB}$, macOS begins swapping memory pages to the internal NVMe SSD. PyTorch execution throughput collapses by up to $10\times - 30\times$.
- **Training Time on 97,669 Images:**
  - Training a dual-branch ViT-Base + Swin-Base backbone across 97,669 images for 40 epochs on M1 Pro MPS requires approximately **18 to 24 hours**.
  - By contrast, an NVIDIA RTX 4090 or A100 (24GB/40GB VRAM) completes the same training run in **1.5 to 2.5 hours** using native FP16 Tensor Cores.

---

## 2. Partitioned Hybrid Engineering Workflow

To maximize speed and efficiency, implement a **partitioned hybrid workflow**:

```
+-------------------------------------------------------------------------------------------------+
|                               PARTITIONED WORKFLOW ARCHITECTURE                                 |
+---------------------------------------------------+---------------------------------------------+
| TIER 1: LOCAL M1 PRO (Rapid Engineering)          | TIER 2: CLOUD GPU (Heavy Compute)           |
+---------------------------------------------------+---------------------------------------------+
| 1. Codebase development, debugging, and linting   | 1. Full-scale backbone training (40 epochs)  |
| 2. Annotation parsing and data sanity checks      | 2. 3-Fold Leave-One-Domain-Out cross-val    |
| 3. Unit-testing `run.py` & Codabench API contract | 3. Large-scale multi-model ensembling       |
| 4. Metric calculation engine verification (mADM)  | 4. Long hyperparameter sweeps & grid search |
| 5. Lightweight adapter prototyping on cached feats| 5. Docker container validation inside Linux |
+---------------------------------------------------+---------------------------------------------+
```

---

## 3. High-Efficiency Local M1 Pro Configuration (PyTorch MPS)

When running exploratory training, adapter learning, or local validation on the M1 Pro, follow these exact settings to prevent memory exhaustion and maximize MPS throughput:

```python
import os
import torch
import torch.nn as nn

# 1. Device Selection with Fallback
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # Enable aggressive memory release on MPS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7" # Cap allocation at 70% RAM
    print("[M1 Pro] Utilizing Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("[M1 Pro] Falling back to CPU")

# 2. Optimal Batch Sizing & Gradient Accumulation
# Using a small physical batch size fits comfortably within unified RAM
PHYSICAL_BATCH_SIZE = 16
ACCUMULATION_STEPS = 4  # Effective Batch Size = 16 * 4 = 64

# 3. Dataloader Worker Configuration
# On macOS, spawning too many workers causes IPC socket overhead. Set num_workers=2
NUM_WORKERS = 2
PIN_MEMORY = False  # pin_memory is not supported/needed on unified memory MPS

# 4. FP16 Mixed Precision via Autocast
def train_step(model, optimizer, scaler, batch_x, batch_y):
    optimizer.zero_grad()
    
    # MPS supports float16 in PyTorch 2.4+
    with torch.autocast(device_type="mps", dtype=torch.float16):
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss = loss / ACCUMULATION_STEPS

    loss.backward()
    return loss.item()
```

---

## 4. Rapid Local Prototyping: The "Cached Feature" Trick

You can develop and optimize the retrieval adapters on the M1 Pro in minutes rather than hours by **pre-extracting and caching backbone features**:

```text
[Day 1 on Cloud GPU]:
Extract backbone embeddings for all 97,669 training images (using Swin-B / ViT-B).
Save as compressed NumPy array: train_features.npy (~200 MB).

[Days 2-10 on Local M1 Pro]:
Load train_features.npy into memory.
Train lightweight retrieval adapters (f_vis_ret, f_text_ret, f_attr_ret) directly on the cached vectors.
Training 50 epochs of adapters on M1 Pro takes ONLY 4 MINUTES!
Perform rapid loss formulation tuning, margin tuning, and alpha sweeps locally with zero cloud costs.
```

---

## 5. Cloud Acceleration Strategy (Free & Low-Cost)

When running full backbone training runs, utilize scalable cloud GPU environments:

### 5.1 Free Cloud Resources
1. **Google Colab (Free Tier):** Provides NVIDIA T4 (16GB VRAM). Run 15–20 epoch training jobs during active sessions.
2. **Kaggle Notebooks (Free Tier):** Provides 30 hours/week of dual NVIDIA T4 GPUs or single P100 (16GB VRAM). Excellent for running LODO cross-validation folds in parallel.

### 5.2 On-Demand High-Performance GPU Options
If running final ensembling or large backbones (Swin-Base + ViT-Base):
- **RunPod / Lambda Labs:** Rent an **NVIDIA RTX 4090 (24GB VRAM)** for $\approx \$0.35 - \$0.44\text{ / hour}$, or an **A100 (80GB VRAM)** for $\approx \$1.10\text{ / hour}$.
- A complete 40-epoch training run costs less than **$1.00 USD** on an RTX 4090 and finishes in under 2 hours.

---

## 6. Local-Cloud Code & Checkpoint Synchronization Workflow

Maintain Git as the single source of truth across the M1 Pro workstation and cloud runners:

```bash
# 1. On Local M1 Pro: Push clean code and configurations
git add models/ losses/ configs/ utils/
git commit -m "feat: implement dual-branch CASA attention and ASL loss"
git push origin main

# 2. On Cloud Runner (RunPod / Colab / Kaggle):
git clone https://github.com/<your-username>/RWS2027.git
cd RWS2027
# Download datasets from Google Drive or bucket
python download_datasets.py
# Execute heavy training
python train_upar_2027.py --config configs/c2t_base.yaml

# 3. Save serialized FP16 weights back to cloud storage / git-lfs:
# model_c2t_fp16.pt (~220 MB)

# 4. Pull weights to Local M1 Pro for final packaging and inspection:
# Package into submission.zip and verify with local test harness
```
