# Track B: Fine-Tune from Best Checkpoint with Domain-Robust Augmentation

## Comprehensive Implementation Plan

---

## 1. Executive Summary

| Item | Detail |
| :--- | :--- |
| **Objective** | Close the 0.113 val-to-test mADM generalization gap (0.7311 → 0.6186) by fine-tuning the best checkpoint with domain-robust photometric and geometric augmentation. |
| **Starting Point** | `checkpoint_best.pt` (Epoch 25/26, validation mADM 0.7311, rank-1 33.80%). |
| **Target** | Server mADM ≥ 0.6600 – 0.6850+ on CodaBench. |
| **Training Duration** | ~9.5–10.5 hours on Kaggle Dual Tesla T4s (16–18 epochs × ~34 min/epoch). |
| **Deadline** | October 22, 2026 at 5:30 PM IST (~26 hours 40 minutes remaining). |
| **Submissions Budget** | 2 submissions remaining today (Oct 21) + 3 submissions tomorrow (Oct 22) = **5 total submissions**. |
| **Core Interventions** | 1. Heavy camera-domain simulation (ColorJitter, RandAugment, RandomErasing, GaussianBlur, Grayscale)<br>2. Soft Label Regularization (Label Smoothing $\epsilon=0.10$)<br>3. Surgical Low-LR Fine-Tuning ($2\times 10^{-6}$ backbone, $2\times 10^{-4}$ head)<br>4. Fresh AdamW optimizer + Cosine Annealing with Warmup<br>5. Direct weight extraction to FP16 without optimizer state bloating. |

---

## 2. Deep Root-Cause Analysis: Why the Gap Exists

### 2.1 The Generalization Collapse (0.7311 Val vs. 0.6186 Test)
On our local validation set (33,407 images from UPAR):
- Validation mADM: **0.7311**
- Attribute Accuracy: **92.54%**
- Rank-1: **33.80%**

On CodaBench Test Server (29,248 gallery images across 893 real-world test queries):
- Test mADM: **0.6186**
- Test Rank-1: **36.59%**
- Test Rank-5: **59.77%**
- Test Rank-10: **68.26%**

### 2.2 Critical Diagnostic Findings
1. **The Visual Metric Embedding is Already World-Class**:
   - The test server Rank-5 (59.77%) and Rank-10 (68.26%) **already match or exceed the Top 3–5 teams on the global leaderboard** (`syouya_tobita`: 58.74%, `gnurt2041`: 58.74%, `linhtinh`: 58.64%).
   - The visual backbone understands person identities and general visual appearance remarkably well.
2. **The Bottleneck is Overconfident Attribute Predictions Under Domain Shift**:
   - In UPAR training data, camera conditions are homogeneous within splits.
   - On the test set, surveillance cameras vary dramatically in mounting angle (45°–90° overhead), color gamut, exposure, motion blur, and ambient illumination.
   - Without aggressive color and occlusion augmentation, the attribute head learned spurious correlations (e.g. associating specific shadows or lighting color casts with specific garment colors).
   - Because hard binary cross-entropy (BCE) pushes logits to $\pm 10$ or $\pm 15$, a single misclassified color attribute (e.g., classifying a gray coat as black due to camera contrast) generates an extreme sigmoid probability error ($\approx 0.999$), directly destroying the L1 attribute distance:
     $$\text{Dist}_{\text{attr}} = \frac{\|q - \sigma(\text{logits})\|_1}{40}$$
   - Since $S_{\text{final}} = 0.70 S_{\text{attr}} + 0.30 S_{\text{metric}}$, errors in $S_{\text{attr}}$ overpower the excellent visual metric embedding.

---

## 3. The 6 Technical Pillars of Track B

### Pillar 1: Domain-Robust Photometric & Spatial Augmentation
Instead of plain resizing and basic horizontal flipping, training batches will pass through an aggressive surveillance-domain simulation pipeline:
1. **RandomResizedCrop (scale 0.80–1.0, ratio 0.85–1.15)**: Simulates varying bounding box tightness, distance to camera, and focal length differences.
2. **Heavy ColorJitter (brightness 0.40, contrast 0.40, saturation 0.40, hue 0.10)**: Prevents the attribute head from overfitting to specific camera white balances or sensor color casts.
3. **RandomGrayscale (p=0.10)**: Forces non-color attributes (age, hair length, clothing types, bags, glasses, hats) to rely strictly on geometric and structural cues rather than chromatic shortcuts.
4. **GaussianBlur (kernel=5, sigma 0.1–2.0, p=0.25)**: Simulates camera out-of-focus and atmospheric haze in outdoor surveillance.
5. **RandAugment (num_ops=2, magnitude=9)**: Applies dynamic combinations of auto-contrast, equalization, posterization, solarization, and sharpness adjustments.
6. **RandomErasing (p=0.50, scale 0.02–0.30, ratio 0.3–3.3)**: Simulates real-world occlusions by street furniture, bicycles, railings, and adjacent pedestrians.

### Pillar 2: Soft Label Regularization ($\epsilon = 0.10$)
Replaces hard binary targets $y \in \{0, 1\}$ with smoothed targets:
$$y_{\text{smooth}} = y \cdot (1 - \epsilon) + (1 - y) \cdot \epsilon = 0.90 \text{ or } 0.10$$
- Prevents logits from exploding into saturation.
- When an unseen camera exhibits atypical lighting, the predicted probability hovers near 0.85 or 0.15 rather than 0.999 or 0.001.
- Limits the worst-case penalty on true positive retrieval candidates.

### Pillar 3: Fine-Tune Initialization Mode (Weight-Only Loading)
- Loads **only** the model weights from `checkpoint_best.pt` (Epoch 26).
- **Discards** the old AdamW momentum, variance buffers, and learning rate scheduler states.
- Starting with fresh optimizer buffers ensures that previous high-momentum directions do not conflict with the new gradient signals induced by heavy data augmentation.

### Pillar 4: Surgical Learning Rate Strategy
Fine-tuning a converged 304M parameter ViT-L/14 model requires a delicate touch:
- **Backbone LR**: $2.0 \times 10^{-6}$ (40% of original $5 \times 10^{-6}$)
- **Head LR**: $2.0 \times 10^{-4}$ (40% of original $5 \times 10^{-4}$)
- **Weight Decay**: $1.0 \times 10^{-4}$ with AdamW $(\beta_1 = 0.9, \beta_2 = 0.999)$.

### Pillar 5: Cosine Annealing with Warmup
- **Warmup (Epochs 1–2)**: Linear ramp from $10\%$ of base LR to $100\%$. Protects pretrained visual token representations from shock during early augmented batches.
- **Cosine Annealing (Epochs 3–18)**: Smooth decay down to $1\%$ of base LR ($2 \times 10^{-8}$ on backbone). Ensures stable convergence into a flat, robust minimum.

### Pillar 6: Gradient Clipping & Numerical Stability
- Max gradient norm clipped to $1.0$ via Accelerate:
  `accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Prevents gradient spikes triggered by extreme RandAugment or RandomErasing patches.

---

## 4. End-to-End Execution Timeline (Remaining 26 Hours)

```
[Oct 21, 03:00 PM - 03:30 PM] | Preparation Phase
  - Upload checkpoint_best.pt to Kaggle as a clean dataset.
  - Set up notebook with Track B code cells.
  - Verify zero RAM leak on startup.

[Oct 21, 03:30 PM - 01:30 AM] | Fine-Tuning Execution Phase (10 Hours)
  - Launch 18-Epoch fine-tuning on 2x Tesla T4 GPUs via Accelerate.
  - Checkpoint saved every epoch.
  - Epoch 1 finishes ~04:05 PM (Go/No-Go validation health check).
  - Training finishes ~01:30 AM.

[Oct 22, 01:30 AM - 02:15 AM] | Model Packaging & Audit Phase
  - Extract best checkpoint weights to FP16 (~583 MB).
  - Verify complete elimination of optimizer states.
  - Run air-gapped sandbox verification of submission zip.

[Oct 22, 08:00 AM - 05:30 PM] | Championship Submissions Phase (Deadline: 05:30 PM)
  - Submission 1 (Morning): Fine-tuned model with Flip TTA + 0.70/0.30 fusion.
  - Submission 2 (Afternoon): Optimal fusion calibration based on Sub 1 metrics.
  - Submission 3 (Late Afternoon): Final refined shot before 05:30 PM deadline.
```

---

## 5. Step-by-Step Kaggle Notebook Architecture

### Cell 1: Environment & Hardware Verification
- Verify Dual Tesla T4 GPUs (14.56 GB VRAM each).
- Verify FP16 mixed precision capability and available system RAM.

### Cell 2: In-Memory Code Injection & Augmentation Setup
- Patch `UPARDatasetV4` in `sota_pipeline/dataset_v4.py` to inject the domain-robust augmentation pipeline for the `train` split, leaving the `val` split clean and deterministic.
- Ensure PIL image conversions are strictly preserved before tensor transformation.

### Cell 3: Training Engine Update (`train_v4.py`)
- Inject `--finetune_from` weight-only loader (discarding optimizer/scheduler buffers).
- Inject `--label_smooth 0.10` into the BCE attribute loss computation.
- Inject `CosineAnnealingWithWarmup` scheduler (2 warmup epochs + 16 cosine decay epochs).
- Inject `accelerator.clip_grad_norm_(..., max_norm=1.0)`.

### Cell 4: RAM Hygiene Cell
- Explicitly run garbage collection and call `libc.so.6` `malloc_trim(0)`.
- Avoid any large-scale disk directory copying (`shutil.copytree`) into memory.

### Cell 5: Accelerate Multi-GPU Training Launch
- Launch 2-process distributed training with batch size 24 per GPU, gradient accumulation 3 (effective batch size 144).
- Total epochs: 18.
- Output directory: `/kaggle/working/v4_finetune_outputs`.

### Cell 6: Post-Training Automated Extraction & Submission Generation
- Load `checkpoint_best.pt` from fine-tuning outputs.
- Strip all optimizer states, casting parameters to FP16.
- Package into `submission_v4_finetune.zip` along with `run.py`, `metadata.yaml`, and vendored `timm`.
- Run automated in-situ sandbox verification to ensure 0 NaNs, correct rank shape, and runtime well within 300s.

---

## 6. What to Monitor During Training

1. **Epoch 1 Go/No-Go Decision**:
   - Training loss will start higher (~2.0–3.0) due to label smoothing and heavy augmentation.
   - Validation mADM will temporarily dip from 0.7311 to ~0.58–0.66 in Epoch 1 because the model is adapting to the new noisy feature distribution. **This is completely normal.**
   - Attribute accuracy should remain ≥ 82%.
2. **Epochs 3–8 (Adaptation & Gain)**:
   - Training loss steadily drops below 1.0.
   - Validation mADM climbs back towards 0.70+.
3. **Epochs 10–18 (Convergence)**:
   - Learning rate decays below $5 \times 10^{-7}$.
   - The model finds flat minima that do not rely on high-frequency lighting shortcuts.
   - Checkpoint with highest validation mADM is automatically saved as `checkpoint_best.pt`.

---

## 7. Expected Outcomes & Honest Assessment

- **Current Baseline (Submission #1007022)**: Server mADM **0.6186** (Rank-1: 36.59%, Rank-5: 59.77%, Rank-10: 68.26%).
- **Realistic Target Post-Fine-Tuning**: Server mADM **0.6500 – 0.6750+**.
- **Stretch Target (Top 3)**: Server mADM **0.6850+**.
  - While single-model fine-tuning cannot match a 5-model ensemble, closing the domain generalization gap through photometric augmentation directly addresses the exact flaw holding our Rank-1 score back.
  - Every 1% improvement in Rank-1 and reduction in attribute L1 error provides direct, compounding gains in mADM.
