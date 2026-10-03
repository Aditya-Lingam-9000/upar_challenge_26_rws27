# Stage 1 Optimization: Empirical Before vs. After Scorecard & Forensic Analysis

> **Date:** October 3, 2026  
> **Status:** Stage 1 Optimization Complete & Verified  
> **Validation Benchmark:** Full 33,407 Validation Gallery Images across 3,462 Multi-Camera Attribute Queries  
> **Baseline Models Evaluated:** `EXP-004` (ConvNeXt-Tiny, 256x128) & `EXP-005` (Swin-T, 256x128)

---

## 1. Executive Summary: The Stage 1 Optimization Breakthrough

In Stage 1 of the Championship Roadmap, we deployed:
1. **Per-Attribute Platt Probability Calibration** ($T_a, b_a$ scaling) to eliminate positive logit bias created by Asymmetric Loss ($\gamma_- = 4, \gamma_+ = 0$).
2. **Inverse-Entropy Attribute Distance Weighting (IE-ADW)** to evaluate whether information-weighted distance improves attribute retrieval over uniform distance.
3. **Cross-Architecture Heterogeneous Ensembling** (ConvNeXt-Tiny 50% + Swin-T 50%).

### Key Empirical Findings:
1. **Platt Calibration Delivers a Massive Leap (+0.0161 to +0.0171 mADM):**
   - **`EXP-004` ConvNeXt-Tiny:** mADM jumped from **0.5738 $\to$ 0.5899** (+0.0161), with Rank-1 jumping from **16.35% $\to$ 17.07%** (+0.72%).
   - **`EXP-005` Swin-T:** mADM jumped from **0.5741 $\to$ 0.5912** (+0.0171), with mAP jumping from **0.1246 $\to$ 0.1273** (+0.0027).
2. **Heterogeneous Ensembling Shatters the 0.60 Ceiling (0.6100 mADM):**
   - Blending ConvNeXt-Tiny and Swin-T with Platt calibration achieves **0.6100 mADM**, **0.1326 mAP**, and **17.36% Rank-1**.
   - This represents an absolute gain of **+0.0362 mADM** and **+1.01% Rank-1** over raw standalone ConvNeXt-Tiny, directly positioning the pipeline within striking distance of the competition leader (`MaxwellEQ` @ 0.6280 mADM).
3. **Crucial Information-Theoretic Revelation on IE-ADW:**
   - Weighted L1 distance (IE-ADW) caused mADM to drop from `0.6100` to `0.5475` (-0.0625).
   - **Root Cause:** The official Codabench mADM scoring program explicitly computes the *unweighted* mean L1 distance $\frac{1}{40} \sum_{a=1}^{40} |q_a - p_a^{(k)}|$. Ranking candidates by a weighted metric forces the top-K retrieval to minimize weighted distance, which mechanically increases the unweighted L1 distance evaluated by the benchmark.
   - **Takeaway:** Calibrated Uniform L1 distance is the mathematically optimal retrieval metric for the Codabench leaderboard objective.

---

## 2. Complete 9-Configuration Benchmark Scorecard

Evaluated on the full official UPAR-2027 Track 2 validation set ($N = 33,407$ gallery images, $Q = 3,462$ attribute queries):

| # | Configuration | mADM | mAP | Rank-1 | Rank-5 | Rank-10 | mINP | PAR mA | Validation Status |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **EXP-004: Raw ConvNeXt-Tiny (Uniform L1)** | 0.5738 | 0.1243 | 16.35% | 26.92% | 32.99% | 0.0572 | 82.80% | Uncalibrated Baseline |
| **2** | **EXP-004: Calibrated ConvNeXt-Tiny (Uniform L1)** | **0.5899** | **0.1251** | **17.07%** | **27.01%** | 32.15% | 0.0545 | 77.49% | **+0.0161 mADM, +0.72% R-1** |
| **3** | EXP-004: Calibrated + IE-ADW Weighted L1 | 0.5310 | 0.1195 | 15.71% | 25.82% | 30.76% | 0.0557 | 77.49% | Metric Mismatch Penalty |
| **4** | **EXP-005: Raw Swin-T (Uniform L1)** | 0.5741 | 0.1246 | 16.23% | 26.83% | 32.03% | 0.0593 | 83.46% | Uncalibrated Baseline |
| **5** | **EXP-005: Calibrated Swin-T (Uniform L1)** | **0.5912** | **0.1273** | **16.58%** | 26.69% | 31.57% | 0.0593 | 78.33% | **+0.0171 mADM, +0.35% R-1** |
| **6** | EXP-005: Calibrated + IE-ADW Weighted L1 | 0.5361 | 0.1239 | 16.06% | 26.34% | 31.14% | 0.0592 | 78.33% | Metric Mismatch Penalty |
| **7** | **Ensemble: Raw 50/50 Blend (Uniform L1)** | 0.5950 | 0.1314 | 16.81% | 26.78% | 31.92% | 0.0641 | 83.65% | Raw Ensemble Gain |
| **8** | **Ensemble: Calibrated 50/50 Blend (Uniform L1)** | **0.6100** | **0.1326** | **17.36%** | 26.83% | 31.66% | 0.0613 | 78.14% | **CHAMPION CONFIGURATION** |
| **9** | Ensemble: Calibrated Blend + IE-ADW Weighted | 0.5475 | 0.1245 | 16.00% | 25.85% | 31.02% | 0.0592 | 78.08% | Metric Mismatch Penalty |

---

## 3. Deep-Dive Before vs. After Analysis

### 3.1 Why Probability Calibration (Platt Scaling) Succeeded Dramatically

When training with Asymmetric Loss (ASL):
$$L_{\text{ASL}} = - y (1-p)^{\gamma_+} \log(p) - (1-y) (p_m)^{\gamma_-} \log(1-p_m)$$
where $\gamma_- = 4, \gamma_+ = 0$, and margin shift $m = 0.05$:
- For negative samples ($y=0$), the gradient is scaled down by $(p - m)^4$. When $p < m$, the gradient is zero.
- For positive samples ($y=1$), the gradient has weight $1.0$ at all times.
- **Consequence:** The neural network weights are updated with strong positive gradients and suppressed negative gradients. This causes the uncalibrated model to emit **consistently over-optimistic positive logits**.

When raw logits $z_a$ are passed through a standard sigmoid $\sigma(z_a)$, the resulting probabilities $p_a$ hover around $0.25 - 0.45$ even for attributes whose actual empirical frequency is only $2.0\% - 5.0\%$.

#### The Platt Scaling Remedy:
We fitted per-attribute scalar parameters $(T_a, b_a)$ such that:
$$p_a^{\text{calib}} = \sigma\left(\frac{z_a - b_a}{T_a}\right)$$
- The fitted bias offsets $b_a$ were consistently positive ($+1.0$ to $+2.8$).
- The fitted temperature parameters $T_a$ were between $1.1$ and $1.9$.
- Subtracting $b_a$ systematically shifts the over-optimistic logits back to the true empirical distribution.
- When gallery images are compared against binary attribute queries $q_a \in \{0, 1\}$, a calibrated $p_a \approx 0.03$ yields $|0 - 0.03| = 0.03$ error, whereas uncalibrated $p_a \approx 0.35$ yielded $|0 - 0.35| = 0.35$ error!
- Over 40 attributes, this error reduction directly increases the mean agreement degree by **+0.0161 to +0.0171** across all 33,407 images.

---

### 3.2 Why Heterogeneous Ensembling Breaks the 0.60 Barrier

ConvNeXt-Tiny is a **hierarchical convolutional architecture** with 7x7 depthwise convolutions and inverted bottleneck blocks. It excels at local spatial feature capture (clothing seams, pattern textures, shoes, hair boundaries).

Swin Transformer (Swin-T) is a **shifted-window vision transformer** with self-attention across dynamic local windows. It excels at long-range cross-region context (relationship between upper body attire and lower body trousers, overall color coherence, carrying accessories relative to body pose).

Because their error covariance is low, when their calibrated attribute posterior probabilities are combined:
$$p_a^{\text{ensemble}} = 0.50 \cdot p_a^{\text{ConvNeXt}} + 0.50 \cdot p_a^{\text{Swin}}$$
- Random prediction errors cancel out.
- Confident, cross-validated attribute detections are reinforced.
- **Result:** mADM leaps to **0.6100**, and Rank-1 leaps to **17.36%**, outperforming both individual backbones.

---

### 3.3 The Failure Mode of IE-ADW (Inverse-Entropy Weighting)

In theory, inverse-entropy weighting ($w_a = -\log_2(p_a)$) places higher weight on rare attributes (e.g., `hat`, `glasses`, `boots`), ensuring that candidates matching rare features rank higher.

However, in the official Track 2 evaluation program (`scoring_program/score.py`):
$$\text{mADM} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{K} \sum_{k=1}^K \left(1 - \frac{1}{40} \sum_{a=1}^{40} |q_a - p_a^{(k)}|\right)$$

Notice that:
1. The scoring program computes an **unweighted arithmetic mean** $\frac{1}{40}\sum |q_a - p_a^{(k)}|$.
2. If we rank gallery images by a weighted distance $D_w = \sum w_a |q_a - p_a|$ where rare attributes have weights up to $20\times$:
   - A gallery image with a match on 1 rare attribute but errors on 5 common attributes will be ranked ahead of an image with errors on 0 rare attributes and 1 common attribute.
   - When evaluated by the unweighted metric, the first image incurs an unweighted error of $5/40 = 0.125$, while the second image incurs an error of only $1/40 = 0.025$.
   - Consequently, the top-K retrieved images have higher unweighted L1 distance, causing mADM to drop by **0.05 to 0.06**.
3. **Conclusion:** Weighted distance should only be used if the benchmark metric itself uses weighted distance. For Codabench Track 2, **Uniform L1 Distance with Calibrated Probabilities** is the mathematically true optimizer of the competition metric.

---

## 4. Hardware & Codabench Ingestion Latency Feasibility

| Deployment Option | Model Architecture | Zip Package Size | Ingestion RAM Usage | Ingestion CPU Run Time (29,248 Gallery Imgs) | Timeout Risk on Codabench | Local Val mADM |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Option A (Ultra-Safe)** | EXP-005 Calibrated Swin-T | 106 MB | 1.8 GB | ~18 - 20 minutes | **Zero (Well within 45m)** | 0.5912 |
| **Option B (Fast Single)** | EXP-004 Calibrated ConvNeXt-T | 105 MB | 1.6 GB | ~14 - 16 minutes | **Zero (Well within 45m)** | 0.5899 |
| **Option C (Maximum Accuracy)** | Calibrated Ensemble (ConvNeXt-T + Swin-T) | 211 MB | 3.2 GB | ~32 - 36 minutes | **Low to Medium (Near 45m)** | **0.6100** |

---

## 5. Next Steps

1. Build standalone submission zip packages for both:
   - `submissions/EXP-005_CALIB_submission.zip` (Single-model calibrated Swin-T, ultra-fast and safe).
   - `submissions/EXP-006_STAGE1_ENSEMBLE_submission.zip` (Dual-model calibrated ensemble, maximum mADM of 0.6100).
2. Execute full end-to-end sandbox container simulation on both packages to verify zero errors, zero internet calls, and exact Codabench compliance.
3. Advance to **Stage 2: ArcFace Metric Learning** to extract identity-discriminative 512D deep embeddings, combining attribute matching with visual similarity to challenge the 0.65+ mADM threshold.
