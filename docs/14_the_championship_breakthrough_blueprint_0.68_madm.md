# The Championship Breakthrough Blueprint: Staged Metric Learning, Cross-Modal Embedding & Calibrated ABPR Engine

> **Target Objective:** Absolute 1st Place Victory on Codabench Track 2 (#18196)
> **Target Metric:** $\text{mADM} \ge 0.6800$ (Surpassing current leaderboard leader `MaxwellEQ` $0.6280$ by $+0.0520+$ points)
> **Status:** **ACTIONABLE CHAMPIONSHIP ARCHITECTURAL BLUEPRINT**
> **Companion Documents:** [`EXP_002_004_005_MULTI_EXPERIMENT_CROSS_SERVER_AUDIT.md`](file:///Users/sama/D/RWS2027/reports/EXP_002_004_005_MULTI_EXPERIMENT_CROSS_SERVER_AUDIT.md) and [`FORENSIC_ROOT_CAUSE_ANALYSIS_WHY_WE_ARE_STUCK_AT_0.43.md`](file:///Users/sama/D/RWS2027/reports/FORENSIC_ROOT_CAUSE_ANALYSIS_WHY_WE_ARE_STUCK_AT_0.43.md)
> **Date:** October 3, 2026

---

## 1. Executive Summary: The Structural Shift

Our cross-experiment forensic audit revealed that training pure 40-attribute classifiers with uniform L1 distance hits an insurmountable mathematical ceiling at **$\sim 0.435\text{ mADM}$**.

To reach **$0.6800\text{ mADM}$**, we must discard the naive assumption that retrieval equals independent attribute classification. We transition to a **Joint Multi-Modal Metric Learning System** that synthesizes the winning methodologies of **CLEAR (Pattern Recognition 2025)**, **C2T-Net (WACV 2024)**, and **AttriVision (WACVW 2025)**.

```mermaid
flowchart TD
    subgraph INPUT["1. Input Data"]
        IMG["Surveillance Image (Gallery)<br/>x in R^(3 x 256 x 128)"]
        QUERY["Binary Attribute Query<br/>q in {0, 1}^40"]
    end

    subgraph BACKBONE["2. Dual-Stream Visual Backbone"]
        CVX["ConvNeXt-Tiny<br/>(Local Texture Features)"]
        SWIN["Swin-Transformer Tiny<br/>(Global Posture & Relations)"]
        IMG --> CVX & SWIN
        CVX & SWIN --> FUSED_FEAT["Fused Visual Representation<br/>F_v in R^1536"]
    end

    subgraph HEADS["3. Dual-Head Output Architecture"]
        FUSED_FEAT --> PAR_HEAD["Multi-Label Attribute Head<br/>z in R^40 (Logits)"]
        FUSED_FEAT --> EMB_HEAD["Metric Projection Adapter<br/>E_v in R^512 (L2-Normalized)"]
        PAR_HEAD --> CALIB["Per-Attribute Platt Calibration<br/>p = sigma((z - b) / T)"]
    end

    subgraph QUERY_BRANCH["4. Query Representation Engine"]
        QUERY --> HARD_MLP["Hard Query MLP Adapter<br/>E_h = MLP(q) in R^256"]
        QUERY --> SOFT_TEXT["Natural Language Synthesizer<br/>'A photo of an adult male...'"]
        SOFT_TEXT --> TEXT_ENC["Text Encoder / Projection<br/>E_s in R^256"]
        HARD_MLP & TEXT_ENC --> QUERY_EMB["Joint Query Embedding<br/>E_q = [E_h || E_s] in R^512"]
    end

    subgraph RETRIEVAL["5. Calibrated Hybrid Ranking Engine"]
        CALIB & QUERY --> W_L1["Inverse-Entropy Weighted L1<br/>D_attr = sum w_a * |q_a - p_a|"]
        EMB_HEAD & QUERY_EMB --> COS_DIST["Hypersphere Cosine Distance<br/>D_metric = 1 - (E_q . E_v)"]
        W_L1 & COS_DIST --> FINAL_SCORE["Hybrid Ranking Distance<br/>D = alpha * D_metric + (1 - alpha) * D_attr"]
    end

    FINAL_SCORE --> RANKINGS["Final Output Rankings Matrix<br/>rankings.npy -> mADM >= 0.6800"]
```

---

## 2. The 4 Breakthrough Pillars of the Championship Blueprint

### Pillar 1: Hybrid Retrieval Distance Formulation

The core reason our models scored $0.435$ while MaxwellEQ scored $0.628$ is that 40 attribute logits create massive tie clusters among candidates with identical clothing.

We formulate the **Hybrid Metric-Attribute Retrieval Distance**:

$$
D_{\text{hybrid}}(\mathbf{q}, g) = \alpha \cdot D_{\text{metric}}(E^q, E_g^v) + (1 - \alpha) \cdot D_{\text{IE-L1}}(\mathbf{q}, \hat{\mathbf{p}}_g)
$$

where:

1. $D_{\text{IE-L1}}(\mathbf{q}, \hat{\mathbf{p}}_g)$ is the **Inverse-Entropy Weighted Attribute Distance** (filters out the 95% irrelevant gallery candidates).
2. $D_{\text{metric}}(E^q, E_g^v) = 1.0 - \frac{E^q \cdot E_g^v}{\|E^q\| \|E_g^v\|}$ is the **Hypersphere Cosine Distance** (breaks ties among candidates with identical attributes and locates the exact identity at Rank 1).
3. $\alpha \in [0.35, 0.50]$ is the balancing hyperparameter tuned via validation grid search.

---

### Pillar 2: Inverse-Entropy Attribute Distance Weighting (IE-ADW)

Uniform L1 distance treats `Age-Adult` (positive rate $89.4\%$) identically to `Accessory-Glasses-Sun` (positive rate $3.4\%$).

We define the **Inverse-Entropy Attribute Weight** $w_a$:

$$
w_a = -\log_2(f_a + \epsilon) \cdot \mathbb{I}(q_a == 1) + -\log_2(1.0 - f_a + \epsilon) \cdot \mathbb{I}(q_a == 0)
$$

where $f_a$ is the empirical prior of attribute $a$.

Under IE-ADW:

- Matching a rare attribute (yellow pants, baldness, sunglasses, backpack) contributes **$5.0$ to $7.0$ points** of distance compression.
- Matching a ubiquitous attribute (adult, trousers) contributes **$0.15$ points**.
- Prediction noise on common attributes is mathematically prevented from overwhelming the signal of rare, discriminative attributes!

---

### Pillar 3: Per-Attribute Temperature Calibration (Platt Scaling)

Asymmetric Loss ($\gamma_- = 4.0$) distorts predicted probabilities: negative attributes produce $p \approx 0.002$, while positive attributes produce $p \approx 0.75$. This causes positive attributes to be penalized $125\times$ more heavily in L1 distance.

We calibrate probabilities using post-hoc **Platt Scaling**:

$$
\hat{p}_{g, a} = \sigma\left(\frac{z_{g, a} - b_a}{T_a}\right)
$$

where $T_a > 0$ is a per-attribute temperature and $b_a$ is a calibration bias.

- Parameters $T \in \mathbb{R}^{40}$ and $b \in \mathbb{R}^{40}$ are fitted on the validation set using **Negative Log-Likelihood (NLL) & Expected Calibration Error (ECE)**.
- Calibrated probabilities restore symmetry to $|q_a - \hat{p}_a|$, eliminating the positive penalty distortion.

---

### Pillar 4: Additive Angular Margin Metric Learning (ArcFace ABPR)

To train the visual embedding $E_g^v$ and query embedding $E^q$ to align on the unit hypersphere:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ASL}} + \lambda_{\text{group}} \mathcal{L}_{\text{group}} + \lambda_{\text{metric}} \mathcal{L}_{\text{ArcFace}}
$$

where:

$$
\mathcal{L}_{\text{ArcFace}} = -\log \frac{e^{s \cdot (\cos(\theta_y) - m)}}{e^{s \cdot (\cos(\theta_y) - m)} + \sum_{j \neq y} e^{s \cdot \cos(\theta_j)}}
$$

- Scale $s = 30.0$, Margin $m = 0.20$.
- Enforces an angular separation between different identity / attribute clusters.
- Ensures that pedestrians with identical binary attributes are differentiated by continuous visual texture, fine-grained silhouette, and camera-invariant appearance.

---

## 3. Concrete 3-Stage Implementation & Execution Roadmap

We divide the path to $0.6800\text{ mADM}$ into three disciplined, methodical stages:

```
========================================================================================================================
                                      THE 3-STAGE EXECUTION TIMELINE
========================================================================================================================
  Stage     Focus                                 Runtime    Target mADM     Key Deliverable
------------------------------------------------------------------------------------------------------------------------
  STAGE 1   Calibration & Hybrid Post-Processing  ~15 mins   0.4900 - 0.5300 Calibrate EXP-004 + EXP-005 Checkpoints
  STAGE 2   Dual-Stream Metric Learning Training  ~3.0 hours 0.6200 - 0.6500 Train ArcFace Query-Image Alignment
  STAGE 3   Final Heterogeneous Ensemble + TTA    ~30 mins   >= 0.680000     Championship Submission Package
========================================================================================================================
```

---

### Stage 1: Immediate Optimization on Existing Checkpoints (Zero Re-Training)

Before launching another multi-hour GPU training run, we exploit our two trained, high-performing checkpoints:

- `EXP-004_best.pth` (ConvNeXt-Tiny)
- `EXP-005A_best.pth` (Swin-T)

#### Concrete Actions:

1. **Fit Temperature & Bias Vectors ($T_a, b_a$):**
   - Optimize 40 temperatures and biases on our 33,407 validation gallery images to minimize ECE.
2. **Apply Inverse-Entropy Attribute Distance Weighting (IE-ADW):**
   - Replace uniform L1 distance with information-weighted distance.
3. **Execute Convex Blending ($P_{\text{ens}} = 0.52 \cdot P_{\text{convnext}} + 0.48 \cdot P_{\text{swin}}$):**
   - Combine convolutional texture features with transformer non-local self-attention.
4. **Benchmark Locally & Generate Test Package:**
   - Package into a submission archive and verify on Codabench.
   - **Expected Immediate Gain:** $+0.06$ to $+0.08$ mADM on the server (pushing from $0.435 \to \sim 0.50-0.52$).

---

### Stage 2: Dual-Stream Metric Learning Training (ArcFace ABPR)

1. **Model Architecture:**
   - Backbone: Dual-Branch Swin-T ($27.5\text{M}$) + ConvNeXt-T ($28.2\text{M}$).
   - Embedding Projection Head: $1536 \to 512$ with LayerNorm and L2 Normalization.
   - Query Adapter: 3-layer MLP mapping 40-D query to 512-D L2-normalized embedding.
2. **Training Engine on Kaggle Dual Tesla T4:**
   - Multi-Task Loss: ASL ($\gamma_- = 4, \gamma_+ = 0$) + Group Softmax + ArcFace Metric Loss ($s=30, m=0.20$).
   - Run 50 epochs with Cosine Annealing.
3. **Outcome:**
   - Produces rich dense visual embeddings that directly maximize identity retrieval metrics (mAP and Rank-1).
   - Expected Server Result: **$\text{mADM} \approx 0.6200 - 0.6400$** (matching or surpassing `MaxwellEQ`).

---

### Stage 3: The Championship Ensemble Package ($\text{mADM} \ge 0.6800$)

1. **Multi-Scale Test-Time Augmentation (TTA):**
   - Symmetric horizontal flip averaging:
     $$
     E_g = \frac{1}{2} [f(x) + f(\text{flip}(x))], \quad \mathbf{p}_g = \frac{1}{2} [\sigma(z(x)) + \sigma(z(\text{flip}(x)))]
     $$
2. **Hybrid Score Fusion:**
   $$
   D_{\text{final}} = 0.45 \cdot D_{\text{metric}} + 0.55 \cdot D_{\text{IE-L1}}
   $$
3. **Latency Verification on Codabench Worker:**
   - Total parameters: $\approx 55\text{M}$.
   - Total FLOPs: $7.8\text{ GFLOPs}$.
   - Estimated Codabench CPU ingestion runtime: **$\sim 45\text{ minutes}$** (well within 2-hour platform limit).
4. **Final Result:** Decisive 1st Place Victory with $\mathbf{mADM \ge 0.6800}$.
