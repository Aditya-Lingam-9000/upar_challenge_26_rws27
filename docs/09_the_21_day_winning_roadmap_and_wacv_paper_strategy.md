# Report 09: The 21-Day Winning Roadmap, Milestone Architecture & WACV 2027 Workshop Paper Strategy

**Document:** Operational Execution Plan & Academic Publication Blueprint  
**Time Horizon:** 21 Days (September 30, 2026 $\to$ October 21, 2026 AoE)  
**Deliverables:** Winning Codabench Model (#1), 8-Page WACV Workshop Technical Paper, Reproducibility Codebase  
**Document Status:** Publication-Grade Execution Roadmap  

---

## 1. Master Strategic Calendar & Operational Milestones

```
Sep 30 - Oct 04          Oct 05 - Oct 10          Oct 11 - Oct 16          Oct 17 - Oct 21
+--------------------+   +--------------------+   +--------------------+   +--------------------+
|      PHASE 1       |   |      PHASE 2       |   |      PHASE 3       |   |      PHASE 4       |
| Foundation, Data   |-->| Dual-Stream System |-->| Domain Invariance  |-->| Ensembling, Final  |
| Prep & Local Bench |   | & Margin Learning  |   | & Metric Tuning    |   | Polish & Win Lock  |
+--------------------+   +--------------------+   +--------------------+   +--------------------+
                                                                                     |
                                                                                     v
                                                                           Oct 21: Challenge Closes
                                                                           Oct 23: Workshop Paper Due
                                                                           Oct 30: Code Audit Due
```

---

## 2. Phase-by-Phase Day-by-Day Implementation Roadmap

### Phase 1: Foundation, Local Validation Engine & Baseline Reproduction (Days 1–5 | Sep 30 – Oct 04)

- **Day 1 (Sep 30): Environment Setup & Dataset Preparation**
  - Create dedicated virtual environment: `python3 -m venv rws_env && source rws_env/bin/activate`.
  - Extract datasets from `RWS27_Data/` (Market-1501, PA-100K, PETA) into `UPAR-Challenge-2027/data/`.
  - Verify directory structure against `UPAR-Challenge-2027/README.md`.
  - Map PETA filenames using `peta_file_mapping.txt`.
- **Day 2 (Oct 01): Local Offline Metric Engine Implementation**
  - Write `metrics/retrieval_metrics.py` implementing exact vectorization of:
    - Degree of Match ($\text{dom}$), Normalized Degree of Match ($\text{ndom}$), and Average Degree of Match (ADM)
    - Mean Average Precision (mAP), Rank-1, Rank-5, Rank-10, and mINP.
  - Verify that this local engine reproduces identical numbers to the starter kit sample baseline ($\approx 0.1567\text{ mADM}$).
- **Day 3 (Oct 02): Partition 3-Fold Leave-One-Domain-Out (LODO) Splits**
  - Generate split CSVs for LODO cross-validation (PA100K + PETA $\to$ Market; Market + PETA $\to$ PA100K; Market + PA100K $\to$ PETA).
  - Compute class weights and positive priors per domain.
- **Day 4 (Oct 03): Baseline Implementation (ConvNeXt-Base)**
  - Implement standard ConvNeXt-Base with a 40-attribute multi-label head.
  - Train on PA-100K + Market + PETA using weighted BCE.
  - Evaluate on local validation set. Target: $\text{mADM} \approx 0.48 - 0.52$.
- **Day 5 (Oct 04): Packaging Pipeline & First Test Submission**
  - Package baseline into `submission.zip` with `run.py` and `metadata.yaml`.
  - Submit to Codabench Phase 1. Confirm successful execution inside the container and verify log outputs.

---

### Phase 2: Dual-Stream System & Margin Learning (Days 6–11 | Oct 05 – Oct 10)

- **Day 6 (Oct 05): Implement C2T-Net Dual-Branch Backbone**
  - Integrate pure PyTorch Swin-B and ViT-B backbones.
  - Implement Channel-Aware Self-Attention (CASA) and Cross-Fused Attention (SVCF / VSCF).
- **Day 7 (Oct 06): Train C2T-Net with Asymmetric Loss (ASL)**
  - Implement ASL ($\gamma_+=0, \gamma_-=4, m=0.05$) and Category Softmax Loss.
  - Fine-tune on Cloud GPU (RTX 4090 / A100) using domain-balanced batch sampling.
  - Target PAR validation metrics: $\text{mA} \ge 81.5\%$, $\text{F1} \ge 85.5\%$.
- **Day 8 (Oct 07): Feature Caching & Pseudo-Description Generation**
  - Freeze the trained C2T-Net backbone.
  - Generate offline pseudo-descriptions for all 7,204 training queries and 3,462 validation queries.
  - Pre-extract and cache visual feature embeddings and query text embeddings using SigLIP / CLIP text encoders.
- **Day 9 (Oct 08): Implement & Train Retrieval Adapters**
  - Construct 3-layer MLP projection adapters: $f^{vis}_{ret}$ (visual feature to 512-D), $f^{attr}_{ret}$ (binary query to 256-D), and $f^{text}_{ret}$ (text embedding to 256-D).
  - Train adapters using Additive Angular Margin Loss ($s=30, m=0.2$).
- **Day 10 (Oct 09): Dynamic Joint Fusion Optimization**
  - Evaluate joint scoring: $S = \alpha S_{\text{attr}} + (1 - \alpha) S_{\text{emb}}$.
  - Perform parameter sweep on validation data to determine optimal $\alpha \approx 0.70$.
- **Day 11 (Oct 10): Second Diagnostic Submission on Codabench**
  - Upload single-model HDSR-Net. Verify container timing ($\approx 3.5\text{ mins}$) and inspect leaderboard score.
  - Target score: $\mathbf{mADM \ge 0.6350}$ (surpassing current 1st place).

---

### Phase 3: Domain Invariance Tuning & Metric Calibration (Days 12–17 | Oct 11 – Oct 16)

- **Day 12 (Oct 11): Domain-Adversarial Feature Alignment**
  - Introduce a Gradient Reversal Layer (GRL) and domain discriminator predicting source dataset (Market vs. PA100K vs. PETA).
  - Train backbone to maximize attribute classification while confusing domain identity, enforcing domain invariance.
- **Day 13 (Oct 12): Surveillance Data Augmentation Tuning**
  - Integrate aspect-ratio-preserving resize with reflection padding.
  - Tune Random Erasing parameters ($p=0.5$, scale $0.02 - 0.20$) and strict ColorJitter bounds (hue $\le 0.03$).
- **Day 14 (Oct 13): Test-Time Augmentation (TTA) Integration**
  - Incorporate horizontal flip TTA into `run.py`.
  - Confirm $+0.8 - 1.2\%$ boost in local validation mADM.
- **Day 15 (Oct 14): Train Complementary Architectures**
  - Train Model B: ConvNeXt-Base with Focal ASL.
  - Train Model C: SigLIP-PAR with region-aware prompts.
- **Day 16 (Oct 15): LODO Cross-Domain Generalization Audit**
  - Evaluate all trained models across the 3 LODO cross-validation folds.
  - Select checkpoints exhibiting the lowest cross-domain variance.
- **Day 17 (Oct 16): Third Leaderboard Diagnostic Submission**
  - Submit updated model incorporating TTA and domain-aligned weights.
  - Target score: $\mathbf{mADM \ge 0.6550}$.

---

### Phase 4: Ensembling, Final Submission & Paper Preparation (Days 18–21 | Oct 17 – Oct 21)

- **Day 18 (Oct 17): Multi-Model Heterogeneous Ensemble**
  - Blend predicted probability matrices and metric embeddings:
    $$\mathbf{P}_{\text{ens}} = 0.50 \cdot \mathbf{P}_{\text{C2T}} + 0.30 \cdot \mathbf{P}_{\text{ConvNeXt}} + 0.20 \cdot \mathbf{P}_{\text{SigLIP}}$$
  - Verify that ensemble inference finishes in $< 1,200\text{ seconds}$ inside the container.
- **Day 19 (Oct 18): Full Local Docker Sandbox Stress Test**
  - Run the packaged archive inside `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` locally with `--network none`.
  - Verify zero runtime warnings, no memory leaks, and perfect output shape.
- **Day 20 (Oct 19): Submit Final Winning Candidate on Codabench**
  - Submit the final ensembled model.
  - Verify status switches to `Finished` and verify leaderboard rank.
  - Target score: $\mathbf{mADM \ge 0.6800}$ (**Securing 1st Place**).
- **Day 21 (Oct 20–21): Final Buffer & Submission Lock-in**
  - Lock in the final submission before Oct 21, 23:59 AoE (Oct 22, 17:29 IST).
  - Archive all training checkpoints, git commits, configuration YAMLs, and logs for verification.

---

## 3. WACV 2027 Workshop Technical Paper Strategy (Due Oct 23 AoE)

Accepted challenge technical papers will be presented at the RWS Workshop and published in the **IEEE/CVF WACV 2027 Workshop Proceedings**.

### 3.1 Paper Requirements & Specifications
- **Template:** Official WACV 2027 LaTeX format (IEEE/CVF double-blind).
- **Page Limit:** **Maximum 8 pages** excluding references.
- **Submission System:** OpenReview (`thecvf.com/WACV/2027/Workshop/RWS`).
- **Deadline:** **October 23, 2026, 23:59 AoE (October 24, 17:29 IST)**.

### 3.2 Recommended Paper Outline & Structure
1. **Title:** *Bridging the Domain Gap in Attribute-Based Person Retrieval via Cross-Transformers and Multi-Modal Metric Alignment*
2. **Abstract (150–200 words):** 
   - Define the challenge of cross-dataset ABPR under hidden domain shifts.
   - Introduce HDSR-Net: dual-branch cross-transformers (C2T-Net) + pseudo-text generation + angular margin metric learning.
   - Highlight 1st place standing on the WACV 2027 UPAR Challenge leaderboard.
3. **Introduction (1.5 pages):**
   - Real-world surveillance applications (forensic search, missing persons, perimeter security).
   - The distinction between image-to-image Re-ID and attribute-to-image retrieval.
   - Historical context: why 2024 methods failed on Track 2 and how HDSR-Net resolves domain collapse.
4. **Related Work (1 page):**
   - Pedestrian Attribute Recognition (PAR).
   - Attribute-Based Person Retrieval (ABPR).
   - Vision-Language Models in Surveillance.
5. **Proposed Methodology (2.5 pages):**
   - Dual-branch architecture: Swin-Transformer (local) + Vanilla ViT (global).
   - Channel-Aware Self-Attention (CASA) and Cross-Fused Attention (SVCF / VSCF).
   - Dual-Stream Query Representation: Hard binary embeddings + Soft pseudo-description embeddings.
   - Compound Loss Function: Asymmetric Loss (ASL) + Category Softmax Constraints + Additive Angular Margin Metric Loss.
   - Dynamic Joint Fusion: Balancing calibrated attribute distances and hyperspherical metric similarities.
6. **Experiments & Results (2 pages):**
   - Dataset statistics and LODO cross-validation setup.
   - Comparative evaluation against official baseline and leaderboard competitors.
   - Comprehensive Ablation Study Table:
     - Impact of CASA and Cross-Fusion vs. standard ViT / Swin.
     - Impact of ASL vs. BCE vs. Focal Loss on extreme minority attributes.
     - Impact of soft pseudo-description queries vs. raw binary queries.
     - Impact of fusion parameter $\alpha$.
     - Qualitative retrieval visualizations: top-10 retrieval strips demonstrating near-miss ranking quality.
7. **Conclusion & Broader Impact (0.5 page):**
   - Summary of key contributions and future directions in cross-domain surveillance retrieval.

---

## 4. Code Submission & Reproducibility Verification (Due Oct 30 AoE)

The top-ranked teams must provide their complete training code and documentation by **October 30, 2026 AoE (October 31, 17:29 IST)**.

### 4.1 Reproducibility Checklist
To guarantee passing the organizers' audit without disqualification:
1. **Deterministic Random Seeds:** Set seeds across Python, NumPy, and PyTorch:
   ```python
   torch.manual_seed(42)
   np.random.seed(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
2. **Configuration Reproducibility:** Store all training hyperparameters in structured YAML config files (`configs/train_c2t.yaml`).
3. **Automated Training Shell Script:** Provide a single executable script:
   ```bash
   bash run_training.sh --config configs/train_c2t.yaml --data_dir ./data
   ```
4. **Fact Sheet Preparation:** Document:
   - Team name, members, and academic affiliations.
   - Hardware requirements (GPU model, VRAM, training time).
   - Detailed method summary, loss formulations, and pre-trained backbone sources.
