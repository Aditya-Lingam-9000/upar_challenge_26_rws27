# Report 1: What We Have Until Now & What Mistakes We Made

**Status**: Ranked **#33 out of 40** with best mADM = 0.5720 (submission ID 994819)
**Gap to #1**: -0.1063 mADM (tannguyenbmt47 at 0.6783)
**Deadline**: October 22, 2026 (4 days remaining)
**Submissions remaining**: 33/50 total, 3/day

---

## 1. What We Built (V3 Architecture)

| Component | What It Is | Problem |
|-----------|-----------|---------|
| **Backbone** | EVA02-Base-Patch16-CLIP-224 (86M params, 768-dim) | **Way too small.** Top teams use ViT-L/14 (304M, 1024-dim) — 3.5× larger |
| **ConvPass** | Custom depthwise separable conv bypasses in ViT blocks | **Unpublished, no benchmarks anywhere.** Zero evidence this helps person retrieval |
| **DMHA** | Custom Deformable Multi-Head Attention with grid sampling | **Unpublished.** Adds complexity but no proven gain on any ReID benchmark |
| **PCNorm** | Custom Fourier Phase-Content Normalization | **Unpublished.** Fourier DG is a real concept, but our implementation is untested |
| **AGCA** | Custom 6-part Anatomical Gaussian Cross-Attention | **Unpublished.** Part-based attention is proven (PCB, TransReID), but AGCA is our invention |
| **ContinuousAttributePrompter** | Small MLP mapping 40 binary attrs → 1024-dim embedding | **Fundamentally inferior to CLIP text encoder with natural language** |
| **LorentzHyperbolicHead** | Hyperbolic embedding space | **No evidence** hyperbolic embeddings help attribute retrieval |
| **Custom Losses** | Continuous Attr-SDM + ASL + Sinkhorn OT + Hyperbolic + LookASAM | **All custom variants** of real losses, but untested combinations |

**Total**: 107.4M params, ~205MB FP16

> [!CAUTION]
> **Every single surgical module we added (ConvPass, DMHA, PCNorm, AGCA, Lorentz) is our own invention with ZERO published benchmarks.** We essentially built a research prototype and expected it to beat competition veterans using proven frameworks.

---

## 2. Complete Submission History

| # | Date | Submission | Score | Delta | Notes |
|---|------|-----------|-------|-------|-------|
| 1 | Oct 01 | EXP-001 | 0.4041 | — | First attempt, ConvNeXt-Tiny |
| 2 | Oct 01 | EXP-002 | 0.4126 | +0.0085 | Minor tweaks |
| 3 | Oct 02 | EXP-003 | 0.4350 | +0.0224 | Larger model |
| 4 | Oct 03 | EXP-005A | 0.4271 | -0.0079 | Regression |
| 5 | Oct 03 | EXP-006 SWIN | 0.4552 | +0.0281 | Swin backbone |
| 6 | Oct 04 | EXP-006 | 0.4507 | -0.0045 | Different variant |
| 7 | Oct 06 | SOTA Championship | 0.4912 | +0.0405 | First V1 SOTA |
| 8 | Oct 15 | submission_latest | 0.5057 | +0.0145 | V2 attempt |
| 9 | Oct 15 | submission_latest | 0.5246 | +0.0189 | V2 improved |
| 10 | Oct 15 | submission_sota_v2 | 0.5279 | +0.0033 | V2 final |
| 11 | Oct 17 | submission_sota_v3 | **0.5720** | **+0.0441** | **BEST EVER** — V3 clean |
| 12 | Oct 17 | submission_sota_v3 | 0.5542 | **-0.0178** | V3 + DBA + zoom crop (WORSE) |

**Trend**: Steady climb from 0.40 to 0.57 over 17 submissions, then **regression** when we added DBA/zoom crop.

---

## 3. The 7 Critical Mistakes

### Mistake 1: Using EVA02-Base Instead of ViT-Large
EVA02-Base has 86M params and 768-dim embeddings. Top person retrieval models use:
- **CLIP ViT-L/14**: 304M params, 1024-dim → Rank-1 95.7% on Market-1501
- **EVA02-Large**: 304M params, 1024-dim
- **SigLIP So400m**: 400M params

Our backbone is **3.5× smaller** than what competition winners use. This is the single biggest gap.

### Mistake 2: Inventing Custom Architecture Instead of Using Proven Frameworks
Top teams use **proven, published frameworks** with verified benchmarks:
- **IRRA** (CVPR 2023): 73.38% Rank-1 on CUHK-PEDES with SDM+ITC+MLM losses
- **CLIP-ReID**: 93.8% Rank-1 on Market-1501, published and reproduced
- **TransReID**: 95.2% Rank-1 on Market-1501 with side-information embeddings
- **SOLIDER**: 96.9% Rank-1 on Market-1501 with semantic conditioning

We used **none of these**. Instead we invented ConvPass, DMHA, PCNorm, AGCA — modules with zero evidence of working.

### Mistake 3: Custom MLP Prompter Instead of CLIP Text Encoder
Our `ContinuousAttributePrompter` is a tiny MLP that maps 40 binary bits → 1024-D. This is fundamentally weak because:
- It has no language understanding
- It cannot capture attribute relationships (e.g., "young female with short hair" as a semantic concept)
- It was trained only on our small dataset

**What top teams do**: Convert binary attributes to **natural language sentences** (e.g., "A young female wearing a black short-sleeved top and blue trousers, carrying a backpack") and encode them with the **CLIP text encoder**. This leverages billions of image-text pairs of pretraining.

### Mistake 4: Training Only 45 Epochs on 97K Images
With a ViT-Base backbone on 97K training images:
- We trained 15 epochs from scratch, then 30 more resumed = 45 total
- Top teams train **120+ epochs** on larger datasets with curriculum learning
- Our model never converged — the 30 resumed epochs showed **zero improvement** (mADM flatlined at ~0.675 locally)

### Mistake 5: Post-Processing Experiments Without Validation
We added Alpha-DBA Gallery Smoothing and 90% Center Zoom Crop TTA directly to the submission without proper ablation:
- **Alpha-DBA**: Blurred fine-grained attribute features, lost 0.018 mADM
- **90% Center Zoom Crop**: Amputated head/foot regions, destroying hat/shoe attributes
- These took our best score (0.5720) down to 0.5542

### Mistake 6: No Domain Generalization Training
The test set contains **unseen surveillance cameras** with severe degradations (75px resolution, compression artifacts, extreme angles from MEVID). We trained on clean Market-1501/PA-100K/PETA images with no surveillance degradation simulation.

### Mistake 7: Overengineered Inference Pipeline
Our test-time pipeline includes 6 complex stages (Prior-Matching Calibration → Alpha-QE → Cascaded Asymmetric Reranking with Surprisal Weighting). While conceptually interesting, this complexity makes it fragile and hard to debug. Simple approaches (attribute L1 distance + cosine similarity with optimal α) work better per the blueprint research.

---

## 4. What Actually Worked (Keep These)

| Component | Contribution | Evidence |
|-----------|-------------|----------|
| Dual-Pass Horizontal Flip TTA | +0.01-0.02 mADM | Standard, proven everywhere |
| Prior-Matching Calibration | +0.005-0.01 mADM | Newton-Raphson logit calibration |
| Alpha-QE (α=3.0, k=5) | +0.005-0.01 mADM | Published technique |
| Top-300 Cascaded Reranking | +0.01-0.02 mADM | Attribute reranking within candidate pool |
| Air-gapped vendored timm | Required | Submission runs correctly |

---

## 5. What Actively Hurt (NEVER Do Again)

| Component | Impact | Why |
|-----------|--------|-----|
| Alpha-DBA Gallery Smoothing | **-0.018 mADM** | Blurs fine-grained attribute features |
| 90% Center Zoom Crop TTA | **Negative** | Cuts off head/feet, destroys hat/shoe attrs |
| Expanding cascade_k > 300 | **Negative** | Injects noise from bad attribute predictions |
| Dynamic Gallery Marginal Calibration | **Negative** | Worse than fixed prior calibration |
| Two-Stage Multi-Hop Alpha-QE | **Negative** | Over-smooths query representations |
| PCA Whitening | **Negative** | Collapses cross-modal alignment |

---

## 6. The Real Gap: Us vs. Competition

| Metric | Us (Best) | #1 | #10 | Gap to #1 | Gap to #10 |
|--------|-----------|-----|------|-----------|------------|
| mADM | 0.5720 | 0.6783 | 0.6595 | **-0.1063** | **-0.0875** |
| Rank-1 | 33.27% | 40.79% | ~38% | -7.52% | ~-5% |
| mAP | 30.22% | 39.52% | ~37% | -9.30% | ~-7% |
| mINP | 20.61% | 27.91% | ~26% | -7.30% | ~-5% |

> [!IMPORTANT]
> The gap is massive: **0.1063 mADM** to #1, **0.0875** to even crack top 10. This is not a post-processing gap — this is a **fundamental model quality gap**. No amount of inference tricks can bridge a 0.10+ mADM deficit. We need a fundamentally better model.
