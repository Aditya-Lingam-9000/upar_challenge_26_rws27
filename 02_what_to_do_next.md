# Report 2: What Needs to Be Done Next & Why

**Remaining Time**: ~4 days (Oct 18-22)
**Remaining Submissions**: 33/50 total, 3/day → max 12 more submissions
**Kaggle Sessions**: ~3-4 more 12-hour GPU sessions possible
**Current Best**: 0.5720 mADM (#33)
**Target**: Top 10 (≥0.66 mADM), ideally Top 5 (≥0.67 mADM)

---

## 1. The Fundamental Problem

Our model's **attribute prediction quality** is the bottleneck. mADM is computed as:

$$\text{mADM} = \frac{1}{Q} \sum_{q=1}^{Q} \text{ADM}(q) = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\min(R_q, G_q)} \sum_{k=1}^{\min(R_q, G_q)} \text{DoM}(q, g_k)$$

Where $\text{DoM}(q, g)$ = fraction of query attributes matching the gallery person. This means:
1. **Attribute accuracy directly determines mADM** — if your attribute predictions are wrong, the degree-of-match for each gallery item is wrong, so your rankings are wrong.
2. **A better backbone produces better attribute predictions**, which directly improves mADM.
3. **Post-processing can only reorder items** — it can't fix fundamentally wrong attribute predictions.

> [!IMPORTANT]
> The gap to #1 is 0.1063 mADM. Based on all evidence, ~80% of this gap (0.085) comes from inferior attribute prediction quality (smaller backbone, worse training), and ~20% (0.021) comes from inference/reranking differences.

---

## 2. Three Parallel Strategies (Priority Order)

### Strategy A: Retrain with CLIP ViT-L/14 + Natural Language Templates ⭐ HIGHEST IMPACT

**Why**: This is what top teams almost certainly use. CLIP ViT-L/14 is the dominant backbone in 2024-2026 person retrieval.

**Architecture**:
- **Visual Backbone**: CLIP ViT-L/14 (304M params, 1024-dim, pretrained on 400M image-text pairs)
- **Text Encoder**: CLIP text transformer (frozen or fine-tuned)
- **Query Encoding**: Convert 40 binary attributes → natural language sentence → CLIP text embedding
- **Training Loss**: SDM (Similarity Distribution Matching) + ITC (Image-Text Contrastive) + multi-label BCE for attribute head

**Template Example**:
```
Query: [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, ...]
→ "An adult female with short hair, wearing a black top and black trousers"
```

**Expected Server Score**: **0.64-0.68 mADM** (based on IRRA achieving 73% Rank-1 on CUHK-PEDES and the blueprint's 0.6894 local validation with α=0.70 fusion)

**Time Required**: 
- Model code: 4-6 hours
- Training on Kaggle Dual T4: 8-10 hours (60-80 epochs)
- Submission packaging: 1-2 hours
- **Total: ~1 Kaggle session (12 hours)**

**Risk**: Medium. CLIP ViT-L/14 FP16 is ~580MB — tight against 500MB ZIP limit. May need FP16 quantization or pruning.

**Proven Benchmarks**:
| Framework | Backbone | Market-1501 Rank-1 | CUHK-PEDES Rank-1 |
|-----------|----------|--------------------|--------------------|
| CLIP-ReID | ViT-B/16 | 93.8% | — |
| IRRA | ViT-B/16 | — | 73.38% |
| TransReID | ViT-B/16 | 95.2% | — |
| SOLIDER | Swin-B | 96.9% | — |
| **Ours needs** | **ViT-L/14** | **≥95%** | **≥70%** |

---

### Strategy B: Pure Strong Attribute Classifier + Simple Fusion ⭐ SAFEST PATH

**Why**: The blueprint research showed that a **pure attribute classifier** with α=0.70 attribute weight achieves 0.6894 mADM on local validation. mADM rewards correct attribute matching, not identity matching. A strong attribute classifier is more important than a strong metric embedding.

**Architecture**:
- **Backbone**: Swin-Base or EVA02-Large (304M params), fine-tuned for multi-label attribute classification
- **Head**: 9 specialized attribute group heads (Age, Gender, Hair, UpperLength, UpperColor, LowerLength, LowerColor, LowerType, Accessories)
- **Metric Stream**: Simple 512-dim embedding head on top of backbone features
- **Fusion**: α=0.70 attribute + 0.30 metric (proven optimal from blueprint ablation)

**Expected Server Score**: **0.62-0.66 mADM**

**Time Required**: 
- Model code: 3-4 hours
- Training on Kaggle Dual T4: 6-8 hours
- **Total: ~1 Kaggle session**

**Risk**: Low. This is a proven approach. The blueprint's ablation table shows α=0.70 is optimal.

**Key Insight**: The blueprint showed:
```
Pure Attribute (α=1.0):  0.6210 mADM
Pure Metric (α=0.0):     0.5120 mADM
α=0.70 Fusion:           0.6894 mADM
```
This means attribute quality matters 2× more than metric quality for mADM.

---

### Strategy C: Inference-Only Improvements to Current V3 Model ⭐ QUICKEST

**Why**: We can squeeze more score from the existing 0.5720 model without any retraining.

**Improvements to Try** (each independently):

| Technique | Expected Gain | Risk | Time |
|-----------|--------------|------|------|
| k-Reciprocal Reranking (k1=20, k2=6, λ=0.3) | +0.01-0.03 | Medium | 2h |
| Gallery Feature Whitening (ZCA, ε=1e-3) | +0.005-0.015 | Low | 1h |
| Pseudo-Relevance Feedback (Rocchio, α=1.0, β=0.5, k=5) | +0.005-0.01 | Low | 1h |
| Multi-scale TTA (224 + 336) | +0.005-0.01 | Low | 1h |
| Confidence-weighted attribute masking | +0.005-0.01 | Low | 1h |

**Expected Server Score**: **0.58-0.60 mADM** (marginal improvement over 0.5720)

**Time Required**: 2-4 hours coding, no training needed

**Risk**: Low for individual improvements, but combining them can regress (as we saw with DBA+zoom crop).

> [!WARNING]
> Strategy C alone will NOT reach Top 10. It can only add +0.01-0.03 to our 0.5720 baseline. We MUST do Strategy A or B to have any chance.

---

## 3. Recommended Execution Plan

### Day 1 (Oct 18): Strategy C + Strategy B Preparation
1. **Morning**: Implement k-reciprocal reranking and gallery whitening in run.py → test locally → submit
2. **Afternoon**: Write Strategy B training code (Swin-Base/EVA02-Large attribute classifier)
3. **Evening**: Launch Strategy B training on Kaggle (runs overnight)

### Day 2 (Oct 19): Strategy B Submission + Strategy A Preparation  
1. **Morning**: Download Strategy B checkpoint → package → submit
2. **Afternoon**: Write Strategy A training code (CLIP ViT-L/14 + natural language templates)
3. **Evening**: Launch Strategy A training on Kaggle (runs overnight)

### Day 3 (Oct 20): Strategy A Submission + Best Ensemble
1. **Morning**: Download Strategy A checkpoint → package → submit
2. **Afternoon**: If A scores well, try ensemble of A+B
3. **Evening**: Final fine-tuning if needed

### Day 4 (Oct 21): Final Submissions
1. Submit best model with optimal post-processing
2. Try k-reciprocal reranking variants on best model
3. Reserve 1 submission for Oct 22 deadline day

---

## 4. What NOT to Do (Learned from Mistakes)

1. ❌ **Do NOT add more custom unpublished modules** — every custom module we added (ConvPass, DMHA, PCNorm, AGCA) was wasted engineering
2. ❌ **Do NOT use Alpha-DBA, Zoom Crop TTA, or PCA Whitening** — proven to hurt
3. ❌ **Do NOT expand cascade_k beyond 300** — injects noise
4. ❌ **Do NOT train from scratch** — always fine-tune from pretrained weights
5. ❌ **Do NOT combine multiple untested improvements in one submission** — test individually first
6. ❌ **Do NOT spend time on hyperbolic embeddings** — no evidence they help this task
