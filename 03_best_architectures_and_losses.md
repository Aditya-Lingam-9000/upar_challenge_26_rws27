# Report 3: Best Architectures, Models & Losses for Next Training

---

## 1. Backbone Selection (Visual Encoder)

### Tier 1: Best for Our Task (Recommended)

| Backbone | Params | Embed Dim | FP16 Size | Market-1501 R1 | Notes |
|----------|--------|-----------|-----------|----------------|-------|
| **CLIP ViT-L/14** | 304M | 1024 | ~580MB | 95.7% (CLIP-ReID) | **Best VL alignment**, has text encoder |
| **EVA02-Large-Patch14-CLIP** | 304M | 1024 | ~580MB | ~96% | Best MIM+CLIP pretrained |
| **SigLIP So400m/14** | 400M | 1152 | ~760MB | Not tested | Sigmoid loss, may be too large for ZIP |

### Tier 2: Good Alternatives (Fits 500MB ZIP)

| Backbone | Params | Embed Dim | FP16 Size | Market-1501 R1 | Notes |
|----------|--------|-----------|-----------|----------------|-------|
| **Swin-Base** | 88M | 1024 | ~168MB | 96.9% (SOLIDER) | Proven in SOLIDER |
| **CLIP ViT-B/16** | 86M | 512 | ~164MB | 93.8% (CLIP-ReID) | Smaller but proven |
| **EVA02-Base-Patch16-CLIP** | 86M | 768 | ~164MB | ~94% | What we currently use |

### Tier 3: Too Large / Not Recommended

| Backbone | Why Not |
|----------|---------|
| CLIP ViT-H/14 (632M) | Exceeds 500MB ZIP limit |
| InternViT-6B | Way too large |
| DINOv2 ViT-L | No text encoder, pure visual |

> [!IMPORTANT]
> **Recommended**: **CLIP ViT-L/14** — it has both a strong visual encoder AND a text encoder for cross-modal alignment. FP16 checkpoint is ~580MB which is tight but feasible with the submission ZIP (model weights only, no optimizer state). If ZIP is too large, fall back to **Swin-Base** or **CLIP ViT-B/16** which both fit easily.

---

## 2. Framework Selection (Training Pipeline)

### Option A: IRRA-Style (CVPR 2023) — RECOMMENDED

**Paper**: "Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval" (Jiang et al., CVPR 2023)

**Architecture**:
```
                    Binary Attributes (40-dim)
                            │
                    Template Expansion
                            │
                    "An adult female with
                     short hair wearing..."
                            │
                    ┌───────┴───────┐
                    │  CLIP Text    │
                    │  Encoder      │
                    │  (Frozen/FT)  │
                    └───────┬───────┘
                            │
                     Text Embedding
                     (768 or 1024-D)
                            │                        Gallery Image
                            │                            │
                    ┌───────┴───────┐           ┌────────┴────────┐
                    │   Text        │           │   CLIP Visual   │
                    │   Projection  │           │   Encoder       │
                    │   Head        │           │   (Fine-tuned)  │
                    └───────┬───────┘           └────────┬────────┘
                            │                            │
                     Query Embed                  Image Embed
                     (512 or 1024-D)              (512 or 1024-D)
                            │                            │
                            └──────── Cosine ────────────┘
                                   Similarity
```

**Losses**:
1. **SDM (Similarity Distribution Matching)**: Aligns image-text similarity distributions with KL-divergence
2. **ITC (Image-Text Contrastive)**: InfoNCE contrastive loss between image and text embeddings
3. **Multi-label BCE**: Auxiliary attribute classification on visual features

**Published Benchmarks**:
- CUHK-PEDES: Rank-1 = 73.38%, mAP = 66.13%
- ICFG-PEDES: Rank-1 = 63.46%
- RSTPReid: Rank-1 = 60.20%

### Option B: CLIP-ReID Style — SIMPLER ALTERNATIVE

**Architecture**: Fine-tune CLIP visual encoder with ID loss + text prompts

**Published Benchmarks**:
- Market-1501: Rank-1 = 93.8%, mAP = 89.8%
- MSMT17: Rank-1 = 85.4%, mAP = 71.6%

### Option C: Pure Multi-Label Attribute Classifier — SAFEST

**Architecture**: Standard backbone + multi-label classification heads, no cross-modal

---

## 3. Loss Functions (Ranked by Importance)

### Must-Have Losses

| Loss | Formula | Purpose | Expected Impact |
|------|---------|---------|----------------|
| **Multi-Label BCE** | $\ell = -\sum_{a=1}^{40} [y_a \log(\sigma(z_a)) + (1-y_a)\log(1-\sigma(z_a))]$ | Attribute prediction accuracy | **Core** — directly determines mADM |
| **Asymmetric Loss (ASL)** | $\ell_{ASL} = -\sum [y(1-p)^{\gamma+} \log p + (1-y)\max(p-m, 0)^{\gamma-} \log(1-p)]$ | Handles severe class imbalance (e.g., Bald: 1.26%, Hat: 4.25%) | +2-5% attribute accuracy |
| **SDM Loss** | $\ell_{SDM} = \text{KL}(p_{\text{sim}} \| p_{\text{target}})$ | Cross-modal alignment without hard mining | +3-5% Rank-1 |
| **ITC / InfoNCE** | $\ell_{ITC} = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$ | Contrastive alignment of paired embeddings | +2-4% Rank-1 |

### Recommended Loss Weights

```python
total_loss = 1.0 * loss_bce_asl      # Primary: attribute accuracy
           + 0.5 * loss_sdm           # Cross-modal alignment  
           + 0.3 * loss_itc           # Contrastive alignment
           + 0.1 * loss_triplet       # Hard-negative mining (optional)
```

### Losses to AVOID

| Loss | Why Avoid |
|------|-----------|
| **ArcFace on attribute clusters** | Overfits to training attribute combinations |
| **Sinkhorn OT** | Too expensive, unstable, unproven on this task |
| **Hyperbolic Loss** | No evidence of benefit, adds complexity |

---

## 4. Training Recipe

### Hyperparameters (Proven Configuration)

```yaml
# Optimizer
optimizer: AdamW
weight_decay: 0.05
betas: [0.9, 0.999]

# Learning Rate Schedule
backbone_lr: 5e-6        # Low for pretrained trunk
head_lr: 5e-4             # Higher for new heads  
scheduler: CosineAnnealingLR
warmup_epochs: 5
total_epochs: 80

# Batch Size (Dual T4 GPUs)
batch_size_per_gpu: 64     # With gradient accumulation 2
effective_batch_size: 256

# Data Augmentation
random_horizontal_flip: 0.5
random_erasing: 0.5        # REA is proven +1-2% Rank-1
color_jitter: 0.3
label_smoothing: 0.1       # Proven +0.5-1% on ReID

# Surveillance Degradation (Critical for Domain Gap)
random_downsample: [64, 96]  # Simulate low-res surveillance
jpeg_compression: [30, 70]   # Simulate video compression
gaussian_blur: [0.1, 2.0]    # Simulate camera defocus
```

### Training Schedule

```
Epochs 1-5:   Warmup (lr: 0 → max), only new heads unfrozen
Epochs 5-20:  Full fine-tuning, high lr
Epochs 20-60: Cosine decay, moderate lr
Epochs 60-80: Fine-tuning at very low lr (1e-6)
```

---

## 5. Attribute-to-Text Template Design

This is **critical** for CLIP-based approaches. The template must produce natural language that CLIP's text encoder can understand.

### Template Structure
```python
def attrs_to_text(attrs: np.ndarray) -> str:
    """Convert 40-dim binary attribute vector to natural language description."""
    parts = []
    
    # Age (indices 0-2)
    if attrs[0]: parts.append("young")
    elif attrs[1]: parts.append("adult")
    elif attrs[2]: parts.append("old")
    
    # Gender (index 3)
    gender = "female" if attrs[3] else "male"
    parts.append(gender)
    
    # Hair (indices 4-6)
    if attrs[4]: parts.append("with short hair")
    elif attrs[5]: parts.append("with long hair")
    elif attrs[6]: parts.append("bald")
    
    # Upper body (indices 7-19)
    upper_len = "short-sleeved" if attrs[7] else "long-sleeved"
    upper_colors = []
    color_names = ["black", "blue", "brown", "green", "grey", 
                   "orange", "pink", "purple", "red", "white", "yellow", "other-colored"]
    for i, c in enumerate(color_names):
        if attrs[8 + i]:
            upper_colors.append(c)
    upper_color = " ".join(upper_colors[:2]) if upper_colors else ""
    parts.append(f"wearing a {upper_color} {upper_len} top")
    
    # Lower body (indices 20-34)
    lower_len = "short" if attrs[20] else "long"
    lower_colors = []
    for i, c in enumerate(color_names):
        if attrs[21 + i]:
            lower_colors.append(c)
    lower_color = " ".join(lower_colors[:2]) if lower_colors else ""
    
    lower_type = "trousers" if attrs[33] else "skirt" if attrs[34] else "pants"
    parts.append(f"and {lower_color} {lower_len} {lower_type}")
    
    # Accessories (indices 35-39)
    accs = []
    if attrs[35]: accs.append("a backpack")
    if attrs[36]: accs.append("a bag")
    if attrs[37]: accs.append("glasses")
    if attrs[38]: accs.append("sunglasses")
    if attrs[39]: accs.append("a hat")
    if accs:
        parts.append(f"carrying {', '.join(accs)}")
    
    return f"A surveillance photo of a {' '.join(parts)}"
```

### Example Outputs
```
[0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,...] 
→ "A surveillance photo of a adult female with short hair wearing a black top and black long trousers"

[1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,...,1,0,0,0,0]
→ "A surveillance photo of a young male with long hair wearing a white short-sleeved top and blue short pants carrying a backpack"
```

---

## 6. Inference Pipeline (Simplified, Proven)

```python
# Step 1: Extract gallery features with Flip TTA
feat = normalize(model(img) + model(flip(img)))
logits = 0.5 * (logits_orig + logits_flip)

# Step 2: Calibrate attribute probabilities to dataset priors
probs = prior_matching_calibration(logits, UPAR_PRIORS)

# Step 3: Compute attribute similarity (L1 distance)
S_attr = 1.0 - (sum|q - p|) / 40.0

# Step 4: Compute metric similarity (cosine)
S_metric = (q @ g.T + 1.0) / 2.0

# Step 5: Fuse with alpha=0.70
S_final = 0.70 * S_attr + 0.30 * S_metric

# Step 6: Optional: k-reciprocal reranking on S_final
```

> [!TIP]
> The fusion weight α=0.70 (attribute-heavy) is proven optimal from the blueprint's ablation table. This reflects mADM's emphasis on attribute matching quality over identity matching.
