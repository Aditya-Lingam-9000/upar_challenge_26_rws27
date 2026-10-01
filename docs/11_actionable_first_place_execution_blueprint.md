# Report 11: Actionable First-Place Execution Blueprint: Step-by-Step Implementation Guide to Win Track 2

**Document:** Tactical Engineering & Execution Manual  
**Target:** 1st Place Standing in Track 2 (ABPR @ WACV 2027 / Codabench #18196)  
**Target Score:** $\mathbf{mADM \ge 0.6800}$ (Surpassing Current #1 `MaxwellEQ` @ $0.6280$)  
**Hardware Setup:** Apple M1 Pro (16GB RAM) + Scalable Cloud GPU (Colab/Kaggle/RunPod)  
**Document Status:** Complete Implementation & Execution Blueprint  

---

## 1. Executive Master Plan: The 4-Stage Trajectory to 1st Place

To take 1st place on the live leaderboard and maintain it through official code verification, we execute a 4-stage engineering progression:

```
+---------------------------------------------------------------------------------------------------------+
|                                    4-STAGE TRAJECTORY TO 1ST PLACE                                      |
+-------------------+---------------------+-----------------------+---------------------------------------+
| Milestone         | Expected mADM Score | Target Timeline       | Primary Engineering Focus             |
+-------------------+---------------------+-----------------------+---------------------------------------+
| Stage 1: Baseline | 0.5000 - 0.5300     | Days 1 - 4 (Oct 1-4)  | Data pipeline, local mADM engine,     |
|                   |                     |                       | ConvNeXt baseline, first Codabench sub|
+-------------------+---------------------+-----------------------+---------------------------------------+
| Stage 2: C2T-Net  | 0.6000 - 0.6300     | Days 5 - 9 (Oct 5-9)  | Swin-B + ViT-B dual backbone, CASA    |
|                   |                     |                       | attention, Asymmetric Loss (ASL)      |
+-------------------+---------------------+-----------------------+---------------------------------------+
| Stage 3: CLEAR    | 0.6500 - 0.6700     | Days 10 - 15(Oct10-15)| Pseudo-text descriptions, soft+hard   |
|   Dual-Stream     |                     |                       | query adapters, angular margin loss   |
+-------------------+---------------------+-----------------------+---------------------------------------+
| Stage 4: Ensemble | 0.6800 - 0.7200     | Days 16 - 20(Oct16-20)| Multi-model blending, TTA, metric     |
|   & Win Lock      | (1ST PLACE)         |                       | calibration, submission lock-in       |
+-------------------+---------------------+-----------------------+---------------------------------------+
```

---

## 2. Phase 1 Immediate Action Items (Next 24–48 Hours)

### Step 2.1: Establish Python 3.11 Virtual Environment on macOS (Apple Silicon M1 Pro)
The Codabench container runs Python 3.11 with PyTorch 2.4.1. To avoid version divergence, configure the local environment with identical dependencies:

```bash
# In project root: /Users/sama/D/RWS2027
cd /Users/sama/D/RWS2027

# Create clean virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install core deep learning packages matching Codabench runtime
pip install torch==2.4.1 torchvision==0.19.1 numpy==1.26.4 pillow==10.4.0

# Install development and scientific utilities
pip install tqdm pyyaml matplotlib scipy

# Verify Apple Metal Performance Shaders (MPS) availability
python3 -c "import torch; print('PyTorch Version:', torch.__version__); print('MPS Available:', torch.backends.mps.is_available())"
```

### Step 2.2: Unpack and Harmonize the Datasets from `RWS27_Data`
The raw archives downloaded in `RWS27_Data` must be unpacked into `UPAR-Challenge-2027/data/`:

```bash
# Ensure target data directory exists
mkdir -p UPAR-Challenge-2027/data

# 1. Extract Market-1501
unzip -q "RWS27_Data/Market/Market-1501-v15.09.15.zip" -d UPAR-Challenge-2027/data/
mv UPAR-Challenge-2027/data/Market-1501-v15.09.15 UPAR-Challenge-2027/data/Market1501

# 2. Extract PA-100K
mkdir -p UPAR-Challenge-2027/data/PA100k
unzip -q "RWS27_Data/PA-100K Dataset/data.zip" -d UPAR-Challenge-2027/data/PA100k/

# 3. Extract PETA and organize images via peta_file_mapping.txt
mkdir -p UPAR-Challenge-2027/data/PETA/images
unzip -q "RWS27_Data/PETA Dataset/PETA.zip" -d UPAR-Challenge-2027/data/PETA/temp_peta

# Run organizing script using the official starter kit mapping
python3 -c "
import shutil, os, numpy as np
from pathlib import Path

mapping_file = 'UPAR-Challenge-2027/peta_file_mapping.txt'
data_dir = Path('UPAR-Challenge-2027/data')
mapping = {row[0]: row[1] for row in np.genfromtxt(mapping_file, dtype=str, delimiter=',')}

src_base = data_dir / 'PETA' / 'temp_peta'
for file_path in src_base.glob('*/*/*/*'):
    if file_path.suffix == '.txt':
        continue
    rel_key = str(file_path.relative_to(src_base))
    # match against mapping
    for orig_key, dest_rel in mapping.items():
        if orig_key.endswith(file_path.name):
            target = data_dir / dest_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file_path), str(target))
            break

shutil.rmtree(str(src_base), ignore_errors=True)
print('PETA dataset organized successfully.')
"
```

### Step 2.3: Verify Annotation Integrity
Check that all image paths in `UPAR-Challenge-2027/data/annotations/train.csv` and `val.csv` exist on disk:

```bash
python3 -c "
import csv, os

def check_csv(csv_path, base_dir):
    missing = 0
    total = 0
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            total += 1
            img_rel = row[0]
            if not os.path.exists(os.path.join(base_dir, img_rel)):
                missing += 1
    print(f'{csv_path}: {total} total, {missing} missing images.')

check_csv('UPAR-Challenge-2027/data/annotations/train.csv', 'UPAR-Challenge-2027/data')
check_csv('UPAR-Challenge-2027/data/annotations/val.csv', 'UPAR-Challenge-2027/data')
"
```

---

## 3. The Offline Metric Engine: Exact Local Replicas of Codabench Scoring

To optimize models without wasting daily submission limits on Codabench (limit: 3/day), we build a local metric engine that reproduces the competition's scoring program down to the 6th decimal place.

### File: `src/metrics/retrieval_metrics.py`

```python
"""
Local Validation Engine: Exactly replicates Codabench Scoring Program
Computes: mADM, mAP, Rank-1, Rank-5, Rank-10, mINP
"""
import numpy as np


def compute_metrics(similarities: np.ndarray, 
                    queries: np.ndarray, 
                    gallery_gt: np.ndarray) -> dict[str, float]:
    """
    Args:
        similarities: (Q, G) float array. Higher value = better match.
        queries: (Q, 40) binary ground truth query attribute matrix.
        gallery_gt: (G, 40) binary ground truth gallery attribute matrix.
    Returns:
        dict containing: mADM, mAP, Rank-1, Rank-5, Rank-10, mINP
    """
    Q, G = similarities.shape
    assert queries.shape == (Q, 40)
    assert gallery_gt.shape == (G, 40)

    # 1. Compute Degree of Match (dom) for every (query, gallery) pair: (Q, G)
    # dom[i, j] = fraction of 40 attributes where gallery j equals query i
    # ||q - y||_1 = sum(y) + q @ (1 - 2y).T
    l1_diff = gallery_gt.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * gallery_gt).T
    dom = 1.0 - (l1_diff / 40.0)  # (Q, G) in [0, 1]

    # Ground truth exact matches: all 40 attributes match (dom == 1.0)
    exact_matches = (l1_diff == 0)  # Boolean array (Q, G)

    # 2. Compute Normalized Degree of Match (ndom): (Q, G)
    mean_dom = dom.mean(axis=1, keepdims=True)  # (Q, 1)
    max_dom = dom.max(axis=1, keepdims=True)   # (Q, 1) guaranteed to be 1.0
    denom = np.maximum(max_dom - mean_dom, 1e-8)
    ndom = np.maximum(0.0, (dom - mean_dom) / denom)

    # 3. Sort gallery candidates for each query (Descending similarity)
    # Using argsort with negative similarity (stable sort preserves gallery order on ties)
    ranking = np.argsort(-similarities, axis=1)  # (Q, G) indices

    ap_list = []
    adm_list = []
    inp_list = []
    rank1_count = 0
    rank5_count = 0
    rank10_count = 0

    for i in range(Q):
        q_order = ranking[i]
        q_matches = exact_matches[i, q_order]      # (G,) boolean
        q_ndom = ndom[i, q_order]                  # (G,) float

        # Find 1-based ranks of exact matches: r_1 < r_2 < ... < r_R
        ranks = np.where(q_matches)[0] + 1         # 1-indexed ranks
        R = len(ranks)

        if R == 0:
            continue  # Every query in challenge has at least 1 match

        # Rank-k indicators
        min_rank = ranks[0]
        if min_rank <= 1:
            rank1_count += 1
        if min_rank <= 5:
            rank5_count += 1
        if min_rank <= 10:
            rank10_count += 1

        # AP = (1/R) * sum_{k=1}^R (k / r_k)
        precisions = np.arange(1, R + 1) / ranks
        ap = precisions.mean()
        ap_list.append(ap)

        # INP = R / r_R
        inp = R / ranks[-1]
        inp_list.append(inp)

        # ADM = (1/R) * sum_{k=1}^R [ (sum_{j=1}^{r_k} ndom(j)) / r_k ]
        cum_ndom = np.cumsum(q_ndom)
        adm_terms = cum_ndom[ranks - 1] / ranks
        adm = adm_terms.mean()
        adm_list.append(adm)

    return {
        "mADM": float(np.mean(adm_list)),
        "mAP": float(np.mean(ap_list)),
        "Rank-1": float(rank1_count / Q),
        "Rank-5": float(rank5_count / Q),
        "Rank-10": float(rank10_count / Q),
        "mINP": float(np.mean(inp_list)),
    }
```

---

## 4. Stage 1: Build the ConvNeXt-Base Baseline (Days 1–4)

Before training complex transformer networks, establish a rock-solid baseline model. This serves as our submission sanity test and will later act as the CNN component in our multi-model ensemble.

### 4.1 Architecture: ConvNeXt-Base Multi-Label Network
```python
"""
File: src/models/convnext_baseline.py
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


class ConvNeXtPAR(nn.Module):
    def __init__(self, num_attributes: int = 40, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Base_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_base(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        
        # Replace 1000-class ImageNet head with 40-attribute multi-label head
        self.backbone.classifier[2] = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_attributes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        logits = self.head(feat)
        return logits
```

### 4.2 Training Protocol
- **Loss:** Weighted Binary Cross-Entropy (pos_weight scaled by empirical positive rate).
- **Optimizer:** AdamW, $\text{lr} = 1\text{e-}4$, $\text{weight\_decay} = 1\text{e-}2$.
- **Scheduler:** Cosine Annealing with 3 epochs warmup.
- **Image Size:** $256 \times 128$.
- **Expected Outcome:** Reaching $\mathbf{\approx 0.50 - 0.52\text{ mADM}}$ on validation and Codabench.

---

## 5. Stage 2: Build C2T-Net Dual-Branch Cross-Transformers (Days 5–9)

C2T-Net is our primary backbone. It won 1st Place in Track 1 of UPAR 2024 and forms the visual core of our system.

### 5.1 Architecture: Swin-B + ViT-B with Channel-Aware Self-Attention (CASA)

```python
"""
File: src/models/c2t_backbone.py
Dual-Branch Cross-Transformer with CASA and Cross-Fused Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CASA(nn.Module):
    """Channel-Aware Self-Attention Module"""
    def __init__(self, channels: int, spatial_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Transpose spatial and channel dimensions
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return x + out  # Residual connection


class CrossFusedAttention(nn.Module):
    """Cross-Fusion between Swin local tokens and ViT global tokens"""
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, token_seq_a: torch.Tensor, token_seq_b: torch.Tensor) -> torch.Tensor:
        # Cross attention: tokens_a attend over tokens_b
        q = self.q_proj(token_seq_a)
        k = self.k_proj(token_seq_b)
        v = self.v_proj(token_seq_b)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        return token_seq_a + out
```

### 5.2 Training C2T-Net with Asymmetric Loss (ASL)
Train C2T-Net on the 97,669 training images using Asymmetric Loss to resolve class imbalance:

```python
"""
File: src/losses/asymmetric_loss.py
"""
import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, 
                 clip_margin: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: (B, 40)
        probs = torch.sigmoid(logits)
        
        # Positive loss
        pos_loss = targets * (1.0 - probs) ** self.gamma_pos * torch.log(probs.clamp(min=self.eps))
        
        # Negative loss with probability shifting
        shifted_neg_probs = (probs - self.clip_margin).clamp(min=0.0)
        neg_loss = (1.0 - targets) * (shifted_neg_probs ** self.gamma_neg) * torch.log((1.0 - shifted_neg_probs).clamp(min=self.eps))
        
        loss = - (pos_loss + neg_loss).sum(dim=1)
        return loss.mean()
```
- **Target Performance at End of Stage 2:** Validation $\text{mA} \ge 81.5\%$, Retrieval $\mathbf{mADM \ge 0.6250}$.

---

## 6. Stage 3: Build the CLEAR Dual-Stream Metric Retrieval System (Days 10–15)

Stage 3 bridges the modality gap between raw bit-vector queries and visual images using language embeddings and angular margin metric learning.

```
                           STAGE 3 DUAL-STREAM ARCHITECTURE
+------------------------------------+             +------------------------------------+
|  Query Vector q in {0, 1}^40       |             |  Pedestrian Image: g in R^(HxW)    |
+------------------------------------+             +------------------------------------+
       |                             |                               |
       |                             |                               v
       v                             v             +------------------------------------+
+-------------+      +---------------+             |  Frozen C2T-Net Backbone (f_par)   |
| Hard Vector |      | Natural Text  |             +------------------------------------+
|   q (40-D)  |      | Pseudo-Prompt |                               |
+-------------+      +---------------+                               v
       |                             |             +------------------------------------+
       v                             v             | Visual Retrieval Adapter f_vis_ret |
+-------------+      +---------------+             +------------------------------------+
| f_attr_ret  |      | SigLIP / CLIP |                               |
|   (256-D)   |      | Text Encoder  |                               v
+-------------+      +---------------+             +------------------------------------+
       |                             |             | Normalized Visual Embedding E^v    |
       |                             v             +------------------------------------+
       |                     +---------------+                               |
       |                     |  f_text_ret   |                               |
       |                     |    (256-D)    |                               |
       |                     +---------------+                               |
       |                             |                                       |
       +--------------+--------------+                                       |
                      |                                                      |
                      v                                                      |
       +-----------------------------+                                       |
       | Fused Query Embedding E^q   |                                       |
       |    E^q = [E^h ; E^s]        |                                       |
       +-----------------------------+                                       |
                      |                                                      |
                      +----------------------+-------------------------------+
                                             |
                                             v
                           +------------------------------------+
                           | Additive Angular Margin Loss       |
                           | ArcFace-Style: s=30.0, margin=0.20 |
                           +------------------------------------+
```

### 6.1 Pseudo-Description Generation Engine
Convert each 40-dimensional binary query into a natural language sentence:

```python
"""
File: src/utils/pseudo_caption.py
"""
ATTRIBUTE_NAMES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun", "Accessory-Hat"
]

def generate_pseudo_caption(q: list[int] | np.ndarray) -> str:
    # 1. Age & Gender
    age = "adult"
    if q[0] == 1: age = "young"
    elif q[2] == 1: age = "old"
    gender = "woman" if q[3] == 1 else "man"

    # 2. Hair
    hair = "short hair"
    if q[5] == 1: hair = "long hair"
    elif q[6] == 1: hair = "bald head"

    # 3. Upper body clothing
    upper_len = "short-sleeved" if q[7] == 1 else "long-sleeved"
    upper_colors = []
    color_names = ["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white", "yellow", "other colored"]
    for idx, cname in enumerate(color_names):
        if q[8 + idx] == 1:
            upper_colors.append(cname)
    upper_color_str = "/".join(upper_colors) if upper_colors else "neutral"

    # 4. Lower body clothing
    lower_len = "short" if q[20] == 1 else "long"
    lower_colors = []
    for idx, cname in enumerate(color_names):
        if q[21 + idx] == 1:
            lower_colors.append(cname)
    lower_color_str = "/".join(lower_colors) if lower_colors else "neutral"
    lower_type = "trousers" if q[33] == 1 else "skirt or dress"

    # 5. Accessories
    accs = []
    if q[35] == 1: accs.append("backpack")
    if q[36] == 1: accs.append("bag")
    if q[37] == 1: accs.append("glasses")
    if q[38] == 1: accs.append("sunglasses")
    if q[39] == 1: accs.append("hat")
    acc_str = f"with {', '.join(accs)}" if accs else "without accessories"

    caption = (
        f"A surveillance photo of a {age} {gender} with {hair}, "
        f"wearing a {upper_color_str} {upper_len} top and {lower_color_str} {lower_len} {lower_type}, "
        f"{acc_str}."
    )
    return caption
```

### 6.2 Additive Angular Margin Metric Loss
Train the retrieval adapters ($f^{vis}_{ret}, f^{text}_{ret}, f^{attr}_{ret}$) with ArcFace-style angular loss:

```python
"""
File: src/losses/angular_margin_loss.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularMarginLoss(nn.Module):
    def __init__(self, scale: float = 30.0, margin: float = 0.20):
        super().__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, visual_embeddings: torch.Tensor, query_embeddings: torch.Tensor) -> torch.Tensor:
        # Both embeddings must be L2-normalized: (B, D)
        cos_theta = visual_embeddings @ query_embeddings.T  # (B, B)
        
        # Ground truth diagonal pairs (i, i) are positive matches
        B = visual_embeddings.size(0)
        target = torch.arange(B, device=visual_embeddings.device)
        
        # Apply additive angular margin to positive matches: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sin_theta = torch.sqrt((1.0 - torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7) ** 2))
        cos_theta_m = cos_theta * np.cos(self.margin) - sin_theta * np.sin(self.margin)
        
        # Select margin cosine for positive pairs, keep original cosine for negatives
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output_logits = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output_logits = output_logits * self.scale
        
        loss = F.cross_entropy(output_logits, target)
        return loss
```
- **Target Performance at End of Stage 3:** Validation $\mathbf{mADM \ge 0.6550 - 0.6700}$.

---

## 7. Stage 4: Multi-Model Ensembling & Winning Submission Lock-In (Days 16–20)

To guarantee beating the current 1st place (`MaxwellEQ` @ 0.6280) on the hidden test set, we construct a 3-model heterogeneous ensemble.

### 7.1 The Tri-Model Ensemble Composition
1. **Model A (50% Weight):** C2T-Net (Swin-B + ViT-B) with CLEAR Dual-Stream Adapters.
2. **Model B (30% Weight):** ConvNeXt-Base trained with Asymmetric Focal Loss.
3. **Model C (20% Weight):** SigLIP-PAR with region-aware prompt alignment.

### 7.2 Ensemble Inference Engine in `run.py`

```python
# In run.py inside rank_gallery:
# Blend predicted probabilities across models
P_ens = 0.50 * P_modelA + 0.30 * P_modelB + 0.20 * P_modelC

# Blend metric embeddings across models
E_v_ens = 0.50 * E_v_modelA + 0.30 * E_v_modelB + 0.20 * E_v_modelC
E_v_norm = E_v_ens / (np.linalg.norm(E_v_ens, axis=1, keepdims=True) + 1e-8)

# Compute fused similarity
l1_dist = P_ens.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * P_ens).T
S_attr = 1.0 - (l1_dist / 40.0)

E_q_norm = E_q / (np.linalg.norm(E_q, axis=1, keepdims=True) + 1e-8)
S_emb = (E_q_norm @ E_v_norm.T + 1.0) / 2.0

# 70/30 Optimal Fusion
similarities = 0.70 * S_attr + 0.30 * S_emb
return {"similarities": similarities.astype(np.float32)}
```

---

## 8. Computing Workflow: Local Apple M1 Pro (16GB) vs. Cloud Acceleration

To complete training and testing within the 21-day timeline, structure compute as follows:

```
+---------------------------------------------------------------------------------------------------------+
|                                    COMPUTE ALLOCATION DECISION TREE                                     |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Task Description         | Platform              | Execution Time      | Resource Cost                  |
+--------------------------+-----------------------+---------------------+--------------------------------+
| Code debugging & linting | Local M1 Pro          | Instantaneous       | Free                           |
| Local data extraction    | Local M1 Pro          | 5 minutes           | Free                           |
| Local metric engine test | Local M1 Pro          | 15 seconds          | Free                           |
| Single image infer check | Local M1 Pro (MPS)    | 2 seconds           | Free                           |
| Adapter fine-tuning      | Local M1 Pro (MPS)    | 4 minutes           | Free                           |
| Full C2T-Net Training    | Cloud (Colab/RunPod)  | 2 hours             | Free (Colab) / $0.80 (RunPod)  |
| 3-Fold LODO Training     | Cloud (Kaggle/RunPod) | 4.5 hours           | Free (Kaggle) / $1.80 (RunPod) |
| Container Mock Docker    | Local Docker / Cloud  | 4 minutes           | Free                           |
+--------------------------+-----------------------+---------------------+--------------------------------+
```

---

## 9. Codabench Air-Gapped Packaging Checklist

Before uploading any archive to Codabench, execute this pre-flight verification:

```bash
# 1. Navigate to submission directory
cd my_submission

# 2. Check root contents (MUST contain run.py and metadata.yaml at root)
ls -la run.py metadata.yaml

# 3. Clean OS artifacts and Python bytecode
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.DS_Store" -delete
find . -name "*.pyc" -delete

# 4. Check serialized checkpoint sizes (Ensure FP16 weights are used)
ls -lh weights/
# Target: weights should be under 350 MB total

# 5. Build archive
zip -r ../submission_v1.zip . -x "*/__pycache__/*" "*.DS_Store" ".git/*"

# 6. Verify zip root layout (run.py must NOT be nested in a subfolder)
unzip -l ../submission_v1.zip | head -n 10
```

---

## 10. The 21-Day Action Calendar: Exactly What to Do Each Day

- **Day 1 (Today, Sep 30):**
  - Create Python 3.11 virtual environment `venv` on M1 Pro.
  - Extract Market-1501, PA-100K, and PETA from `RWS27_Data/`.
  - Implement `retrieval_metrics.py` (Local Evaluation Engine).
- **Day 2 (Oct 01):**
  - Verify that local metric engine correctly reproduces sample baseline score ($0.1567\text{ mADM}$).
  - Implement `convnext_baseline.py` and run 5 epochs locally to verify gradient flow.
- **Day 3 (Oct 02):**
  - Train ConvNeXt-Base on Cloud GPU for 25 epochs.
  - Evaluate on local validation set. Target: $\text{mADM} \approx 0.50$.
- **Day 4 (Oct 03):**
  - Package ConvNeXt-Base into `submission_stage1.zip`.
  - **Submit to Codabench (Submission 1).** Verify container logs and establish our first leaderboard score.
- **Day 5 (Oct 04):**
  - Implement C2T-Net dual-branch backbone (Swin-B + ViT-B with CASA).
  - Implement Asymmetric Loss (`asymmetric_loss.py`) with hard negative margin.
- **Day 6 (Oct 05):**
  - Launch C2T-Net training on Cloud GPU (RTX 4090 / A100).
- **Day 7 (Oct 06):**
  - Evaluate C2T-Net checkpoint. Target PAR $\text{mA} \ge 81.5\%$.
  - Evaluate raw attribute retrieval. Target: $\text{mADM} \approx 0.61 - 0.62$.
- **Day 8 (Oct 07):**
  - Freeze C2T-Net backbone.
  - Implement `pseudo_caption.py` and pre-extract text embeddings for all queries.
- **Day 9 (Oct 08):**
  - Train retrieval adapters ($f^{vis}_{ret}, f^{text}_{ret}, f^{attr}_{ret}$) on M1 Pro using Additive Angular Margin Loss.
- **Day 10 (Oct 09):**
  - Tune fusion parameter $\alpha$ ($S = \alpha S_{\text{attr}} + (1-\alpha) S_{\text{emb}}$) on validation data.
- **Day 11 (Oct 10):**
  - Package C2T-Net + CLEAR adapters into `submission_stage2.zip`.
  - **Submit to Codabench (Submission 2).** Target score: $\mathbf{\ge 0.6350\text{ mADM}}$ (Taking 1st Place).
- **Day 12 (Oct 11):**
  - Set up 3-Fold Leave-One-Domain-Out (LODO) cross-validation to test cross-dataset robustness.
- **Day 13 (Oct 12):**
  - Add surveillance data augmentation (reflection padding, random erasing, hue limits).
- **Day 14 (Oct 13):**
  - Implement Test-Time Augmentation (TTA: horizontal flip) inside `run.py`.
- **Day 15 (Oct 14):**
  - Train complementary models: ConvNeXt-Base (ASL) and SigLIP-PAR.
- **Day 16 (Oct 15):**
  - Execute LODO cross-validation audit. Select top checkpoints with minimum domain variance.
- **Day 17 (Oct 16):**
  - Package updated single model with TTA.
  - **Submit to Codabench (Submission 3).** Target: $\mathbf{\ge 0.6550\text{ mADM}}$.
- **Day 18 (Oct 17):**
  - Construct Tri-Model Ensemble (C2T-Net + ConvNeXt + SigLIP).
  - Verify ensemble runtime is $< 1,200\text{ seconds}$ inside local Docker sandbox.
- **Day 19 (Oct 18):**
  - Perform local stress-testing with memory leak checks.
- **Day 20 (Oct 19):**
  - **Submit Final Winning Ensemble to Codabench (Final Submission).**
  - Target score: $\mathbf{\ge 0.6800\text{ mADM}}$ (**Locking 1st Place**).
- **Day 21 (Oct 20–21 AoE):**
  - Verify leaderboard standing.
  - Archive git commits, training scripts, and FP16 checkpoints for organizer verification.
- **Post-Challenge (Oct 22–23 AoE):**
  - Write and submit the 8-page WACV 2027 Workshop Technical Paper via OpenReview.
- **Post-Challenge (Oct 24–30 AoE):**
  - Submit the verified code and fact sheet for the official winner audit.
