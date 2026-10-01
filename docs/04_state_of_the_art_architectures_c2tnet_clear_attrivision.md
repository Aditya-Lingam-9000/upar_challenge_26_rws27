# Report 04: State-of-the-Art Architectures: Deconstruction of C2T-Net, CLEAR, AttriVision & Modern PAR/ABPR Systems

**Document:** Architectural Analysis & Literature Deconstruction  
**Key Frameworks Analyzed:** C2T-Net (WACVW 2024), CLEAR (Pattern Recognition 2025), AttriVision (WACVW 2025), SOLIDER (CVPR 2023), PromptPAR (2024), VLM-PAR (2025/2026), UniPAR (2026)  
**Document Status:** Publication-Grade Architectural Blueprint  

---

## 1. Architectural Evolution in Pedestrian Attribute Recognition & Retrieval

Historically, pedestrian attribute recognition relied on standard Convolutional Neural Networks (ResNet-50) augmented with spatial attention masks or Part-based Convolutional Baselines (PCB). However, between 2023 and 2026, the literature experienced a major paradigm shift driven by:
1. **Vision Transformers (ViT & Swin):** Capturing long-range contextual spatial relationships across body parts.
2. **Channel-Spatial Dual Attention:** Disentangling overlapping attributes.
3. **Cross-Modal Vision-Language Alignment:** Framing attribute retrieval not as Euclidean distance between bit vectors, but as multi-modal matching between visual features and natural language semantic embeddings.

```
                  CHRONOLOGICAL ARCHITECTURAL EVOLUTION (2023 - 2026)
---------------------------------------------------------------------------------------
2023: ConvNeXt Baseline (Specker et al.)
      ResNet-50 / ConvNeXt-Base + FC Head + Weighted BCE + EMA
      Result: Outperformed all complex ad-hoc CNNs in 2023/2024.
         |
2024: C2T-Net (Doanh C. Bui et al. - 1st Place UPAR 2024 Track 1)
      Dual-Branch: Swin Transformer (Local) + Vanilla ViT (Global)
      CASA (Channel-Aware Attention) + SVCF/VSCF Cross-Attention Fusion.
         |
2024/2025: AttriVision (Mehran Adibi Sedeh et al. - WACVW 2025)
      Reformulates PAR as Image-Text Retrieval using CLIP ViT-B/16.
      Attribute prompts + Focal Cross-Entropy for class imbalance.
         |
2025: CLEAR (Doanh C. Bui et al. - Pattern Recognition 2025 / arXiv:2403.06119)
      Unified PAR & Retrieval: Cross-Transformers + Pseudo-Text Descriptions
      Hard + Soft Query Embeddings (GPT/SigLIP) + Additive Angular Margin Learning.
         |
2026: VLM-PAR & UniPAR
      Unified multi-modal transformers with frozen SigLIP 2 encoders & late deep fusion.
---------------------------------------------------------------------------------------
```

---

## 2. In-Depth Deconstruction of C2T-Net (1st Place UPAR 2024 Track 1)

C2T-Net (*Channel-Aware Cross-Fused Transformer-Style Networks for Pedestrian Attribute Recognition*) was introduced by Doanh C. Bui et al. and won 1st Place in Track 1 of the UPAR 2024 Challenge.

```
                                  C2T-NET DUAL-BRANCH PIPELINE
                                 +----------------------------+
                                 |   Input Image (256 x 128)  |
                                 +----------------------------+
                                                |
                        +-----------------------+-----------------------+
                        |                                               |
                        v                                               v
        +-------------------------------+               +-------------------------------+
        |    Swin Transformer Branch    |               |   Vanilla ViT / EVA Branch    |
        |  (Hierarchical Local Windows) |               |    (Full Sequence Global)     |
        +-------------------------------+               +-------------------------------+
                        |                                               |
                        v                                               v
        +-------------------------------+               +-------------------------------+
        |   Channel-Aware SA (CASA)     |               |    Multi-Head Self-Attention  |
        | Transpose(Group) -> Attn -> Up|               |       Global [CLS] Token      |
        +-------------------------------+               +-------------------------------+
                        |                                               |
                        +-----------------------+-----------------------+
                                                |
                                                v
                               +---------------------------------+
                               |    Cross-Fused Self-Attention   |
                               |    - SVCF: Swin attends to ViT  |
                               |    - VSCF: ViT attends to Swin  |
                               +---------------------------------+
                                                |
                                                v
                               +---------------------------------+
                               |   Independent LayerNorm (LNs, LNv|
                               +---------------------------------+
                                                |
                                                v
                               +---------------------------------+
                               |    Multi-Label Classifier Head  |
                               |      (40 Sigmoid Predictions)   |
                               +---------------------------------+
```

### 2.1 The Core Insight of C2T-Net
The authors recognized that pedestrian attributes operate at different spatial scales:
- **Global Attributes:** Gender, Age, and Body Build require an understanding of overall proportions and context across the entire image sequence.
- **Local Attributes:** Shoes, Bag, Glasses, Hat, and Hair require high-resolution local attention within constrained spatial windows.

Vanilla ViT excels at global long-range dependencies, while Swin Transformer excels at fine-grained local dependencies. C2T-Net unifies both within a parallel dual-branch network.

### 2.2 Channel-Aware Self-Attention (CASA)
Standard self-attention computes dependencies across spatial patch tokens:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
This models spatial interactions, but ignores correlations across feature channels (e.g., color channels versus edge/texture channels). 

CASA transposes the feature map $z_s \in \mathbb{R}^{C \times H \times W}$ into a grouped matrix $z_s^T \in \mathbb{R}^{(H \cdot W) \times C}$, allowing the spatial positions to act as keys and queries that compute attention weights across feature channels:
$$att_{\text{CASA}} = \text{softmax}\left(\frac{Q_{\text{CASA}} K_{\text{CASA}}^T}{\sqrt{d_c}}\right) \in \mathbb{R}^{(H \cdot W) \times (H \cdot W)}$$
$$z_s^{\text{CASA}} = z_s^T + att_{\text{CASA}} \odot z_s^T$$
This mechanism dynamically boosts feature channels that correspond to active attribute regions while suppressing background clutter.

### 2.3 Cross-Fused Self-Attention (SVCF & VSCF)
At the final stage of the dual backbones:
- **Swin-to-ViT Cross-Fusion (SVCF):** The flattened spatial feature vector of the Swin branch attends to the complete token sequence of the Vanilla ViT branch:
  $$SVCF = \text{Attention}(Q_{\text{Swin}}, K_{\text{ViT}}, V_{\text{ViT}})$$
- **ViT-to-Swin Cross-Fusion (VSCF):** The `[CLS]` token from the ViT branch attends across the spatial token grid of the Swin branch:
  $$VSCF = \text{Attention}(Q_{\text{ViT}}, K_{\text{Swin}}, V_{\text{Swin}})$$
- **Independent Layer Normalization:** Because the two branches capture distinct feature distributions, independent LayerNorm operations ($LN_s$ and $LN_v$) are applied before concatenation:
  $$\mathbf{z} = \text{Concat}(LN_s(\mathbf{z}_s), LN_v(\mathbf{z}_v))$$
  $$\hat{\mathbf{y}} = \sigma(W_{\text{head}} \mathbf{z} + b)$$

---

## 3. In-Depth Deconstruction of CLEAR (Pattern Recognition 2025)

CLEAR (*Cross-Transformers with Pre-trained Language Model is All you need for Person Attribute Recognition and Retrieval*), authored by Doanh C. Bui, Thinh V. Le, Ba Hung Ngo, and Tae Jong Choi (arXiv:2403.06119, published in *Pattern Recognition* 2025), is the **most directly relevant prior work** for UPAR Track 2.

### 3.1 The Two-Phase Training Paradigm
CLEAR unifies PAR and ABPR into a staged, decoupled architecture:
1. **Phase 1 (PAR Backbone Training):** Train the dual-branch Cross-Transformer backbone on multi-label attribute classification using binary cross-entropy and label smoothing.
2. **Phase 2 (Retrieval Adapter Training via Margin Learning):** Freeze the trained Cross-Transformer backbone ($f_{\text{par}}$). Attach lightweight projection adapters ($f^{vis}_{ret}$, $f^{text}_{ret}$, $f^{attr}_{ret}$) and train them with Additive Angular Margin Loss to align visual person embeddings with query embeddings.

### 3.2 Dual-Stream Query Representation: Soft + Hard Embeddings
Rather than feeding only the 40-dimensional binary query into an MLP, CLEAR extracts complementary semantics:

#### 1. Hard Query Embedding ($E^h$)
The discrete 40-bit one-hot encoded vector $\mathbf{h} \in \{0, 1\}^{40}$ is projected through adapter $f^{attr}_{ret}$ (a 3-layer MLP with ReLU):
$$E^h = f^{attr}_{ret}(\mathbf{h}) \in \mathbb{R}^{d_{\text{query}}}$$

#### 2. Soft Pseudo-Description Query ($E^s$)
Discrete attributes are converted into natural descriptive syntax using a deterministic text template:
```text
"This is a photo of [AGE] [GENDER] with [HAIR LENGTH] hair, 
dressed in a [LOWER COLOR] [LOWER BODY CLOTHING] and a [UPPER COLOR] [UPPER BODY CLOTHING] 
with [SLEEVES LENGTH] sleeves, wearing [ACCESSORIES], carrying [BAG] and [BACKPACK]."
```
- The generated sentence is passed into a pre-trained language model (GPT / CLIP text encoder) to obtain word token embeddings $\{s_1, s_2, \dots, s_{N_w}\}$.
- Token features are pooled/flattened and projected through adapter $f^{text}_{ret}$:
  $$E^s = f^{text}_{ret}(\mathbf{s}) \in \mathbb{R}^{d_{\text{query}}}$$
- **Combined Query Vector:**
  $$E^q = [E^h \,\|\, E^s] \in \mathbb{R}^{2 \cdot d_{\text{query}}}$$

### 3.3 Additive Angular Margin Learning
To align visual features with query embeddings, CLEAR adopts ArcFace-style angular margin learning:
$$\mathcal{L}_{\text{margin}} = -\log \frac{e^{s \cdot (\cos(\theta_{y}) - m)}}{e^{s \cdot (\cos(\theta_{y}) - m)} + \sum_{j \neq y} e^{s \cdot \cos(\theta_j)}}$$
where $\theta$ is the angle between the normalized visual feature $E^v = f^{vis}_{ret}(z_{\text{img}})$ and query embedding $E^q$, scale factor $s=30$, and margin $m=0.2$.

### 3.4 CLEAR's Published Results on UPAR-2024
On the UPAR-2024 official test set (which defeated all competitors during the 2024 challenge):
- CLEAR surpassed the official baseline by **+2.9% in Rank-1** and **+1.4% in mAP**.
- On Market-1501 retrieval, CLEAR achieved state-of-the-art: **Rank-1 = 56.8%**, **mAP = 44.0%**.

---

## 4. In-Depth Deconstruction of AttriVision (WACVW 2025) & PromptPAR

AttriVision (*Advancing Generalization in Pedestrian Attribute Recognition using CLIP*, Sedeh et al., WACV 2025 Workshop) explores cross-dataset generalization using Vision-Language Models.

```
                          ATTRIVISION / PROMPTPAR MECHANISM
    +------------------------+                     +------------------------+
    | Pedestrian Image Crop  |                     |  40 Attribute Prompts  |
    | (Upper, Lower, Full)   |                     | "a photo of a person   |
    +------------------------+                     |  wearing black pants"  |
                |                                  +------------------------+
                v                                               |
    +------------------------+                                  v
    |   CLIP Image Encoder   |                     +------------------------+
    |     (ViT-B / 16)       |                     |   CLIP Text Encoder    |
    +------------------------+                     +------------------------+
                |                                               |
                +-----------------------+-----------------------+
                                        |
                                        v
                       +---------------------------------+
                       | Cross-Modal Multi-Label Head    |
                       |    Focal Cross-Entropy Loss     |
                       +---------------------------------+
```

### 4.1 Reformulating Attribute Classification as Multi-Modal Retrieval
AttriVision re-frames the classification of attribute $a$ as computing cosine similarity between the image embedding $v \in \mathbb{R}^D$ and textual prompt embeddings:
$$t_a^+ = \text{TextEncoder}(\text{"a pedestrian with } [a]\text{"})$$
$$t_a^- = \text{TextEncoder}(\text{"a pedestrian without } [a]\text{"})$$
The probability that attribute $a$ is present is computed via temperature-scaled softmax:
$$p_a = \frac{e^{\langle v, t_a^+ \rangle / \tau}}{e^{\langle v, t_a^+ \rangle / \tau} + e^{\langle v, t_a^- \rangle / \tau}}$$

### 4.2 Handling Class Imbalance via Focal Cross-Entropy
To prevent ubiquitous classes from dominating gradient updates, AttriVision incorporates Focal Cross-Entropy:
$$\mathcal{L}_{\text{Focal}} = -\sum_{a=1}^{40} \alpha_a (1 - p_a)^\gamma \log(p_a)$$
with focusing parameter $\gamma = 2.0$ and inverse class frequency weighting $\alpha_a$.

---

## 5. Architectural Comparison Matrix

```
=========================================================================================================
                                  COMPARATIVE ARCHITECTURAL MATRIX
=========================================================================================================
Feature / Metric        ConvNeXt Baseline       C2T-Net (2024)       CLEAR (2025)         AttriVision (2025)
---------------------------------------------------------------------------------------------------------
Backbone Architecture   ConvNeXt-Base           Swin-B + ViT-B       Swin-B + ViT-B       CLIP ViT-B/16
Feature Fusion          Global Average Pool     SVCF + VSCF Cross    Cross-Transformers   Cross-Attention
Channel Attention       None                    CASA Module          CASA Module          None
Query Representation    Raw Bit-Vector (40-D)   Raw Bit-Vector (40-D)Dual: Hard + Soft    Text Prompts
Language Integration    None                    None                 Pretrained LM (GPT)  CLIP Text Encoder
Metric Learning Loss    None (Direct L1)        None (Direct L1)     Angular Margin Loss  Focal CE
Test-Domain Robustness  Moderate (0.584 mADM)   High                 Very High            High
Submittable in Container Yes (Single Backbone)  Yes (Vendored)       Yes (Vendored)       Yes (Vendored)
=========================================================================================================
```

---

## 6. Synthesis: The Optimal Architecture for 1st Place

To achieve the winning score ($> 0.68\text{ mADM}$), our system must synthesize the strengths of each prior work:
1. **From C2T-Net:** Adopt the dual-branch Swin-B + ViT-B backbone with CASA and cross-fused attention to extract local and global pedestrian features.
2. **From CLEAR:** Implement the dual-stream retrieval adapters ($f^{vis}_{ret}, f^{text}_{ret}, f^{attr}_{ret}$), generate pseudo-descriptions, and optimize additive angular margin loss.
3. **From AttriVision:** Integrate SigLIP / CLIP text representations and focal loss principles to handle extreme attribute imbalance.
4. **Dynamic Fusion:** Output both continuous attribute probabilities and dense cross-modal embeddings, combining them into a joint similarity score that eliminates ties and maximizes mADM.
