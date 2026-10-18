"""V4 CLIP ViT-L/14 Championship Pipeline: Models & Prompter Architectures.

This module provides the neural network architectures for the WACV 2027 UPAR Challenge
Track 2 (Attribute-based Person Retrieval) V4 Championship Pipeline:
1. `attrs_to_text`: Synthesizes fluent, grammatically sound English descriptions from 40-D
   binary attribute vectors for OpenCLIP text supervision.
2. `CLIPAttributeRetrievalModel`: ViT-L/14 visual backbone with dual-head outputs:
   - 768-dim metric projection for cross-modal similarity alignment.
   - 40-dim multi-label attribute classification logits for ASL supervision.
3. `ContinuousAttributePrompter`: Multi-layer query encoder mapping 40 binary/continuous
   attributes to the 768-dim metric space with learned attribute axes and multi-temperature ensembling.
4. `InferenceVisualModel`: Lightweight inference wrapper instantiated without pretrained weights
   to ensure air-gapped test-time execution with exact state_dict compatibility.
5. Constants: `CANONICAL_ATTRIBUTES` and `UPAR_ATTR_PRIORS`.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Canonical 40 UPAR Attributes & Empirical Training Priors
# =====================================================================
CANONICAL_ATTRIBUTES: List[str] = [
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

UPAR_ATTR_PRIORS: np.ndarray = np.array([
    0.0110, 0.9650, 0.0237, 0.4292, 0.6546, 0.3114, 0.0126, 0.6288, 0.2556, 0.0821,
    0.0199, 0.0435, 0.1621, 0.0058, 0.0105, 0.0253, 0.0694, 0.2250, 0.0497, 0.0753,
    0.3751, 0.4877, 0.1788, 0.0448, 0.0150, 0.1719, 0.0021, 0.0155, 0.0008, 0.0039,
    0.0447, 0.0017, 0.0372, 0.9038, 0.0958, 0.2085, 0.2931, 0.1394, 0.0218, 0.0425
], dtype=np.float32)

_COLOR_NAMES: List[str] = [
    "black", "blue", "brown", "green", "grey", "orange",
    "pink", "purple", "red", "white", "yellow", "multi-colored"
]


# =====================================================================
# 1. Attribute to Natural Language Prompt Synthesizer
# =====================================================================
def attrs_to_text(attrs: Union[np.ndarray, torch.Tensor, List[Union[int, float]]]) -> str:
    """Converts a 40-dimensional binary attribute vector into a fluent English description.

    Produces fluent English descriptions following the canonical template:
    "A surveillance photo of an adult female with short hair wearing a black short-sleeved top
    and blue long trousers carrying a backpack."

    Args:
        attrs: 40-dimensional attribute vector (binary {0, 1} or continuous in [0, 1]).
               Accepts np.ndarray, torch.Tensor, or List of numbers.

    Returns:
        Fluent English sentence describing the pedestrian.

    Raises:
        ValueError: If `attrs` does not contain exactly 40 elements.
    """
    if isinstance(attrs, torch.Tensor):
        vec = attrs.detach().cpu().numpy().flatten()
    else:
        vec = np.asarray(attrs, dtype=np.float32).flatten()

    if vec.size != 40:
        raise ValueError(f"Expected 40 attributes, got vector with {vec.size} elements.")

    # Threshold probabilities / soft continuous values at 0.5
    b = (vec >= 0.5).astype(bool)

    # 1. Demographics: Age and Gender
    has_demo = any(b[:4])
    if has_demo:
        is_female = b[3]
        gender_str = "female" if is_female else "male"

        if b[0]:
            age_str = "young"
        elif b[2]:
            age_str = "older"
        elif b[1]:
            age_str = "adult"
        else:
            age_str = None

        if age_str is not None:
            article = "an" if age_str in ("adult", "older") else "a"
            person_str = f"{article} {age_str} {gender_str}"
        else:
            article = "a"
            person_str = f"{article} {gender_str}"
    else:
        person_str = "a pedestrian"

    # 2. Hair characteristics
    hair_str = ""
    if b[6]:  # Bald
        hair_str = "who is bald"
    elif b[4]:  # Short hair
        hair_str = "with short hair"
    elif b[5]:  # Long hair
        hair_str = "with long hair"

    # 3. Upper body clothing (indices 7 to 19)
    upper_colors = [_COLOR_NAMES[i] for i in range(12) if b[8 + i]]
    has_upper = b[7] or len(upper_colors) > 0

    upper_desc = ""
    if has_upper:
        color_phrase = " and ".join(upper_colors) if upper_colors else ""
        len_phrase = "short-sleeved" if b[7] else "long-sleeved"
        parts = [color_phrase, len_phrase, "top"]
        upper_desc = " ".join([p for p in parts if p])

    # 4. Lower body clothing (indices 20 to 34)
    lower_colors = [_COLOR_NAMES[i] for i in range(12) if b[21 + i]]
    is_lower_short = b[20]
    is_skirt = b[34]
    has_lower = is_lower_short or b[33] or b[34] or len(lower_colors) > 0

    lower_desc = ""
    if has_lower:
        color_phrase = " and ".join(lower_colors) if lower_colors else ""
        if is_skirt:
            l_type = "short skirt" if is_lower_short else "long skirt"
        else:
            l_type = "shorts" if is_lower_short else "long trousers"
        parts = [color_phrase, l_type]
        lower_desc = " ".join([p for p in parts if p])

    # Combine clothing clauses
    clothing_str = ""
    if upper_desc and lower_desc:
        clothing_str = f"wearing a {upper_desc} and {lower_desc}"
    elif upper_desc:
        clothing_str = f"wearing a {upper_desc}"
    elif lower_desc:
        clothing_str = f"wearing {lower_desc}"

    # 5. Accessories: Carrying & Wearing
    carrying: List[str] = []
    if b[35]:  # Backpack
        carrying.append("a backpack")
    if b[36]:  # Bag
        carrying.append("a bag")

    wearing_acc: List[str] = []
    if b[39]:  # Hat
        wearing_acc.append("a hat")
    if b[38]:  # Sunglasses
        wearing_acc.append("sunglasses")
    elif b[37]:  # Glasses-Normal
        wearing_acc.append("glasses")

    acc_phrases: List[str] = []
    if wearing_acc:
        acc_phrases.append(f"wearing {' and '.join(wearing_acc)}")
    if carrying:
        acc_phrases.append(f"carrying {' and '.join(carrying)}")

    # Assemble complete description sentence
    clauses: List[str] = [f"A surveillance photo of {person_str}"]
    if hair_str:
        clauses.append(hair_str)
    if clothing_str:
        clauses.append(clothing_str)
    if acc_phrases:
        clauses.extend(acc_phrases)

    return " ".join(clauses) + "."


# =====================================================================
# 2. CLIP ViT-L/14 Dual-Head Retrieval Model (Training)
# =====================================================================
class CLIPAttributeRetrievalModel(nn.Module):
    """V4 Championship Training Model with ViT-L/14 CLIP visual backbone.

    Extracts high-capacity visual representations from 224x224 pedestrian crops:
    1. Metric Head: Projects the 1024-dim CLS token to a 768-dim L2-normalized shared
       metric space for cross-modal similarity alignment with OpenCLIP ViT-L/14 text.
    2. Attribute Head: Predicts 40 multi-label attribute classification logits from
       the 1024-dim CLS token under Asymmetric Loss (ASL) supervision.

    Args:
        backbone_name: Name of the timm pretrained ViT backbone. Defaults to
                       'vit_large_patch14_clip_224.openai' (304M parameters).
        embed_dim: Dimension of the shared multimodal metric space (768).
        num_attrs: Number of target binary attributes (40).
        pretrained: Whether to download/load official pretrained weights.
    """

    def __init__(
        self,
        backbone_name: str = "vit_large_patch14_clip_224.openai",
        embed_dim: int = 768,
        num_attrs: int = 40,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim
        self.num_attrs = num_attrs

        # 304M parameter ViT-L/14 CLIP visual transformer (1024 feature dim)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = getattr(self.backbone, "num_features", 1024)

        # 1024 -> 768 shared multimodal metric projection head
        self.metric_head = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 1024 -> 512 -> 40 multi-label pedestrian attribute prediction head
        self.attr_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_attrs),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ViT-L/14 backbone and dual heads.

        Args:
            images: [B, 3, 224, 224] pedestrian RGB image tensor.

        Returns:
            Tuple of:
            - metric_embed: [B, embed_dim] L2-normalized metric embeddings.
            - attr_logits: [B, num_attrs] Raw classification logits before sigmoid.
        """
        feats = self.backbone(images)  # [B, 1024]
        metric_raw = self.metric_head(feats)  # [B, embed_dim]
        metric_embed = F.normalize(metric_raw, p=2, dim=-1)
        attr_logits = self.attr_head(feats)  # [B, num_attrs]
        return metric_embed, attr_logits

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward pass to maintain unified API across pipelines."""
        return self.forward(images)


# =====================================================================
# 3. Continuous Multi-Attribute Manifold Prompter
# =====================================================================
class ContinuousAttributePrompter(nn.Module):
    """Query encoder mapping 40 binary/continuous attributes into 768-dim metric space.

    Employs a dual-pathway architecture:
    1. Weighted Axis Projection: Maps bipolarized attribute indicators [-1, +1] through
       learnable continuous attribute axes in R^hidden_dim weighted by empirical surprisal.
    2. Direct Projection: Direct 3-layer MLP projection preserving global attribute combinations.
    3. Multi-Temperature Test-Time Ensembling: Generates sharpened query representations
       across diverse temperature scales to maximize retrieval precision.

    Args:
        num_attrs: Number of pedestrian attributes (40).
        embed_dim: Target shared metric space dimension (768).
        hidden_dim: Intermediate hidden dimension (512).
    """

    def __init__(
        self,
        num_attrs: int = 40,
        embed_dim: int = 768,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_attrs = num_attrs
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 40 Learnable continuous semantic attribute axes: [40, hidden_dim]
        self.attr_axes = nn.Parameter(torch.randn(num_attrs, hidden_dim) * 0.02)

        # Information-theoretic attribute prior weights w_a (normalized surprisal)
        self.register_buffer("attr_weights", torch.ones(num_attrs, dtype=torch.float32))

        # Manifold Refinement Network: hidden_dim -> hidden_dim -> embed_dim
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Direct Projection Network: num_attrs -> hidden_dim -> embed_dim
        self.direct_proj = nn.Sequential(
            nn.Linear(num_attrs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Initialize with empirical UPAR training set priors
        self.set_empirical_weights(UPAR_ATTR_PRIORS)

    def set_empirical_weights(self, frequencies: np.ndarray) -> None:
        """Sets information-theoretic surprisal weights w_a = sqrt(-log2(p_a)).

        Rare attributes (e.g. Bald 1.26%, Hat 4.25%) receive higher weights,
        while ultra-frequent attributes (e.g. Adult 96.5%) are softly downweighted.

        Args:
            frequencies: 1D array of shape [num_attrs] containing training frequencies.
        """
        p = np.clip(np.asarray(frequencies, dtype=np.float32), 1e-4, 1.0 - 1e-4)
        surprisal = -np.log2(p)
        w = np.sqrt(surprisal)
        w = w / np.mean(w)
        self.attr_weights.copy_(
            torch.from_numpy(w).to(dtype=self.attr_weights.dtype, device=self.attr_weights.device)
        )

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """Encodes attribute vectors into L2-normalized 768-dim metric representations.

        Args:
            attrs: [B, 40] continuous or binary attribute vectors in [0, 1].

        Returns:
            [B, embed_dim] L2-normalized query embeddings.
        """
        if attrs.dim() == 1:
            attrs = attrs.unsqueeze(0)

        attrs = attrs.to(device=self.attr_axes.device, dtype=self.attr_axes.dtype)
        weights = self.attr_weights.to(device=attrs.device, dtype=attrs.dtype)

        # Map binary [0, 1] -> [-1, 1] bipolar orientation
        bipolar_q = 2.0 * attrs - 1.0  # [B, 40]

        # Weighted attribute combinations: [B, 40]
        weighted_q = bipolar_q * weights.unsqueeze(0)

        # Semantic axis aggregation: [B, 40] @ [40, hidden_dim] -> [B, hidden_dim]
        cont_embed = torch.matmul(weighted_q, self.attr_axes)

        # Non-linear refinement + direct residual pathway
        refined = self.refiner(cont_embed)
        direct = self.direct_proj(attrs)

        fused = refined + direct
        return F.normalize(fused, p=2, dim=-1)

    def forward_multi_temp(
        self,
        attrs: torch.Tensor,
        temps: List[float] = [1.25, 1.80, 2.50, 4.50],
        weights: List[float] = [0.25, 0.35, 0.25, 0.15],
    ) -> torch.Tensor:
        """Multi-temperature attribute sharpening ensemble for test-time query expansion.

        Evaluates query representations across temperature scales and blends them with
        prespecified ensemble weights, restoring prior state upon completion.

        Args:
            attrs: [B, 40] attribute vectors.
            temps: List of temperature scale exponents.
            weights: List of corresponding mixture weights (must sum to ~1.0).

        Returns:
            [B, embed_dim] L2-normalized ensembled query embedding.
        """
        orig_weights = self.attr_weights.clone()
        embeds: List[torch.Tensor] = []

        for t, w in zip(temps, weights):
            scaled_w = torch.pow(orig_weights.abs() + 1e-8, 2.0 / t)
            scaled_w = scaled_w / scaled_w.mean().clamp(min=1e-6)
            self.attr_weights.copy_(scaled_w)
            embeds.append(w * self.forward(attrs))

        # Restore original attribute weights
        self.attr_weights.copy_(orig_weights)

        fused = torch.stack(embeds, dim=0).sum(dim=0)
        return F.normalize(fused, p=2, dim=-1)


# =====================================================================
# 4. Lightweight Inference Visual Model (Air-Gapped Submission)
# =====================================================================
class InferenceVisualModel(nn.Module):
    """Lightweight inference wrapper for air-gapped test evaluation.

    Instantiates the exact same ViT-L/14 architecture, metric head, and attribute head
    as `CLIPAttributeRetrievalModel`, but sets `pretrained=False` and cleans the model
    name to avoid any network connection attempts during evaluation.

    Has 100% state_dict parameter key compatibility with `CLIPAttributeRetrievalModel`.

    Args:
        backbone_name: Architecture name without pretrained repository tags. Defaults
                       to 'vit_large_patch14_clip_224'.
        embed_dim: Dimension of metric space (768).
        num_attrs: Number of attributes (40).
    """

    def __init__(
        self,
        backbone_name: str = "vit_large_patch14_clip_224",
        embed_dim: int = 768,
        num_attrs: int = 40,
    ) -> None:
        super().__init__()
        # Strip repository identifier (e.g. '.openai') to guarantee offline initialization
        clean_backbone = backbone_name.replace(".openai", "")
        self.backbone_name = clean_backbone
        self.embed_dim = embed_dim
        self.num_attrs = num_attrs

        # ViT-L/14 backbone instantiated without downloading weights
        self.backbone = timm.create_model(clean_backbone, pretrained=False, num_classes=0)
        feat_dim = getattr(self.backbone, "num_features", 1024)

        # 1024 -> 768 metric head
        self.metric_head = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # 1024 -> 512 -> 40 attribute head
        self.attr_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_attrs),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extracts metric embeddings and attribute logits.

        Args:
            images: [B, 3, 224, 224] normalized image tensor.

        Returns:
            Tuple of:
            - metric_embed: [B, embed_dim] L2-normalized embeddings.
            - attr_logits: [B, num_attrs] Classification logits.
        """
        feats = self.backbone(images)
        metric_raw = self.metric_head(feats)
        metric_embed = F.normalize(metric_raw, p=2, dim=-1)
        attr_logits = self.attr_head(feats)
        return metric_embed, attr_logits

    def encode_image(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward pass."""
        return self.forward(images)
