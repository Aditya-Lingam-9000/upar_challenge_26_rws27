"""Stage 2 Standalone Submission Packaging Engine for Codabench Track 2.

Builds an air-gapped, zero-internet-dependent zip submission packaging:
  - Swin-T backbone (or ConvNeXt-Tiny) + Multi-Head PAR Classifier
  - Stage 2 VisualProjector (768 -> 512) and Dual-Branch QueryEncoder ([E_h || E_s] -> 512)
  - Hardcoded Platt calibration parameters (T_a, b_a)
  - Hybrid Scorer (Attribute L1 + Metric Cosine Distance)
  - Pure PyTorch, zero external dependencies, robust edge-case handling.
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
import numpy as np
import torch


def generate_stage2_run_py(
    model_type: str = "swin_t",
    lambda_metric: float = 0.25,
    height: int = 256,
    width: int = 128,
    T_params: list[float] | None = None,
    b_params: list[float] | None = None,
) -> str:
    """Generate self-contained pure PyTorch run.py for Stage 2 submission."""

    T_str = repr(list(T_params)) if T_params is not None else "[1.5] * 40"
    b_str = repr(list(b_params)) if b_params is not None else "[1.0] * 40"

    code = f'''"""Codabench Track 2 Submission Entrypoint: Stage 2 Metric Learning + Calibrated PAR.

Strictly compatible with pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime.
Zero external package dependencies beyond torch, torchvision, numpy, and PIL.
Conforms to Codabench Ingestion Contract:
  - load_model()
  - rank_gallery(sample: dict) -> {{"distances": np.ndarray}}
"""
from __future__ import annotations

import os
from typing import Any
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

# Canonical list of 40 UPAR attributes in challenge training order
CANONICAL_ATTRIBUTES = [
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
    "Accessory-Glasses-Sun", "Accessory-Hat",
]

# Hardcoded fitted Platt calibration parameters
PLATT_T = np.array({T_str}, dtype=np.float32)
PLATT_B = np.array({b_str}, dtype=np.float32)

LAMBDA_METRIC = {lambda_metric}
IMG_HEIGHT = {height}
IMG_WIDTH = {width}

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BACKBONE_WEIGHTS = os.path.join(ASSET_DIR, "model_backbone.pth")
METRIC_HEADS_WEIGHTS = os.path.join(ASSET_DIR, "metric_heads.pth")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_BACKBONE: nn.Module | None = None
_VIS_PROJECTOR: nn.Module | None = None
_QUERY_ENCODER: nn.Module | None = None


# Architecture Definitions
class SwinPAR(nn.Module):
    def __init__(self, num_attributes: int = 40):
        super().__init__()
        self.backbone = models.swin_t(weights=None)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_attributes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits


class VisualProjector(nn.Module):
    def __init__(self, in_features: int = 768, embedding_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return F.normalize(feat, p=2, dim=-1)


class QueryEncoder(nn.Module):
    def __init__(self, num_attributes: int = 40, embedding_dim: int = 512):
        super().__init__()
        half_dim = embedding_dim // 2
        self.hard_mlp = nn.Sequential(
            nn.Linear(num_attributes, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, half_dim),
            nn.LayerNorm(half_dim, eps=1e-6),
        )
        self.attr_token_embeddings = nn.Parameter(torch.randn(num_attributes, half_dim) * 0.02)
        self.semantic_adapter = nn.Sequential(
            nn.LayerNorm(half_dim, eps=1e-6),
            nn.Linear(half_dim, half_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(half_dim, half_dim),
            nn.LayerNorm(half_dim, eps=1e-6),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        e_h = self.hard_mlp(q)
        attr_weights = q / (q.sum(dim=-1, keepdim=True).clamp(min=1.0))
        semantic_pooled = torch.matmul(attr_weights, self.attr_token_embeddings)
        e_s = self.semantic_adapter(semantic_pooled)
        e_q = torch.cat([e_h, e_s], dim=-1)
        return F.normalize(e_q, p=2, dim=-1)


_INFERENCE_TRANSFORMS = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model() -> None:
    \"\"\"Loads backbone and Stage 2 metric heads onto compute device.\"\"\"
    global _BACKBONE, _VIS_PROJECTOR, _QUERY_ENCODER
    if _BACKBONE is not None:
        return
    print(f"Loading Stage 2 model onto {{_DEVICE}}...", flush=True)

    # 1. Instantiate networks
    _BACKBONE = SwinPAR(num_attributes=40)
    _VIS_PROJECTOR = VisualProjector(in_features=768, embedding_dim=512)
    _QUERY_ENCODER = QueryEncoder(num_attributes=40, embedding_dim=512)

    # 2. Load backbone weights with multi-version PyTorch compatibility
    try:
        bb_sd = torch.load(BACKBONE_WEIGHTS, map_location="cpu", weights_only=True)
    except Exception:
        bb_sd = torch.load(BACKBONE_WEIGHTS, map_location="cpu")
    if isinstance(bb_sd, dict) and "model_state_dict" in bb_sd:
        bb_sd = bb_sd["model_state_dict"]
    bb_sd = {{k.replace("module.", ""): v for k, v in bb_sd.items() if not k.startswith("embedding_head.")}}
    _BACKBONE.load_state_dict(bb_sd, strict=False)

    # 3. Load metric head weights with multi-version PyTorch compatibility
    try:
        mh_sd = torch.load(METRIC_HEADS_WEIGHTS, map_location="cpu", weights_only=True)
    except Exception:
        mh_sd = torch.load(METRIC_HEADS_WEIGHTS, map_location="cpu")
    if "vis_projector" in mh_sd:
        _VIS_PROJECTOR.load_state_dict(mh_sd["vis_projector"], strict=False)
        _QUERY_ENCODER.load_state_dict(mh_sd["query_encoder"], strict=False)
    else:
        _VIS_PROJECTOR.load_state_dict({{k.replace("vis_projector.", ""): v for k, v in mh_sd.items() if "vis_projector" in k}}, strict=False)
        _QUERY_ENCODER.load_state_dict({{k.replace("query_encoder.", ""): v for k, v in mh_sd.items() if "query_encoder" in k}}, strict=False)

    _BACKBONE = _BACKBONE.to(_DEVICE).eval()
    _VIS_PROJECTOR = _VIS_PROJECTOR.to(_DEVICE).eval()
    _QUERY_ENCODER = _QUERY_ENCODER.to(_DEVICE).eval()
    print("Stage 2 model and metric heads successfully loaded and ready for inference.", flush=True)


@torch.no_grad()
def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    \"\"\"Official Codabench entrypoint.

    Args:
      sample: dict with:
        - "gallery": list of dicts with "image_path"
        - "attribute_names": list of 40 attribute names
        - "queries": (Q, 40) binary query array/list

    Returns:
      {{"distances": distances}}: (Q, G) float32 numpy array.
    \"\"\"
    global _BACKBONE, _VIS_PROJECTOR, _QUERY_ENCODER
    if _BACKBONE is None:
        load_model()

    gallery = sample.get("gallery", [])
    queries_raw = np.asarray(sample.get("queries", []), dtype=np.float32)
    queries_raw = np.nan_to_num(queries_raw, nan=0.0)
    queries_raw = np.clip(queries_raw, 0.0, 1.0)
    attribute_names = sample.get("attribute_names", CANONICAL_ATTRIBUTES)

    num_samples = len(gallery)
    num_queries = len(queries_raw)

    # Edge case 1: Empty gallery
    if num_samples == 0:
        return {{"distances": np.zeros((num_queries, 0), dtype=np.float32)}}

    # Edge case 2: Empty queries
    if num_queries == 0:
        return {{"distances": np.zeros((0, num_samples), dtype=np.float32)}}

    # Realign queries if attribute ordering differs
    if attribute_names != CANONICAL_ATTRIBUTES:
        print("Realigning query attributes to match model canonical ordering...", flush=True)
        name_to_col = {{name: idx for idx, name in enumerate(attribute_names)}}
        aligned_queries = np.zeros((num_queries, 40), dtype=np.float32)
        for canon_idx, name in enumerate(CANONICAL_ATTRIBUTES):
            if name in name_to_col:
                aligned_queries[:, canon_idx] = queries_raw[:, name_to_col[name]]
        queries = aligned_queries
    else:
        queries = queries_raw

    # Project queries to metric space: (Q, 512)
    q_tensor = torch.from_numpy(queries).float().to(_DEVICE)
    q_emb = _QUERY_ENCODER(q_tensor).cpu().numpy()

    # Process gallery in batches
    batch_size = 32
    all_calib_probs = []
    all_vis_embs = []
    use_autocast = _DEVICE.type == "cuda"

    print(f"Executing inference on {{num_samples}} gallery images (batch_size={{batch_size}})...", flush=True)

    for i in range(0, num_samples, batch_size):
        batch_items = gallery[i : i + batch_size]
        tensors = []

        for item in batch_items:
            img_path = item.get("image_path") or item.get("image") or ""
            try:
                with Image.open(img_path) as img:
                    tensors.append(_INFERENCE_TRANSFORMS(img.convert("RGB")))
            except Exception:
                # Graceful corrupt image fallback: neutral zero tensor
                tensors.append(torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32))

        batch_tensor = torch.stack(tensors, dim=0).to(_DEVICE)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats, logits = _BACKBONE(batch_tensor)
                v_embs = _VIS_PROJECTOR(feats)
        else:
            feats, logits = _BACKBONE(batch_tensor)
            v_embs = _VIS_PROJECTOR(feats)

        # Apply Platt calibration: p = sigma((logits - b) / T)
        logits_np = logits.float().cpu().numpy()
        calib_logits = np.clip((logits_np - PLATT_B[None, :]) / PLATT_T[None, :], -20.0, 20.0)
        probs = 1.0 / (1.0 + np.exp(-calib_logits))

        all_calib_probs.append(probs)
        all_vis_embs.append(v_embs.float().cpu().numpy())

        del batch_tensor, feats, logits, v_embs

        processed = i + len(batch_items)
        if (i // batch_size) % 50 == 0 or processed >= num_samples:
            if _DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{{processed}}/{{num_samples}}] Gallery processed ({{processed/num_samples*100:.1f}}%)", flush=True)

    probs = np.vstack(all_calib_probs).astype(np.float32)  # (G, 40)
    v_embs = np.vstack(all_vis_embs).astype(np.float32)     # (G, 512)

    # 1. Attribute continuous L1 distance: (Q, G)
    # sum_a |q_a - p_a| / 40.0 = (sum(p) + q @ (1 - 2p).T) / 40.0
    attr_dist = (probs.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * probs).T) / 40.0

    # 2. Metric cosine distance: 0.5 * (1 - q_emb @ v_emb.T): (Q, G)
    cos_sim = q_emb @ v_embs.T
    metric_dist = 0.5 * (1.0 - cos_sim)

    # 3. Hybrid fusion
    distances = (1.0 - LAMBDA_METRIC) * attr_dist + LAMBDA_METRIC * metric_dist

    # Sanitize distances: strictly no NaN/Inf
    distances = np.nan_to_num(distances, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
    print(f"Rank gallery complete. Distance matrix shape: {{distances.shape}}", flush=True)
    return {{"distances": distances}}
'''
    return code


def build_stage2_submission(
    output_zip: str = "submissions/EXP-006_STAGE2_SWIN_METRIC_submission.zip",
    swin_zip: str = "submissions/EXP-005A_submission.zip",
    metric_heads_path: str = "checkpoints/stage2_swin_metric_heads.pth",
    calibration_path: str = "checkpoints/stage1_calibration_params.npz",
    lambda_metric: float = 0.25,
) -> str:
    """Builds and packages self-contained Stage 2 submission zip."""
    print("=" * 80)
    print(f"BUILDING STAGE 2 SUBMISSION: {output_zip}")
    print("=" * 80)

    staging_dir = "/tmp/stage2_submission_staging"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(os.path.join(staging_dir, "assets"), exist_ok=True)

    # 1. Load calibration parameters
    calib_paths = [calibration_path, "assets/stage1_calibration_params.npz", "checkpoints/stage1_calibration_params.npz"]
    calib_file = next((cp for cp in calib_paths if os.path.exists(cp)), None)
    if calib_file is not None:
        calib = np.load(calib_file)
        T_params = calib["T_005"]
        b_params = calib["b_005"]
    else:
        T_params = [1.5287, 1.9120, 1.4530, 1.0267, 1.7007, 1.4921, 1.5512, 1.3312, 1.2643, 1.3411, 1.4912, 1.5231, 1.3482, 1.6021, 1.6432, 1.6082, 1.4451, 1.3452, 1.6071, 1.5291, 1.1451, 1.1952, 1.2341, 1.4481, 1.5781, 1.3352, 1.6441, 1.6571, 1.6352, 1.5221, 1.4601, 1.6131, 1.5081, 1.0362, 1.2691, 1.2381, 1.3621, 1.4771, 1.4691, 1.3241]
        b_params = [2.7781, 0.9276, 1.4545, 1.4719, 1.1888, 1.1121, 1.6231, 1.1541, 1.1921, 1.2191, 1.4131, 1.4721, 1.2481, 1.5431, 1.5841, 1.5491, 1.3711, 1.2441, 1.5481, 1.4641, 1.1741, 1.1631, 1.2041, 1.3831, 1.5081, 1.2441, 1.5861, 1.5991, 1.5781, 1.4611, 1.3971, 1.5551, 1.4441, 0.9711, 1.1831, 1.1671, 1.2981, 1.4131, 1.4041, 1.2631]

    # 2. Generate run.py
    run_py_content = generate_stage2_run_py(
        model_type="swin_t",
        lambda_metric=lambda_metric,
        height=256,
        width=128,
        T_params=T_params,
        b_params=b_params,
    )
    with open(os.path.join(staging_dir, "run.py"), "w", encoding="utf-8") as f:
        f.write(run_py_content)
    print("Generated self-contained run.py with embedded Platt calibration.")

    # 3. Generate metadata.yaml (canonical Codabench code submission format)
    metadata_yaml = """command: python3 -c "print('Submission package ready. The ingestion program imports run.py')"
"""
    with open(os.path.join(staging_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        f.write(metadata_yaml)

    # 4. Extract backbone weights from swin_zip
    print(f"Extracting backbone weights from {swin_zip}...")
    temp_zip_dir = "/tmp/extract_swin_weights"
    if os.path.exists(temp_zip_dir):
        shutil.rmtree(temp_zip_dir)
    with zipfile.ZipFile(swin_zip, "r") as zf:
        zf.extractall(temp_zip_dir)

    src_bb = os.path.join(temp_zip_dir, "assets", "model.pth")
    dst_bb = os.path.join(staging_dir, "assets", "model_backbone.pth")
    shutil.copy2(src_bb, dst_bb)
    shutil.rmtree(temp_zip_dir)
    print(f"Copied backbone weights ({os.path.getsize(dst_bb)/(1024*1024):.1f} MB).")

    # 5. Copy metric heads
    dst_mh = os.path.join(staging_dir, "assets", "metric_heads.pth")
    shutil.copy2(metric_heads_path, dst_mh)
    print(f"Copied Stage 2 metric heads ({os.path.getsize(dst_mh)/(1024*1024):.1f} MB).")

    # 6. Create zip package
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging_dir):
            for file in files:
                abs_f = os.path.join(root, file)
                rel_f = os.path.relpath(abs_f, staging_dir)
                zf.write(abs_f, rel_f)

    total_mb = os.path.getsize(output_zip) / (1024 * 1024)
    shutil.rmtree(staging_dir)
    print(f"\n>> Stage 2 Submission Package successfully built: {output_zip} ({total_mb:.1f} MB)")
    return output_zip


def generate_stage2_dual_run_py(
    lambda_metric: float = 0.25,
    height: int = 256,
    width: int = 128,
    T_cvx: list[float] | None = None,
    b_cvx: list[float] | None = None,
    T_swin: list[float] | None = None,
    b_swin: list[float] | None = None,
) -> str:
    """Generate self-contained pure PyTorch run.py for Stage 2 Dual-Stream Fused submission."""
    T_cvx_str = repr(list(T_cvx)) if T_cvx is not None else "[1.5] * 40"
    b_cvx_str = repr(list(b_cvx)) if b_cvx is not None else "[1.0] * 40"
    T_swin_str = repr(list(T_swin)) if T_swin is not None else "[1.5] * 40"
    b_swin_str = repr(list(b_swin)) if b_swin is not None else "[1.0] * 40"

    code = f'''"""Codabench Track 2 Submission Entrypoint: Dual-Stream Fused Metric Learning + Calibrated PAR.

Strictly compatible with pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime.
Zero external package dependencies beyond torch, torchvision, numpy, and PIL.
Conforms to Codabench Ingestion Contract:
  - load_model()
  - rank_gallery(sample: dict) -> {{"distances": np.ndarray}}
"""
from __future__ import annotations

import os
from typing import Any
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

# Canonical list of 40 UPAR attributes in challenge training order
CANONICAL_ATTRIBUTES = [
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
    "Accessory-Glasses-Sun", "Accessory-Hat",
]

# Hardcoded fitted Platt calibration parameters
PLATT_T_CVX = np.array({T_cvx_str}, dtype=np.float32)
PLATT_B_CVX = np.array({b_cvx_str}, dtype=np.float32)
PLATT_T_SWIN = np.array({T_swin_str}, dtype=np.float32)
PLATT_B_SWIN = np.array({b_swin_str}, dtype=np.float32)

LAMBDA_METRIC = {lambda_metric}
IMG_HEIGHT = {height}
IMG_WIDTH = {width}

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CONVNEXT_WEIGHTS = os.path.join(ASSET_DIR, "model_convnext.pth")
SWIN_WEIGHTS = os.path.join(ASSET_DIR, "model_swin.pth")
METRIC_HEADS_WEIGHTS = os.path.join(ASSET_DIR, "metric_heads.pth")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CVX: nn.Module | None = None
_SWIN: nn.Module | None = None
_VIS_PROJECTOR: nn.Module | None = None
_QUERY_ENCODER: nn.Module | None = None


# Architecture Definitions
class ConvNeXtPAR(nn.Module):
    def __init__(self, num_attributes: int = 40):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        in_features = 768
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_attributes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.features(x)
        feat_vec = self.avgpool(feat_map).flatten(1)
        logits = self.classifier(feat_vec)
        return feat_vec, logits


class SwinPAR(nn.Module):
    def __init__(self, num_attributes: int = 40):
        super().__init__()
        self.backbone = models.swin_t(weights=None)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_attributes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return features, logits


class VisualProjector(nn.Module):
    def __init__(self, in_features: int = 1536, embedding_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, embedding_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return F.normalize(feat, p=2, dim=-1)


class QueryEncoder(nn.Module):
    def __init__(self, num_attributes: int = 40, embedding_dim: int = 512):
        super().__init__()
        half_dim = embedding_dim // 2
        self.hard_mlp = nn.Sequential(
            nn.Linear(num_attributes, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, half_dim),
            nn.LayerNorm(half_dim, eps=1e-6),
        )
        self.attr_token_embeddings = nn.Parameter(torch.randn(num_attributes, half_dim) * 0.02)
        self.semantic_adapter = nn.Sequential(
            nn.LayerNorm(half_dim, eps=1e-6),
            nn.Linear(half_dim, half_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(half_dim, half_dim),
            nn.LayerNorm(half_dim, eps=1e-6),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        e_h = self.hard_mlp(q)
        attr_weights = q / (q.sum(dim=-1, keepdim=True).clamp(min=1.0))
        semantic_pooled = torch.matmul(attr_weights, self.attr_token_embeddings)
        e_s = self.semantic_adapter(semantic_pooled)
        e_q = torch.cat([e_h, e_s], dim=-1)
        return F.normalize(e_q, p=2, dim=-1)


_INFERENCE_TRANSFORMS = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model() -> None:
    \"\"\"Loads Dual-Stream backbones and Stage 2 metric heads onto compute device.\"\"\"
    global _CVX, _SWIN, _VIS_PROJECTOR, _QUERY_ENCODER
    if _CVX is not None and _SWIN is not None:
        return
    print(f"Loading Stage 2 Dual-Stream models onto {{_DEVICE}}...", flush=True)

    _CVX = ConvNeXtPAR(num_attributes=40)
    _SWIN = SwinPAR(num_attributes=40)
    _VIS_PROJECTOR = VisualProjector(in_features=1536, embedding_dim=512)
    _QUERY_ENCODER = QueryEncoder(num_attributes=40, embedding_dim=512)

    # 1. Load ConvNeXt weights
    try:
        cvx_sd = torch.load(CONVNEXT_WEIGHTS, map_location="cpu", weights_only=True)
    except Exception:
        cvx_sd = torch.load(CONVNEXT_WEIGHTS, map_location="cpu")
    if isinstance(cvx_sd, dict) and "model_state_dict" in cvx_sd:
        cvx_sd = cvx_sd["model_state_dict"]
    cvx_sd = {{k.replace("module.", ""): v for k, v in cvx_sd.items() if not k.startswith("embedding_head.")}}
    _CVX.load_state_dict(cvx_sd, strict=False)

    # 2. Load Swin-T weights
    try:
        swin_sd = torch.load(SWIN_WEIGHTS, map_location="cpu", weights_only=True)
    except Exception:
        swin_sd = torch.load(SWIN_WEIGHTS, map_location="cpu")
    if isinstance(swin_sd, dict) and "model_state_dict" in swin_sd:
        swin_sd = swin_sd["model_state_dict"]
    swin_sd = {{k.replace("module.", ""): v for k, v in swin_sd.items() if not k.startswith("embedding_head.")}}
    _SWIN.load_state_dict(swin_sd, strict=False)

    # 3. Load metric head weights (1536-D)
    try:
        mh_sd = torch.load(METRIC_HEADS_WEIGHTS, map_location="cpu", weights_only=True)
    except Exception:
        mh_sd = torch.load(METRIC_HEADS_WEIGHTS, map_location="cpu")
    if "vis_projector" in mh_sd:
        _VIS_PROJECTOR.load_state_dict(mh_sd["vis_projector"], strict=False)
        _QUERY_ENCODER.load_state_dict(mh_sd["query_encoder"], strict=False)
    else:
        _VIS_PROJECTOR.load_state_dict({{k.replace("vis_projector.", ""): v for k, v in mh_sd.items() if "vis_projector" in k}}, strict=False)
        _QUERY_ENCODER.load_state_dict({{k.replace("query_encoder.", ""): v for k, v in mh_sd.items() if "query_encoder" in k}}, strict=False)

    _CVX = _CVX.to(_DEVICE).eval()
    _SWIN = _SWIN.to(_DEVICE).eval()
    _VIS_PROJECTOR = _VIS_PROJECTOR.to(_DEVICE).eval()
    _QUERY_ENCODER = _QUERY_ENCODER.to(_DEVICE).eval()
    print("Dual-Stream Stage 2 models and metric heads successfully loaded and ready for inference.", flush=True)


@torch.no_grad()
def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    global _CVX, _SWIN, _VIS_PROJECTOR, _QUERY_ENCODER
    if _CVX is None:
        load_model()

    gallery = sample.get("gallery", [])
    queries_raw = np.asarray(sample.get("queries", []), dtype=np.float32)
    queries_raw = np.nan_to_num(queries_raw, nan=0.0)
    queries_raw = np.clip(queries_raw, 0.0, 1.0)
    attribute_names = sample.get("attribute_names", CANONICAL_ATTRIBUTES)

    num_samples = len(gallery)
    num_queries = len(queries_raw)

    if num_samples == 0:
        return {{"distances": np.zeros((num_queries, 0), dtype=np.float32)}}
    if num_queries == 0:
        return {{"distances": np.zeros((0, num_samples), dtype=np.float32)}}

    if attribute_names != CANONICAL_ATTRIBUTES:
        name_to_col = {{name: idx for idx, name in enumerate(attribute_names)}}
        aligned_queries = np.zeros((num_queries, 40), dtype=np.float32)
        for canon_idx, name in enumerate(CANONICAL_ATTRIBUTES):
            if name in name_to_col:
                aligned_queries[:, canon_idx] = queries_raw[:, name_to_col[name]]
        queries = aligned_queries
    else:
        queries = queries_raw

    q_tensor = torch.from_numpy(queries).float().to(_DEVICE)
    q_emb = _QUERY_ENCODER(q_tensor).cpu().numpy()

    batch_size = 32
    all_calib_probs = []
    all_vis_embs = []
    use_autocast = _DEVICE.type == "cuda"

    print(f"Executing Dual-Stream inference on {{num_samples}} gallery images (batch_size={{batch_size}})...", flush=True)

    for i in range(0, num_samples, batch_size):
        batch_items = gallery[i : i + batch_size]
        tensors = []

        for item in batch_items:
            img_path = item.get("image_path") or item.get("image") or ""
            try:
                with Image.open(img_path) as img:
                    tensors.append(_INFERENCE_TRANSFORMS(img.convert("RGB")))
            except Exception:
                tensors.append(torch.zeros(3, IMG_HEIGHT, IMG_WIDTH, dtype=torch.float32))

        batch_tensor = torch.stack(tensors, dim=0).to(_DEVICE)

        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                f_cvx, l_cvx = _CVX(batch_tensor)
                f_swin, l_swin = _SWIN(batch_tensor)
                f_fused = torch.cat([f_cvx, f_swin], dim=-1)
                v_embs = _VIS_PROJECTOR(f_fused)
        else:
            f_cvx, l_cvx = _CVX(batch_tensor)
            f_swin, l_swin = _SWIN(batch_tensor)
            f_fused = torch.cat([f_cvx, f_swin], dim=-1)
            v_embs = _VIS_PROJECTOR(f_fused)

        l_cvx_np = l_cvx.float().cpu().numpy()
        l_swin_np = l_swin.float().cpu().numpy()

        p_cvx = 1.0 / (1.0 + np.exp(-np.clip((l_cvx_np - PLATT_B_CVX[None, :]) / PLATT_T_CVX[None, :], -20.0, 20.0)))
        p_swin = 1.0 / (1.0 + np.exp(-np.clip((l_swin_np - PLATT_B_SWIN[None, :]) / PLATT_T_SWIN[None, :], -20.0, 20.0)))
        probs = 0.5 * (p_cvx + p_swin)

        all_calib_probs.append(probs)
        all_vis_embs.append(v_embs.float().cpu().numpy())

        del batch_tensor, f_cvx, l_cvx, f_swin, l_swin, f_fused, v_embs

        processed = i + len(batch_items)
        if (i // batch_size) % 50 == 0 or processed >= num_samples:
            if _DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[{{processed}}/{{num_samples}}] Gallery processed ({{processed/num_samples*100:.1f}}%)", flush=True)

    probs = np.vstack(all_calib_probs).astype(np.float32)
    v_embs = np.vstack(all_vis_embs).astype(np.float32)

    attr_dist = (probs.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * probs).T) / 40.0
    cos_sim = q_emb @ v_embs.T
    metric_dist = 0.5 * (1.0 - cos_sim)

    distances = (1.0 - LAMBDA_METRIC) * attr_dist + LAMBDA_METRIC * metric_dist
    distances = np.nan_to_num(distances, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
    print(f"Rank gallery complete. Distance matrix shape: {{distances.shape}}", flush=True)
    return {{"distances": distances}}
'''
    return code


def build_stage2_dual_fused_submission(
    output_zip: str = "submissions/EXP-006_STAGE2_DUAL_FUSED_CHAMPIONSHIP_submission.zip",
    convnext_zip: str = "submissions/EXP-004_submission.zip",
    swin_zip: str = "submissions/EXP-005A_submission.zip",
    metric_heads_path: str = "checkpoints/stage2_dual_fused_metric_heads.pth",
    calibration_path: str = "checkpoints/stage1_calibration_params.npz",
    lambda_metric: float = 0.25,
) -> str:
    """Builds and packages self-contained Stage 2 Dual-Stream Fused championship submission zip."""
    print("=" * 80)
    print(f"BUILDING STAGE 2 DUAL-STREAM CHAMPIONSHIP SUBMISSION: {output_zip}")
    print("=" * 80)

    staging_dir = "/tmp/stage2_dual_submission_staging"
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(os.path.join(staging_dir, "assets"), exist_ok=True)

    # 1. Load calibration parameters for both models
    calib_paths = [calibration_path, "assets/stage1_calibration_params.npz", "checkpoints/stage1_calibration_params.npz"]
    calib_file = next((cp for cp in calib_paths if os.path.exists(cp)), None)
    if calib_file is not None:
        calib = np.load(calib_file)
        T_cvx = calib["T_004"]
        b_cvx = calib["b_004"]
        T_swin = calib["T_005"]
        b_swin = calib["b_005"]
    else:
        T_cvx = [1.5416, 1.8003, 1.3536, 1.0762, 1.6780, 1.4883, 1.5458, 1.3283, 1.2592, 1.3323, 1.4851, 1.5165, 1.3414, 1.5971, 1.6384, 1.6033, 1.4398, 1.3406, 1.6017, 1.5242, 1.1394, 1.1895, 1.2291, 1.4429, 1.5724, 1.3303, 1.6394, 1.6521, 1.6309, 1.5173, 1.4552, 1.6083, 1.5034, 1.0315, 1.2646, 1.2332, 1.3571, 1.4721, 1.4649, 1.3195]
        b_cvx = [2.8053, 0.9997, 1.3599, 1.4451, 1.2190, 1.1278, 1.6392, 1.1691, 1.2062, 1.2343, 1.4283, 1.4871, 1.2625, 1.5582, 1.5993, 1.5641, 1.3857, 1.2589, 1.5627, 1.4789, 1.1891, 1.1782, 1.2185, 1.3982, 1.5231, 1.2584, 1.6012, 1.6143, 1.5932, 1.4761, 1.4123, 1.5701, 1.4592, 0.9852, 1.1983, 1.1821, 1.3129, 1.4281, 1.4192, 1.2783]
        T_swin = [1.5287, 1.9120, 1.4530, 1.0267, 1.7007, 1.4921, 1.5512, 1.3312, 1.2643, 1.3411, 1.4912, 1.5231, 1.3482, 1.6021, 1.6432, 1.6082, 1.4451, 1.3452, 1.6071, 1.5291, 1.1451, 1.1952, 1.2341, 1.4481, 1.5781, 1.3352, 1.6441, 1.6571, 1.6352, 1.5221, 1.4601, 1.6131, 1.5081, 1.0362, 1.2691, 1.2381, 1.3621, 1.4771, 1.4691, 1.3241]
        b_swin = [2.7781, 0.9276, 1.4545, 1.4719, 1.1888, 1.1121, 1.6231, 1.1541, 1.1921, 1.2191, 1.4131, 1.4721, 1.2481, 1.5431, 1.5841, 1.5491, 1.3711, 1.2441, 1.5481, 1.4641, 1.1741, 1.1631, 1.2041, 1.3831, 1.5081, 1.2441, 1.5861, 1.5991, 1.5781, 1.4611, 1.3971, 1.5551, 1.4441, 0.9711, 1.1831, 1.1671, 1.2981, 1.4131, 1.4041, 1.2631]

    # 2. Generate run.py
    run_py_content = generate_stage2_dual_run_py(
        lambda_metric=lambda_metric,
        height=256,
        width=128,
        T_cvx=T_cvx,
        b_cvx=b_cvx,
        T_swin=T_swin,
        b_swin=b_swin,
    )
    with open(os.path.join(staging_dir, "run.py"), "w", encoding="utf-8") as f:
        f.write(run_py_content)
    print("Generated self-contained dual run.py with embedded Platt calibration.")

    # 3. Generate metadata.yaml
    metadata_yaml = """command: python3 -c "print('Submission package ready. The ingestion program imports run.py')"
"""
    with open(os.path.join(staging_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        f.write(metadata_yaml)

    # 4. Extract ConvNeXt weights
    print(f"Extracting ConvNeXt weights from {convnext_zip}...")
    temp_zip_dir = "/tmp/extract_cvx_weights"
    if os.path.exists(temp_zip_dir):
        shutil.rmtree(temp_zip_dir)
    with zipfile.ZipFile(convnext_zip, "r") as zf:
        zf.extractall(temp_zip_dir)
    src_cvx = os.path.join(temp_zip_dir, "assets", "model.pth")
    dst_cvx = os.path.join(staging_dir, "assets", "model_convnext.pth")
    shutil.copy2(src_cvx, dst_cvx)
    shutil.rmtree(temp_zip_dir)
    print(f"Copied ConvNeXt weights ({os.path.getsize(dst_cvx)/(1024*1024):.1f} MB).")

    # 5. Extract Swin-T weights
    print(f"Extracting Swin-T weights from {swin_zip}...")
    temp_zip_dir = "/tmp/extract_swin_weights"
    if os.path.exists(temp_zip_dir):
        shutil.rmtree(temp_zip_dir)
    with zipfile.ZipFile(swin_zip, "r") as zf:
        zf.extractall(temp_zip_dir)
    src_swin = os.path.join(temp_zip_dir, "assets", "model.pth")
    dst_swin = os.path.join(staging_dir, "assets", "model_swin.pth")
    shutil.copy2(src_swin, dst_swin)
    shutil.rmtree(temp_zip_dir)
    print(f"Copied Swin-T weights ({os.path.getsize(dst_swin)/(1024*1024):.1f} MB).")

    # 6. Copy metric heads (1536-D)
    dst_mh = os.path.join(staging_dir, "assets", "metric_heads.pth")
    shutil.copy2(metric_heads_path, dst_mh)
    print(f"Copied Dual-Stream Stage 2 metric heads ({os.path.getsize(dst_mh)/(1024*1024):.1f} MB).")

    # 7. Create zip package
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging_dir):
            for file in files:
                abs_f = os.path.join(root, file)
                rel_f = os.path.relpath(abs_f, staging_dir)
                zf.write(abs_f, rel_f)

    total_mb = os.path.getsize(output_zip) / (1024 * 1024)
    shutil.rmtree(staging_dir)
    print(f"\n>> Stage 2 Dual-Stream Championship Package successfully built: {output_zip} ({total_mb:.1f} MB)")
    return output_zip


if __name__ == "__main__":
    build_stage2_submission()
    build_stage2_dual_fused_submission()

