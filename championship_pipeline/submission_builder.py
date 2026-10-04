"""Championship Submission Packager for Codabench Track 2.

Packages trained CrossModalChampionshipNet into an air-gapped, zero-crash submission zip:
  - run.py (implements rank_gallery in continuous 1024-D metric space)
  - metadata.yaml
  - assets/model_state.pth (FP16 weights, ~60 MB payload, well below 200 MB limit)
  - Zero external package dependencies beyond torch, torchvision, numpy, and PIL.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
import torch

RUN_PY_CONTENT = '''"""Codabench Track 2 Championship Entrypoint.

Self-contained pure PyTorch cross-modal inference module.
Strictly compatible with pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime.
Zero external package dependencies beyond torch, torchvision, numpy, and PIL.
Computes gallery ranking directly in continuous 1024-D hyperspherical metric space.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WEIGHTS_PATH = os.path.join(ASSET_DIR, "model_state.pth")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL: nn.Module | None = None
_IMG_HEIGHT = 256
_IMG_WIDTH = 128

_TRANSFORM = T.Compose([
    T.Resize((_IMG_HEIGHT, _IMG_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Canonical domain vocabulary (85 words)
VOCABULARY = [
    "<PAD>", "<UNK>", "<SOS>", "<EOS>",
    "this", "is", "a", "photo", "of", "young", "adult", "elderly", "person",
    "female", "male", "with", "short", "long", "bald", "head", "hair",
    "dressed", "in", "sleeved", "upper", "garment", "and", "lower", "clothing",
    "trousers", "pants", "skirt", "dress", "carrying", "wearing", "no", "prominent",
    "backpack", "bag", "glasses", "sunglasses", "hat", "cap", "full", "length",
    "black", "blue", "brown", "green", "grey", "orange", "pink", "purple",
    "red", "white", "yellow", "other", "colored", "neutral", "dark", "bright"
]
WORD_TO_IDX = {word: idx for idx, word in enumerate(VOCABULARY)}
COLOR_NAMES = [
    "black", "blue", "brown", "green", "grey",
    "orange", "pink", "purple", "red", "white", "yellow", "other colored"
]


class StandaloneVisualStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.swin_t(weights=None)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.visual_adapter = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        raw = self.visual_adapter(feat)
        return F.normalize(raw, p=2, dim=-1)


class StandaloneQueryStream(nn.Module):
    def __init__(self, vocab_size: int = len(VOCABULARY), max_length: int = 40):
        super().__init__()
        self.max_length = max_length
        self.hard_adapter = nn.Sequential(
            nn.Linear(40, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.token_embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, 256) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True
        )
        self.text_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.text_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, hard_attr: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raw_hard = self.hard_adapter(hard_attr)
        seq_len = tokens.size(1)
        x_text = self.token_embedding(tokens) + self.pos_embedding[:, :seq_len, :]
        padding_mask = ~mask
        text_feat = self.text_transformer(x_text, src_key_padding_mask=padding_mask)
        mask_exp = mask.unsqueeze(-1).float()
        pooled = (text_feat * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)
        raw_soft = self.text_adapter(pooled)
        raw_query = torch.cat([raw_hard, raw_soft], dim=-1)
        return F.normalize(raw_query, p=2, dim=-1)


class StandaloneCrossModalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_stream = StandaloneVisualStream()
        self.query_stream = StandaloneQueryStream()


def query_to_tokens(q: np.ndarray, max_len: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
    # Age
    if q[0] == 1: age = "young"
    elif q[2] == 1: age = "elderly"
    else: age = "adult"

    gender = "female" if q[3] == 1 else "male"
    if q[6] == 1: hair = "bald head"
    elif q[5] == 1: hair = "long hair"
    else: hair = "short hair"

    upper_len = "short sleeved" if q[7] == 1 else "long sleeved"
    upper_colors = [COLOR_NAMES[i] for i in range(12) if q[8 + i] == 1]
    upper_str = " ".join(upper_colors) if upper_colors else "neutral colored"

    lower_type = "skirt dress" if q[34] == 1 else "trousers pants"
    lower_len = "short" if q[20] == 1 else "full length"
    lower_colors = [COLOR_NAMES[i] for i in range(12) if q[21 + i] == 1]
    lower_str = " ".join(lower_colors) if lower_colors else "dark colored"

    acc = []
    if q[35] == 1: acc.append("backpack")
    if q[36] == 1: acc.append("bag")
    if q[37] == 1: acc.append("glasses")
    if q[38] == 1: acc.append("sunglasses")
    if q[39] == 1: acc.append("hat")
    acc_str = "carrying wearing " + " ".join(acc) if acc else "with no prominent accessories"

    text = f"this is a photo of {age} {gender} with {hair} dressed in {upper_str} {upper_len} upper garment and {lower_str} {lower_len} {lower_type} {acc_str}"
    clean = re.findall(r"\\b[a-zA-Z]+\\b", text.lower())
    toks = [WORD_TO_IDX.get("<SOS>")]
    for w in clean:
        toks.append(WORD_TO_IDX.get(w, WORD_TO_IDX["<UNK>"]))
        if len(toks) >= max_len - 1:
            break
    toks.append(WORD_TO_IDX.get("<EOS>"))
    seq_len = len(toks)
    pad = max_len - seq_len
    toks = toks + [WORD_TO_IDX["<PAD>"]] * pad
    mask = [True] * seq_len + [False] * pad
    return torch.tensor(toks, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def load_model() -> None:
    global _MODEL
    if _MODEL is not None:
        return
    print(f"Loading Championship Cross-Modal weights from {WEIGHTS_PATH}...", flush=True)
    model = StandaloneCrossModalNet()
    ckpt = torch.load(WEIGHTS_PATH, map_location=_DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # Filter out auxiliary classifier head if present
    cleaned = {}
    for k, v in sd.items():
        k_clean = k[7:] if k.startswith("module.") else k
        if not k_clean.startswith("visual_stream.aux_classifier."):
            cleaned[k_clean] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    print("CrossModalChampionshipNet successfully initialized.", flush=True)


@torch.no_grad()
def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """Official Codabench entrypoint computing 1024-D continuous metric ranking."""
    if _MODEL is None:
        load_model()

    gallery = sample["gallery"]
    queries = np.asarray(sample["queries"], dtype=np.float32)
    num_gallery = len(gallery)
    num_queries = len(queries)

    print(f"Ranking {num_gallery} gallery candidates against {num_queries} queries...", flush=True)

    # 1. Encode gallery images
    batch_size = 32
    all_gal = []
    for i in range(0, num_gallery, batch_size):
        items = gallery[i : i + batch_size]
        tensors = []
        for it in items:
            try:
                with Image.open(it.get("image_path", "")) as img:
                    t = _TRANSFORM(img.convert("RGB"))
            except Exception:
                t = torch.zeros(3, _IMG_HEIGHT, _IMG_WIDTH, dtype=torch.float32)
            tensors.append(t)
        batch_t = torch.stack(tensors, dim=0).to(_DEVICE)
        norm_v = _MODEL.visual_stream(batch_t)
        all_gal.append(norm_v.cpu().numpy())

    gal_feats = np.vstack(all_gal).astype(np.float32)
    gal_feats /= (np.linalg.norm(gal_feats, axis=1, keepdims=True) + 1e-8)

    # 2. Encode queries
    tok_list = []
    mask_list = []
    for q in queries:
        t, m = query_to_tokens(q)
        tok_list.append(t)
        mask_list.append(m)

    hard_t = torch.tensor(queries, dtype=torch.float32).to(_DEVICE)
    toks_t = torch.stack(tok_list).to(_DEVICE)
    mask_t = torch.stack(mask_list).to(_DEVICE)

    norm_q = _MODEL.query_stream(hard_t, toks_t, mask_t)
    query_feats = norm_q.cpu().numpy().astype(np.float32)
    query_feats /= (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)

    # 3. Compute continuous hyperspherical cosine distance: d = 1.0 - (q @ g.T)
    distances = 1.0 - np.dot(query_feats, gal_feats.T)
    distances = np.nan_to_num(distances, nan=2.0, posinf=2.0, neginf=0.0).astype(np.float32)

    print(f"Distance matrix computed: shape={distances.shape}, min={distances.min():.4f}, max={distances.max():.4f}", flush=True)
    return {"distances": distances}
'''

METADATA_YAML = """command: python3 run.py
description: Championship Cross-Modal Metric Learning System (Swin-T + ArcFace)
"""


def build_championship_submission(
    model: torch.nn.Module,
    output_zip_path: str,
    save_fp16: bool = True,
) -> str:
    """Pack model weights and pure PyTorch entrypoint into an air-gapped zip file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_zip_path)), exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        assets_dir = os.path.join(tmp_dir, "assets")
        os.makedirs(assets_dir, exist_ok=True)

        # Write run.py and metadata.yaml
        with open(os.path.join(tmp_dir, "run.py"), "w", encoding="utf-8") as f:
            f.write(RUN_PY_CONTENT)
        with open(os.path.join(tmp_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
            f.write(METADATA_YAML)

        # Save weights in FP16 to compress payload below 65 MB
        weights_path = os.path.join(assets_dir, "model_state.pth")
        state_dict = model.state_dict()
        if save_fp16:
            fp16_state_dict = {}
            for k, v in state_dict.items():
                if v.dtype == torch.float32:
                    fp16_state_dict[k] = v.half()
                else:
                    fp16_state_dict[k] = v
            torch.save(fp16_state_dict, weights_path)
        else:
            torch.save(state_dict, weights_path)

        # Build zip archive
        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(os.path.join(tmp_dir, "run.py"), arcname="run.py")
            z.write(os.path.join(tmp_dir, "metadata.yaml"), arcname="metadata.yaml")
            z.write(weights_path, arcname="assets/model_state.pth")

    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"[Submission Builder] Created: {output_zip_path} ({zip_size_mb:.2f} MB)", flush=True)
    assert zip_size_mb < 200.0, f"Submission size {zip_size_mb:.2f} MB exceeds 200 MB limit!"
    return output_zip_path
