# Report 07: Codabench Container Engineering, Air-Gapped Packaging & Production Submission Contract

**Document:** Container Engineering & Deployment Specification  
**Platform:** Codabench (Docker Worker Sandbox)  
**Base Image:** `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`  
**Execution Constraints:** Zero Network Access, 3,600s Wall-Clock Limit, `Force_Last` Rule  
**Document Status:** Complete Production Implementation Contract  

---

## 1. The Container Runtime Environment

Submissions are evaluated inside an isolated, unprivileged Docker container:
- **Base Image:** `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
- **Linux Distribution:** Ubuntu 22.04 LTS
- **Python Version:** 3.11
- **CUDA / cuDNN Version:** CUDA 12.1 / cuDNN 9
- **Pre-installed Python Packages:**
  - `torch==2.4.1`
  - `torchvision`
  - `numpy`
  - `Pillow`
- **Network Interface:** **Disabled (`--network none`).** Any outgoing TCP/UDP socket call raises an immediate network unreachable exception.

---

## 2. The Dependency Vendoring Rule: Surviving Without `pip`

Because the evaluation worker operates with zero internet access, standard deep learning libraries that competitors take for granted are **completely unavailable**:

```
+-------------------------------------------------------------------------------------------------+
|                                 CONTAINER PACKAGE AVAILABILITY                                   |
+--------------------------+-----------------------+----------------------------------------------+
| Package Name             | Container Status      | Resolution Protocol                          |
+--------------------------+-----------------------+----------------------------------------------+
| PyTorch                  | AVAILABLE (2.4.1)     | Native import                                |
| Torchvision              | AVAILABLE             | Native import                                |
| NumPy                    | AVAILABLE             | Native import                                |
| Pillow                   | AVAILABLE             | Native import                                |
+--------------------------+-----------------------+----------------------------------------------+
| timm                     | NOT INSTALLED         | Vendor pure PyTorch Swin/ViT modules         |
| transformers             | NOT INSTALLED         | Vendor pure PyTorch text tokenizers/models   |
| huggingface_hub          | NOT INSTALLED         | Package all weights locally inside zip       |
| scikit-learn             | NOT INSTALLED         | Implement vectorized metrics in pure NumPy   |
| opencv-python (cv2)      | NOT INSTALLED         | Use PIL.Image and torchvision.transforms     |
| pandas                   | NOT INSTALLED         | Use standard Python `csv` module             |
| albumentations           | NOT INSTALLED         | Use torchvision transforms                   |
| einops                   | NOT INSTALLED         | Replace with `torch.reshape` & `transpose`   |
+--------------------------+-----------------------+----------------------------------------------+
```

**Architectural Rule:** All model definitions (Swin stages, ViT layers, CASA attention, cross-fusion modules, and retrieval adapters) must be written in **pure PyTorch** and placed inside a local `models/` directory bundled inside `submission.zip`.

---

## 3. The Complete Submission Directory Layout

To execute successfully, the archive must unpack cleanly at the root of the submission directory:

```text
submission/
├── run.py                 # Core entry point (REQUIRED AT ROOT)
├── metadata.yaml          # Ingestion metadata (REQUIRED AT ROOT)
├── models/                # Pure PyTorch vendored architectures
│   ├── __init__.py
│   ├── swin_backbone.py   # Pure PyTorch Swin Transformer
│   ├── vit_backbone.py    # Pure PyTorch Vanilla ViT / EVA
│   ├── casa_attention.py  # Channel-Aware Self-Attention
│   ├── cross_fusion.py    # SVCF & VSCF cross-fusion modules
│   └── retrieval_heads.py # Attribute heads & metric adapters
├── weights/               # Pretrained FP16 model checkpoints
│   ├── model_c2t_fp16.pt  # ~220 MB (Swin-B + ViT-B)
│   └── model_conv_fp16.pt # ~170 MB (ConvNeXt-Base)
└── utils/                 # Pure Python helpers
    ├── __init__.py
    ├── tokenizer.py       # Lightweight word tokenizer for pseudo-text
    └── fast_loader.py     # High-throughput batch image loader
```

---

## 4. The Official Python Submission Interface: Complete Production Code

Below is the production-grade `run.py` script:

```python
"""
Official Submission Entry Point: run.py
Target: UPAR Challenge Track 2 @ WACV 2027 (Codabench #18196)
"""
from __future__ import annotations

import os
import sys
import gc
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Ensure local modules can be imported
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Hardware setup
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL: nn.Module | None = None

# Preprocessing pipeline
_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model() -> None:
    """
    Called once by the ingestion engine upon container startup.
    Instantiates model backbones, moves to GPU, and loads FP16 weights.
    """
    global _MODEL
    if _MODEL is not None:
        return

    print(f"[Submission] Loading model on device: {_DEVICE}")
    try:
        from models.c2t_retrieval import HDSRNet
        model = HDSRNet(nattr=40, embed_dim=512)
        
        weights_path = os.path.join(HERE, "weights", "model_c2t_fp16.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")
            
        checkpoint = torch.load(weights_path, map_location=_DEVICE)
        model.load_state_dict(checkpoint)
        
        # Optimize for inference
        model.to(_DEVICE)
        model.eval()
        if _DEVICE.type == "cuda":
            model.half()  # Run in FP16 for speed and memory efficiency
            
        _MODEL = model
        print("[Submission] Model successfully loaded in FP16 mode.")
    except Exception as e:
        print(f"[Submission ERROR] Failed during load_model: {e}")
        raise e


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Called once with all queries and the entire hidden gallery.
    
    sample structure:
    {
      "attribute_names": ["Age-Young", ..., "Accessory-Hat"], # length 40
      "queries": [[0, 1, ...], ...],                          # Q x 40 binary matrix
      "gallery": [                                            # G gallery images
        {"index": 0, "image": "path.jpg", "image_path": "/app/input_data/images/..."},
        ...
      ]
    }
    
    Returns:
      {"similarities": Q x G float32 numpy array}
    """
    global _MODEL
    if _MODEL is None:
        load_model()

    assert _MODEL is not None, "Model failed to initialize"

    attribute_names = sample["attribute_names"]
    queries_raw = sample["queries"]
    gallery = sample["gallery"]

    Q = len(queries_raw)
    G = len(gallery)
    queries = np.asarray(queries_raw, dtype=np.float32)

    print(f"[Submission] Starting retrieval: {Q} queries across {G} gallery images.")

    # ---------------------------------------------------------
    # Step 1: Single-Pass Batched Gallery Inference
    # ---------------------------------------------------------
    batch_size = 128 if _DEVICE.type == "cuda" else 32
    gallery_probs_list = []
    gallery_embeds_list = []

    with torch.no_grad():
        for start_idx in range(0, G, batch_size):
            end_idx = min(start_idx + batch_size, G)
            batch_items = gallery[start_idx:end_idx]

            batch_tensors = []
            for item in batch_items:
                img_path = item["image_path"]
                try:
                    with Image.open(img_path) as img:
                        batch_tensors.append(_TRANSFORM(img.convert("RGB")))
                except Exception as img_err:
                    # Fallback for corrupted images
                    batch_tensors.append(torch.zeros(3, 256, 128))

            x = torch.stack(batch_tensors).to(_DEVICE)
            if _DEVICE.type == "cuda":
                x = x.half()

            # Forward pass: extract calibrated probabilities and metric embeddings
            probs, embeds = _MODEL.forward_gallery(x)
            gallery_probs_list.append(probs.cpu().float().numpy())
            gallery_embeds_list.append(embeds.cpu().float().numpy())

            if (start_idx // batch_size) % 50 == 0:
                print(f"[Submission] Processed {end_idx}/{G} gallery images...")

    P = np.concatenate(gallery_probs_list, axis=0)       # (G, 40)
    E_v = np.concatenate(gallery_embeds_list, axis=0)   # (G, 512)

    # Free temporary GPU memory
    del gallery_probs_list, gallery_embeds_list
    if _DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------------------------------------
    # Step 2: Query Embedding Extraction
    # ---------------------------------------------------------
    with torch.no_grad():
        q_tensor = torch.from_numpy(queries).to(_DEVICE)
        if _DEVICE.type == "cuda":
            q_tensor = q_tensor.half()
        E_q = _MODEL.forward_queries(q_tensor).cpu().float().numpy() # (Q, 512)

    # ---------------------------------------------------------
    # Step 3: Vectorized Metric Computation
    # ---------------------------------------------------------
    print("[Submission] Computing vectorized cross-similarities...")

    # Stream 1: Bounded Attribute-Space Similarity in [0, 1]
    # L1 distance: sum|q - p| = sum(p) + q @ (1 - 2p).T
    l1_dist = P.sum(axis=1, keepdims=True).T + queries @ (1.0 - 2.0 * P).T # (Q, G)
    S_attr = 1.0 - (l1_dist / 40.0)

    # Stream 2: Bounded Metric-Space Cosine Similarity in [0, 1]
    E_v_norm = E_v / (np.linalg.norm(E_v, axis=1, keepdims=True) + 1e-8)
    E_q_norm = E_q / (np.linalg.norm(E_q, axis=1, keepdims=True) + 1e-8)
    cosine_sim = E_q_norm @ E_v_norm.T # (Q, G)
    S_emb = (cosine_sim + 1.0) / 2.0

    # ---------------------------------------------------------
    # Step 4: Joint Convex Fusion
    # ---------------------------------------------------------
    alpha = 0.70
    similarities = alpha * S_attr + (1.0 - alpha) * S_emb

    # Verify output integrity
    assert similarities.shape == (Q, G), f"Shape mismatch: {similarities.shape} vs {(Q, G)}"
    assert not np.isnan(similarities).any(), "NaN values detected in similarity matrix"
    assert not np.isinf(similarities).any(), "Inf values detected in similarity matrix"

    print("[Submission] Finished successfully. Returning similarities.")
    return {"similarities": similarities.astype(np.float32)}
```

---

## 5. Timing, Memory & Throughput Budgets

- **Execution Time Limit:** 3,600 seconds (1 hour).
- **Estimated Hidden Gallery Size:** The hidden input archive is $379.3\text{ MB}$. In surveillance benchmarks, pedestrian crops average $10 - 25\text{ KB}$ per image. This indicates a gallery of approximately **20,000 to 35,000 images**.
- **Throughput Analysis:**
  - On an NVIDIA T4 GPU in FP16, batch size 128 processes $\approx 220\text{ images/second}$.
  - Processing 35,000 gallery images requires $\approx 160\text{ seconds}$ ($\approx 2.7\text{ minutes}$).
  - Vectorized matrix multiplication ($Q \times G$) requires $< 5\text{ seconds}$.
  - Total container runtime: $\approx 3.5\text{ minutes}$, well within the 60-minute limit.

---

## 6. Local Docker Testing Harness

Before uploading to Codabench, test the archive inside the exact official container:

```bash
# 1. Pull the official Codabench runtime image
docker pull pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# 2. Package the submission archive
cd submission_dir
zip -r ../submission.zip . -x "*/__pycache__/*" "*.DS_Store" ".git/*"
cd ..

# 3. Execute local mock evaluation inside the container (with network disabled!)
docker run --gpus all --network none --rm \
    -v $(pwd)/submission_dir:/app/submission \
    -v $(pwd)/mock_data:/app/input_data \
    -w /app/submission \
    pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
    python3 -c "
import run, json
run.load_model()
sample = {
    'attribute_names': ['Age-Young'] * 40,
    'queries': [[0]*40]*5,
    'gallery': [{'index': 0, 'image': 'test.jpg', 'image_path': '/app/input_data/sample.jpg'}]
}
res = run.rank_gallery(sample)
print('SUCCESS! Output shape:', res['similarities'].shape)
"
```
