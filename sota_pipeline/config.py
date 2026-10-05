"""Configuration Engine for SOTA Championship Person Retrieval Pipeline.

Defines all hyperparameters, loss balances, model backbones, dataset paths,
and multi-session hardware orchestration parameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SOTAConfig:
    """Master configuration for unconstrained championship model and training."""

    # ---------------------------------------------------------
    # Experiment Metadata
    # ---------------------------------------------------------
    exp_id: str = "EXP-SOTA-001"
    description: str = "Unconstrained SOTA Architecture: EVA-02 / ViT-L + 3-Strip PCB + Soft SDM + MLM + ASL"
    seed: int = 42

    # ---------------------------------------------------------
    # Environment & Paths
    # ---------------------------------------------------------
    dataset_root: Optional[str] = None
    output_dir: str = "sota_outputs"
    checkpoint_dir: str = "checkpoints_sota"
    
    # ---------------------------------------------------------
    # Image & Geometric Specifications
    # ---------------------------------------------------------
    img_height: int = 336
    img_width: int = 168
    num_patch_rows: int = 24
    num_patch_cols: int = 12
    patch_size: int = 14
    
    # Anatomical 3-Strip Boundaries along Y-axis (Rows)
    strip_head_rows: tuple = (0, 6)    # Rows 0-6: Head, Hair, Hat, Glasses
    strip_torso_rows: tuple = (6, 16)  # Rows 6-16: Torso, Jacket, Shirt, Backpack
    strip_legs_rows: tuple = (16, 24)  # Rows 16-24: Pants, Shorts, Skirt, Footwear

    # ---------------------------------------------------------
    # Architecture Backbones & Dimensions
    # ---------------------------------------------------------
    # Preferred backbones: 'eva02_large', 'clip_vit_l14', 'torchvision_vit_b16'
    model_name: str = "eva02_large"
    embed_dim: int = 1024
    num_attrs: int = 40
    gem_p_init: float = 3.0
    use_mlm_head: bool = True
    mlm_mask_prob: float = 0.15

    # ---------------------------------------------------------
    # Loss Formulation & Weights
    # ---------------------------------------------------------
    lambda_sdm: float = 1.0       # Soft-Target Similarity Distribution Matching
    lambda_mlm: float = 0.5       # Cross-Modal Masked Language Modeling
    lambda_asl: float = 1.0       # Asymmetric Multi-Label Loss for 40 attributes
    lambda_strip: float = 0.5     # Anatomical 3-Strip Consistency Loss
    
    # Multi-Instance SDM Parameters
    sdm_epsilon_hamming: int = 1  # Samples with <= 1 attribute difference share positive mass
    sdm_tau_init: float = 14.28   # Initial logit scale (1/0.07)
    
    # Asymmetric Loss (ASL) Parameters
    asl_gamma_pos: float = 0.0    # No downweighting of hard positive attributes
    asl_gamma_neg: float = 4.0    # Heavy polynomial suppression of dominant negative traits
    asl_clip_margin: float = 0.05 # Margin threshold to zero out easy background negatives

    # ---------------------------------------------------------
    # Prompt Engine & Augmentations
    # ---------------------------------------------------------
    attr_dropout_prob: float = 0.15
    use_synonym_jitter: bool = True
    max_text_tokens: int = 77
    
    # Surveillance Degradation Simulation (SRDM on MEVID)
    srdm_prob: float = 0.70
    srdm_min_height: int = 64
    srdm_max_height: int = 120
    srdm_blur_prob: float = 0.50
    srdm_jpeg_prob: float = 0.60

    # ---------------------------------------------------------
    # Optimization & Multi-Session Schedule
    # ---------------------------------------------------------
    epochs: int = 70
    warmup_epochs: int = 5
    batch_size: int = 64          # Per-GPU batch size (128 global on Dual T4)
    gradient_accumulation_steps: int = 2
    
    backbone_lr: float = 5e-5     # Conservative learning rate for pretrained backbone
    adapter_lr: float = 3.5e-4    # Higher learning rate for newly initialized heads
    min_lr: float = 1e-6          # Minimum learning rate for CosineAnnealingLR
    weight_decay: float = 1e-4
    use_amp: bool = True          # Automatic Mixed Precision (torch.cuda.amp)
    
    # Multi-Session Kaggle Resumption
    max_session_hours: float = 10.75 # Graceful exit before 12h Kaggle timeout
    eval_every_epochs: int = 1
    save_every_epochs: int = 5

    # ---------------------------------------------------------
    # Post-Processing & Reranking
    # ---------------------------------------------------------
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.30
    rrf_k0: int = 60
    rrf_metric_weight: float = 0.65
    rrf_attr_weight: float = 0.35

    @classmethod
    def resolve_dataset_root(cls, user_path: Optional[str] = None) -> str:
        """Autodetects dataset location across local workstation and Kaggle environments.
        Prioritizes Rank 1 HDF5 binary container (.h5) for instant startup.
        """
        candidates = [
            user_path,
            os.environ.get("UPAR_DATA_ROOT"),
            # Rank 1: HDF5 Binary Container (Kaggle & Local)
            "/kaggle/input/upar-binary-dataset/upar_binary_dataset.h5",
            "/kaggle/input/upar-binary-dataset",
            "/kaggle/input/upar-hdf5-dataset/upar_binary_dataset.h5",
            "/kaggle/input/upar-hdf5-dataset",
            "/kaggle/input/upar-challenge-2027-h5/upar_binary_dataset.h5",
            "/kaggle/input/upar-challenge-2027-h5",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "upar_binary_dataset.h5")),
            # Legacy Directories (Kaggle & Local)
            "/kaggle/input/datasets/hinewk/upar-challenge-2027-data/data",
            "/kaggle/input/upar-challenge-2027-data/data",
            "/kaggle/input/upar-dataset/UPAR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "upar_dataset", "UPAR")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RWS27_Data")),
        ]
        for c in candidates:
            if not c or not os.path.exists(c):
                continue

            # Case A: Explicit HDF5 file
            if os.path.isfile(c) and c.endswith((".h5", ".hdf5")):
                return os.path.abspath(c)

            # Case B: Directory containing an HDF5 file
            if os.path.isdir(c):
                default_h5 = os.path.join(c, "upar_binary_dataset.h5")
                if os.path.isfile(default_h5):
                    return os.path.abspath(default_h5)
                for f in os.listdir(c):
                    if f.endswith((".h5", ".hdf5")):
                        return os.path.abspath(os.path.join(c, f))

            # Case C: Legacy directory with dataset_all.pkl or Market folder
            if os.path.isfile(os.path.join(c, "dataset_all.pkl")) or os.path.isdir(os.path.join(c, "Market")):
                return os.path.abspath(c)
            if os.path.isfile(c) and c.endswith(".pkl"):
                return os.path.dirname(os.path.abspath(c))

        # Default fallback
        for c in candidates:
            if c and os.path.exists(c):
                return os.path.abspath(c)
                
        raise FileNotFoundError(
            "Could not automatically resolve UPAR dataset root. "
            "Please specify via --dataset_root or set UPAR_DATA_ROOT environment variable."
        )
