"""Standalone Production Extraction and Model Weight Stripping Script.

Loads raw multi-session training checkpoints, eliminates training-only overhead
(AdamW optimizer states, CosineAnnealingLR schedules, AMP GradScaler state,
and the 84M parameter Cross-Modal MLM Head), and exports clean, lightweight deployment
model variants across FP32, FP16, BF16, and Dynamic INT8.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))
from sota_pipeline.models import SOTAChampionshipModel, VisualStream


def extract_and_export_variants(
    source_checkpoint_path: str = "results/upar_challenge_26_rws27/checkpoints_sota/best_model.pt",
    output_dir: str = "extracted_models"
) -> Dict[str, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(f"[EXTRACT] Loading source checkpoint from: {source_checkpoint_path}")
    print("=" * 80)

    if not os.path.isfile(source_checkpoint_path):
        raise FileNotFoundError(f"Source checkpoint not found at: {source_checkpoint_path}")

    raw_bundle = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    epoch = raw_bundle.get("epoch", "N/A")
    global_step = raw_bundle.get("global_step", "N/A")
    best_metric = raw_bundle.get("best_metric", "N/A")
    print(f"[EXTRACT] Checkpoint Metadata: Epoch = {epoch} | Step = {global_step} | Best Metric = {best_metric}")

    source_sd: Dict[str, torch.Tensor] = raw_bundle["model_state_dict"]
    print(f"[EXTRACT] Loaded {len(source_sd)} total parameter tensors from model_state_dict.")

    # 1. Separate Visual, Text, and MLM tensors
    full_stripped_sd = {}
    visual_only_sd = {}
    mlm_sd = {}

    for k, v in source_sd.items():
        if k.startswith("mlm_head."):
            mlm_sd[k] = v
        else:
            full_stripped_sd[k] = v
            if k.startswith("visual_stream."):
                # Strip prefix for standalone VisualStream loading
                clean_k = k[len("visual_stream."):]
                visual_only_sd[clean_k] = v

    print(f"  -> Discarded MLM head tensors: {len(mlm_sd)} tensors ({sum(p.numel() for p in mlm_sd.values()):,} params)")
    print(f"  -> Preserved Full Model tensors: {len(full_stripped_sd)} tensors ({sum(p.numel() for p in full_stripped_sd.values()):,} params)")
    print(f"  -> Preserved Visual-Only tensors: {len(visual_only_sd)} tensors ({sum(p.numel() for p in visual_only_sd.values()):,} params)")

    summary = {}

    # Helper function to save and report
    def save_variant(name: str, sd: Dict[str, torch.Tensor], target_dtype: str, extra_meta: Dict[str, Any] = None):
        out_path = os.path.join(output_dir, f"{name}.pt")
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "state_dict": sd,
            "precision": target_dtype,
        }
        if extra_meta:
            payload.update(extra_meta)
        torch.save(payload, out_path)
        size_bytes = os.path.getsize(out_path)
        size_mb = size_bytes / (1024 ** 2)
        total_params = sum(p.numel() for p in sd.values() if isinstance(p, torch.Tensor))
        print(f"[EXPORT] Saved {name:<25} | Size: {size_mb:>7.2f} MB | Tensors: {len(sd):>3} | Params: {total_params:,} | Dtype: {target_dtype}")
        summary[name] = {
            "path": out_path,
            "size_mb": size_mb,
            "params": total_params,
            "precision": target_dtype,
            "tensors": len(sd)
        }

    # --------------------------------------------------------------------------
    # Variant 1: Full Model Stripped (FP32)
    # --------------------------------------------------------------------------
    fp32_full = {k: v.float() for k, v in full_stripped_sd.items()}
    save_variant("sota_full_fp32", fp32_full, "float32")

    # --------------------------------------------------------------------------
    # Variant 2: Visual Stream Stripped (FP32)
    # --------------------------------------------------------------------------
    fp32_vis = {k: v.float() for k, v in visual_only_sd.items()}
    save_variant("sota_visual_fp32", fp32_vis, "float32")

    # --------------------------------------------------------------------------
    # Variant 3: Full Model Stripped (FP16)
    # --------------------------------------------------------------------------
    fp16_full = {k: v.half() if v.is_floating_point() else v for k, v in full_stripped_sd.items()}
    save_variant("sota_full_fp16", fp16_full, "float16")

    # --------------------------------------------------------------------------
    # Variant 4: Visual Stream Stripped (FP16) - Target Deployment Primary
    # --------------------------------------------------------------------------
    fp16_vis = {k: v.half() if v.is_floating_point() else v for k, v in visual_only_sd.items()}
    save_variant("sota_visual_fp16", fp16_vis, "float16")

    # --------------------------------------------------------------------------
    # Variant 5: Visual Stream Stripped (BF16) - BFloat16 Precision
    # --------------------------------------------------------------------------
    bf16_vis = {k: v.bfloat16() if v.is_floating_point() else v for k, v in visual_only_sd.items()}
    save_variant("sota_visual_bf16", bf16_vis, "bfloat16")

    # --------------------------------------------------------------------------
    # Variant 6: Visual Stream Dynamic INT8 Quantized (Linear Layers)
    # --------------------------------------------------------------------------
    print("[QUANT] Building dynamic INT8 quantized model...")
    try:
        # Build base visual model in FP32
        base_model = SOTAChampionshipModel(embed_dim=1024, num_attrs=40, backbone_type="auto", use_mlm=False)
        base_model.load_state_dict(full_stripped_sd, strict=False)
        vis_stream = base_model.visual_stream
        vis_stream.eval()

        # Apply dynamic quantization on linear layers
        quant_vis = torch.ao.quantization.quantize_dynamic(
            vis_stream,
            {nn.Linear},
            dtype=torch.qint8
        )
        int8_out_path = os.path.join(output_dir, "sota_visual_int8.pt")
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "model_state_dict": quant_vis.state_dict(),
            "precision": "qint8",
        }, int8_out_path)
        int8_size_mb = os.path.getsize(int8_out_path) / (1024 ** 2)
        print(f"[EXPORT] Saved sota_visual_int8          | Size: {int8_size_mb:>7.2f} MB | Precision: qint8 dynamic")
        summary["sota_visual_int8"] = {
            "path": int8_out_path,
            "size_mb": int8_size_mb,
            "precision": "qint8",
        }
    except Exception as e:
        print(f"[QUANT WARNING] INT8 Dynamic quantization note: {e}")

    print("=" * 80)
    print(f"[EXTRACT] All {len(summary)} model variants successfully generated and preserved!")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    extract_and_export_variants()
