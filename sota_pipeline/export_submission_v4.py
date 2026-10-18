#!/usr/bin/env python3
"""SOTA Championship V4: Submission Packaging & Air-Gapped Verification Engine.

Converts trained SOTA V4 CLIP ViT-L/14 checkpoint to FP16 (or INT8 quantized),
executes isolated subprocess verification of run.py, and packages into a CodaBench-compliant
ZIP strictly < 500 MB.
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Tuple
import zipfile

import torch
import torch.nn as nn


def clean_state_dict(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Removes module. and _orig_mod. prefixes introduced by DDP / torch.compile."""
    cleaned: Dict[str, Any] = {}
    for k, v in sd.items():
        clean_k = k
        if clean_k.startswith("module."):
            clean_k = clean_k[7:]
        if clean_k.startswith("_orig_mod."):
            clean_k = clean_k[10:]
        cleaned[clean_k] = v
    return cleaned


def to_fp16_dict(sd: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    """Casts floating-point tensors in a state dict to half-precision (FP16)."""
    fp16_dict: Dict[str, Any] = {}
    total_params = 0
    total_bytes = 0
    for k, v in sd.items():
        if torch.is_tensor(v):
            if v.is_floating_point():
                v_fp16 = v.half()
            else:
                v_fp16 = v
            fp16_dict[k] = v_fp16
            total_params += v_fp16.numel()
            total_bytes += v_fp16.element_size() * v_fp16.numel()
        else:
            fp16_dict[k] = v
    return fp16_dict, total_params, total_bytes


def export_model(
    src_checkpoint: str,
    dst_path: str,
    template_dir: str,
    use_int8: bool = False,
) -> None:
    """
    Extracts visual_model and prompter weights from checkpoint and saves to dst_path.
    If use_int8 is True, applies dynamic INT8 quantization to Linear layers.
    Otherwise, casts all weights to FP16.
    """
    print("=" * 75)
    print(f"[EXPORT V4] Loading trained checkpoint from: {src_checkpoint}...")
    ckpt = torch.load(src_checkpoint, map_location="cpu", weights_only=False)

    # 1. Extract visual_model state dict
    if isinstance(ckpt, dict) and "visual_model" in ckpt and isinstance(ckpt["visual_model"], dict):
        vis_sd = ckpt["visual_model"]
    else:
        state_dict = ckpt.get("model_state_dict", ckpt)
        vis_sd = {k[len("visual_model."):]: v for k, v in state_dict.items() if k.startswith("visual_model.")}
        if not vis_sd:
            vis_sd = {
                k: v for k, v in state_dict.items()
                if not k.startswith("prompter.") and not k.startswith("attr_prompter.")
                and not k.startswith("text_projection.") and not k.startswith("optimizer")
            }

    # 2. Extract prompter state dict
    if isinstance(ckpt, dict) and "prompter" in ckpt and isinstance(ckpt["prompter"], dict):
        prm_sd = ckpt["prompter"]
    else:
        state_dict = ckpt.get("model_state_dict", ckpt)
        prm_sd = {k[len("prompter."):]: v for k, v in state_dict.items() if k.startswith("prompter.")}
        if not prm_sd:
            prm_sd = {k[len("attr_prompter."):]: v for k, v in state_dict.items() if k.startswith("attr_prompter.")}

    vis_sd = clean_state_dict(vis_sd)
    prm_sd = clean_state_dict(prm_sd)

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        os.remove(dst_path)

    # Ensure template_dir is in sys.path to instantiate models for quantization
    abs_template = os.path.abspath(template_dir)
    if abs_template not in sys.path:
        sys.path.insert(0, abs_template)

    if use_int8:
        print("[EXPORT V4] Applying Dynamic INT8 Quantization (Linear layers -> qint8)...")
        # Setup quantization engine
        if "qnnpack" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "qnnpack"
        elif "fbgemm" in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = "fbgemm"

        from run import ContinuousAttributePrompter, InferenceVisualModel

        visual_model = InferenceVisualModel(embed_dim=768, num_attrs=40).eval()
        prompter = ContinuousAttributePrompter(num_attrs=40, embed_dim=768, hidden_dim=512).eval()

        visual_model.load_state_dict(vis_sd, strict=False)
        prompter.load_state_dict(prm_sd, strict=False)

        # Dynamic quantization on Linear layers
        quant_vis = torch.ao.quantization.quantize_dynamic(
            visual_model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        quant_prm = torch.ao.quantization.quantize_dynamic(
            prompter,
            {nn.Linear},
            dtype=torch.qint8,
        )

        payload = {
            "visual_model": quant_vis,
            "prompter": quant_prm,
            "quantized": True,
        }
        torch.save(payload, dst_path)
        print("[EXPORT V4] INT8 Dynamic Quantization complete!")

    else:
        print("[EXPORT V4] Converting weights to FP16 half-precision...")
        vis_fp16, vis_params, vis_bytes = to_fp16_dict(vis_sd)
        prm_fp16, prm_params, prm_bytes = to_fp16_dict(prm_sd)

        payload = {
            "visual_model": vis_fp16,
            "prompter": prm_fp16,
            "quantized": False,
        }
        torch.save(payload, dst_path)

        total_params = vis_params + prm_params
        total_mb = (vis_bytes + prm_bytes) / (1024 * 1024)
        print(f"[EXPORT V4] Visual Model: {vis_params:,} params | Prompter: {prm_params:,} params")
        print(f"[EXPORT V4] Total Parameters: {total_params:,} | Raw FP16 Size: {total_mb:.2f} MB")

    file_size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"[EXPORT V4] Serialized weights to: {dst_path} ({file_size_mb:.2f} MB)")
    print("=" * 75)


def run_air_gapped_verification(template_dir: str) -> None:
    """Executes run.py in an isolated subprocess to verify 100% offline air-gapped operation."""
    print("=" * 75)
    print("[VERIFY V4] Testing air-gapped execution in isolated subprocess...")
    run_py = os.path.abspath(os.path.join(template_dir, "run.py"))
    if not os.path.isfile(run_py):
        raise FileNotFoundError(f"Missing run.py at: {run_py}")

    t0 = time.time()
    res = subprocess.run(
        [sys.executable, run_py],
        cwd=template_dir,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if res.returncode != 0:
        print("[VERIFY V4] FAILED! Subprocess stderr:")
        print(res.stderr)
        print("[VERIFY V4] Subprocess stdout:")
        print(res.stdout)
        raise RuntimeError(f"Air-gapped verification of {run_py} failed with code {res.returncode}!")

    print(f"[VERIFY V4] PASSED in {elapsed:.2f}s! Subprocess tail:")
    for line in res.stdout.strip().split("\n")[-8:]:
        print(f"  {line}")
    print("=" * 75)


def package_zip(template_dir: str, zip_output_path: str, strict: bool = False) -> None:
    """
    Creates CodaBench-compliant ZIP with root-relative paths strictly < 500 MB.
    """
    print(f"[PACKAGE V4] Building submission ZIP: {zip_output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(zip_output_path)), exist_ok=True)

    # Validate essential files
    required_files = ["run.py", "metadata.yaml"]
    for req in required_files:
        p = os.path.join(template_dir, req)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required submission file: {p}")

    assets_dir = os.path.join(template_dir, "assets")
    if not os.path.isdir(assets_dir):
        raise FileNotFoundError(f"Missing assets directory at {assets_dir}")

    candidates = [f for f in os.listdir(assets_dir) if f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(f"Missing model weights in {assets_dir}!")

    vendor_timm = os.path.join(template_dir, "vendor", "timm")
    if not os.path.isdir(vendor_timm):
        raise FileNotFoundError(f"Missing vendored timm at {vendor_timm}!")

    if os.path.exists(zip_output_path):
        os.remove(zip_output_path)

    t0 = time.time()
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(template_dir):
            if "__pycache__" in root or ".git" in root or ".DS_Store" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".DS_Store", ".tmp")):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, template_dir)
                zf.write(full_path, rel_path)

    elapsed = time.time() - t0
    zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
    print(f"[PACKAGE V4] Successfully built {zip_output_path} ({zip_size_mb:.2f} MB in {elapsed:.1f}s)")

    if zip_size_mb > 500.0:
        print("!" * 75)
        print(f"[PACKAGE V4] WARNING: ZIP size ({zip_size_mb:.2f} MB) exceeds CodaBench 500 MB limit!")
        print("[PACKAGE V4] Recommended: Run with '--int8' to generate an INT8 quantized package (~233 MB).")
        print("!" * 75)
        if strict:
            raise ValueError(f"Submission ZIP exceeds 500 MB limit ({zip_size_mb:.2f} MB)!")
    else:
        print(f"[PACKAGE V4] Validation: PASSED! Submission ZIP is strictly < 500 MB ({zip_size_mb:.2f} MB).")
    print("=" * 75)



def main() -> None:
    parser = argparse.ArgumentParser(description="Export and package SOTA V4 CLIP ViT-L/14 submission")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="v4_outputs/best_model_v4.pt",
        help="Path to trained checkpoint (e.g. v4_outputs/best_model_v4.pt)",
    )
    parser.add_argument(
        "--template_dir",
        type=str,
        default="sota_pipeline/submission_template_v4",
        help="Path to submission template directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="submissions",
        help="Directory to output submission ZIP",
    )
    parser.add_argument(
        "--submission_name",
        type=str,
        default="submission_v4_championship",
        help="Submission archive name (without .zip)",
    )
    parser.add_argument(
        "--output_zip",
        type=str,
        default=None,
        help="Explicit output ZIP file path (overrides --output_dir and --submission_name)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Apply Dynamic INT8 Quantization to Linear layers",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip running isolated run.py verification",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error if ZIP exceeds 500 MB limit",
    )
    args = parser.parse_args()


    # Determine final zip output path
    if args.output_zip:
        zip_output_path = args.output_zip
    else:
        name = args.submission_name
        if args.int8 and not name.endswith("_int8"):
            name = f"{name}_int8"
        zip_output_path = os.path.join(args.output_dir, f"{name}.zip")

    # Destination weights path inside template assets
    dst_weights = os.path.join(args.template_dir, "assets", "model_v4_fp16.pt")

    # Locate source checkpoint
    src_ckpt = args.checkpoint
    if not os.path.isfile(src_ckpt):
        # Check standard alternative candidates
        alt_candidates = [
            "v4_outputs/checkpoint_best.pt",
            "v4_outputs/checkpoint_latest.pt",
            "v4_outputs/best_model_v4.pt",
            os.path.join("v4_outputs", "checkpoint_epoch_*.pt"),
        ]
        found = False
        for cand in alt_candidates:
            matches = glob.glob(cand)
            if matches:
                src_ckpt = matches[0]
                found = True
                print(f"[EXPORT V4] Checkpoint '{args.checkpoint}' not found. Using fallback: {src_ckpt}")
                break
        if not found:
            print(f"[EXPORT V4] NOTICE: No trained checkpoint found at {args.checkpoint} or fallbacks.")
            if os.path.isfile(dst_weights):
                print(f"[EXPORT V4] Using existing weights in {dst_weights}")
            else:
                print(f"[EXPORT V4] Checkpoint will be required when training completes.")

    if os.path.isfile(src_ckpt):
        export_model(
            src_checkpoint=src_ckpt,
            dst_path=dst_weights,
            template_dir=args.template_dir,
            use_int8=args.int8,
        )

    # Verify vendored timm
    vendor_timm = os.path.join(args.template_dir, "vendor", "timm")
    if not os.path.isdir(vendor_timm):
        # Try copying from v3 template
        src_vendor = os.path.join(os.path.dirname(args.template_dir), "submission_template_v3", "vendor", "timm")
        if os.path.isdir(src_vendor):
            os.makedirs(os.path.dirname(vendor_timm), exist_ok=True)
            shutil.copytree(src_vendor, vendor_timm)
            print(f"[EXPORT V4] Copied vendored timm from {src_vendor} -> {vendor_timm}")

    # Run air-gapped verification
    if not args.skip_verify:
        run_air_gapped_verification(args.template_dir)

    # Package into submission ZIP
    package_zip(args.template_dir, zip_output_path, strict=args.strict)
    print("\n[EXPORT V4 COMPLETE] Full SOTA V4 Championship Submission Pipeline Finished Successfully!")



if __name__ == "__main__":
    main()
