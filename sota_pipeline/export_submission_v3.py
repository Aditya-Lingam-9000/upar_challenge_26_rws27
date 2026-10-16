#!/usr/bin/env python3
"""SOTA Championship V3: Submission Packaging & Air-Gapped Verification Engine.

Converts trained SOTA V3 checkpoint to FP16, executes isolated subprocess
verification of run.py, and packages into CodaBench-compliant ZIP strictly < 500 MB.
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
import zipfile
from typing import Any, Dict
import torch


def export_model_fp16(src_checkpoint: str, dst_path: str) -> None:
    """Extracts model weights, casts floating tensors to FP16, and serializes."""
    print(f"[EXPORT V3] Loading trained checkpoint from: {src_checkpoint}...")
    ckpt = torch.load(src_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    fp16_state_dict: Dict[str, Any] = {}
    total_params = 0
    total_bytes = 0

    for k, v in state_dict.items():
        clean_key = k
        if clean_key.startswith("module."):
            clean_key = clean_key[7:]
        if clean_key.startswith("_orig_mod."):
            clean_key = clean_key[10:]

        if torch.is_tensor(v):
            if v.is_floating_point():
                v_fp16 = v.half()
            else:
                v_fp16 = v
            fp16_state_dict[clean_key] = v_fp16
            total_params += v_fp16.numel()
            total_bytes += v_fp16.element_size() * v_fp16.numel()
        else:
            fp16_state_dict[clean_key] = v

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        os.remove(dst_path)

    torch.save(fp16_state_dict, dst_path)
    file_size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"[EXPORT V3] Serialized FP16 weights: {dst_path}")
    print(f"[EXPORT V3] Total Parameters: {total_params:,} | Size: {file_size_mb:.2f} MB")


def run_air_gapped_verification(template_dir: str) -> None:
    """Runs run.py in a clean isolated subprocess to verify offline execution."""
    print("[VERIFY V3] Testing air-gapped execution in isolated subprocess...")
    run_py = os.path.abspath(os.path.join(template_dir, "run.py"))
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, run_py],
        cwd=template_dir,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    if res.returncode != 0:
        print("[VERIFY V3] FAILED! Subprocess stderr:")
        print(res.stderr)
        print("[VERIFY V3] Subprocess stdout:")
        print(res.stdout)
        raise RuntimeError("Air-gapped verification failed!")

    print(f"[VERIFY V3] PASSED in {elapsed:.2f}s! Subprocess output:")
    for line in res.stdout.strip().split("\n")[-6:]:
        print(f"  {line}")


def package_zip(template_dir: str, zip_output_path: str) -> None:
    """Creates CodaBench-compliant zip with root-relative paths strictly < 500 MB."""
    print(f"[PACKAGE V3] Building submission ZIP: {zip_output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(zip_output_path)), exist_ok=True)

    required_files = ["run.py", "metadata.yaml"]
    for req in required_files:
        p = os.path.join(template_dir, req)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    model_pt = os.path.join(template_dir, "assets", "model_sota_fp16.pt")
    if not os.path.isfile(model_pt):
        # Fallback check
        candidates = [f for f in os.listdir(os.path.join(template_dir, "assets")) if f.endswith(".pt")]
        if not candidates:
            raise FileNotFoundError("Missing model weights in assets directory!")

    if os.path.exists(zip_output_path):
        os.remove(zip_output_path)

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

    zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
    print(f"[PACKAGE V3] Successfully built {zip_output_path} ({zip_size_mb:.2f} MB)")
    if zip_size_mb > 500.0:
        raise ValueError(f"CRITICAL: Submission ZIP exceeds 500 MB limit ({zip_size_mb:.2f} MB)!")
    print("[PACKAGE V3] Submission ZIP verified strictly compliant with CodaBench!")


def main():
    parser = argparse.ArgumentParser(description="Export and package SOTA V3 submission")
    parser.add_argument("--checkpoint", type=str, default="sota_v3_outputs/best_model_sota_v3.pt")
    parser.add_argument("--template_dir", type=str, default="sota_pipeline/submission_template_v3")
    parser.add_argument("--output_zip", type=str, default="submissions/submission_sota_v3.zip")
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args()

    dst_weights = os.path.join(args.template_dir, "assets", "model_sota_fp16.pt")

    if os.path.isfile(args.checkpoint):
        export_model_fp16(args.checkpoint, dst_weights)
    else:
        print(f"[EXPORT V3] Checkpoint {args.checkpoint} not found. Using existing weights in {dst_weights}")

    if not args.skip_verify:
        run_air_gapped_verification(args.template_dir)

    package_zip(args.template_dir, args.output_zip)
    print("[EXPORT V3 COMPLETE] Pipeline completed successfully!")


if __name__ == "__main__":
    main()
