#!/usr/bin/env python3
"""
SOTA Championship V2: Submission Packaging & Verification Engine.

Converts trained checkpoint to FP16, copies assets, executes self-contained
air-gapped validation test, and packages into a CodaBench-compliant ZIP.
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
from typing import Dict, Any
import numpy as np
import torch


def export_model_fp16(src_checkpoint: str, dst_path: str) -> None:
    """Extracts model weights, casts floating tensors to FP16, and saves."""
    print(f"[EXPORT] Loading trained checkpoint from: {src_checkpoint}...")
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
    print(f"[EXPORT] Serialized FP16 weights: {dst_path}")
    print(f"[EXPORT] Total Parameters: {total_params:,} | Size: {file_size_mb:.2f} MB")


def run_air_gapped_verification(template_dir: str) -> None:
    """Runs run.py in a clean subprocess from template_dir to verify offline execution."""
    print("[VERIFY] Testing air-gapped execution in isolated subprocess...")
    run_py = os.path.abspath(os.path.join(template_dir, "run.py"))
    t0 = time.time()
    res = subprocess.run(
        [sys.executable, run_py],
        cwd=template_dir,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - t0
    if res.returncode != 0:
        print("[VERIFY] FAILED! Subprocess stderr:")
        print(res.stderr)
        print("[VERIFY] Subprocess stdout:")
        print(res.stdout)
        raise RuntimeError("Air-gapped verification failed!")
    print(f"[VERIFY] PASSED in {elapsed:.2f}s! Subprocess output:")
    for line in res.stdout.strip().split("\n")[-6:]:
        print(f"  {line}")


def package_zip(template_dir: str, zip_output_path: str) -> None:
    """Creates CodaBench-compliant zip with root-relative paths."""
    print(f"[PACKAGE] Building submission ZIP: {zip_output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(zip_output_path)), exist_ok=True)

    required_files = ["run.py", "metadata.yaml"]
    for req in required_files:
        p = os.path.join(template_dir, req)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    model_pt = os.path.join(template_dir, "assets", "model_sota_fp16.pt")
    vocab_gz = os.path.join(template_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.isfile(model_pt):
        raise FileNotFoundError(f"Missing model weights: {model_pt}")
    if not os.path.isfile(vocab_gz):
        raise FileNotFoundError(f"Missing BPE vocab: {vocab_gz}")

    with zipfile.ZipFile(zip_output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(template_dir):
            dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".DS_Store"]]
            for f in files:
                if f.endswith((".pyc", ".DS_Store")):
                    continue
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, template_dir)
                zf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
    print(f"[PACKAGE] Successfully packaged {zip_output_path} ({zip_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export and package SOTA Championship V2 submission.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_sota_v2/best_model.pt", help="Source checkpoint")
    parser.add_argument("--template_dir", type=str, default="sota_pipeline/submission_template_v2", help="Template directory")
    parser.add_argument("--output_zip", type=str, default="submissions/submission_sota_v2.zip", help="Destination ZIP path")
    parser.add_argument("--skip_verify", action="store_true", help="Skip self-test verification")
    args = parser.parse_args()

    target_weights = os.path.join(args.template_dir, "assets", "model_sota_fp16.pt")
    if os.path.isfile(args.checkpoint):
        export_model_fp16(args.checkpoint, target_weights)
    else:
        print(f"[WARN] Checkpoint {args.checkpoint} not found. Using existing {target_weights} if available.")

    if not args.skip_verify:
        run_air_gapped_verification(args.template_dir)

    package_zip(args.template_dir, args.output_zip)
    print("[COMPLETE] Submission package ready for upload to CodaBench!")


if __name__ == "__main__":
    main()
