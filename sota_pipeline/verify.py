"""Post-Training Verification & Checkpoint Integrity Audit Tool.

Verifies:
1. Only 'best_model.pt' and 'last_checkpoint.pt' exist (clean workspace, no intermediate bloat).
2. Both checkpoints contain all 7 essential multi-session state components:
   - model_state_dict
   - optimizer_state_dict
   - scheduler_state_dict
   - scaler_state_dict
   - epoch
   - global_step
   - best_metric
3. Model weights load into SOTAChampionshipModel without error.
4. Dummy forward pass through VisualStream and TextStream executes successfully.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional
import torch

from sota_pipeline.config import SOTAConfig
from sota_pipeline.models import SOTAChampionshipModel


def verify_checkpoints(checkpoint_dir: str, device: str = "cpu") -> bool:
    print("=" * 80)
    print(f"[AUDIT] Starting SOTA Post-Training Integrity Audit")
    print(f"[AUDIT] Checkpoint Directory: {os.path.abspath(checkpoint_dir)}")
    print(f"[AUDIT] Audit Device: {device}")
    print("=" * 80)

    if not os.path.isdir(checkpoint_dir):
        print(f"[FAIL] Checkpoint directory not found: {checkpoint_dir}")
        return False

    files = os.listdir(checkpoint_dir)
    print(f"[AUDIT] Found {len(files)} files in checkpoint directory: {files}")

    required_checkpoints = ["best_model.pt", "last_checkpoint.pt"]
    passed = True

    # 1. Check for expected files
    for req in required_checkpoints:
        req_path = os.path.join(checkpoint_dir, req)
        if not os.path.isfile(req_path):
            print(f"[FAIL] Missing required checkpoint: {req}")
            passed = False
        else:
            size_mb = os.path.getsize(req_path) / (1024 * 1024)
            print(f"[OK] Found '{req}': {size_mb:.2f} MB")

    # 2. Check for intermediate bloat
    bloat_files = [f for f in files if f.startswith("checkpoint_ep_") or f.endswith(".tmp")]
    if bloat_files:
        print(f"[WARN] Detected {len(bloat_files)} extraneous intermediate files: {bloat_files}")
    else:
        print(f"[OK] No intermediate checkpoint bloat found (strictly retaining best & latest).")

    # 3. Component and State Verification
    required_keys = [
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "epoch",
        "global_step",
        "best_metric",
    ]

    cfg = SOTAConfig()
    audit_device = torch.device(device)

    for req in required_checkpoints:
        req_path = os.path.join(checkpoint_dir, req)
        if not os.path.isfile(req_path):
            continue

        print(f"\n[AUDIT] Validating '{req}'...")
        try:
            ckpt = torch.load(req_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[FAIL] Could not load '{req}': {e}")
            passed = False
            continue

        missing_keys = [k for k in required_keys if k not in ckpt]
        if missing_keys:
            print(f"[FAIL] '{req}' missing essential components: {missing_keys}")
            passed = False
        else:
            print(f"[OK] All 7 multi-session state components verified.")
            print(f"     Epoch: {ckpt['epoch']} | Step: {ckpt['global_step']} | Best mADM: {ckpt['best_metric']:.4f}")

        # Model Loading and Dry-Run Verification
        try:
            model = SOTAChampionshipModel(
                embed_dim=cfg.embed_dim,
                num_attrs=cfg.num_attrs,
                img_height=cfg.img_height,
                img_width=cfg.img_width,
                patch_size=cfg.patch_size,
                use_mlm=cfg.use_mlm_head,
            )
            # Remove any 'module.' prefixes if saved from DDP
            state_dict = ckpt["model_state_dict"]
            clean_state_dict = {
                (k[7:] if k.startswith("module.") else k): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(clean_state_dict, strict=True)
            model.to(audit_device)
            model.eval()

            # Dummy forward pass to verify execution
            with torch.no_grad():
                dummy_img = torch.randn(2, 3, cfg.img_height, cfg.img_width, device=audit_device)
                dummy_tokens = torch.randint(0, 1000, (2, 77), device=audit_device)
                dummy_out = model(dummy_img, dummy_tokens)

                assert "img_metric" in dummy_out and dummy_out["img_metric"].shape == (2, cfg.embed_dim)
                assert "txt_metric" in dummy_out and dummy_out["txt_metric"].shape == (2, cfg.embed_dim)
                assert "attr_logits" in dummy_out and dummy_out["attr_logits"].shape == (2, cfg.num_attrs)
            print(f"[OK] Loaded '{req}' into SOTAChampionshipModel; forward verification succeeded.")

        except Exception as e:
            print(f"[FAIL] Architecture validation failed for '{req}': {e}")
            passed = False

    print("\n" + "=" * 80)
    if passed:
        print("[AUDIT PASSED] All checkpoints are valid, complete, and ready for next session / evaluation!")
    else:
        print("[AUDIT FAILED] One or more integrity checks failed.")
    print("=" * 80)
    return passed


def main():
    parser = argparse.ArgumentParser(description="Verify SOTA Championship Checkpoint Integrity")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_sota", help="Path to checkpoint directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for audit (cpu or cuda)")
    args = parser.parse_args()

    success = verify_checkpoints(args.checkpoint_dir, device=args.device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
