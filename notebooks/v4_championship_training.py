#!/usr/bin/env python3
"""
===================================================================================
 V4 CLIP ViT-L/14 CHAMPIONSHIP TRAINING — KAGGLE DUAL TESLA T4 NOTEBOOK
===================================================================================
 Competition: RWS2027 - UPAR Challenge Track 2: Attribute-based Person Retrieval
 Model: CLIP ViT-L/14 (304M params) + Natural Language Attribute Templates
 Target: ≥0.68 mADM on CodaBench server
 Hardware: Kaggle Dual Tesla T4 (14.5GB VRAM each)
 Sessions: 2-3 × 12h sessions with full checkpoint resume
===================================================================================

INSTRUCTIONS:
  This script is designed to be pasted cell-by-cell into a Kaggle notebook.
  Each cell is demarcated by a comment: ## === CELL N: <description> ===
  
  SESSION 1: Run all cells. Training runs epochs 1-50 (~10 hours).
  SESSION 2: Run cells 1-3, then cell 6 with --resume_from flag. Runs epochs 51-100.
  SESSION 3: Run cells 1-3, then cell 6 with --resume_from flag. Runs epochs 101-120.
  FINAL: Run cells 7-9 to export and package the submission.

PREREQUISITES:
  - Kaggle GPU notebook (Tesla T4 × 2)
  - Dataset: hinewk/upar-binary-dataset mounted as input
  - Secret: GITHUB_TOKEN with repo access
"""

# ============================================================================
## === CELL 1: HARDWARE INSPECTION & ENVIRONMENT SETUP ===
# ============================================================================
print("=" * 80)
print("[HARDWARE] Inspecting Dual Tesla T4 Accelerators & Host Environment...")
print("=" * 80)

import torch
import os, sys, subprocess

print(f"CUDA Available: {torch.cuda.is_available()} | Device Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  -> GPU {i}: {props.name} | Total VRAM: {props.total_mem / 1e9:.2f} GB")

cpu_count = os.cpu_count() or 4
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
except ImportError:
    ram_gb = 31.0
print(f"Host CPU Cores: {cpu_count} | Total RAM: {ram_gb:.2f} GB")
print("=" * 80)


# ============================================================================
## === CELL 2: INSTALL DEPENDENCIES ===
# ============================================================================
print("[DEPS] Installing OpenCLIP, Timm, and Accelerate dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "open_clip_torch>=2.24.0", "timm>=1.0.0", "accelerate>=0.33.0", "h5py"
])

# Verify imports
import timm
import open_clip
import accelerate
import h5py
print(f"[DEPS] timm={timm.__version__}, open_clip={open_clip.__version__}, "
      f"accelerate={accelerate.__version__}, h5py={h5py.__version__}")
print("[DEPS] All dependencies installed successfully!")


# ============================================================================
## === CELL 3: CLONE REPOSITORY ===
# ============================================================================
import os

# Get GitHub token from Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    GITHUB_TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
    print(f"[AUTH] Successfully retrieved GITHUB_TOKEN from Kaggle Secrets!")
except Exception as e:
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
    if not GITHUB_TOKEN:
        print(f"[AUTH] WARNING: No GITHUB_TOKEN found. Clone may fail for private repos.")
    else:
        print(f"[AUTH] Using GITHUB_TOKEN from environment variable.")

REPO_URL = f"https://{GITHUB_TOKEN}@github.com/Aditya-Lingam-9000/upar_challenge_26_rws27.git"
WORK_DIR = "/kaggle/working/upar_challenge_26_rws27"

if os.path.isdir(WORK_DIR):
    print(f"[GIT] Repository already exists at {WORK_DIR}, pulling latest...")
    subprocess.check_call(["git", "-C", WORK_DIR, "pull", "--ff-only"])
else:
    print(f"[GIT] Cloning repository into {WORK_DIR}...")
    subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, WORK_DIR])

print(f"[GIT] Successfully synced repository!")
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)
print(f"[SYSTEM] Working directory set to: {os.getcwd()}")


# ============================================================================
## === CELL 4: AUTO-DETECT DATASET ===
# ============================================================================
import glob

H5_CANDIDATES = [
    "/kaggle/input/upar-binary-dataset/upar_binary_dataset.h5",
    "/kaggle/input/datasets/hinewk/upar-binary-dataset/upar_binary_dataset.h5",
    "/kaggle/input/upar-binary-dataset/*.h5",
    os.path.join(WORK_DIR, "upar_binary_dataset.h5"),
]

H5_PATH = None
for pattern in H5_CANDIDATES:
    matches = glob.glob(pattern)
    if matches:
        H5_PATH = matches[0]
        break

if H5_PATH is None:
    # Search all input directories
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith('.h5'):
                H5_PATH = os.path.join(root, f)
                break
        if H5_PATH:
            break

assert H5_PATH is not None, "ERROR: Could not find HDF5 dataset! Ensure it's mounted as a Kaggle input dataset."
print(f"[DATASET] Found HDF5 dataset at: {H5_PATH}")
print(f"[DATASET] File size: {os.path.getsize(H5_PATH) / 1e6:.1f} MB")

# Quick validation
import h5py
with h5py.File(H5_PATH, 'r') as f:
    train_n = f['train/labels'].shape[0]
    val_n = f['val/labels'].shape[0]
    num_attrs = f['train/labels'].shape[1]
    print(f"[DATASET] Train: {train_n:,} images | Val: {val_n:,} images | Attributes: {num_attrs}")


# ============================================================================
## === CELL 5: VERIFY MODEL LOADS ===
# ============================================================================
import timm
import torch

print("[VERIFY] Testing CLIP ViT-L/14 model creation via timm...")
test_model = timm.create_model('vit_large_patch14_clip_224.openai', pretrained=True, num_classes=0)
dummy_input = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = test_model(dummy_input)
print(f"[VERIFY] Backbone output shape: {out.shape}")  # Should be [2, 1024]
assert out.shape == (2, 1024), f"Unexpected output shape: {out.shape}"

# Count params
total_params = sum(p.numel() for p in test_model.parameters())
print(f"[VERIFY] Backbone parameters: {total_params:,} ({total_params * 2 / 1e6:.1f} MB in FP16)")

# Verify CLIP text encoder
import open_clip
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-L-14')
test_text = tokenizer(["A photo of a person", "A surveillance photo of a young male"])
with torch.no_grad():
    text_out = clip_model.encode_text(test_text)
print(f"[VERIFY] CLIP text encoder output shape: {text_out.shape}")  # Should be [2, 768]
text_dim = text_out.shape[1]
print(f"[VERIFY] Text embedding dimension: {text_dim}")

del test_model, clip_model, dummy_input, test_text
torch.cuda.empty_cache()
print("[VERIFY] All model verifications passed!")


# ============================================================================
## === CELL 6: LAUNCH TRAINING ===
# ============================================================================
# For SESSION 1: Run without --resume_from
# For SESSION 2+: Run with --resume_from pointing to latest checkpoint

import subprocess, sys, os

OUTPUT_DIR = "/kaggle/working/v4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- CONFIGURE THESE FOR EACH SESSION ----
RESUME_FROM = None  # Set to checkpoint path for sessions 2+
# Example: RESUME_FROM = "/kaggle/working/v4_outputs/checkpoint_epoch_50.pt"

EPOCHS = 120        # Total epochs (across all sessions)
BATCH_SIZE = 32     # Per-GPU batch size
GRAD_ACCUM = 2      # Gradient accumulation steps
BACKBONE_LR = 5e-6  # Backbone learning rate
HEAD_LR = 5e-4      # Head learning rate
SAVE_EVERY = 5      # Save checkpoint every N epochs
EVAL_EVERY = 5      # Evaluate every N epochs
# -------------------------------------------

cmd = [
    sys.executable, os.path.join(WORK_DIR, "sota_pipeline", "train_v4.py"),
    "--h5_path", H5_PATH,
    "--output_dir", OUTPUT_DIR,
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--grad_accum", str(GRAD_ACCUM),
    "--backbone_lr", str(BACKBONE_LR),
    "--head_lr", str(HEAD_LR),
    "--save_every", str(SAVE_EVERY),
    "--eval_every", str(EVAL_EVERY),
]

if RESUME_FROM and os.path.isfile(RESUME_FROM):
    cmd.extend(["--resume_from", RESUME_FROM])
    print(f"[TRAINING] Resuming from checkpoint: {RESUME_FROM}")
else:
    print(f"[TRAINING] Starting fresh training for {EPOCHS} epochs")

print(f"[TRAINING] Command: {' '.join(cmd)}")
print(f"[TRAINING] Output directory: {OUTPUT_DIR}")
print(f"[TRAINING] Effective batch size: {BATCH_SIZE} × {GRAD_ACCUM} × {torch.cuda.device_count()} = {BATCH_SIZE * GRAD_ACCUM * max(1, torch.cuda.device_count())}")
print("=" * 80)

# Use accelerate launch for multi-GPU
accelerate_cmd = [
    sys.executable, "-m", "accelerate.commands.launch",
    "--multi_gpu",
    "--num_processes", str(torch.cuda.device_count()),
    "--mixed_precision", "fp16",
] + cmd[1:]  # Remove the python executable, accelerate adds it

print(f"[TRAINING] Accelerate command: {' '.join(accelerate_cmd)}")

process = subprocess.Popen(
    accelerate_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd=WORK_DIR,
)

# Stream output
for line in iter(process.stdout.readline, ''):
    print(line, end='', flush=True)

process.wait()
print(f"\n[TRAINING] Process exited with code: {process.returncode}")

if process.returncode != 0:
    print("[TRAINING] WARNING: Training ended with non-zero exit code!")
else:
    print("[TRAINING] Training completed successfully!")


# ============================================================================
## === CELL 7: LIST CHECKPOINTS ===
# ============================================================================
import os, glob

OUTPUT_DIR = "/kaggle/working/v4_outputs"
print(f"\n[CHECKPOINTS] Contents of {OUTPUT_DIR}:")
for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.pt"))):
    size_mb = os.path.getsize(f) / 1e6
    print(f"  {os.path.basename(f)}: {size_mb:.1f} MB")


# ============================================================================
## === CELL 8: EXPORT FP16 MODEL & PACKAGE SUBMISSION ZIP ===
# ============================================================================
import subprocess, sys, os

OUTPUT_DIR = "/kaggle/working/v4_outputs"
WORK_DIR = "/kaggle/working/upar_challenge_26_rws27"

# Find best checkpoint
best_ckpt = os.path.join(OUTPUT_DIR, "best_model_v4.pt")
if not os.path.isfile(best_ckpt):
    # Fallback to latest checkpoint
    import glob
    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, "checkpoint_epoch_*.pt")))
    if ckpts:
        best_ckpt = ckpts[-1]
        print(f"[EXPORT] Using latest checkpoint: {best_ckpt}")
    else:
        raise FileNotFoundError(f"No checkpoints found in {OUTPUT_DIR}")
else:
    print(f"[EXPORT] Using best checkpoint: {best_ckpt}")

subprocess.check_call([
    sys.executable, os.path.join(WORK_DIR, "sota_pipeline", "export_submission_v4.py"),
    "--checkpoint", best_ckpt,
    "--output_dir", os.path.join(WORK_DIR, "submissions"),
    "--submission_name", "submission_v4_championship",
])

# List final submission
import glob
for f in sorted(glob.glob(os.path.join(WORK_DIR, "submissions", "submission_v4*"))):
    if f.endswith('.zip'):
        size_mb = os.path.getsize(f) / 1e6
        print(f"\n[SUBMISSION] {os.path.basename(f)}: {size_mb:.1f} MB")
        if size_mb > 500:
            print("[SUBMISSION] WARNING: ZIP exceeds 500MB! Running INT8 quantization...")
            subprocess.check_call([
                sys.executable, os.path.join(WORK_DIR, "sota_pipeline", "export_submission_v4.py"),
                "--checkpoint", best_ckpt,
                "--output_dir", os.path.join(WORK_DIR, "submissions"),
                "--submission_name", "submission_v4_championship_int8",
                "--int8",
            ])


# ============================================================================
## === CELL 9: PUSH TO GITHUB & COPY TO OUTPUT ===
# ============================================================================
import subprocess, shutil, os, glob

WORK_DIR = "/kaggle/working/upar_challenge_26_rws27"

# Push updated code to GitHub
try:
    subprocess.check_call(["git", "-C", WORK_DIR, "add", "-A"])
    subprocess.check_call(["git", "-C", WORK_DIR, "commit", "-m",
                          "feat(v4): CLIP ViT-L/14 championship model training complete"])
    subprocess.check_call(["git", "-C", WORK_DIR, "push"])
    print("[GIT] Successfully pushed to GitHub!")
except subprocess.CalledProcessError as e:
    print(f"[GIT] Git push failed (may need manual push): {e}")

# Copy submission ZIP and best checkpoint to Kaggle output
for f in glob.glob(os.path.join(WORK_DIR, "submissions", "submission_v4*.zip")):
    dst = os.path.join("/kaggle/working", os.path.basename(f))
    shutil.copy2(f, dst)
    print(f"[OUTPUT] Copied {os.path.basename(f)} to /kaggle/working/ ({os.path.getsize(dst)/1e6:.1f} MB)")

# Also copy best model checkpoint
OUTPUT_DIR = "/kaggle/working/v4_outputs"
best_model = os.path.join(OUTPUT_DIR, "best_model_v4.pt")
if os.path.isfile(best_model):
    dst = os.path.join("/kaggle/working", "best_model_v4.pt")
    shutil.copy2(best_model, dst)
    print(f"[OUTPUT] Copied best_model_v4.pt to /kaggle/working/ ({os.path.getsize(dst)/1e6:.1f} MB)")

print("\n" + "=" * 80)
print("[DONE] V4 Championship Training Pipeline Complete!")
print("=" * 80)
print("Next steps:")
print("1. Download submission_v4_championship.zip from the Kaggle output")
print("2. Upload to CodaBench: https://www.codabench.org/competitions/18196/")
print("3. Wait for evaluation (< 300 seconds on A10 GPU)")
print("=" * 80)
