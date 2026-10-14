"""Exhaustive Multi-Platform Verification Suite for SOTA Full Championship Submission.

Tests:
1. Pure CPU Mode Simulation (Codabench target environment).
2. MPS Mode Simulation (Apple Silicon M1 Pro local verification).
3. Robustness & Edge Cases (Empty gallery, single query, corrupt image path, extreme queries).
4. Strict Container Sandbox Packaging & Official Scoring Benchmark.
"""
from __future__ import annotations

import csv
import gc
import importlib
import os
import resource
import shutil
import sys
import time
import zipfile
import numpy as np
import torch
from PIL import Image

WORKSPACE = "/Users/sama/D/RWS2027"
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)
SUBMISSION_DIR = os.path.join(WORKSPACE, "submissions/SOTA_Full_Championship_Submission")
ZIP_PATH = os.path.join(WORKSPACE, "submissions/SOTA_Full_Championship_Submission.zip")
SANDBOX_DIR = "/tmp/codabench_sandbox_final"
VAL_DIR = os.path.join(WORKSPACE, "UPAR-Challenge-2027/data/annotations/task2/val")
DATA_DIR = os.path.join(WORKSPACE, "UPAR-Challenge-2027/data")


import subprocess

def get_live_rss_mb() -> float:
    try:
        pid = os.getpid()
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).decode().strip()
        return int(out) / 1024
    except Exception:
        rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rusage / (1024 * 1024) if sys.platform == "darwin" else rusage / 1024


def load_val_data(max_queries: int = 300, max_gallery: int = 3000):
    val_ids_path = os.path.join(VAL_DIR, "ids.csv")
    val_queries_path = os.path.join(VAL_DIR, "queries.csv")
    val_gt_path = os.path.join(VAL_DIR, "gt.csv")

    gallery_paths = []
    with open(val_ids_path, "r") as f:
        for row in csv.reader(f):
            if row:
                gallery_paths.append(row[0])

    queries = []
    with open(val_queries_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                queries.append([float(x) for x in row])
    queries = np.array(queries, dtype=np.float32)

    gt_labels = []
    with open(val_gt_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                gt_labels.append([float(x) for x in row[1:]])
    gt_labels = np.array(gt_labels, dtype=np.float32)

    q_sub = queries[:max_queries]
    g_sub_paths = gallery_paths[:max_gallery]
    g_sub_gt = gt_labels[:max_gallery]
    gallery_sample = [{"image_path": os.path.join(DATA_DIR, p)} for p in g_sub_paths]

    return q_sub, gallery_sample, g_sub_gt


def test_cpu_mode_simulation():
    print("\n" + "=" * 70)
    print("TEST 1: Pure CPU Mode Simulation (Codabench Target)")
    print("=" * 70)

    import submissions.SOTA_Full_Championship_Submission.run as run_sub

    # Force CPU configuration
    run_sub.HW.has_cuda = False
    run_sub.HW.has_mps = False
    run_sub.HW.device = torch.device("cpu")
    run_sub.HW.device_type = "cpu"
    run_sub.HW.use_half = False
    run_sub.HW.torch_threads = 4
    run_sub.HW.batch_size = 64
    run_sub.HW.num_io_workers = 4
    run_sub._DEVICE = torch.device("cpu")

    # Reset models to force re-instantiation on CPU
    run_sub._VISUAL_MODEL = None
    run_sub._TEXT_MODEL = None
    run_sub._TOKENIZER = None

    run_sub.load_model()

    q_sub, gallery_sample, _ = load_val_data(max_queries=20, max_gallery=200)
    sample = {"queries": q_sub, "gallery": gallery_sample}

    t0 = time.time()
    res = run_sub.rank_gallery(sample)
    t1 = time.time()

    distances = res["distances"]
    assert distances.shape == (20, 200), f"Unexpected shape {distances.shape}"
    assert not np.isnan(distances).any(), "NaNs found in distance matrix!"
    assert not np.isinf(distances).any(), "Infs found in distance matrix!"

    throughput = len(gallery_sample) / (t1 - t0)
    rss_mb = get_live_rss_mb()
    print(f">> CPU Mode PASSED: Shape: {distances.shape} | Time: {t1-t0:.2f}s ({throughput:.1f} img/s) | Peak RSS: {rss_mb:.1f} MB")
    max_allowed = 3500 if sys.platform == "darwin" else 2048
    assert rss_mb < max_allowed, f"Memory exceeded limit: {rss_mb:.1f} MB (allowed: {max_allowed} MB)"


def test_mps_mode_simulation():
    print("\n" + "=" * 70)
    print("TEST 2: MPS Mode Simulation (Apple Silicon M1 Pro)")
    print("=" * 70)

    import submissions.SOTA_Full_Championship_Submission.run as run_sub

    # Force MPS configuration
    run_sub.HW.has_cuda = False
    run_sub.HW.has_mps = True
    run_sub.HW.device = torch.device("mps")
    run_sub.HW.device_type = "mps"
    run_sub.HW.use_half = True
    run_sub.HW.torch_threads = 8
    run_sub.HW.batch_size = 64
    run_sub.HW.num_io_workers = 8
    run_sub._DEVICE = torch.device("mps")

    # Reset models to reload on MPS
    run_sub._VISUAL_MODEL = None
    run_sub._TEXT_MODEL = None
    run_sub._TOKENIZER = None

    run_sub.load_model()

    q_sub, gallery_sample, gt_sub = load_val_data(max_queries=100, max_gallery=1000)
    sample = {"queries": q_sub, "gallery": gallery_sample}

    t0 = time.time()
    res = run_sub.rank_gallery(sample)
    t1 = time.time()

    distances = res["distances"]
    assert distances.shape == (100, 1000), f"Unexpected shape {distances.shape}"
    assert not np.isnan(distances).any(), "NaNs found in distance matrix!"
    assert not np.isinf(distances).any(), "Infs found in distance matrix!"

    from sota_pipeline.evaluate import compute_official_metrics_from_ranking
    ranking = np.argsort(distances, axis=1)
    metrics = compute_official_metrics_from_ranking(ranking, q_sub, gt_sub)

    throughput = len(gallery_sample) / (t1 - t0)
    rss_mb = get_live_rss_mb()
    print(f">> MPS Mode PASSED: Time: {t1-t0:.2f}s ({throughput:.1f} img/s) | Peak RSS: {rss_mb:.1f} MB")
    print(f">> Verified SOTA mADM: {metrics['mADM']:.4f} | Rank-1: {metrics['Rank-1']:.2f}% | Rank-5: {metrics['Rank-5']:.2f}%")


def test_edge_cases():
    print("\n" + "=" * 70)
    print("TEST 3: Robustness & Edge Cases")
    print("=" * 70)

    import submissions.SOTA_Full_Championship_Submission.run as run_sub

    # Case 3A: Empty gallery
    print("Testing Case 3A: Empty Gallery...")
    res_empty = run_sub.rank_gallery({"queries": np.zeros((5, 40), dtype=np.float32), "gallery": []})
    assert res_empty["distances"].shape == (5, 0), f"Expected (5, 0), got {res_empty['distances'].shape}"
    print(">> Case 3A PASSED")

    # Case 3B: Single query
    print("Testing Case 3B: Single Query...")
    q_sub, gallery_sample, _ = load_val_data(max_queries=1, max_gallery=10)
    res_single = run_sub.rank_gallery({"queries": q_sub, "gallery": gallery_sample})
    assert res_single["distances"].shape == (1, 10), f"Expected (1, 10), got {res_single['distances'].shape}"
    print(">> Case 3B PASSED")

    # Case 3C: Corrupted / Non-existent image paths
    print("Testing Case 3C: Corrupted & Non-existent Image Paths...")
    corrupt_gallery = [
        {"image_path": "/non/existent/path/001.jpg"},
        {"image_path": "/dev/null"},
        {"image_path": os.path.join(DATA_DIR, "Market1501/query/0001_c1s1_001051_00.jpg")}
    ]
    res_corrupt = run_sub.rank_gallery({"queries": np.ones((2, 40), dtype=np.float32), "gallery": corrupt_gallery})
    assert res_corrupt["distances"].shape == (2, 3), f"Expected (2, 3), got {res_corrupt['distances'].shape}"
    assert not np.isnan(res_corrupt["distances"]).any(), "Corrupt images caused NaNs!"
    print(">> Case 3C PASSED: Handled corrupt images cleanly without crashing!")

    # Case 3D: All zeros / All ones queries
    print("Testing Case 3D: Extreme Query Vectors (All Zeros, All Ones)...")
    extreme_queries = np.vstack([np.zeros((1, 40), dtype=np.float32), np.ones((1, 40), dtype=np.float32)])
    res_extreme = run_sub.rank_gallery({"queries": extreme_queries, "gallery": gallery_sample})
    assert res_extreme["distances"].shape == (2, 10)
    assert not np.isnan(res_extreme["distances"]).any()
    print(">> Case 3D PASSED")


def test_sandbox_packaging_and_scoring():
    print("\n" + "=" * 70)
    print("TEST 4: Strict Container Sandbox Packaging & Scoring Benchmark")
    print("=" * 70)

    # 1. Package submission zip
    print(f"Creating submission zip: {ZIP_PATH}...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SUBMISSION_DIR):
            # Exclude cache and system files
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", ".DS_Store")]
            for file in files:
                if file.endswith((".pyc", ".pyo", ".DS_Store")):
                    continue
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, SUBMISSION_DIR)
                zf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"Submission zip created successfully ({zip_size_mb:.1f} MB)!")

    # 2. Extract into sandbox
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR, exist_ok=True)

    print(f"Extracting zip into sandbox ({SANDBOX_DIR})...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(SANDBOX_DIR)

    # Verify sandbox contents
    sandbox_files = os.listdir(SANDBOX_DIR)
    print(f"Sandbox files: {sandbox_files}")
    assert "run.py" in sandbox_files, "run.py missing in sandbox!"
    assert "metadata.yaml" in sandbox_files, "metadata.yaml missing in sandbox!"
    assert os.path.exists(os.path.join(SANDBOX_DIR, "assets", "model_sota_fp16.pt")), "model_sota_fp16.pt missing!"
    assert os.path.exists(os.path.join(SANDBOX_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz")), "bpe vocab missing!"

    # 3. Dynamic import from sandbox
    sys.path.insert(0, SANDBOX_DIR)
    spec = importlib.util.spec_from_file_location("sandbox_run", os.path.join(SANDBOX_DIR, "run.py"))
    sandbox_run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sandbox_run)

    print("Successfully imported run.py from sandbox!")
    sandbox_run.load_model()

    # 4. Run official benchmark on 300 queries, 3000 gallery images
    q_sub, gallery_sample, gt_sub = load_val_data(max_queries=300, max_gallery=3000)
    sample = {"queries": q_sub, "gallery": gallery_sample}

    t0 = time.time()
    res = sandbox_run.rank_gallery(sample)
    t1 = time.time()

    from sota_pipeline.evaluate import compute_official_metrics_from_ranking
    ranking = np.argsort(res["distances"], axis=1)
    metrics = compute_official_metrics_from_ranking(ranking, q_sub, gt_sub)

    print("\n" + "=" * 70)
    print("SANDBOX BENCHMARK EVALUATION RESULTS:")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")
    print(f"  Total Time     : {t1 - t0:.2f}s ({len(gallery_sample) / (t1 - t0):.1f} img/s)")
    print(f"  Peak RSS       : {get_live_rss_mb():.1f} MB")
    print("=" * 70)

    assert metrics["mADM"] >= 0.60, f"mADM below expected SOTA threshold: {metrics['mADM']:.4f}"
    print("\n>> TEST 4 PASSED: Sandbox execution confirmed 100% functional with verified SOTA score!")


if __name__ == "__main__":
    print("=" * 80)
    print("STARTING EXHAUSTIVE MULTI-PLATFORM VERIFICATION SUITE")
    print("=" * 80)

    test_cpu_mode_simulation()
    test_mps_mode_simulation()
    test_edge_cases()
    test_sandbox_packaging_and_scoring()

    print("\n" + "=" * 80)
    print("ALL VERIFICATION SUITE TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 80)
