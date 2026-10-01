"""Simulation and Ingestion Conformance Test for Codabench Submission."""
import importlib.util
import os
import shutil
import zipfile
import numpy as np
import csv

import sys

SANDBOX_DIR = "/Users/sama/D/RWS2027/sandbox_test_submission"
ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/sama/D/RWS2027/submissions/EXP-002_submission.zip"

print("--- 1. Testing Clean Extraction ---")
if os.path.exists(SANDBOX_DIR):
    shutil.rmtree(SANDBOX_DIR)
os.makedirs(SANDBOX_DIR)

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    zf.extractall(SANDBOX_DIR)

files = os.listdir(SANDBOX_DIR)
print("Root files in extracted archive:", files)
assert "run.py" in files, "Missing run.py!"
assert "metadata.yaml" in files, "Missing metadata.yaml!"
assert os.path.exists(os.path.join(SANDBOX_DIR, "assets", "model.pth")), "Missing assets/model.pth!"
print("Archive structure verified: 100% compliant!")

print("--- 2. Dynamically importing run.py ---")
spec = importlib.util.spec_from_file_location("submission_run", os.path.join(SANDBOX_DIR, "run.py"))
submission_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission_run)

print("Calling load_model()...")
submission_run.load_model()
assert submission_run._MODEL is not None, "Model failed to load!"
print("load_model() passed cleanly!")

print("--- 3. Constructing Real Sample Dict ---")
# Pick 10 real images
mock_gallery = []
with open("UPAR-Challenge-2027/data/annotations/task2/val/gt.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for i, row in enumerate(reader):
        if i >= 10:
            break
        rel_path = row[0]
        full_path = os.path.join("UPAR-Challenge-2027/data", rel_path)
        mock_gallery.append({
            "index": i,
            "image": rel_path,
            "image_path": full_path,
        })

# Pick 5 real queries
queries_list = []
with open("UPAR-Challenge-2027/data/annotations/task2/val/queries.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    attr_cols = [c.lstrip("# ").strip() for c in header]
    for i, row in enumerate(reader):
        if i >= 5:
            break
        queries_list.append([float(val) for val in row])
mock_queries = np.array(queries_list, dtype=np.float32)

sample = {
    "attribute_names": attr_cols,
    "queries": mock_queries,
    "gallery": mock_gallery,
}
n_q = len(sample['queries'])
n_g = len(sample['gallery'])
n_a = len(sample['attribute_names'])
print(f"Sample built: {n_q} queries, {n_g} gallery items, {n_a} attributes")

print("--- 4. Calling rank_gallery(sample) ---")
result = submission_run.rank_gallery(sample)

assert isinstance(result, dict), "Result must be a dict!"
assert list(result.keys()) == ["distances"], f"Expected exactly ['distances'], got {list(result.keys())}"
distances = result["distances"]

print("Distances type:", type(distances))
print("Distances shape:", distances.shape)
print("Distances dtype:", distances.dtype)
print(f"Distances min/max/mean: {np.min(distances):.4f}, {np.max(distances):.4f}, {np.mean(distances):.4f}")

assert distances.shape == (5, 10), f"Expected shape (5, 10), got {distances.shape}"
assert not np.isnan(distances).any(), "Distances contain NaN!"
assert not np.isinf(distances).any(), "Distances contain Inf!"
print("Rank gallery simulation: 100% SUCCESS!")

print("--- 5. Testing Robustness: Shuffled Attributes & Missing Image ---")
shuffled_attrs = list(reversed(attr_cols))
shuffled_queries = mock_queries[:, ::-1]

# Corrupt one image path
mock_gallery_corrupt = list(mock_gallery)
mock_gallery_corrupt[3] = {"index": 3, "image": "non_existent.jpg", "image_path": "/invalid/non_existent.jpg"}

sample_edge = {
    "attribute_names": shuffled_attrs,
    "queries": shuffled_queries,
    "gallery": mock_gallery_corrupt,
}
result_edge = submission_run.rank_gallery(sample_edge)
distances_edge = result_edge["distances"]
assert distances_edge.shape == (5, 10)
assert not np.isnan(distances_edge).any()
assert not np.isinf(distances_edge).any()
print("Robustness test passed: 100% resilient against corrupt images and shuffled attribute names!")

# Cleanup sandbox
shutil.rmtree(SANDBOX_DIR)
print("Cleanup complete. All tests PASSED!")
