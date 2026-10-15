"""High-Throughput RAM-Cached Dataset Loader & Surveillance Degradation Engine.

Features:
1. High-Throughput RAM Cache: Preloads images into system memory in ~25-45s,
   completely eliminating disk I/O bottlenecks during training.
2. Surveillance Degradation Simulation (SRDM): Replicates MEVID 75px blurry video tracklets,
   motion blur, and DVR compression artifacts.
3. Dynamic Multi-Modal Batch Formatter: Delivers image tensors, 40-D attribute vectors,
   full tokenized descriptions, masked token sequences for MLM, and sub-strip tokens.
4. Robust Path Resolver: Seamlessly runs on local Apple Silicon and Kaggle Cloud Dual T4.
"""
from __future__ import annotations

import io
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from sota_pipeline.prompt_engine import (
    CANONICAL_ATTRIBUTES,
    DynamicPromptEngine,
)


class SurveillanceResolutionDrop:
    """Simulates distant security cameras where subject crops span <75px."""

    def __init__(self, min_height: int = 48, max_height: int = 96, p: float = 0.50):
        self.min_height = min_height
        self.max_height = max_height
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            w, h = img.size
            target_h = random.randint(self.min_height, self.max_height)
            target_w = max(16, int(w * (target_h / h)))
            down = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            return down.resize((w, h), Image.Resampling.BILINEAR)
        return img


class SurveillanceMotionBlur:
    """Simulates rapid camera pan or pedestrian motion blur in surveillance tracklets."""

    def __init__(self, p: float = 0.30):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(1.0, 2.5)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class SurveillanceJPEGCompression:
    """Simulates surveillance DVR encoding and blocking compression artifacts."""

    def __init__(self, min_quality: int = 30, max_quality: int = 65, p: float = 0.40):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            quality = random.randint(self.min_quality, self.max_quality)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        return img


def get_sota_transforms(
    height: int = 336,
    width: int = 168,
    is_train: bool = True,
    use_srdm: bool = True
) -> Callable:
    """Builds transform pipeline with surveillance degradation simulation."""
    if is_train:
        transforms_list = [
            T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
        ]
        if use_srdm:
            transforms_list.extend([
                SurveillanceResolutionDrop(min_height=48, max_height=96, p=0.50),
                SurveillanceMotionBlur(p=0.35),
                SurveillanceJPEGCompression(min_quality=35, max_quality=70, p=0.40),
            ])
        transforms_list.extend([
            T.RandomHorizontalFlip(p=0.50),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return T.Compose(transforms_list)
    else:
        return T.Compose([
            T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class SOTAUPARDataset(Dataset):
    """
    High-Performance RAM-Cached Multi-Modal Dataset.
    Supports continuous 1024-D cross-modal retrieval, MLM, and strip consistency.
    Supports both Rank 1 Single HDF5 Binary Container (.h5) and legacy directories.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        img_height: int = 336,
        img_width: int = 168,
        use_srdm: bool = True,
        preload_to_ram: bool = True,
        max_samples: Optional[int] = None,
        prompt_engine: Optional[DynamicPromptEngine] = None,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.is_train = (split == "train")
        self.transform = get_sota_transforms(
            height=img_height,
            width=img_width,
            is_train=self.is_train,
            use_srdm=use_srdm
        )
        self.prompt_engine = prompt_engine or DynamicPromptEngine(p_dropout=0.15 if self.is_train else 0.0)

        # Detect Rank 1 HDF5 binary container
        self.is_h5 = False
        self.h5_path: Optional[str] = None
        self._h5_handle: Optional[Any] = None

        if os.path.isfile(dataset_root) and dataset_root.endswith((".h5", ".hdf5")):
            self.is_h5 = True
            self.h5_path = os.path.abspath(dataset_root)
        elif os.path.isdir(dataset_root):
            h5_default = os.path.join(dataset_root, "upar_binary_dataset.h5")
            if os.path.isfile(h5_default):
                self.is_h5 = True
                self.h5_path = h5_default
            else:
                for fname in os.listdir(dataset_root):
                    if fname.endswith((".h5", ".hdf5")):
                        self.is_h5 = True
                        self.h5_path = os.path.join(dataset_root, fname)
                        break

        # Load metadata
        self.samples, self.labels = self._load_metadata()

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
            self.labels = self.labels[:max_samples]

        self.attr_labels = self.labels
        self.num_samples = len(self.samples)
        src_desc = f"HDF5: {os.path.basename(self.h5_path)}" if self.is_h5 else "Directory"
        print(f"[{split.upper()}] Initialized dataset with {self.num_samples:,} samples ({src_desc}).")

        # High-Speed RAM Pre-caching
        self.ram_cache: Optional[List[Optional[bytes]]] = None
        if preload_to_ram and self.num_samples > 0:
            self._preload_images_to_ram()

    def _load_metadata(self) -> Tuple[List[str], np.ndarray]:
        """Loads image names and 40-D attribute labels from HDF5 or dataset_all.pkl."""
        if self.is_h5 and self.h5_path:
            import h5py
            with h5py.File(self.h5_path, "r") as h5f:
                if self.split not in h5f:
                    raise KeyError(f"Split '{self.split}' not found in HDF5 {self.h5_path}. Available: {list(h5f.keys())}")
                labels = h5f[self.split]["labels"][:]
                raw_names = h5f[self.split]["image_names"][:]
                samples = [
                    (n.decode("utf-8") if isinstance(n, bytes) else str(n))
                    for n in raw_names
                ]
            return samples, labels.astype(np.float32)

        # Check annotations CSV (e.g. UPAR-Challenge-2027/data/annotations/val.csv or train.csv)
        csv_candidates = [
            os.path.join(self.dataset_root, "annotations", f"{self.split}.csv"),
            os.path.join(self.dataset_root, f"{self.split}.csv"),
            os.path.join(os.path.dirname(self.dataset_root), "annotations", f"{self.split}.csv"),
            os.path.join(self.dataset_root, "annotations", "task1", f"{self.split}", "gt.csv"),
            os.path.join(self.dataset_root, "annotations", "task2", f"{self.split}", "gt.csv"),
        ]
        for csv_path in csv_candidates:
            if os.path.isfile(csv_path):
                import csv
                samples = []
                labels_list = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row in reader:
                        if not row or len(row) < 41:
                            continue
                        rel_path = row[0].lstrip("# ").strip()
                        img_path = os.path.join(self.dataset_root, rel_path)
                        if not os.path.isfile(img_path):
                            img_path = os.path.join(os.path.dirname(self.dataset_root), rel_path)
                        samples.append(img_path)
                        labels_list.append([float(x) for x in row[1:41]])
                print(f"[Dataset] Successfully loaded {len(samples):,} samples from {csv_path}")
                return samples, np.asarray(labels_list, dtype=np.float32)

        pkl_path = os.path.join(self.dataset_root, "dataset_all.pkl")
        if not os.path.isfile(pkl_path):
            # Check parent directory
            pkl_path = os.path.join(os.path.dirname(self.dataset_root), "dataset_all.pkl")

        if os.path.isfile(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            partition = data.get("partition", {})
            split_key = "train" if self.is_train else "val"
            indices = partition.get(split_key, [])
            if isinstance(indices, list) and len(indices) > 0:
                split_indices = indices[-1] if len(indices) == 9 else indices[0]
            else:
                split_indices = np.arange(len(data["image_name"]))

            raw_names = data["image_name"]
            raw_labels = data["label"]

            samples = []
            labels_list = []
            for idx in split_indices:
                img_rel = raw_names[idx]
                img_path = os.path.join(self.dataset_root, img_rel)
                if not os.path.isfile(img_path):
                    img_path = os.path.join(self.dataset_root, "UPAR", img_rel)

                samples.append(img_path)
                labels_list.append(raw_labels[idx])

            return samples, np.asarray(labels_list, dtype=np.float32)

        # Fallback: scan directory structure
        print(f"[Dataset] Neither CSV nor dataset_all.pkl found. Scanning directory...")
        samples = []
        labels_list = []
        for root, _, files in os.walk(self.dataset_root):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append(os.path.join(root, f))
                    labels_list.append(np.zeros(40, dtype=np.float32))

        return samples, np.asarray(labels_list, dtype=np.float32)

    def _preload_images_to_ram(self):
        """Preloads compressed image bytes into host system RAM."""
        if self.is_h5 and self.h5_path:
            import h5py
            import time
            print(f"[RAM Cache] Preloading {self.num_samples:,} images from HDF5 container into host RAM...")
            t0 = time.time()
            with h5py.File(self.h5_path, "r") as h5f:
                ds = h5f[self.split]["image_bytes"]
                self.ram_cache = [bytes(ds[i]) for i in range(self.num_samples)]
            elapsed = time.time() - t0
            total_mb = sum(len(b) for b in self.ram_cache if b is not None) / (1024 * 1024)
            print(f"[RAM Cache] Preloaded {len(self.ram_cache):,} images ({total_mb:.1f} MB) in {elapsed:.2f}s! All disk I/O eliminated.")
            return

        print(f"[RAM Cache] Preloading {self.num_samples:,} images into host RAM (16 threads)...")
        self.ram_cache = [None] * self.num_samples

        def _load_single(i):
            path = self.samples[i]
            if os.path.isfile(path):
                try:
                    with open(path, "rb") as fp:
                        self.ram_cache[i] = fp.read()
                except Exception:
                    self.ram_cache[i] = None

        with ThreadPoolExecutor(max_workers=16) as pool:
            pool.map(_load_single, range(self.num_samples))

        valid_count = sum(1 for b in self.ram_cache if b is not None)
        total_mb = sum(len(b) for b in self.ram_cache if b is not None) / (1024 * 1024)
        print(f"[RAM Cache] Preloaded {valid_count:,}/{self.num_samples:,} images ({total_mb:.1f} MB) in RAM! Disk I/O eliminated.")

    def _get_h5_handle(self):
        """Lazy per-worker HDF5 handle to prevent multi-processing fork locks."""
        if self._h5_handle is None and self.is_h5 and self.h5_path:
            import h5py
            self._h5_handle = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5_handle

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. Load image (RAM Cache > Lazy HDF5 > Disk File)
        img_bytes = self.ram_cache[idx] if self.ram_cache is not None else None
        if img_bytes is not None:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif self.is_h5:
            h5_handle = self._get_h5_handle()
            raw_b = bytes(h5_handle[self.split]["image_bytes"][idx])
            img = Image.open(io.BytesIO(raw_b)).convert("RGB")
        else:
            path = self.samples[idx]
            if os.path.isfile(path):
                img = Image.open(path).convert("RGB")
            else:
                # Synthetic dummy crop if missing
                img = Image.new("RGB", (168, 336), color=(128, 128, 128))

        img_tensor = self.transform(img)

        # 2. Extract 40-D attribute vector
        attr_vec = self.labels[idx]

        # 3. Dynamic text prompt synthesis
        prompt = self.prompt_engine.synthesize_prompt(attr_vec, is_train=self.is_train)
        tokens = self.prompt_engine.tokenize(prompt).squeeze(0) # [77]

        # 4. Optional Masked Language Modeling (15% tokens masked with <MASK>)
        masked_tokens = tokens.clone()
        masked_labels = torch.full_like(tokens, fill_value=-100)
        if self.is_train:
            mask_prob = 0.15
            for pos in range(1, len(tokens)): # Skip <SOS>
                if tokens[pos].item() in (0, 3): # Skip PAD/EOS
                    continue
                if random.random() < mask_prob:
                    masked_labels[pos] = tokens[pos].clone()
                    masked_tokens[pos] = 4 # <MASK> token ID

        # 5. Anatomical Sub-strip text tokens
        prompt_head = self.prompt_engine.synthesize_prompt(attr_vec, is_train=self.is_train, anatomical_strip="head")
        prompt_torso = self.prompt_engine.synthesize_prompt(attr_vec, is_train=self.is_train, anatomical_strip="torso")
        prompt_legs = self.prompt_engine.synthesize_prompt(attr_vec, is_train=self.is_train, anatomical_strip="legs")

        tok_head = self.prompt_engine.tokenize(prompt_head).squeeze(0)
        tok_torso = self.prompt_engine.tokenize(prompt_torso).squeeze(0)
        tok_legs = self.prompt_engine.tokenize(prompt_legs).squeeze(0)

        return {
            "image": img_tensor,
            "attr_labels": torch.from_numpy(attr_vec),
            "text_tokens": tokens,
            "masked_tokens": masked_tokens,
            "masked_labels": masked_labels,
            "strip_tokens": (tok_head, tok_torso, tok_legs),
            "sample_idx": idx,
        }

    def get_all_labels(self) -> np.ndarray:
        """Returns all 40-D attribute labels as a float32 numpy array."""
        return np.asarray(self.labels, dtype=np.float32)

    def get_query_subset(self, num_queries: int = 500) -> np.ndarray:
        """Deterministically samples a subset of queries from the dataset labels."""
        all_labels = self.get_all_labels()
        Q = min(num_queries, len(all_labels))
        indices = np.linspace(0, len(all_labels) - 1, Q, dtype=int)
        return all_labels[indices]
