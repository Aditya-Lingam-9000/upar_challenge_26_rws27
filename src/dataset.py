"""High-throughput Dataset Loaders and Samplers for UPAR 2027 Challenge.

Features:
  1. UPARDataset: Multi-attribute dataset loader with robust CSV parsing, domain tagging,
     and surveillance augmentations (RandomErasing, ColorJitter, RandomHorizontalFlip).
  2. DomainBalancedBatchSampler: Samples balanced batches across Market1501, PA100k, and PETA
     to prevent gradient bias towards the dominant domain (PA-100k: 79k vs PETA: 8.6k).
  3. UPARRetrievalEvalDataset: Gallery and query loaders for Task 2 evaluation.
"""
from __future__ import annotations

import csv
import os
from typing import Callable, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T


# Canonical list of 40 UPAR attributes in challenge order
UPAR_40_ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other",
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun", "Accessory-Hat",
]

DOMAIN_NAME_TO_ID = {
    "Market1501": 0,
    "PA100k": 1,
    "PETA": 2,
}


def get_default_transforms(
    height: int = 256,
    width: int = 128,
    is_train: bool = True,
    erasing_prob: float = 0.5,
) -> T.Compose:
    """Build PyTorch transforms for pedestrian surveillance images."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if is_train:
        transform_list = [
            T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10, padding_mode="edge"),
            T.RandomCrop((height, width)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.ToTensor(),
            normalize,
        ]
        if erasing_prob > 0.0:
            transform_list.append(
                T.RandomErasing(
                    p=erasing_prob,
                    scale=(0.02, 0.25),
                    ratio=(0.3, 3.3),
                    value="random",
                )
            )
        return T.Compose(transform_list)
    else:
        return T.Compose([
            T.Resize((height, width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            normalize,
        ])


class UPARDataset(Dataset):
    """PyTorch Dataset for UPAR multi-label pedestrian attribute recognition."""

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        transform: Optional[Callable] = None,
        height: int = 256,
        width: int = 128,
        is_train: bool = True,
    ) -> None:
        self.data_root = data_root
        self.csv_path = csv_path
        self.transform = transform or get_default_transforms(height, width, is_train)

        self.image_paths: list[str] = []
        self.labels: list[np.ndarray] = []
        self.domain_ids: list[int] = []

        self._load_annotations()

    def _load_annotations(self) -> None:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Annotations file not found: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Robust header parsing
            first_col = header[0].lstrip("# ").strip().lower()
            if first_col in ["image", "image_path", "img"]:
                self.attribute_names = [h.strip() for h in header[1:]]
            else:
                self.attribute_names = [header[0].lstrip("# ").strip()] + [h.strip() for h in header[1:]]

            assert len(self.attribute_names) == 40, (
                f"Expected 40 attributes, got {len(self.attribute_names)} in {self.csv_path}"
            )

            for row in reader:
                if not row or len(row) < 41:
                    continue
                rel_path = row[0].strip()
                attr_vals = np.array([float(val.strip()) for val in row[1:41]], dtype=np.float32)

                # Determine domain ID from path
                if rel_path.startswith("Market1501"):
                    did = DOMAIN_NAME_TO_ID["Market1501"]
                elif rel_path.startswith("PA100k"):
                    did = DOMAIN_NAME_TO_ID["PA100k"]
                elif rel_path.startswith("PETA"):
                    did = DOMAIN_NAME_TO_ID["PETA"]
                else:
                    did = 0

                self.image_paths.append(rel_path)
                self.labels.append(attr_vals)
                self.domain_ids.append(did)

        self.labels = np.array(self.labels, dtype=np.float32)
        self.domain_ids = np.array(self.domain_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        rel_path = self.image_paths[index]
        full_path = os.path.join(self.data_root, rel_path)

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image at {full_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.from_numpy(self.labels[index])
        domain_id = int(self.domain_ids[index])

        return image, label, domain_id, rel_path


class DomainBalancedBatchSampler(Sampler):
    """Batch sampler that ensures proportional representation from Market1501, PA100k, and PETA.

    Default ratio:
      - PA-100k: 50%
      - Market1501: 25%
      - PETA: 25%
    """
    def __init__(
        self,
        domain_ids: np.ndarray,
        batch_size: int = 64,
        drop_last: bool = False,
        domain_weights: tuple[float, float, float] = (0.25, 0.50, 0.25),
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.domain_weights = np.array(domain_weights, dtype=np.float32)
        self.domain_weights /= self.domain_weights.sum()

        self.domain_indices: dict[int, np.ndarray] = {
            d: np.where(domain_ids == d)[0] for d in range(3)
        }
        self.total_samples = len(domain_ids)
        self.num_batches = self.total_samples // batch_size

    def __iter__(self):
        # Permute indices within each domain
        shuffled_indices = {
            d: np.random.permutation(indices) for d, indices in self.domain_indices.items()
        }
        ptrs = {d: 0 for d in range(3)}

        # Determine batch counts per domain
        # e.g., batch_size = 64: Market ~16, PA100k ~32, PETA ~16
        counts = np.round(self.batch_size * self.domain_weights).astype(int)
        counts[0] = self.batch_size - (counts[1] + counts[2])  # ensure exact sum

        for _ in range(self.num_batches):
            batch = []
            for d in range(3):
                needed = counts[d]
                idx_pool = shuffled_indices[d]
                ptr = ptrs[d]

                if ptr + needed > len(idx_pool):
                    # Reshuffle and wrap around
                    shuffled_indices[d] = np.random.permutation(self.domain_indices[d])
                    idx_pool = shuffled_indices[d]
                    ptr = 0

                batch.extend(idx_pool[ptr : ptr + needed].tolist())
                ptrs[d] = ptr + needed

            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches


class Task2RetrievalEvalData:
    """Helper container for Task 2 offline validation data.

    Loads:
      - 3,462 queries: (Q, 40) binary array
      - 33,407 gallery ground-truth: (G, 40) binary array
      - Gallery image relative paths: list of G strings
      - Query-to-gallery mapping: dict mapping query_index -> list of matching gallery indices
    """
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        task2_dir = os.path.join(data_root, "annotations", "task2", "val")
        queries_path = os.path.join(task2_dir, "queries.csv")
        gt_path = os.path.join(task2_dir, "gt.csv")
        ids_path = os.path.join(task2_dir, "ids.csv")

        # 1. Load queries
        self.queries = []
        with open(queries_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)  # header
            for row in reader:
                if row:
                    self.queries.append([float(x.strip()) for x in row])
        self.queries = np.array(self.queries, dtype=np.float32)

        # 2. Load gallery gt
        self.gallery_paths = []
        self.gallery_gt = []
        with open(gt_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)  # header
            for row in reader:
                if row:
                    self.gallery_paths.append(row[0].strip())
                    self.gallery_gt.append([float(x.strip()) for x in row[1:41]])
        self.gallery_gt = np.array(self.gallery_gt, dtype=np.float32)

        # 3. Load ids mapping (maps image -> query index)
        self.image_to_query_idx = {}
        with open(ids_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    self.image_to_query_idx[row[0].strip()] = int(row[1].strip())
