"""UPAR Challenge Track 2: V4 CLIP ViT-L/14 Championship Dataset & Augmentation Pipeline.

This module provides high-throughput dataset loading and realistic surveillance
degradation transforms tailored for the WACV 2027 UPAR Challenge (Track 2:
Attribute-based Person Retrieval).

Core Features:
1. SurveillanceDegradation:
   - Downsampling to low resolution ([48, 128] px shorter side) + bicubic restoration
   - Random JPEG compression ([20, 70] quality)
   - Gaussian blur (kernel [3, 5], sigma [0.5, 2.0], p=0.3)
   - Horizontal linear motion blur (kernel [3, 7], p=0.1)
2. UPARDataset:
   - Memory-efficient lazy image loading from HDF5 binary container
   - Transparent handling of both numpy byte arrays and raw bytes objects
   - Unified support for dataset key aliasing ('images' <-> 'image_bytes')
   - Pre-extracted labels cached in memory for instant multi-label indexing
   - Automated conversion of 40-dim binary attribute vectors to natural language descriptions
   - Multi-worker multiprocessing safety with automatic per-worker HDF5 connection isolation
   - Fault-tolerant corrupted sample recovery with random fallback
3. Dataloader factory:
   - Standard train/val transforms with CLIP-specific bicubic resize and normalization
   - Custom collator preserving variable-length natural language text prompts
"""

from __future__ import annotations

import io
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ==============================================================================
# Canonical 40 UPAR Attributes & Dataset Priors
# ==============================================================================
CANONICAL_ATTRIBUTES: Tuple[str, ...] = (
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
    "Accessory-Glasses-Sun", "Accessory-Hat"
)

UPAR_PRIORS: Tuple[float, ...] = (
    0.0110, 0.9650, 0.0237, 0.4292, 0.6546, 0.3114, 0.0126, 0.6288, 0.2556, 0.0821,
    0.0199, 0.0435, 0.1621, 0.0058, 0.0105, 0.0253, 0.0694, 0.2250, 0.0497, 0.0753,
    0.3751, 0.4877, 0.1788, 0.0448, 0.0150, 0.1719, 0.0021, 0.0155, 0.0008, 0.0039,
    0.0447, 0.0017, 0.0372, 0.9038, 0.0958, 0.2085, 0.2931, 0.1394, 0.0218, 0.0425
)

CLIP_MEAN: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


# ==============================================================================
# Attribute to Text Description Conversion
# ==============================================================================
try:
    from sota_pipeline.models_v4 import attrs_to_text  # type: ignore
except ImportError:
    try:
        from models_v4 import attrs_to_text  # type: ignore
    except ImportError:
        def attrs_to_text(attrs: Union[np.ndarray, torch.Tensor, Sequence[float]]) -> str:
            """Convert 40-dim binary attribute vector to natural language description.

            Args:
                attrs: 40-element binary attribute vector (array, tensor, or sequence).

            Returns:
                Natural language description suitable for CLIP text encoder.
            """
            if hasattr(attrs, "detach"):
                attrs = attrs.detach().cpu().numpy()
            attrs_arr = np.asarray(attrs, dtype=np.float32).squeeze()
            parts: List[str] = []

            # Age (indices 0-2)
            if attrs_arr[0] > 0.5:
                parts.append("young")
            elif attrs_arr[1] > 0.5:
                parts.append("adult")
            elif attrs_arr[2] > 0.5:
                parts.append("old")

            # Gender (index 3)
            gender = "female" if attrs_arr[3] > 0.5 else "male"
            parts.append(gender)

            # Hair (indices 4-6)
            if attrs_arr[4] > 0.5:
                parts.append("with short hair")
            elif attrs_arr[5] > 0.5:
                parts.append("with long hair")
            elif attrs_arr[6] > 0.5:
                parts.append("bald")

            # Upper body (indices 7-19)
            upper_len = "short-sleeved" if attrs_arr[7] > 0.5 else "long-sleeved"
            upper_colors: List[str] = []
            color_names = [
                "black", "blue", "brown", "green", "grey",
                "orange", "pink", "purple", "red", "white", "yellow", "other-colored"
            ]
            for i, c in enumerate(color_names):
                if attrs_arr[8 + i] > 0.5:
                    upper_colors.append(c)
            upper_color = " ".join(upper_colors[:2]) if upper_colors else ""
            if upper_color:
                parts.append(f"wearing a {upper_color} {upper_len} top")
            else:
                parts.append(f"wearing a {upper_len} top")

            # Lower body (indices 20-34)
            lower_len = "short" if attrs_arr[20] > 0.5 else "long"
            lower_colors: List[str] = []
            for i, c in enumerate(color_names):
                if attrs_arr[21 + i] > 0.5:
                    lower_colors.append(c)
            lower_color = " ".join(lower_colors[:2]) if lower_colors else ""

            lower_type = "trousers" if attrs_arr[33] > 0.5 else ("skirt" if attrs_arr[34] > 0.5 else "pants")
            if lower_color:
                parts.append(f"and {lower_color} {lower_len} {lower_type}")
            else:
                parts.append(f"and {lower_len} {lower_type}")

            # Accessories (indices 35-39)
            accs: List[str] = []
            if attrs_arr[35] > 0.5:
                accs.append("a backpack")
            if attrs_arr[36] > 0.5:
                accs.append("a bag")
            if attrs_arr[37] > 0.5:
                accs.append("glasses")
            if attrs_arr[38] > 0.5:
                accs.append("sunglasses")
            if attrs_arr[39] > 0.5:
                accs.append("a hat")
            if accs:
                parts.append(f"carrying {', '.join(accs)}")

            desc = " ".join(" ".join(parts).split())
            return f"A surveillance photo of a {desc}"


# ==============================================================================
# Surveillance Image Degradation Engine
# ==============================================================================
class SurveillanceDegradation:
    """Callable transform simulating real-world surveillance camera degradations.

    Applies realistic optical, sensor, and compression artifacts:
    - Triggered with overall probability `p` (default 0.5).
    - Random downsampling: resizes shorter side to [48, 128] px, then back to original size via BICUBIC.
    - Random JPEG compression: encodes with quality in [20, 70] to simulate DVR codec blocking.
    - Random Gaussian blur: kernel size from [3, 5], sigma in [0.5, 2.0] with probability 0.3.
    - Random motion blur: horizontal 1D linear kernel of size [3, 7] with probability 0.1.

    Operates strictly on PIL Image objects and returns PIL Image objects.
    Applied before standard spatial/photometric training transforms.
    """

    def __init__(
        self,
        p: float = 0.5,
        min_shorter_side: int = 48,
        max_shorter_side: int = 128,
        jpeg_quality_range: Tuple[int, int] = (20, 70),
        gaussian_blur_p: float = 0.3,
        gaussian_kernel_sizes: Sequence[int] = (3, 5),
        gaussian_sigma_range: Tuple[float, float] = (0.5, 2.0),
        motion_blur_p: float = 0.1,
        motion_kernel_sizes: Sequence[int] = (3, 5, 7),
    ) -> None:
        """Initialize the surveillance degradation transform.

        Args:
            p: Overall probability of applying the degradation chain.
            min_shorter_side: Minimum target length for downsampled shorter side.
            max_shorter_side: Maximum target length for downsampled shorter side.
            jpeg_quality_range: Lower and upper bounds for JPEG compression quality.
            gaussian_blur_p: Probability of applying Gaussian blur within degradation chain.
            gaussian_kernel_sizes: Allowable kernel sizes for Gaussian blur.
            gaussian_sigma_range: Min and max standard deviation for Gaussian blur.
            motion_blur_p: Probability of applying horizontal motion blur within degradation chain.
            motion_kernel_sizes: Allowable kernel lengths for horizontal motion blur.
        """
        self.p = float(p)
        self.min_shorter_side = int(min_shorter_side)
        self.max_shorter_side = int(max_shorter_side)
        self.jpeg_quality_range = (int(jpeg_quality_range[0]), int(jpeg_quality_range[1]))
        self.gaussian_blur_p = float(gaussian_blur_p)
        self.gaussian_kernel_sizes = tuple(gaussian_kernel_sizes)
        self.gaussian_sigma_range = (float(gaussian_sigma_range[0]), float(gaussian_sigma_range[1]))
        self.motion_blur_p = float(motion_blur_p)
        self.motion_kernel_sizes = tuple(motion_kernel_sizes)

        # Resolves backwards compatibility across PIL / Pillow versions
        self._bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)

    def _apply_downsampling(self, img: Image.Image) -> Image.Image:
        """Downsample image to low resolution shorter side and restore back using BICUBIC."""
        w, h = img.size
        shorter_side = min(w, h)
        if shorter_side <= self.min_shorter_side:
            return img

        target_short = random.randint(self.min_shorter_side, min(self.max_shorter_side, shorter_side))
        scale = float(target_short) / float(shorter_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        down = img.resize((new_w, new_h), resample=self._bicubic)
        return down.resize((w, h), resample=self._bicubic)

    def _apply_jpeg_compression(self, img: Image.Image) -> Image.Image:
        """Simulate lossy CCTV/DVR compression artifacts."""
        quality = random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    def _apply_gaussian_blur(self, img: Image.Image) -> Image.Image:
        """Simulate optical defocus with random kernel size and sigma."""
        k = random.choice(self.gaussian_kernel_sizes)
        sigma = random.uniform(self.gaussian_sigma_range[0], self.gaussian_sigma_range[1])
        return TF.gaussian_blur(img, kernel_size=k, sigma=sigma)

    def _apply_motion_blur(self, img: Image.Image) -> Image.Image:
        """Simulate horizontal camera pan or pedestrian linear motion blur."""
        k = random.choice(self.motion_kernel_sizes)
        arr = torch.from_numpy(np.array(img, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
        # Horizontal 1D averaging kernel [1, 1, 1, k] replicated across 3 RGB channels
        kernel = (torch.ones((1, 1, 1, k), dtype=torch.float32) / float(k)).repeat(3, 1, 1, 1)
        pad = k // 2
        blurred = F.conv2d(arr, kernel, padding=(0, pad), groups=3)
        blurred_np = blurred.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
        return Image.fromarray(blurred_np)

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply surveillance degradation pipeline to a PIL image with probability p.

        Args:
            img: Input PIL Image in RGB format.

        Returns:
            Degraded PIL Image in RGB format.
        """
        if random.random() >= self.p:
            return img

        # 1. Downsampling to low resolution & bicubic reconstruction
        img = self._apply_downsampling(img)

        # 2. Lossy JPEG compression artifacts
        img = self._apply_jpeg_compression(img)

        # 3. Gaussian defocus blur (p=0.3)
        if random.random() < self.gaussian_blur_p:
            img = self._apply_gaussian_blur(img)

        # 4. Horizontal linear motion blur (p=0.1)
        if random.random() < self.motion_blur_p:
            img = self._apply_motion_blur(img)

        return img


# ==============================================================================
# HDF5 Path Resolution & Multi-Worker Safe Proxy
# ==============================================================================
def resolve_h5_path(path: str) -> str:
    """Resolve HDF5 dataset path across Kaggle and local environments.

    Args:
        path: Specified file or directory path.

    Returns:
        Absolute resolved path to an existing .h5/.hdf5 container.
    """
    if os.path.isfile(path):
        return os.path.abspath(path)

    candidates = [
        path,
        "/kaggle/input/datasets/hinewk/upar-binary-dataset/upar_binary_dataset.h5",
        "/kaggle/input/upar-binary-dataset/upar_binary_dataset.h5",
        "upar_binary_dataset.h5",
        os.path.join(os.getcwd(), "upar_binary_dataset.h5"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "upar_binary_dataset.h5"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # Search inside directory if a directory was provided
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith((".h5", ".hdf5")):
                cand_path = os.path.join(path, fname)
                if os.path.isfile(cand_path):
                    return os.path.abspath(cand_path)

    return os.path.abspath(path)


def _extract_image_bytes(raw: Any) -> bytes:
    """Extract raw byte stream from various HDF5 storage representations.

    Handles numpy uint8 byte arrays, variable length byte buffers, and Python bytes.

    Args:
        raw: Raw data entry retrieved from HDF5 dataset.

    Returns:
        Standard Python bytes sequence representing encoded image (JPEG/PNG).
    """
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    if hasattr(raw, "tobytes"):
        return raw.tobytes()
    if isinstance(raw, (np.ndarray, np.void)):
        return raw.tobytes()
    return bytes(raw)


class _H5GroupProxy:
    """Transparent proxy for an h5py.Group that aliases 'images' and 'image_bytes'."""

    def __init__(self, grp: h5py.Group) -> None:
        self._grp = grp

    def __getitem__(self, key: str) -> Any:
        try:
            val = self._grp[key]
        except KeyError:
            if key == "images" and "image_bytes" in self._grp:
                val = self._grp["image_bytes"]
            elif key == "image_bytes" and "images" in self._grp:
                val = self._grp["images"]
            else:
                raise
        if isinstance(val, h5py.Group):
            return _H5GroupProxy(val)
        return val

    def __contains__(self, key: str) -> bool:
        if key in self._grp:
            return True
        if key == "images" and "image_bytes" in self._grp:
            return True
        if key == "image_bytes" and "images" in self._grp:
            return True
        return False

    def __len__(self) -> int:
        return len(self._grp)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._grp, name)

    def __repr__(self) -> str:
        return repr(self._grp)


class _H5FileProxy:
    """Transparent proxy for an h5py.File that aliases 'images' and 'image_bytes' across paths."""

    def __init__(self, f: h5py.File) -> None:
        self._f = f

    def __getitem__(self, key: str) -> Any:
        try:
            val = self._f[key]
        except KeyError:
            if "/images" in key and key.replace("/images", "/image_bytes") in self._f:
                val = self._f[key.replace("/images", "/image_bytes")]
            elif "/image_bytes" in key and key.replace("/image_bytes", "/images") in self._f:
                val = self._f[key.replace("/image_bytes", "/images")]
            elif key == "images" and "image_bytes" in self._f:
                val = self._f["image_bytes"]
            elif key == "image_bytes" and "images" in self._f:
                val = self._f["images"]
            else:
                raise
        if isinstance(val, h5py.Group):
            return _H5GroupProxy(val)
        return val

    def __contains__(self, key: str) -> bool:
        if key in self._f:
            return True
        if "/images" in key and key.replace("/images", "/image_bytes") in self._f:
            return True
        if "/image_bytes" in key and key.replace("/image_bytes", "/images") in self._f:
            return True
        return False

    def __len__(self) -> int:
        return len(self._f)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._f, name)

    def __repr__(self) -> str:
        return repr(self._f)


# ==============================================================================
# UPARDataset
# ==============================================================================
class UPARDataset(Dataset):
    """Main UPAR Challenge person retrieval dataset backed by an HDF5 container.

    Features:
    - Lazy on-the-fly image decompression directly from HDF5 variable-length byte streams.
    - Full in-memory attribute array storage for instantaneous batch indexing.
    - Automatic text generation via `attrs_to_text()` for joint vision-language alignment.
    - Fault-tolerant corrupted sample recovery that draws random valid replacements.
    - Process-safe fork/spawn serialization: cleanly reconnects HDF5 in PyTorch worker processes.
    - Robust aliasing between dataset keys ('images' vs 'image_bytes').
    """

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        surveillance_aug: bool = True,
    ) -> None:
        """Initialize UPARDataset.

        Args:
            h5_path: File system path to the upar_binary_dataset.h5 container.
            split: Data partition, typically 'train' or 'val'.
            transform: Custom torchvision transform. If None, default transform for
                the chosen split is automatically constructed.
            surveillance_aug: If True and split == 'train', applies SurveillanceDegradation.
        """
        super().__init__()
        self.h5_path = resolve_h5_path(h5_path)
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"HDF5 dataset container not found at: {self.h5_path}")

        self.split = str(split).strip()
        self.surveillance_aug = bool(surveillance_aug)

        # Initialize internal HDF5 handle and transparent proxy
        self._raw_h5: Optional[h5py.File] = h5py.File(self.h5_path, "r")
        self._proxy: Optional[_H5FileProxy] = _H5FileProxy(self._raw_h5)

        if self.split not in self._raw_h5:
            raise KeyError(
                f"Split '{self.split}' not found in HDF5 container at {self.h5_path}. "
                f"Available groups: {list(self._raw_h5.keys())}"
            )

        split_grp = self._raw_h5[self.split]

        # Resolve labels key ('labels' or 'attributes')
        label_key = "labels" if "labels" in split_grp else "attributes"
        if label_key not in split_grp:
            raise KeyError(f"Neither 'labels' nor 'attributes' found in group '{self.split}'")

        # Load labels into memory as float32 numpy array
        self.labels: np.ndarray = np.asarray(split_grp[label_key][:], dtype=np.float32)
        self.length: int = len(self.labels)

        # Resolve image storage key ('images' or 'image_bytes')
        self.img_key: str = "images" if "images" in split_grp else "image_bytes"

        # Configure transforms
        if transform is not None:
            self.transform: Optional[Callable] = transform
        else:
            if self.split == "train":
                self.transform = get_train_transform()
            else:
                self.transform = get_val_transform()

        # Configure surveillance degradation (train split only)
        if self.split == "train" and self.surveillance_aug:
            self.surveillance_degradation: Optional[SurveillanceDegradation] = SurveillanceDegradation(p=0.5)
        else:
            self.surveillance_degradation = None

    @property
    def h5(self) -> _H5FileProxy:
        """Return multi-worker safe HDF5 file proxy in read-only mode."""
        if self._raw_h5 is None or not getattr(self._raw_h5, "id", None) or not self._raw_h5.id.valid:
            self._raw_h5 = h5py.File(self.h5_path, "r")
            self._proxy = _H5FileProxy(self._raw_h5)
        return self._proxy  # type: ignore

    def __len__(self) -> int:
        """Return total number of samples in the dataset split."""
        return self.length

    def __getstate__(self) -> Dict[str, Any]:
        """Support pickle serialization across PyTorch DataLoader worker subprocesses."""
        state = self.__dict__.copy()
        # Drop open file handles prior to serialization
        state["_raw_h5"] = None
        state["_proxy"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore unpickled dataset object in a child process."""
        self.__dict__.update(state)
        self._raw_h5 = None
        self._proxy = None

    def _load_single_item(self, idx: int) -> Dict[str, Any]:
        """Load and transform a single sample from HDF5.

        Args:
            idx: Sample index in [0, length - 1].

        Returns:
            Dictionary containing 'image', 'labels', 'text', and 'index'.
        """
        # Retrieve encoded image bytes via H5 proxy
        raw_entry = self.h5[f"{self.split}/{self.img_key}"][idx]
        raw_bytes = _extract_image_bytes(raw_entry)

        # Decode JPEG / PNG into PIL RGB Image
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        # Apply surveillance degradation (train split only, if enabled)
        if self.surveillance_degradation is not None:
            img = self.surveillance_degradation(img)

        # Apply standard transform (resize, augmentations, tensor conversion, normalization)
        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = TF.to_tensor(img)

        # Retrieve attribute vector and format text description
        label_vec = self.labels[idx]
        label_tensor = torch.from_numpy(label_vec).float()
        text_desc = attrs_to_text(label_vec)

        return {
            "image": img_tensor,
            "labels": label_tensor,
            "text": text_desc,
            "index": int(idx),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve transformed sample with fallback recovery against corrupted images.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
            - 'image': transformed image tensor [3, 224, 224]
            - 'labels': attribute tensor [40] float32
            - 'text': natural language text description (string)
            - 'index': integer index
        """
        try:
            return self._load_single_item(idx)
        except Exception:
            # Fallback recovery: attempt up to 10 random valid samples
            for _ in range(10):
                fallback_idx = random.randint(0, self.length - 1)
                try:
                    return self._load_single_item(fallback_idx)
                except Exception:
                    continue
            raise RuntimeError(f"Corrupted image sample at index {idx}; fallback attempts exhausted.")

    def close(self) -> None:
        """Close open HDF5 file handles."""
        if self._raw_h5 is not None:
            try:
                self._raw_h5.close()
            except Exception:
                pass
            self._raw_h5 = None
            self._proxy = None

    def __enter__(self) -> UPARDataset:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


# ==============================================================================
# Standard Transforms (CLIP ViT-L/14 Resolution 224x224)
# ==============================================================================
def get_train_transform() -> T.Compose:
    """Build standard training data augmentation pipeline for CLIP ViT-L/14.

    Pipeline:
    1. Resize to (224, 224) with BICUBIC interpolation
    2. RandomHorizontalFlip(p=0.5)
    3. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
    4. RandomGrayscale(p=0.1)
    5. ToTensor()
    6. Normalize with standard CLIP mean and std
    7. RandomErasing(p=0.5, scale=(0.02, 0.4))

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
        T.RandomErasing(p=0.5, scale=(0.02, 0.4), value="random"),
    ])


def get_val_transform() -> T.Compose:
    """Build deterministic validation transform pipeline for CLIP ViT-L/14.

    Pipeline:
    1. Resize to (224, 224) with BICUBIC interpolation
    2. ToTensor()
    3. Normalize with standard CLIP mean and std

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
    ])


# ==============================================================================
# Collate Function & DataLoader Construction
# ==============================================================================
def upar_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function handling image tensors, attribute tensors, and text lists.

    Preserves text descriptions as Python strings while stacking tensor components.

    Args:
        batch: Sequence of sample dictionaries returned by UPARDataset.__getitem__.

    Returns:
        Collated dictionary containing:
        - 'image': FloatTensor of shape [B, 3, 224, 224]
        - 'labels': FloatTensor of shape [B, 40]
        - 'text': List[str] of length B
        - 'index': LongTensor of shape [B]
        - Convenience aliases: 'images', 'texts', 'indices'
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    texts = [str(item["text"]) for item in batch]
    indices = torch.tensor([int(item["index"]) for item in batch], dtype=torch.long)

    return {
        "image": images,
        "labels": labels,
        "text": texts,
        "index": indices,
        # Convenience aliases
        "images": images,
        "texts": texts,
        "indices": indices,
    }


def build_dataloaders(
    h5_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    surveillance_aug: bool = True,
    val_batch_size: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build PyTorch training and validation DataLoaders for UPAR Challenge.

    Args:
        h5_path: Path to upar_binary_dataset.h5 container.
        batch_size: Mini-batch size for training (default: 32).
        num_workers: Number of background worker processes (default: 4).
        surveillance_aug: Whether to enable surveillance degradation on train set.
        val_batch_size: Mini-batch size for validation (defaults to batch_size if None).
        pin_memory: Whether to pin host memory for accelerated CUDA transfers.
        persistent_workers: Whether to keep DataLoader worker processes alive across epochs.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    resolved_path = resolve_h5_path(h5_path)

    # 1. Instantiate datasets
    train_dataset = UPARDataset(
        h5_path=resolved_path,
        split="train",
        surveillance_aug=surveillance_aug,
    )
    val_dataset = UPARDataset(
        h5_path=resolved_path,
        split="val",
        surveillance_aug=False,
    )

    v_batch = batch_size if val_batch_size is None else int(val_batch_size)
    use_persistent = persistent_workers and (num_workers > 0)

    # 2. Instantiate DataLoaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=upar_collate_fn,
        persistent_workers=use_persistent,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=v_batch,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=upar_collate_fn,
        persistent_workers=use_persistent,
    )

    return train_loader, val_loader
