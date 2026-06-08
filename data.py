"""Dataset utilities for Assignment 3 robustness.

Preprocessing intentionally matches the provided task template exactly:
    images = torch.from_numpy(data["images"]).float() / 255.0

No ImageNet normalization, no dataset mean/std normalization.
The submitted checkpoint contains only weights, so any preprocessing not done by
the server would create a train/test mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class SplitData:
    train_images: torch.Tensor
    train_labels: torch.Tensor
    val_images: torch.Tensor
    val_labels: torch.Tensor


class RobustnessDataset(Dataset):
    """Tensor dataset with optional CIFAR-style augmentation.

    Images must already be float tensors in [0, 1] with shape (N, 3, 32, 32).
    Augmentations preserve the [0, 1] scale and do not add normalization.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = self.images[idx]
        y = self.labels[idx]
        if self.augment:
            x = random_crop_flip(x)
        return x, y


def random_crop_flip(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Random crop with padding + random horizontal flip for one CHW image."""
    # x: (3, 32, 32), values [0, 1]
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[2])

    x_pad = F.pad(x.unsqueeze(0), (padding, padding, padding, padding), mode="reflect").squeeze(0)
    _, h, w = x_pad.shape
    top = torch.randint(0, h - 32 + 1, (1,)).item()
    left = torch.randint(0, w - 32 + 1, (1,)).item()
    return x_pad[:, top : top + 32, left : left + 32]


def load_npz_exact_template(npz_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load .npz using the same preprocessing as the assignment template.

    Expected keys are "images" and "labels". A few fallback key names are accepted
    only to make local debugging easier.
    """
    data = np.load(npz_path)
    image_key = "images" if "images" in data.files else None
    label_key = "labels" if "labels" in data.files else None

    if image_key is None:
        for k in ["x", "X", "data", "train_images"]:
            if k in data.files:
                image_key = k
                break
    if label_key is None:
        for k in ["y", "Y", "targets", "train_labels"]:
            if k in data.files:
                label_key = k
                break

    if image_key is None or label_key is None:
        raise KeyError(f"Could not find image/label arrays. Available keys: {data.files}")

    images_np = data[image_key]
    labels_np = data[label_key]

    # Accept NHWC only as a convenience, but assignment says NCHW.
    if images_np.ndim != 4:
        raise ValueError(f"Expected images with 4 dims, got shape {images_np.shape}")
    if images_np.shape[1:] == (32, 32, 3):
        images_np = np.transpose(images_np, (0, 3, 1, 2))
    if images_np.shape[1:] != (3, 32, 32):
        raise ValueError(f"Expected images shape (N, 3, 32, 32), got {images_np.shape}")

    images = torch.from_numpy(images_np).float() / 255.0
    labels = torch.from_numpy(labels_np).long()

    if labels.min().item() < 0 or labels.max().item() > 8:
        raise ValueError(f"Labels must be in [0, 8], got min={labels.min().item()} max={labels.max().item()}")
    if images.min().item() < 0.0 or images.max().item() > 1.0:
        raise ValueError("Images should be in [0, 1] after dividing by 255.0")

    return images.contiguous(), labels.contiguous()


def make_stratified_split(
    images: torch.Tensor,
    labels: torch.Tensor,
    val_size: float = 0.1,
    seed: int = 42,
) -> SplitData:
    """Create stratified train/validation split based on class labels."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    idx = np.arange(len(labels))
    train_idx, val_idx = next(splitter.split(idx, labels.cpu().numpy()))

    train_idx_t = torch.from_numpy(train_idx).long()
    val_idx_t = torch.from_numpy(val_idx).long()

    return SplitData(
        train_images=images[train_idx_t],
        train_labels=labels[train_idx_t],
        val_images=images[val_idx_t],
        val_labels=labels[val_idx_t],
    )


def make_loaders(
    npz_path: str,
    val_size: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    augment_train: bool = True,
) -> tuple[DataLoader, DataLoader, SplitData]:
    images, labels = load_npz_exact_template(npz_path)
    split = make_stratified_split(images, labels, val_size=val_size, seed=seed)

    train_ds = RobustnessDataset(split.train_images, split.train_labels, augment=augment_train)
    val_ds = RobustnessDataset(split.val_images, split.val_labels, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, split
