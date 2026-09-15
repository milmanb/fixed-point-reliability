"""Fashion-MNIST as dense tensors in [0, 1]."""

from pathlib import Path

import torch
from torchvision.datasets import FashionMNIST

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_fashion_mnist(split="train", root=None, dtype=torch.float32):
    """Return images of shape (N, 1, 28, 28) in [0, 1] and integer class labels (N,)."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(root) if root is not None else REPO_ROOT / "data"
    dataset = FashionMNIST(root=str(root), train=(split == "train"), download=True)
    images = dataset.data.unsqueeze(1).to(dtype) / 255.0
    return images, dataset.targets.clone()
