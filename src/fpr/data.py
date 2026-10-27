"""Fashion-MNIST and BSD68 crops as dense tensors in [0, 1], and the paths of the per-image dumps."""

from pathlib import Path

import torch
from torchvision.datasets import FashionMNIST

REPO_ROOT = Path(__file__).resolve().parents[2]


def per_image_dump(folder):
    """Path of a result folder's per-image dump; exits with instructions if it is missing.

    The dumps are git-ignored. scripts/download_dumps.py fetches the committed results' dumps,
    and scripts/evaluate_models.py recreates them (hours).
    """
    path = REPO_ROOT / folder / "per_image.csv.gz"
    if not path.exists():
        raise SystemExit(f"{Path(folder) / 'per_image.csv.gz'} is missing: download the dumps with "
                         "`python scripts/download_dumps.py`, or recreate them with "
                         "scripts/evaluate_models.py (README, step 3)")
    return path


def load_bsd68_crops(root=None, size=128, dtype=torch.float32):
    """Non-overlapping size x size crops of the 68 BSD68 test images, in [0, 1].

    Each 481 x 321 image gives a centered 3 x 2 grid of crops (2 x 3 when portrait), so size 128
    gives 408 crops. Returns the crops (N, 1, size, size) and the index of each crop's source image
    (N,), in file order and row-major within an image.
    """
    import numpy as np
    from PIL import Image

    root = Path(root) if root is not None else REPO_ROOT / "data" / "bsd68"
    paths = sorted(root.glob("test*.png"))
    if len(paths) != 68:
        raise SystemExit(f"{root} holds {len(paths)} of the 68 BSD68 images: run "
                         "`python scripts/download_pretrained.py`")
    crops, sources = [], []
    for index, path in enumerate(paths):
        image = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        rows, cols = image.shape[0] // size, image.shape[1] // size
        top, left = (image.shape[0] - rows * size) // 2, (image.shape[1] - cols * size) // 2
        for i in range(rows):
            for j in range(cols):
                crops.append(image[top + i * size:top + (i + 1) * size,
                                   left + j * size:left + (j + 1) * size])
                sources.append(index)
    x = torch.from_numpy(np.stack(crops)).unsqueeze(1).to(dtype)
    return x, torch.tensor(sources)


def load_fashion_mnist(split="train", root=None, dtype=torch.float32):
    """Return images of shape (N, 1, 28, 28) in [0, 1] and integer class labels (N,)."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(root) if root is not None else REPO_ROOT / "data"
    dataset = FashionMNIST(root=str(root), train=(split == "train"), download=True)
    images = dataset.data.unsqueeze(1).to(dtype) / 255.0
    return images, dataset.targets.clone()
