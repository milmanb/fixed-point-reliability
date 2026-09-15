"""Forward operators A_s and corrupted observations y = A_s x + n.

Calling a degradation on a clean batch x returns (y, A), where A is the
per-sample linear operator that produced y. The measurement residual
r_A(y) = |A f(y) - y| needs A at inference time, so the random mask or kernel
must travel with the observation.

All randomness is drawn on the CPU from an explicit torch.Generator, so a seed
gives the same corruptions on every machine and device.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _randn(x, gen):
    return torch.randn(x.shape, generator=gen, dtype=x.dtype).to(x.device)


def gaussian_kernel1d(std, dtype=torch.float64):
    radius = max(1, math.ceil(3 * std))
    t = torch.arange(-radius, radius + 1, dtype=dtype)
    kernel = torch.exp(-0.5 * (t / std) ** 2)
    return kernel / kernel.sum()


def gaussian_blur(z, std):
    """Separable Gaussian blur with zero padding.

    With a symmetric kernel and zero padding the operator matrix is symmetric,
    which keeps adjoints trivial if data-consistency steps are added later.
    Fashion-MNIST backgrounds are black, so zero padding matches the data.
    """
    kernel = gaussian_kernel1d(std, dtype=z.dtype).to(z.device)
    radius = (kernel.numel() - 1) // 2
    channels = z.shape[1]
    z = F.conv2d(z, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
                 padding=(0, radius), groups=channels)
    return F.conv2d(z, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
                    padding=(radius, 0), groups=channels)


@dataclass(frozen=True)
class GaussianNoise:
    """Denoising: A = I, y = x + sigma * n. Values are not clipped, so the noise stays Gaussian."""

    sigma: float
    family = "noise"

    @property
    def level(self):
        return self.sigma

    @property
    def label(self):
        return f"noise(sigma={self.sigma:g})"

    def __call__(self, x, gen):
        return x + self.sigma * _randn(x, gen), lambda z: z


@dataclass(frozen=True)
class GaussianBlur:
    """Deblurring: A = Gaussian blur of width `std`, plus weak Gaussian noise."""

    std: float
    sigma: float = 0.02
    family = "blur"

    @property
    def level(self):
        return self.std

    @property
    def label(self):
        return f"blur(std={self.std:g})"

    def __call__(self, x, gen):
        def A(z):
            return gaussian_blur(z, self.std)

        return A(x) + self.sigma * _randn(x, gen), A


@dataclass(frozen=True)
class PixelMask:
    """Random inpainting: each pixel is dropped (set to 0) with probability `drop`."""

    drop: float
    sigma: float = 0.02
    family = "pixel_mask"

    @property
    def level(self):
        return self.drop

    @property
    def label(self):
        return f"pixel_mask(drop={self.drop:g})"

    def __call__(self, x, gen):
        keep = (torch.rand(x.shape, generator=gen) >= self.drop).to(x.dtype).to(x.device)

        def A(z):
            return keep * z

        return A(x + self.sigma * _randn(x, gen)), A


@dataclass(frozen=True)
class BoxMask:
    """Box inpainting: one `size` x `size` square at a uniformly random position is set to 0."""

    size: int
    sigma: float = 0.02
    family = "box_mask"

    @property
    def level(self):
        return self.size

    @property
    def label(self):
        return f"box_mask(size={self.size})"

    def __call__(self, x, gen):
        n, _, h, w = x.shape
        top = torch.randint(0, h - self.size + 1, (n, 1), generator=gen)
        left = torch.randint(0, w - self.size + 1, (n, 1), generator=gen)
        rows, cols = torch.arange(h).view(1, h), torch.arange(w).view(1, w)
        in_rows = (rows >= top) & (rows < top + self.size)
        in_cols = (cols >= left) & (cols < left + self.size)
        box = in_rows[:, :, None] & in_cols[:, None, :]
        keep = (~box).unsqueeze(1).to(x.dtype).to(x.device)

        def A(z):
            return keep * z

        return A(x + self.sigma * _randn(x, gen)), A
