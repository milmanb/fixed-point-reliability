"""Pretrained standard denoisers for the natural-image check: DnCNN-B and Restormer.

Both are blind Gaussian denoisers for grayscale images in [0, 1], trained on natural images with
Gaussian noise only, so blur, masks and noise above their training range are unseen corruptions,
as for the Fashion-MNIST models.

    dncnn_b    DnCNN-B (Zhang et al., IEEE TIP 2017): 20 convolutions with ReLU and a residual
               output y - n(y), trained on sigma in [0, 55]/255. Weights: dncnn_gray_blind.pth of
               the KAIR release v1.0, with batch normalization merged into the convolutions.
    restormer  Restormer (Zamir et al., CVPR 2022): a transformer U-Net, trained on sigma in
               [0, 50]/255. Weights: the authors' gaussian_gray_denoising_blind.pth (Hugging Face
               deepinv/Restormer); the architecture comes from the deepinv library.

scripts/download_pretrained.py fetches the weights into data/pretrained/. Outputs are clipped to
[0, 1] by default, as when they are saved as images (Denoiser, clip=False turns this off). On the
68 full BSD68 images, with Gaussian noise of sigma = 15, 25, 50 (/255) drawn as in the authors'
test scripts, these wrappers give a PSNR of 31.62, 29.16, 26.23 dB (DnCNN-B) and 31.95, 29.51,
26.62 dB (Restormer). The papers report 31.61, 29.16, 26.23 and 31.95, 29.51, 26.62
(scripts/check_pretrained.py, results/pretrained/psnr.csv).
"""

import torch
import torch.nn.functional as F
from torch import nn

from fpr.data import REPO_ROOT


class DnCNN(nn.Module):
    """DnCNN with merged batch normalization; the layer indices match KAIR's state dicts."""

    def __init__(self, channels=1, width=64, depth=20):
        super().__init__()
        layers = [nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(width, channels, 3, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, y):
        return y - self.model(y)


def load_dncnn(path):
    net = DnCNN()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net


class ReflectPadded(nn.Module):
    """Restormer as its authors test it: reflect-pad to a multiple of 8, run, crop.

    deepinv's own forward pads by replication, which lowers the BSD68 PSNR of these weights at full
    image size (rows restormer_replicate of results/pretrained/psnr.csv). With reflection they match
    the paper.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, y):
        h, w = y.shape[-2:]
        padded = F.pad(y, (0, (-w) % 8, 0, (-h) % 8), mode="reflect")
        return self.net.forward_restormer(padded)[..., :h, :w]


def load_restormer(path):
    try:
        from deepinv.models import Restormer  # optional dependency, needed only here (tested: 0.5.0)
    except ImportError:
        raise SystemExit('Restormer needs deepinv: pip install -e ".[pretrained]"') from None

    # The authors' configuration (Denoising/Options/GaussianGrayDenoising_Restormer.yml), stated so
    # that a change of deepinv's defaults cannot change the network silently.
    net = Restormer(in_channels=1, out_channels=1, dim=48, num_blocks=(4, 6, 6, 8),
                    num_refinement_blocks=4, heads=(1, 2, 4, 8), ffn_expansion_factor=2.66, bias=False,
                    LayerNorm_type="BiasFree", dual_pixel_task=False, pretrained=None)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["params"])
    return ReflectPadded(net)


MODELS = {  # name -> (weights under data/, loader)
    "dncnn_b": ("pretrained/dncnn/dncnn_gray_blind.pth", load_dncnn),
    "restormer": ("pretrained/restormer/gaussian_gray_denoising_blind.pth", load_restormer),
}


class Denoiser:
    """Evaluation adapter, as fpr.models.Restorer: float32 on its own device, in chunks of
    `batch_size` images, returned in y's dtype and device. With clip=True (the default) the output
    is clipped to [0, 1], as when it is saved as an image. The clip can create exact fixed points:
    where the network's output leaves [0, 1] on both passes, the clipped values agree and g is 0."""

    differentiable = True

    def __init__(self, net, name, device, batch_size=32, clip=True):
        self.net = net.to(device).eval()
        self.name = name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.clip = clip
        if self.device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = False  # as in fpr.models.Restorer
            torch.backends.cuda.matmul.allow_tf32 = False

    @torch.no_grad()
    def __call__(self, y):
        parts = []
        for part in y.split(self.batch_size):
            out = self.net(part.to(self.device, torch.float32))
            parts.append((out.clamp(0, 1) if self.clip else out).to(y))
        return torch.cat(parts)


def load_denoiser(name, device=None, batch_size=32, root=None, clip=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    relative, loader = MODELS[name]
    path = (root or REPO_ROOT / "data") / relative
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `python scripts/download_pretrained.py`")
    return Denoiser(loader(path), name, device, batch_size=batch_size, clip=clip)
