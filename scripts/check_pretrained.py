"""Check the pretrained denoisers against their papers: PSNR on the 68 full BSD68 images.

The protocol of the authors' test scripts (KAIR's main_test_dncnn.py, Restormer's
test_gaussian_gray_denoising.py): Gaussian noise of sigma/255 is added to the image in [0, 1]
without clipping, with numpy reseeded to 0 for every image. The output is clipped and rounded to 8
bits, and the PSNR against the 8-bit image is averaged over the images. The papers report, for
sigma = 15, 25, 50: DnCNN-B 31.61, 29.16, 26.23 dB (Zhang et al. 2017, Table II) and Restormer
31.95, 29.51, 26.62 dB (Zamir et al. 2022, Table 4, one model for all noise levels). The rows
restormer_replicate run Restormer through deepinv's own forward, which pads by replication. The
evaluation pads by reflection, as the authors do (fpr.pretrained.ReflectPadded). Writes
results/pretrained/psnr.csv.

    python scripts/check_pretrained.py
"""

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fpr.data import REPO_ROOT
from fpr.pretrained import MODELS, Denoiser, load_denoiser

PUBLISHED = {"dncnn_b": (31.61, 29.16, 26.23), "restormer": (31.95, 29.51, 26.62)}
SIGMAS = (15, 25, 50)


def psnr(restored, clean):
    """PSNR in dB of outputs in [0, 1] against clean images in [0, 1], both rounded to 8 bits."""
    mse = np.mean((np.round(restored * 255.0) - np.round(clean * 255.0)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


def noisy(image, sigma):
    """The authors' noisy input: numpy reseeded to 0 for every image, the noise added in float32."""
    y = image.copy()
    y += np.random.RandomState(0).normal(0, sigma / 255.0, image.shape)
    return y


def mean_psnr(model, images, sigma):
    values = [psnr(model(torch.from_numpy(noisy(image, sigma))[None, None]).numpy()[0, 0], image)
              for image in images]
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    paths = sorted((REPO_ROOT / "data" / "bsd68").glob("test*.png"))
    if len(paths) != 68:
        raise SystemExit("data/bsd68 is incomplete: run `python scripts/download_pretrained.py`")
    images = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]
    rows = []
    for name in MODELS:
        model = load_denoiser(name, device=args.device, batch_size=1)
        variants = {name: model}
        if name == "restormer":  # deepinv's own forward, with replication padding
            variants["restormer_replicate"] = Denoiser(model.net.net, "restormer_replicate", args.device,
                                                       batch_size=1)
        for variant, denoiser in variants.items():
            for sigma, published in zip(SIGMAS, PUBLISHED[name], strict=True):
                value = mean_psnr(denoiser, images, sigma)
                rows.append({"model": variant, "sigma": sigma, "psnr": value, "published": published})
                print(f"{variant:19s} sigma {sigma:2d}: {value:.2f} dB (paper {published:.2f})", flush=True)
    out = REPO_ROOT / "results" / "pretrained" / "psnr.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
