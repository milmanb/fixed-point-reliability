"""The report's Fig. 1b: one stable-but-wrong test image per architecture.

For each model without the idempotence penalty (seed 0): the input y, the output f(y), the second
pass f(f(y)) and the truth x of a test image whose error is in the top 10% over all levels and
whose residual g is below its median. The bottleneck pick is the first of plot_models.py's random
picks; the skip pick is drawn at random from that model's stable-but-wrong box-mask images. The
images are recomputed from the checkpoints with the evaluation seed and checked against the
per-image dumps. Writes results/examples.pdf (for the report) and results/examples.png.

    python scripts/plot_examples.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fpr.data import REPO_ROOT, load_fashion_mnist, per_image_dump  # noqa: E402
from fpr.evaluation import CONDITIONS, observe  # noqa: E402
from fpr.models import load_restorer  # noqa: E402
from fpr.plotting import INK  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_models import select_failures  # noqa: E402  (the same rule and picks)

TITLES = ["input $y$", "output $f(y)$", "$f(f(y))$", "truth $x$"]


def model_rows(folder, model):
    frame = pd.read_csv(per_image_dump(folder), usecols=["condition", "family", "model", "image", "e", "g"])
    return frame[frame["model"] == model].reset_index(drop=True)


def picks():
    rows, _, _ = select_failures(model_rows("results/models", "dae_lam0_seed0"), 6, 0)
    _, candidates, _ = select_failures(model_rows("results/models_skip", "unet_lam0_seed0"), 1, 0)
    box = candidates[candidates["family"] == "box_mask"].sample(n=1, random_state=0).iloc[0]
    return [("bottleneck", "dae_lam0_seed0", rows.iloc[0]), ("skip", "unet_lam0_seed0", box)]


def restore(model, row, x):
    """y, f(y), f(f(y)) and x of one test image, checked against the per-image dump."""
    index = int(row["image"])
    condition = {c.label: c for c in CONDITIONS}[row["condition"]]
    y = observe(condition, x, 0)[0][index:index + 1]  # drawn for the whole batch, as in evaluation
    restorer, _ = load_restorer(REPO_ROOT / "checkpoints" / f"{model}.pt", device="cpu")
    with torch.no_grad():
        fy = restorer(y)
        ffy = restorer(fy)
    e = (fy - x[index:index + 1]).abs().mean().item()
    g = (ffy - fy).abs().mean().item()
    if not (np.isclose(e, row["e"], rtol=1e-3) and np.isclose(g, row["g"], rtol=1e-2)):
        raise RuntimeError(f"{model}: recomputed e={e:.5f}, g={g:.6f}; dump e={row['e']:.5f}, g={row['g']:.6f}")
    return [y[0, 0], fy[0, 0], ffy[0, 0], x[index, 0]]


def ordinal(fraction):
    n = round(100 * fraction)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def corruption(label):
    if label.startswith("pixel_mask"):
        return f"{100 * float(label.split('=')[1].rstrip(')')):.0f}% pixel mask"
    return f"box, side {label.split('=')[1].rstrip(')')}"


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    x = load_fashion_mnist("test", dtype=torch.float64)[0][:10000]
    # Drawn one report column wide; the report places it at 0.86 of the column, so the 8 pt titles and
    # 7.5 pt labels print at about 7 and 6.5 pt.
    fig, axes = plt.subplots(2, 4, figsize=(3.35, 1.95))
    for i, (architecture, model, row) in enumerate(picks()):
        for j, image in enumerate(restore(model, row, x)):
            ax = axes[i, j]
            ax.imshow(np.clip(image.numpy(), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(TITLES[j], fontsize=8, color=INK, pad=2.5)
        axes[i, 0].set_ylabel(f"{architecture}\n{corruption(row['condition'])}", fontsize=7.5, color=INK, labelpad=2)
        axes[i, 1].set_xlabel(f"$e$: {ordinal(row['e_pct'])} pct.", fontsize=7.5, color=INK, labelpad=1.5)
        axes[i, 2].set_xlabel(f"$g$: {ordinal(row['g_pct'])} pct.", fontsize=7.5, color=INK, labelpad=1.5)
        print(f"{architecture}: {row['condition']}, image {int(row['image'])}, "
              f"e in the {ordinal(row['e_pct'])} and g in the {ordinal(row['g_pct'])} percentile")
    fig.tight_layout(pad=0.15, h_pad=0.5, w_pad=0.15)
    out = REPO_ROOT / "results" / "examples"
    fig.savefig(out.with_suffix(".pdf"), metadata={"CreationDate": None})  # reruns give the same file
    fig.savefig(out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {out}.pdf and .png")


if __name__ == "__main__":
    main()
