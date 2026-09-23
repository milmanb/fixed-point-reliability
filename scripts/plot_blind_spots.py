"""The report's Figure 2: which corruption families the residual g and ensemble disagreement miss.

For each family, the share of its worst-quartile errors that a signal flags (its top quartile),
with both quartiles taken over all 13 levels of one model, as in the flagged columns of Table 1.
A random score flags 25%. Models without the idempotence penalty; mean over the three seeds, and
the whiskers span the seeds. Reads the per-image dumps of both result folders and writes
results/blind_spots.csv, results/blind_spots.pdf (for the report) and results/blind_spots.png.

    python scripts/plot_blind_spots.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpr.data import REPO_ROOT
from fpr.plotting import FAMILIES, INK, INK_SECONDARY, MUTED, SERIES, style_axes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_table import flagged_share, per_image_columns  # noqa: E402  (same definition as Table 1)

ARCHITECTURES = {"bottleneck": ("results/models", "dae_lam0"), "skip": ("results/models_skip", "unet_lam0")}
SIGNALS = {"g": "residual $g$", "dis": "disagreement"}
SHORT = {"noise": "noise", "blur": "blur", "pixel_mask": "pixel\nmask", "box_mask": "box\nmask"}


def family_shares(architecture, folder, group):
    path = REPO_ROOT / folder / "per_image.csv.gz"
    frame = pd.read_csv(path, usecols=per_image_columns(path))
    frame = frame[frame["model"].str.fullmatch(rf"{group}_seed\d")]
    rows = []
    for family in FAMILIES:
        # Share of the model's worst-quartile errors that come from this family.
        worst = [(seed["family"][seed["e"] >= seed["e"].quantile(0.75)] == family).mean()
                 for _, seed in frame.groupby("model")]
        for signal in SIGNALS:
            per_seed = flagged_share(frame, family, signal)
            rows.append({"architecture": architecture, "group": group, "family": family,
                         "signal": signal, "flagged": per_seed.mean(), "flagged_min": per_seed.min(),
                         "flagged_max": per_seed.max(), "seeds": len(per_seed),
                         "worst_quartile_share": float(np.mean(worst))})
    return rows


def plot(table, path):
    import matplotlib.pyplot as plt

    # Drawn at the printed size of one report column, so the 6-7 pt text stays readable.
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.32), sharey=True)
    width = 0.38
    x = np.arange(len(FAMILIES))
    for ax, architecture in zip(axes, ARCHITECTURES):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        rows = table[table["architecture"] == architecture].set_index(["signal", "family"])
        for k, (signal, color) in enumerate(zip(SIGNALS, (SERIES[0], SERIES[1]))):
            part = rows.loc[signal].reindex(FAMILIES)
            offset = (k - 0.5) * width
            ax.bar(x + offset, part["flagged"], width=width * 0.92, color=color, label=SIGNALS[signal],
                   zorder=2)
            ax.errorbar(x + offset, part["flagged"],
                        yerr=[part["flagged"] - part["flagged_min"], part["flagged_max"] - part["flagged"]],
                        fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=3)
        ax.axhline(0.25, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        ax.set_title(architecture, fontsize=7, color=INK, pad=3)
        ax.set_xticks(x, [SHORT[f] for f in FAMILIES], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("worst errors flagged", fontsize=6.5, color=INK_SECONDARY)
    axes[1].text(len(FAMILIES) - 0.45, 0.27, "chance", fontsize=6, color=MUTED, ha="right", va="bottom")
    axes[1].legend(loc="upper right", frameon=False, fontsize=6, handlelength=1.0, borderaxespad=0.1,
                   labelspacing=0.3)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})  # reruns give the same file
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    table = pd.DataFrame([row for architecture, (folder, group) in ARCHITECTURES.items()
                          for row in family_shares(architecture, folder, group)])
    out = REPO_ROOT / "results" / "blind_spots"
    table.to_csv(out.with_suffix(".csv"), index=False, float_format="%.4f")
    plot(table, out)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out}.csv, .pdf and .png")


if __name__ == "__main__":
    main()
