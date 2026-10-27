"""The report's Figure 2: which corruption families the residual g and ensemble disagreement miss.

For each family, the share of its worst-quartile errors that a signal flags (its top quartile),
with both quartiles taken over all 13 levels of one model, as in the flagged columns of Table 1.
A random score flags 25%. Models without the idempotence penalty; mean over the three seeds, and
the whiskers span the seeds. When the dump results/pretrained/per_image.csv.gz exists, a
third panel shows the residual of the two pretrained denoisers on BSD68 crops (one run each).
Reads the per-image dumps and writes results/blind_spots.csv, results/blind_spots.pdf (for the
report) and results/blind_spots.png.

    python scripts/plot_blind_spots.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpr.data import REPO_ROOT, per_image_dump
from fpr.plotting import FAMILIES, INK, MUTED, SERIES, style_axes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_table import flagged_share, per_image_columns  # noqa: E402  (same definition as Table 1)

ARCHITECTURES = {"bottleneck": ("results/models", "dae_lam0"), "skip": ("results/models_skip", "unet_lam0")}
PRETRAINED = {"DnCNN-B": ("results/pretrained", "dncnn_b"), "Restormer": ("results/pretrained", "restormer")}
SIGNALS = {"g": "residual $g$", "dis": "disagreement"}
SHORT = {"noise": "noise", "blur": "blur", "pixel_mask": "pixel", "box_mask": "box"}


def family_shares(architecture, folder, group):
    path = per_image_dump(folder)
    frame = pd.read_csv(path, usecols=per_image_columns(path))
    frame = frame[frame["model"].str.fullmatch(rf"{group}(_seed\d)?")]
    rows = []
    for family in FAMILIES:
        # Share of the model's worst-quartile errors that come from this family.
        worst = [(seed["family"][seed["e"] >= seed["e"].quantile(0.75)] == family).mean()
                 for _, seed in frame.groupby("model")]
        for signal in SIGNALS:
            per_seed = flagged_share(frame, family, signal)
            if per_seed.empty:  # the pretrained runs have no disagreement
                continue
            rows.append({"architecture": architecture, "group": group, "family": family,
                         "signal": signal, "flagged": per_seed.mean(), "flagged_min": per_seed.min(),
                         "flagged_max": per_seed.max(), "seeds": len(per_seed),
                         "worst_quartile_share": float(np.mean(worst))})
    return rows


def plot(table, path):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    pretrained = [a for a in PRETRAINED if a in set(table["architecture"])]
    panels = [(a, [(s, label, SERIES[k]) for k, (s, label) in enumerate(SIGNALS.items())])
              for a in ARCHITECTURES]
    if pretrained:
        # One hue for the pretrained denoisers (their residual g), told apart by lightness.
        greens = (SERIES[2], "#0d6b4a")
        panels.append(("BSD68", [(a, a, greens[k]) for k, a in enumerate(pretrained)]))
    # Drawn at the printed size of one report column, so the text prints at 7-8 pt.
    fig, axes = plt.subplots(1, len(panels), figsize=(3.35, 1.5), sharey=True)
    width = 0.38
    x = np.arange(len(FAMILIES))
    handles = {}
    for ax, (title, bars) in zip(axes, panels, strict=True):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        for k, (key, label, color) in enumerate(bars):
            if title == "BSD68":
                part = table[(table["architecture"] == key) & (table["signal"] == "g")]
            else:
                part = table[(table["architecture"] == title) & (table["signal"] == key)]
            part = part.set_index("family").reindex(FAMILIES)
            offset = (k - 0.5) * width
            handles[label] = ax.bar(x + offset, part["flagged"], width=width * 0.92, color=color,
                                    zorder=2)
            spread = part["flagged_max"] - part["flagged_min"]
            if (spread > 0).any():  # single runs have no range over seeds
                ax.errorbar(x + offset, part["flagged"],
                            yerr=[part["flagged"] - part["flagged_min"], part["flagged_max"] - part["flagged"]],
                            fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.2, zorder=3)
        ax.axhline(0.25, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        ax.set_title(title, fontsize=8, color=INK, pad=3)
        ax.set_xticks(x, [SHORT[f] for f in FAMILIES], fontsize=7)
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("worst errors flagged", fontsize=7.5, color=INK)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), frameon=False,
               fontsize=7, handlelength=0.9, columnspacing=1.1, borderaxespad=0.1)
    fig.tight_layout(pad=0.2, w_pad=0.5, rect=(0, 0.1, 1, 1))
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None})  # reruns give the same file
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    sources = dict(ARCHITECTURES)
    if (REPO_ROOT / "results" / "pretrained" / "per_image.csv.gz").exists():
        sources |= PRETRAINED
    else:
        print("note: results/pretrained/per_image.csv.gz is missing, so the figure leaves out the BSD68 panel")
    table = pd.DataFrame([row for architecture, (folder, group) in sources.items()
                          for row in family_shares(architecture, folder, group)])
    out = REPO_ROOT / "results" / "blind_spots"
    table.to_csv(out.with_suffix(".csv"), index=False, float_format="%.4f")
    plot(table, out)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out}.csv, .pdf and .png")


if __name__ == "__main__":
    main()
