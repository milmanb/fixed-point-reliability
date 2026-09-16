"""Shared figure style: validated categorical colors, text inks, recessive axes, and labels.

SERIES holds categorical slots 1-3 of the reference palette. The three pass the
colour-vision-deficiency and contrast checks for all pairs on a light surface, so
they are safe in scatter plots. For more categories, highlight one and gray the rest.
"""

import re

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, GRID, NA_FILL = "#c3c2b7", "#e1e0d9", "#f0efec"
CONTEXT = "#c3c2b7"
SEQUENTIAL = ("#fcfcfb", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b")


FAMILIES = ("noise", "blur", "pixel_mask", "box_mask")
FAMILY_LABELS = {"noise": "noise", "blur": "blur", "pixel_mask": "pixel mask", "box_mask": "box mask"}
_LEVEL_LABELS = {"sigma": "$\\sigma$ = {}", "std": "width {}", "drop": "{:.0%}", "size": "side {}"}


def pretty_condition(label):
    """noise(sigma=0.1) -> noise, sigma = 0.1;  pixel_mask(drop=0.75) -> pixel mask, 75%."""
    match = re.fullmatch(r"(\w+)\((\w+)=([\d.]+)\)", label)
    if not match:
        return label
    family, key, value = match.groups()
    template = _LEVEL_LABELS.get(key, "{}")
    text = template.format(float(value) if key == "drop" else value.rstrip("."))
    return f"{FAMILY_LABELS.get(family, family)}, {text}"


def family_scatter(rows, path, title, per_condition=600, seed=0, families=FAMILIES, width=10.5):
    """Signal vs. true error: one row per (label, frame, signal), one column per corruption family.

    Each panel highlights one family in blue over all conditions in gray, so it shows whether
    the signal separates that family's errors from the rest. Panels in a row share the y axis.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    fig, axes = plt.subplots(len(rows), len(families), figsize=(width, 0.9 + 2.2 * len(rows)),
                             sharex=True, sharey="row", squeeze=False)
    for i, (label, frame, signal) in enumerate(rows):
        n = min(per_condition, int(frame.groupby("condition").size().min()))
        sample = frame.groupby("condition", group_keys=False).sample(n=n, random_state=seed)
        pooled = spearmanr(frame[signal], frame["e"]).statistic
        for j, family in enumerate(families):
            ax = axes[i, j]
            style_axes(ax)
            ax.scatter(sample["e"], sample[signal], s=3, color=CONTEXT, alpha=0.35, linewidths=0,
                       rasterized=True)
            highlighted = sample[sample["family"] == family]
            ax.scatter(highlighted["e"], highlighted[signal], s=4, color=SERIES[0], alpha=0.55,
                       linewidths=0, rasterized=True)
            within = frame[frame["family"] == family]
            ax.text(0.97, 0.95, rf"$\rho$ in family = {spearmanr(within[signal], within['e']).statistic:+.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK_SECONDARY)
            if i == 0:
                ax.set_title(FAMILY_LABELS.get(family, family), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{label}\n" + rf"pooled $\rho$ = {pooled:+.2f}", fontsize=8,
                              color=INK_SECONDARY)
            if i == len(rows) - 1:
                ax.set_xlabel("true error e", fontsize=8, color=INK_SECONDARY)
    if title:
        fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (0.9 + 2.2 * len(rows))))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def style_axes(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, width=0.8, length=3)
    if grid:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
