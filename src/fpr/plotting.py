"""Shared figure style: validated categorical colors, text inks and recessive axes.

SERIES holds categorical slots 1-3 of the reference palette. The three pass the
colour-vision-deficiency and contrast checks for all pairs on a light surface, so
they are safe in scatter plots. For more categories, highlight one and gray the rest.
"""

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS, GRID, NA_FILL = "#c3c2b7", "#e1e0d9", "#f0efec"
CONTEXT = "#c3c2b7"
SEQUENTIAL = ("#fcfcfb", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b")


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
