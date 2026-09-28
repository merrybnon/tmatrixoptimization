"""The house palette and matplotlib style, shared by every figure.

Kept free of torch so a figure that only reads ``data/processed/`` runs in the
default environment.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# The validated categorical slots, in fixed order, assigned to entities rather
# than to rank: split 1 is always train, split 2 always val, whatever is drawn.
TRAIN, VAL, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"

# One hue, light to dark, for magnitude. A rainbow would invent structure.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281", "#0d366b"]
)
# Two poles and a neutral gray midpoint, for signed error.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#6da7ec", "#f0efec", "#e87b7b", "#8f1f1f"]
)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SOFT,
        "axes.titlesize": 10,
        "axes.titlecolor": INK,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "figure.dpi": 140,
    })
