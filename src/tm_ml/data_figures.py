"""Figures describing a processed drop, before any model touches it.

    pixi run python -m tm_ml.data_figures --drop Tom1000

PNGs into ``data/processed/<drop>/`` beside the arrays they describe, each
stamped with a provenance footer:

- ``target_statistics.png``  histogram and survival curve of the targets, on a
                             linear and a log axis, with summary statistics
- ``example_matrices.png``   the four highest-target matrices, the four nearest
                             the median and the four lowest, in log10

Outside the Snakemake workflow: it is for looking at the data, and nothing
downstream reads what it writes.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tm_ml.ingest import PROCESSED_DIR, git_commit  # noqa: E402
from tm_ml.style import INK, INK_SOFT, SEQUENTIAL, TRAIN, VAL, style  # noqa: E402

N_BINS = 40
N_EXAMPLES = 4
LOG_CLIP = 1e-14


def footer(fig, meta):
    """Provenance, so a figure that leaves the drop directory still says what it is."""
    fig.text(
        0.005, 0.004,
        f"drop {meta['drop']}, n {meta['n']}  |  {meta['source']}  |  "
        f"ingested at git {str(meta['git_commit'])[:8]}  |  "
        f"drawn at git {str(git_commit())[:8]}, "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        fontsize=6, color=INK_SOFT, ha="left", va="bottom",
    )


def target_statistics(targets, meta, out):
    """Histogram and survival curve, each on a linear and a log axis."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.14, hspace=0.38, wspace=0.2)
    median = float(np.median(targets))
    ordered = np.sort(targets)
    # Fraction of matrices whose target exceeds t, stepping down at each target.
    survival = 1.0 - np.arange(1, len(ordered) + 1) / len(ordered)
    bins = {
        "linear": np.linspace(ordered[0], ordered[-1], N_BINS + 1),
        "log": np.geomspace(ordered[0], ordered[-1], N_BINS + 1),
    }

    for row, scale in enumerate(("linear", "log")):
        ax = axes[row, 0]
        ax.hist(targets, bins=bins[scale], color=TRAIN, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("matrices")
        ax.set_title(f"Histogram, {scale} bins")

        ax = axes[row, 1]
        ax.step(np.concatenate([[ordered[0]], ordered]),
                np.concatenate([[1.0], survival]), where="post", color=TRAIN)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction with target > t")
        ax.set_title(f"Survival, {scale} axis")

        for ax in axes[row]:
            ax.set_xscale(scale)
            ax.set_xlabel("target t (decay exponent)")
            ax.axvline(median, color=VAL, linestyle="--", linewidth=1.2, label="median")
            ax.legend(loc="upper right")

    fig.suptitle(f"Targets of {meta['drop']}", fontsize=12, color=INK, x=0.005, ha="left")
    fig.text(
        0.5, 0.05,
        f"over the {len(targets)} matrices:   mean {targets.mean():.1f}   "
        f"median {median:.1f}   maximum {targets.max():.1f}   minimum {targets.min():.1f}",
        fontsize=10, color=INK, ha="center", va="center",
    )
    footer(fig, meta)
    fig.savefig(out)
    plt.close(fig)


def log_matrix(T):
    """log10 of the off-diagonal entries, the structural zero diagonal left blank."""
    off = ~np.eye(T.shape[-1], dtype=bool)
    return np.where(off, np.log10(np.clip(T, LOG_CLIP, None)), np.nan)


def example_indices(targets):
    """Indices for the highest, nearest-median and lowest rows, in that order."""
    order = np.argsort(targets)
    nearest = np.argsort(np.abs(targets - np.median(targets)))[:N_EXAMPLES]
    return [
        ("highest", order[::-1][:N_EXAMPLES]),
        ("nearest median", nearest[np.argsort(targets[nearest])]),
        ("lowest", order[:N_EXAMPLES]),
    ]


def example_matrices(matrices, targets, meta, out):
    """Three rows of four matrices in log10 on one shared scale."""
    rows = example_indices(targets)
    logged = {i: log_matrix(matrices[i]) for _, indices in rows for i in indices}
    # A floor from the entries drawn rather than the clip: most entries sit in
    # the top few decades, and a floor at -14 would flatten them.
    floor = float(np.nanpercentile(np.stack(list(logged.values())), 2))

    fig = plt.figure(figsize=(11, 8.6))
    grid = fig.add_gridspec(
        len(rows), N_EXAMPLES + 1, width_ratios=[1] * N_EXAMPLES + [0.05],
        left=0.06, right=0.93, top=0.91, bottom=0.05, wspace=0.12, hspace=0.3,
    )
    for row, (label, indices) in enumerate(rows):
        for column, i in enumerate(indices):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(logged[i], cmap=SEQUENTIAL, vmin=floor, vmax=0.0,
                              interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"#{i}   target {targets[i]:.1f}", fontsize=9)
            if column == 0:
                ax.set_ylabel(label, fontsize=9)
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, N_EXAMPLES]))
    colorbar.set_label("log₁₀ T", fontsize=8, color=INK_SOFT)
    colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Example matrices of {meta['drop']}, by target", fontsize=12,
                 color=INK, x=0.005, ha="left")
    footer(fig, meta)
    fig.savefig(out)
    plt.close(fig)


def data_figures(drop, processed_dir=None):
    processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR / drop
    matrices = np.load(processed_dir / "matrices.npy").astype(np.float64)
    targets = np.load(processed_dir / "targets.npy").astype(np.float64)
    meta = json.loads((processed_dir / "meta.json").read_text())

    style()
    target_statistics(targets, meta, processed_dir / "target_statistics.png")
    example_matrices(matrices, targets, meta, processed_dir / "example_matrices.png")
    return processed_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drop", required=True, help="drop name under data/processed/")
    parser.add_argument("--processed-dir", help="override data/processed/<drop>/")
    args = parser.parse_args()

    out = data_figures(args.drop, processed_dir=args.processed_dir)
    print(f"wrote target_statistics.png and example_matrices.png to {out}")


if __name__ == "__main__":
    main()
