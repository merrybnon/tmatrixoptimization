"""Diagnostic figures for a trained run.

    pixi run -e ml visualize --run TMVAE_b0p15-bw20-e300-lff0p05-schcosine_Tom1000

Four PNGs into the run directory, each stamped with a provenance footer — run,
checkpoint epoch and score, drop, split seed, git hash, timestamp — so a figure
is self-describing wherever it ends up:

- ``training_curve.png``  the loss terms and the three schedules, per epoch
- ``predictions.png``     the property, predicted against true, on the test split
- ``reconstruction.png``  example matrices, and error against the size of the
                          true entry, which is where forward KL is blind
- ``latent.png``          how many latent dimensions are actually carrying code

Reads `history.csv` and `metrics.json`, and re-runs the model for the examples,
so it needs `evaluate.py` to have gone first.
"""

import argparse
import csv
import json
from datetime import datetime, timezone

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from tm_ml import evaluate as evaluate_module
from tm_ml import paths
from tm_ml.ingest import git_commit
from tm_ml.models import TMVAEConfig, off_diagonal

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

N_EXAMPLES = 3


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


def footer(fig, metrics):
    """Provenance, so a figure that escapes the run directory still says what it is."""
    fig.text(
        0.005, 0.004,
        f"{metrics['run']}  |  epoch {metrics['best_epoch']} "
        f"({metrics['best_metric']} {metrics['best_score']:.4f})  |  "
        f"drop {metrics['drop']}, split seed {metrics['split_seed']}, "
        f"n_test {metrics['n_test']}  |  git {str(metrics['git_commit'])[:8]}  |  "
        f"drawn {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        fontsize=6, color=INK_SOFT, ha="left", va="bottom",
    )


def read_history(run_dir):
    with open(run_dir / paths.HISTORY) as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def training_curve(history, metrics, out):
    """Six panels, one measure each. Never two y-scales on one axis."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    epoch = history["epoch"]

    panels = [
        ("Objective (at configured weights)", "score", "nats"),
        ("Reconstruction, per-row KL", "recon", "nats"),
        ("Property term", "prop", "squared error"),
    ]
    for ax, (title, term, ylabel) in zip(axes[0], panels):
        ax.plot(epoch, history[f"train_{term}"], color=TRAIN, label="train")
        ax.plot(epoch, history[f"val_{term}"], color=VAL, label="val")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("epoch")
        ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.plot(epoch, history["val_kl"], color=TRAIN, label="total")
    ax.plot(epoch, history["val_kl_node"], color=VAL, linestyle="--", label="node")
    ax.plot(epoch, history["val_kl_global"], color=THIRD, label="graph-level")
    ax.set_title("Prior KL (val) — collapse shows up here first")
    ax.set_ylabel("nats")
    ax.set_xlabel("epoch")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(epoch, history["val_log_recon"], color=TRAIN)
    ax.set_title("Log-space reconstruction error (val)")
    ax.set_ylabel("mean |log T − log T̂|")
    ax.set_xlabel("epoch")
    ax.annotate(
        "unweighted, so weak links count",
        xy=(0.5, 0.92), xycoords="axes fraction", ha="center",
        fontsize=7, color=INK_SOFT,
    )

    # Schedules share a scale only because each is drawn as a fraction of its
    # own configured value — a second y-axis would be the dual-axis mistake.
    ax = axes[1, 2]
    for name, colour in (("lr", TRAIN), ("beta", VAL), ("gamma", THIRD)):
        series = history[name]
        peak = series.max()
        ax.plot(epoch, series / peak if peak else series, color=colour,
                label=f"{name} (max {peak:g})")
    ax.set_title("Schedules, each as a fraction of its own maximum")
    ax.set_ylabel("fraction of configured value")
    ax.set_xlabel("epoch")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="lower right")

    fig.suptitle("Training", fontsize=12, color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def predictions(data, scaler, metrics, out):
    """Predicted against true, and the residual, in real decay-exponent units."""
    log_y = scaler.inverse(data["y"].numpy())
    log_y_hat = scaler.inverse(data["y_hat"].numpy())
    true, predicted = np.exp(log_y), np.exp(log_y_hat)
    slope, intercept, _ = evaluate_module.fit_line(log_y, log_y_hat)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    limits = [min(true.min(), predicted.min()) * 0.9, max(true.max(), predicted.max()) * 1.1]
    ax.plot(limits, limits, color=INK_SOFT, linewidth=1.0, linestyle="--", label="exact")
    ax.scatter(true, predicted, s=22, color=TRAIN, edgecolor=SURFACE,
               linewidth=0.5, label="test example")
    # Prediction regressed on truth, against the identity line. The gap between
    # them is resolution, not error: where they cross is the only exponent the
    # model gets right on average.
    grid = np.linspace(np.log(limits[0]), np.log(limits[1]), 200)
    ax.plot(np.exp(grid), np.exp(slope * grid + intercept), color=INK,
            linewidth=1.4, label=f"fit, resolution {slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("true decay exponent")
    ax.set_ylabel("predicted")
    ax.set_title(
        f"Property prediction — R² {metrics['test_r2']:.3f} on log y, "
        f"median relative error {metrics['test_median_relative_error']:.1%}"
    )
    ax.legend(loc="upper left")

    ax = axes[1]
    residual = log_y_hat - log_y
    ax.axhline(0.0, color=INK_SOFT, linewidth=1.0, linestyle="--")
    ax.scatter(true, residual, s=22, color=TRAIN, edgecolor=SURFACE, linewidth=0.5)
    # The same fit, seen as the tilt it puts on the residual: slope − 1.
    ax.plot(np.exp(grid), (slope - 1.0) * grid + intercept, color=INK, linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("true decay exponent")
    ax.set_ylabel("log ŷ − log y")
    # Resolution below 1 tilts this cloud and is the model being imperfect;
    # calibration away from 1 is the one that means the numbers are wrong.
    ax.set_title(
        f"Residual in log space — resolution {metrics['test_resolution_slope']:.2f}, "
        f"calibration {metrics['test_calibration_slope']:.2f}"
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def reconstruction(data, metrics, out):
    """Example matrices in log10, and error against the size of the true entry."""
    T = data["T"][:N_EXAMPLES].numpy()
    T_hat = data["T_hat"][:N_EXAMPLES].numpy()
    n = T.shape[-1]
    off = off_diagonal(n, torch.device("cpu")).numpy()

    log_true = np.where(off, np.log10(np.clip(T, 1e-14, None)), np.nan)
    log_pred = np.where(off, np.log10(np.clip(T_hat, 1e-14, None)), np.nan)
    difference = log_pred - log_true

    fig = plt.figure(figsize=(11.5, 3.1 * N_EXAMPLES + 3.4))
    grid = fig.add_gridspec(N_EXAMPLES + 1, 3, height_ratios=[1] * N_EXAMPLES + [1.15])
    spread = np.nanmax(np.abs(difference))
    # A shared scale from the data rather than the 1e-14 clamp: almost every
    # entry lives in the top three decades, so a floor at -14 would spend the
    # whole ramp on a handful of structural extremes and flatten the rest.
    floor = float(np.nanpercentile(log_true, 2))

    for row in range(N_EXAMPLES):
        for column, (values, title, cmap, norm) in enumerate((
            (log_true[row], "true, log₁₀ T", SEQUENTIAL, None),
            (log_pred[row], "reconstructed, log₁₀ T̂", SEQUENTIAL, None),
            (difference[row], "log₁₀(T̂ / T) — red is over-predicted", DIVERGING,
             TwoSlopeNorm(vcenter=0.0, vmin=-spread, vmax=spread)),
        )):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(
                values, cmap=cmap, norm=norm,
                vmin=floor if norm is None else None,
                vmax=0.0 if norm is None else None,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"test example {row}", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    ax = fig.add_subplot(grid[N_EXAMPLES, :])
    table = metrics["_by_magnitude"]
    centres = [0.5 * (r["true_min"] + r["true_max"]) for r in table]
    ax.plot(centres, [r["mean_abs_log_error"] for r in table],
            color=TRAIN, marker="o", markersize=5, label="mean |log T − log T̂|")
    ax.set_xscale("log")
    ax.set_xlabel("true entry (decile midpoint)")
    ax.set_ylabel("mean absolute log error")
    # The claim in this title is only true at lambda_log = 0. The log-space term
    # weights every entry equally, so once it is on, the left of the plot is
    # exactly what is being paid for and saying otherwise reads as a defect.
    lambda_log = metrics.get("lambda_log") or 0.0
    ax.set_title(
        "Reconstruction error by size of the true entry — forward KL weights each "
        "term by T_ij, so the left of this plot is nearly free"
        if not lambda_log else
        "Reconstruction error by size of the true entry — log-space term at "
        f"lambda_log {lambda_log:g} weights every entry equally"
    )
    for row in (table[0], table[-1]):
        ax.annotate(
            f"{row['mean_abs_log_error']:.2f}  (×{np.exp(row['mean_abs_log_error']):.0f})",
            xy=(0.5 * (row["true_min"] + row["true_max"]), row["mean_abs_log_error"]),
            textcoords="offset points", xytext=(0, 9), fontsize=7.5, color=INK, ha="center",
        )
    ax.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def latent(metrics, out):
    """Whether the bottleneck is the width it claims to be."""
    eigenvalues = np.array(metrics["latent_pca_eigenvalues"])
    d = len(eigenvalues)
    index = np.arange(1, d + 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    ax.semilogy(index, np.clip(eigenvalues, 1e-12, None), color=TRAIN, marker="o",
                markersize=6, label="Cov(μ) eigenvalue")
    ax.axhline(metrics["latent_noise_floor_var"], color=VAL, linestyle="--",
               linewidth=1.2, label=f"posterior noise E[σ²] = {metrics['latent_noise_floor_var']:.2f}")
    ax.set_xlabel("component")
    ax.set_ylabel("variance")
    ax.set_title(
        f"Latent spectrum — {metrics['latent_pca_components_above_noise']} of {d} "
        "components clear the noise"
    )
    ax.set_xticks(index)
    ax.legend(loc="upper right")

    ax = axes[1]
    kl_per_dim = np.array(metrics["latent_kl_per_dim"])
    ax.bar(index, np.sort(kl_per_dim)[::-1], color=TRAIN, width=0.7,
           label=f"total {kl_per_dim.sum():.2f} nats per node")
    ax.set_xlabel("latent dimension, ordered by rate")
    ax.set_ylabel("nats")
    ax.set_title(
        f"Per-dimension rate — {metrics['latent_dims_to_90pct_kl']} dimensions "
        "carry 90% of it"
    )
    ax.set_xticks(index)
    ax.legend(loc="upper right")

    ax = axes[2]
    labels = ["active\nunits", "PCA 90%", "above\nnoise", "dims to\n90% KL"]
    values = [
        metrics["latent_active_units"], metrics["latent_pca_components_90"],
        metrics["latent_pca_components_above_noise"], metrics["latent_dims_to_90pct_kl"],
    ]
    ax.bar(labels, values, color=TRAIN, width=0.62)
    ax.axhline(d, color=INK_SOFT, linestyle="--", linewidth=1.2)
    ax.annotate(f"nominal width {d}", xy=(0.02, d), xytext=(0, 4),
                textcoords="offset points", fontsize=7.5, color=INK_SOFT)
    for x, value in enumerate(values):
        ax.annotate(str(value), xy=(x, value), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, d * 1.25)
    ax.set_ylabel("dimensions per node")
    ax.set_title("Effective width, four ways")

    fig.suptitle("Latent structure", fontsize=12, color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def visualize(run, num_threads=1):
    torch.set_num_threads(num_threads)
    style()

    run_dir, checkpoint, cfg, model, splits, scaler, device = evaluate_module.load_run(run)
    metrics_path = run_dir / paths.METRICS
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} does not exist; run evaluate first")
    metrics = json.loads(metrics_path.read_text())

    model_cfg = TMVAEConfig(**checkpoint["model_config"])
    data = evaluate_module.collect(model, splits.test, model_cfg, device, cfg["batch_size"])
    history = read_history(run_dir)

    written = []
    for name, draw in (
        ("training_curve.png", lambda p: training_curve(history, metrics, p)),
        ("predictions.png", lambda p: predictions(data, scaler, metrics, p)),
        ("reconstruction.png", lambda p: reconstruction(data, metrics, p)),
        ("latent.png", lambda p: latent(metrics, p)),
    ):
        path = run_dir / name
        draw(path)
        written.append(path)
        print(f"  wrote {path}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory name under results/")
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    visualize(args.run, args.num_threads)


if __name__ == "__main__":
    main()
