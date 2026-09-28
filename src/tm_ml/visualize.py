"""Diagnostic figures for a trained run.

    pixi run -e ml visualize --run TMVAE_b0p15-bw20-e300-lff0p05-schcosine_Tom1000

PNGs into ``figures/`` in the run directory, each stamped with a provenance
footer — run, checkpoint epoch and score, drop, split seed, git hash, timestamp —
so a figure is self-describing wherever it ends up:

- ``training_curve.png``  the loss terms and the three schedules, per epoch
- ``predictions.png``     the property, predicted against true, on the test split
- ``reconstruction.png``  example matrices, and error against the size of the
                          true entry, which is where forward KL is blind
- ``spatial_graph.png``   example landscapes in space, positions recovered from
                          T and from T̂ by inverting the generator, the
                          reconstruction aligned to the truth
- ``latent.png``          how many latent dimensions are actually carrying code
- ``latent_property.png`` where graphs sit in the latent, coloured by the
                          property: the global against node-side axes, and
                          the sorted PCA with and without the global
- ``latent_property_3D.png`` the predicted exponent as a partial-dependence
                          surface over the global dim and sorted-160 PC1,
                          as log ŷ and as ŷ, with the test graphs standing on it
- ``latent_property_3D.html`` the same as a page to turn and hover, plotly.js
                          embedded so it opens offline

and into ``figures/interpolation_and_traversals/``:

- ``traversal_gbd.png``   decoded matrices stepping along the global dimension
- ``traversal_PC1.png``   the same along sorted-160 PC1
- ``traversal_PC2.png``   and PC2
- ``interpolation.png``   straight lines between test graphs, nodes matched first
- ``*_spatial.png``       each of the above with every decoded matrix drawn as
                          the landscape it implies, as in ``spatial_graph.png``

and into ``figures/optimization/``:

- ``maximize_g#.png``     BFGS ascent of the predicted exponent from test
                          example #, penalized by distance from the start; decoded along the
                          path, then the path in the graph-level latent views
- ``minimize_g#.png``     the same descending
- ``*_g#_unreg.png``      either with no penalty

Reads `history.csv` and `metrics.json`, and re-runs the model for the examples,
so it needs `evaluate.py` to have gone first.
"""

import argparse
import csv
import functools
import json
import textwrap
from datetime import datetime, timezone

import matplotlib
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm, to_hex  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from tm_ml import evaluate as evaluate_module
from tm_ml import landscape, optimize, paths
from tm_ml.ingest import git_commit
from tm_ml.models import TMVAEConfig, off_diagonal
from tm_ml.style import (
    DIVERGING, GRID, INK, INK_SOFT, PROPERTY, PROPERTY_SOLID, SEQUENTIAL, SURFACE, THIRD,
    TRAIN, VAL, style,
)

N_EXAMPLES = 3
# Traversal steps, in units of the axis's between-graph σ.
TRAVERSAL_STEPS = np.arange(-3, 4)
INTERPOLATION_STEPS = 7
# Decoded matrices drawn along an optimization path.
N_OPTIMIZATION_COLUMNS = 7
# The weight λ on ‖x − x₀‖² for each optimization variant, the distance from
# the start. `optimize.py` says why the penalty is not anchored at the origin.
OPTIMIZATION_LAMBDA = {"": 0.1, "_unreg": 0.0}
# The sign s on log ŷ for each direction.
OPTIMIZATION_SENSE = {"maximize": 1, "minimize": -1}
# The partial-dependence surface: grid points per axis, the half-width of the
# grid on each axis, centred on zero, and the (elevation, azimuth) it is seen from.
PDP_GRID = 100
PDP_GLOBAL_EXTENT = 4.0
PDP_PC1_EXTENT = 7.0
PDP_VIEW = (25, -60)
# Below this rate a dimension is prior noise, not code.
ACTIVE_DIM_MIN_KL = 0.01
LOG_FLOOR = 1e-12


def provenance(metrics):
    """Run, checkpoint, drop, split seed, git hash and timestamp, as one line."""
    return (
        f"{metrics['run']}  |  epoch {metrics['best_epoch']} "
        f"({metrics['best_metric']} {metrics['best_score']:.4f})  |  "
        f"drop {metrics['drop']}, split seed {metrics['split_seed']}, "
        f"n_test {metrics['n_test']}  |  git {str(metrics['git_commit'])[:8]}  |  "
        f"drawn {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )


def footer(fig, metrics):
    """Provenance, so a figure that escapes the run directory still says what it is."""
    fig.text(0.005, 0.004, provenance(metrics), fontsize=6, color=INK_SOFT,
             ha="left", va="bottom")


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
    ax.scatter(true, predicted, s=22, color=PROPERTY_SOLID, edgecolor=SURFACE,
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
    ax.scatter(true, residual, s=22, color=PROPERTY_SOLID, edgecolor=SURFACE, linewidth=0.5)
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


def log_matrix(T):
    """log10 of the off-diagonal entries, the structural zero diagonal left blank."""
    off = off_diagonal(T.shape[-1], torch.device("cpu")).numpy()
    return np.where(off, np.log10(np.clip(T, 1e-14, None)), np.nan)


def sorted_descriptor(mu):
    """Each dimension's node values sorted, flattened: ``(graphs, n_nodes * d_latent)``.

    Sorting each dimension's 20 node values independently is invariant to
    relabelling, which the raw flattened 160-vector is not: node slot i holds
    a different node in every graph, so a projection of it would describe the
    labelling. This keeps each dimension's marginal over nodes; what it gives
    up is the joint, which node held which combination across dimensions.
    """
    return np.sort(mu, axis=1).reshape(len(mu), -1)


def unsort(descriptor, order):
    """Sorted descriptors ``(k, n_nodes * d_latent)`` back onto nodes as ``(k, n_nodes, d_latent)``.

    ``order`` is ``np.argsort(mu, axis=1)`` of the latents the descriptors came
    from, so rank r of dimension d returns to the node that held rank r — the
    only inverse the sort has once a move has let values cross ranks.
    """
    moved = descriptor.reshape(order.shape)
    z = np.empty_like(moved)
    np.put_along_axis(z, order, moved, axis=1)
    return z


def principal_axes(matrix):
    """Column mean, singular values and right singular vectors, rows as samples."""
    mean = matrix.mean(0)
    _, singular, right = np.linalg.svd(matrix - mean, full_matrices=False)
    return mean, singular, right


def reconstruction(data, metrics, out):
    """Example matrices in log10, and error against the size of the true entry."""
    log_true = log_matrix(data["T"][:N_EXAMPLES].numpy())
    log_pred = log_matrix(data["T_hat"][:N_EXAMPLES].numpy())
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
    # Which objective the run actually trained against, since the shape of this
    # curve means opposite things under each. Forward KL weights by T_ij and so
    # buys the left of the plot nothing; the log-space term weights every entry
    # alike, so once it is on, the left is what is being paid for and calling it
    # free reads as a defect.
    lambda_log = metrics.get("lambda_log") or 0.0
    lambda_recon = metrics.get("lambda_recon", 1.0)
    head = "Reconstruction error by size of the true entry — "
    if not lambda_log:
        tail = "forward KL weights each term by T_ij, so the left of this plot is nearly free"
    elif not lambda_recon:
        tail = f"log-space term only at lambda_log {lambda_log:g}, every entry weighted alike"
    else:
        tail = (f"lambda_recon {lambda_recon:g} weights by T_ij, lambda_log "
                f"{lambda_log:g} weights every entry alike")
    ax.set_title(head + tail)
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


def spatial_graph(data, metrics, out):
    """Example landscapes laid out in space, from T and from T̂.

    Positions come from inverting the generator, `landscape.positions`. The
    reconstruction keeps the input's node labels, so its layout is aligned onto
    the true one by Procrustes on those labels, which settles the rotation,
    reflection and shift that T alone leaves open. Both are scaled to fill the
    unit square first, so the alignment does not fit a scale of its own.
    """
    fig, axes = plt.subplots(N_EXAMPLES, 2, figsize=(11.0, 5.4 * N_EXAMPLES))
    for row in range(N_EXAMPLES):
        T = data["T"][row].numpy().astype(np.float64)
        T_hat = data["T_hat"][row].numpy().astype(np.float64)
        true_xy, _ = landscape.positions(T)
        recon_xy, _ = landscape.positions(T_hat)
        recon_xy = landscape.align(recon_xy, true_xy, scale=False)
        error = np.sqrt(((recon_xy - true_xy) ** 2).sum(axis=1).mean())
        heaviest = max(T.max(), T_hat.max())
        bounds = landscape.limits(true_xy, recon_xy)
        for ax, matrix, xy, title in (
            (axes[row, 0], T, true_xy, "true, positions from T"),
            (axes[row, 1], T_hat, recon_xy,
             f"reconstructed, from T̂ aligned to true — RMS offset {error:.3f}"),
        ):
            landscape.draw(ax, matrix, xy, heaviest, bounds)
            ax.set_xlabel("x")
            ax.set_title(title)
        axes[row, 0].set_ylabel(f"test example {row}\n\ny", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def latent(metrics, out):
    """Whether the bottleneck is the width it claims to be.

    One row per latent group: the node latent always, and the graph-level one
    below it when the run has one. Kept apart because the global slot is the
    one that collapses, and a pooled figure would hide it.
    """
    groups = [("latent", "Node latent", "per node")]
    if metrics.get("d_global"):
        groups.append(("global", "Global latent", "per graph"))

    fig, axes = plt.subplots(len(groups), 3, figsize=(13, 4.2 * len(groups)), squeeze=False)
    for row, (prefix, name, unit) in zip(axes, groups):
        latent_row(row, metrics, prefix, name, unit)

    fig.suptitle("Latent structure", fontsize=12, color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def latent_row(axes, metrics, prefix, name, unit):
    """Spectrum, per-dimension rate and effective width for one latent group."""
    eigenvalues = np.array(metrics[f"{prefix}_pca_eigenvalues"])
    d = len(eigenvalues)
    index = np.arange(1, d + 1)
    noise = metrics[f"{prefix}_noise_floor_var"]

    ax = axes[0]
    ax.semilogy(index, np.clip(eigenvalues, 1e-12, None), color=TRAIN, marker="o",
                markersize=6, label="Cov(μ) eigenvalue")
    ax.axhline(noise, color=VAL, linestyle="--", linewidth=1.2,
               label=f"posterior noise E[σ²] = {noise:.2f}")
    ax.set_xlabel("component")
    ax.set_ylabel("variance")
    ax.set_title(
        f"{name} spectrum — {metrics[f'{prefix}_pca_components_above_noise']} of {d} "
        "components clear the noise"
    )
    ax.set_xticks(index)
    ax.legend(loc="upper right")

    ax = axes[1]
    kl_per_dim = np.array(metrics[f"{prefix}_kl_per_dim"])
    ax.bar(index, np.sort(kl_per_dim)[::-1], color=TRAIN, width=0.7,
           label=f"total {kl_per_dim.sum():.2f} nats {unit}")
    ax.set_xlabel(f"{name.lower()} dimension, ordered by rate")
    ax.set_ylabel("nats")
    ax.set_title(
        f"{name} rate — {metrics[f'{prefix}_dims_to_90pct_kl']} dimensions "
        "carry 90% of it"
    )
    ax.set_xticks(index)
    ax.legend(loc="upper right")

    ax = axes[2]
    labels = ["active\nunits", "PCA 90%", "above\nnoise", "dims to\n90% KL"]
    values = [
        metrics[f"{prefix}_active_units"], metrics[f"{prefix}_pca_components_90"],
        metrics[f"{prefix}_pca_components_above_noise"], metrics[f"{prefix}_dims_to_90pct_kl"],
    ]
    ax.bar(labels, values, color=TRAIN, width=0.62)
    ax.axhline(d, color=INK_SOFT, linestyle="--", linewidth=1.2)
    ax.annotate(f"nominal width {d}", xy=(0.02, d), xytext=(0, 4),
                textcoords="offset points", fontsize=7.5, color=INK_SOFT)
    for x, value in enumerate(values):
        ax.annotate(str(value), xy=(x, value), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_ylim(0, d * 1.25)
    ax.set_ylabel(f"dimensions {unit}")
    ax.set_title(f"{name} effective width, four ways")


def latent_axes(mu, log_y, metrics, d_global):
    """Which node dimensions the graph-level panels plot, ranked two ways.

    Shared by `latent_property` and `optimization`, so an optimization path is
    drawn in exactly the axes `latent_property.png` shows. ``by_rate`` orders
    the live dimensions by KL, ``by_property`` by |r| of the graph mean with
    log y; ``top_global`` is the highest-rate global dimension, or None.
    """
    n_graphs, _, d_latent = mu.shape
    flat = mu.reshape(-1, d_latent)
    graph_mean = mu.mean(1)
    kl_per_dim = np.asarray(
        metrics.get("latent_kl_per_dim") or np.zeros(d_latent), dtype=float
    )
    # A dead dimension sits at mu = 0, sigma = 1. It carries nothing, and left in
    # it would win the correlation ranking on sampling noise alone.
    live = np.flatnonzero((kl_per_dim > ACTIVE_DIM_MIN_KL) & (flat.var(0) > 0))
    if live.size == 0:
        live = np.arange(d_latent)

    by_rate = live[np.argsort(kl_per_dim[live])[::-1]]
    enough = n_graphs > 2 and log_y.std() > 0
    if enough:
        correlation = np.array([
            np.corrcoef(graph_mean[:, d], log_y)[0, 1] for d in range(d_latent)
        ])
        by_property = live[np.argsort(np.abs(correlation[live]))[::-1]]
    else:
        correlation = np.full(d_latent, np.nan)
        by_property = by_rate

    global_kl = np.asarray(metrics.get("global_kl_per_dim") or np.zeros(d_global), dtype=float)
    return {
        "kl_per_dim": kl_per_dim, "live": live, "by_rate": by_rate,
        "by_property": by_property, "correlation": correlation, "enough": enough,
        "global_kl": global_kl, "top_global": int(np.argmax(global_kl)) if d_global else None,
    }


def latent_property(data, scaler, metrics, out):
    """Where graphs land in the latent, and whether that place tracks the property.

    One point per graph throughout, coloured by its decay exponent. A run with a
    global latent gets two rows. The first puts the global dimension against
    the best node-side axis by each criterion — rate, property, sorted PC1 —
    all as graph means, like the global itself. The second is the sorted PCA,
    alone and with the global appended, both raw and weighted by sqrt(n_nodes).
    Raw, the global is one column beside 160 and barely moves the components;
    weighted, it counts as much as one node dimension, which fills n_nodes
    sorted slots. Neither weight is the right one, so both are drawn and the
    global's loading is written on each. A run without one gets the sorted PCA
    alone.
    """
    mu = data["mu"].numpy()
    log_y = scaler.inverse(data["y"].numpy())
    # Colour in the units the property is quoted in; the axis ranking below
    # stays on log y, which is what the head regresses and what R² is measured
    # against, so the two are not the same scale by design.
    decay = np.exp(log_y)
    norm = plt.Normalize(decay.min(), decay.max())
    n_graphs, n_nodes, d_latent = mu.shape
    graph_mean = mu.mean(1)

    mu_global = data["mu_global"].numpy()
    d_global = mu_global.shape[1]
    ranked = latent_axes(mu, log_y, metrics, d_global)
    by_rate, by_property = ranked["by_rate"], ranked["by_property"]
    n_rows = 2 if d_global else 1

    # Margins held in inches, so the extra row adds height instead of squeezing.
    height = 4.4 * n_rows + 0.6
    fig = plt.figure(figsize=(16.5, height))
    grid = fig.add_gridspec(
        n_rows, 4, width_ratios=[1, 1, 1, 0.038],
        left=0.05, right=0.935, top=1 - 0.92 / height, bottom=0.88 / height,
        wspace=0.30, hspace=0.42,
    )
    axes = [[fig.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(n_rows)]
    bar = fig.add_subplot(grid[:, 3])

    def graph_scatter(ax, x, y, xlabel, ylabel, title):
        ax.scatter(x, y, c=decay, cmap=PROPERTY, norm=norm, s=34, alpha=0.95,
                   edgecolor=SURFACE, linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    n_sorted = n_nodes * d_latent
    descriptor = sorted_descriptor(mu)

    def pca_panel(ax, matrix, name, title):
        """PC1 against PC2 of `matrix`, one point per graph; returns the loadings."""
        if n_graphs <= 2:
            ax.set_axis_off()
            ax.set_title(f"{title} — needs 3+ graphs")
            return None, None
        mean, singular, right = principal_axes(matrix)
        coordinates = (matrix - mean) @ right[:2].T
        captured = (singular[:2] ** 2).sum() / max((singular ** 2).sum(), LOG_FLOOR)
        graph_scatter(ax, coordinates[:, 0], coordinates[:, 1], f"{name} PC1",
                      f"{name} PC2", f"{title} — top 2 hold {captured:.1%}")
        return coordinates, right[:2]

    sorted_row = axes[-1]
    coordinates, _ = pca_panel(sorted_row[0], descriptor, f"sorted-{n_sorted}",
                               f"All {n_sorted}, sorted invariant")

    if d_global:
        global_kl = np.asarray(
            metrics.get("global_kl_per_dim") or np.zeros(d_global), dtype=float
        )
        # The highest-rate global dimension. On a collapsed run it is still drawn,
        # though it is then noise.
        top = int(np.argmax(global_kl))
        g = mu_global[:, top]
        g_label = f"global dim {top}"

        for ax, dim, criterion in (
            (axes[0][0], by_rate[0], "highest-rate"),
            (axes[0][1], by_property[0], "top property"),
        ):
            graph_scatter(ax, g, graph_mean[:, dim], g_label, f"latent dim {dim}",
                          f"Global against the {criterion} node dim")
        ax = axes[0][2]
        if coordinates is not None:
            graph_scatter(ax, g, coordinates[:, 0], g_label, f"sorted-{n_sorted} PC1",
                          f"Global against sorted-{n_sorted} PC1")
        else:
            ax.set_axis_off()

        n_joint = n_sorted + d_global
        centred_global = mu_global - mu_global.mean(0)
        for ax, weight, name in (
            (sorted_row[1], 1.0, "unweighted"),
            (sorted_row[2], np.sqrt(n_nodes), f"global ×√{n_nodes}"),
        ):
            _, right = pca_panel(ax, np.c_[descriptor, weight * centred_global],
                                 f"sorted-{n_joint}", f"Sorted + global, {name}")
            if right is not None:
                loading = np.linalg.norm(right[:, n_sorted:], axis=1)
                ax.annotate(
                    f"global loading: PC1 {loading[0]:.2f}, PC2 {loading[1]:.2f}",
                    xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                    fontsize=7.5, color=INK_SOFT,
                )
    else:
        for ax in sorted_row[1:]:
            ax.set_axis_off()

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PROPERTY), cax=bar).set_label(
        "decay exponent", fontsize=8, color=INK_SOFT)
    bar.tick_params(labelsize=7)
    fig.suptitle("Latent space against the property", fontsize=12, color=INK,
                 x=0.005, ha="left")
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def pdp_surface(model, data, scaler, metrics, device):
    """The predicted exponent as a surface over the global dim and sorted-160 PC1.

    ŷ depends on every latent coordinate, so the surface is a partial
    dependence: at each grid point (g, p) every test graph has its global dim
    set to g and its PC1 score moved to p, keeping everything else at its own
    encoding, and the prediction is averaged over the test graphs. That is
    the effect of these two coordinates with the rest distributed as in the
    data, not a slice through any one graph.

    The PC1 move is the traversal's: shift the sorted descriptor along the
    component and put the values back on the nodes they came from, so every
    other component and the within-graph spread are untouched.

    Two panels, each averaging what it plots: the mean of log ŷ, and the mean
    of ŷ, which is not exp of the first. Computed once and drawn twice, as
    `latent_property_3D`'s PNG and `latent_property_3D_html`'s page. Returns
    ``{"message": ...}`` instead when there is nothing to draw.
    """
    mu = data["mu"].numpy()
    mu_global = data["mu_global"].numpy()
    log_y = scaler.inverse(data["y"].numpy())
    n_graphs, n_nodes, d_latent = mu.shape
    d_global = mu_global.shape[1]
    n_sorted = n_nodes * d_latent
    if not d_global:
        return {"message": "Latent property 3D — this run has no global latent"}
    if n_graphs <= 2:
        return {"message": "Latent property 3D — needs 3+ test graphs"}
    decay = np.exp(log_y)

    top = latent_axes(mu, log_y, metrics, d_global)["top_global"]
    global_kl = np.asarray(metrics.get("global_kl_per_dim") or np.zeros(d_global), dtype=float)
    g = mu_global[:, top]

    descriptor = sorted_descriptor(mu)
    mean, _, right = principal_axes(descriptor)
    direction = right[0]
    score = (descriptor - mean) @ direction
    # An eigenvector has no sign of its own; +PC1 is made to raise log y, as
    # in the PC1 traversal.
    if score.std() > 0 and np.corrcoef(score, log_y)[0, 1] < 0:
        direction, score = -direction, -score
    order = np.argsort(mu, axis=1)

    g_grid = np.linspace(-PDP_GLOBAL_EXTENT, PDP_GLOBAL_EXTENT, PDP_GRID)
    p_grid = np.linspace(-PDP_PC1_EXTENT, PDP_PC1_EXTENT, PDP_GRID)

    # One batch per PC1 value: every g on the grid for every test graph.
    pdp_log = np.empty((PDP_GRID, PDP_GRID))
    pdp_raw = np.empty((PDP_GRID, PDP_GRID))
    with torch.no_grad():
        for i, p in enumerate(p_grid):
            z = unsort(descriptor + np.outer(p - score, direction), order)
            z = np.tile(z, (PDP_GRID, 1, 1))
            z_global = np.tile(mu_global, (PDP_GRID, 1))
            z_global[:, top] = np.repeat(g_grid, n_graphs)
            log_y_hat = model.predict(
                *(torch.as_tensor(a, dtype=torch.float32, device=device) for a in (z, z_global))
            ).cpu().numpy()
            log_y_hat = scaler.inverse(log_y_hat).reshape(PDP_GRID, n_graphs)
            # Mean over the test graphs, the axis the partial dependence integrates out.
            pdp_log[i] = log_y_hat.mean(1)
            pdp_raw[i] = np.exp(log_y_hat).mean(1)

    panels = []
    for surface, colour, true, zlabel, title in (
        (pdp_log, np.exp(pdp_log), log_y, "log ŷ",
         f"Mean of log ŷ over the {n_graphs} test graphs"),
        (pdp_raw, pdp_raw, decay, "ŷ",
         f"Mean of ŷ over the {n_graphs} test graphs — not exp of the left"),
    ):
        under = RegularGridInterpolator((p_grid, g_grid), surface, bounds_error=False,
                                        fill_value=None)(np.c_[score, g])
        # Over the test graphs: how far they stand from the surface, against
        # how far they spread about their own mean.
        note = (f"test graphs to surface, RMS {np.sqrt(np.mean((true - under) ** 2)):.3g}\n"
                f"test graphs about their mean, std {true.std():.3g}")
        panels.append({"surface": surface, "colour": colour, "true": true, "under": under,
                       "zlabel": zlabel, "title": title, "note": note})

    # Colour is the decay exponent on both panels, on a log scale so the two
    # read alike and the skewed upper end does not take the whole ramp.
    return {
        "g": g, "score": score, "decay": decay, "g_grid": g_grid, "p_grid": p_grid,
        "panels": panels,
        "low": min(decay.min(), np.exp(pdp_log).min(), pdp_raw.min()),
        "high": max(decay.max(), np.exp(pdp_log).max(), pdp_raw.max()),
        "xlabel": f"global dim {top} — {global_kl[top]:.2f} nats",
        "ylabel": f"sorted-{n_sorted} PC1",
        "title": (f"Partial dependence of the predicted exponent on global dim {top} "
                  f"and sorted-{n_sorted} PC1"),
    }


def latent_property_3D(pdp, metrics, out):
    """`pdp_surface` as a still: two 3D panels, test graphs stemmed to the surface."""
    if "message" in pdp:
        return placeholder(metrics, out, pdp["message"])
    g, score, decay = pdp["g"], pdp["score"], pdp["decay"]
    G, P = np.meshgrid(pdp["g_grid"], pdp["p_grid"])
    norm = LogNorm(pdp["low"], pdp["high"])

    fig = plt.figure(figsize=(16, 7.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03], left=0.01, right=0.95,
                            top=0.88, bottom=0.1, wspace=0.08)
    for column, panel in enumerate(pdp["panels"]):
        ax = fig.add_subplot(grid[0, column], projection="3d")
        faces = PROPERTY(norm(panel["colour"]))
        faces[..., 3] = 0.72
        ax.plot_surface(G, P, panel["surface"], facecolors=faces, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
        for x, y, top_z, bottom_z in zip(g, score, panel["true"], panel["under"]):
            ax.plot([x, x], [y, y], [bottom_z, top_z], color=INK_SOFT, lw=0.5, alpha=0.5)
        ax.scatter(g, score, panel["true"], c=decay, cmap=PROPERTY, norm=norm, s=18,
                   edgecolor=INK, linewidth=0.3, depthshade=False)

        ax.text2D(0.02, 0.97, panel["note"], transform=ax.transAxes, va="top",
                  fontsize=8, color=INK_SOFT)
        ax.set_xlabel(pdp["xlabel"])
        ax.set_ylabel(pdp["ylabel"])
        ax.set_zlabel(panel["zlabel"])
        ax.set_title(panel["title"])
        ax.view_init(elev=PDP_VIEW[0], azim=PDP_VIEW[1])
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color(SURFACE)
        ax.tick_params(labelsize=7)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PROPERTY),
                       cax=fig.add_subplot(grid[0, 2]))
    bar.set_label("decay exponent — partial dependence on the surface, true for test graphs",
                  fontsize=8, color=INK_SOFT)
    bar.ax.tick_params(labelsize=7)
    fig.suptitle(pdp["title"], fontsize=12, color=INK, x=0.005, ha="left")
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def latent_property_3D_html(pdp, metrics, out):
    """`pdp_surface` as a page to turn and hover, plotly.js embedded so it opens offline.

    Colour carries log₁₀ of the exponent on one shared axis, the LogNorm of
    the PNG, with the bar labelled in the exponent itself. Hovering a test
    graph gives its index, coordinates, true value and the surface under it.
    """
    if "message" in pdp:
        fig = go.Figure()
        fig.add_annotation(text=pdp["message"], x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=16, color=INK_SOFT))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return write_html(fig, metrics, out)
    g, score, decay = pdp["g"], pdp["score"], pdp["decay"]
    low, high = np.log10(pdp["low"]), np.log10(pdp["high"])
    ticks = [v for v in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
             if low <= np.log10(v) <= high]

    fig = make_subplots(
        1, 2, specs=[[{"type": "scene"}] * 2], horizontal_spacing=0.02,
        subplot_titles=[f"{p['title']}<br><span style='font-size:11px;color:{INK_SOFT}'>"
                        f"{p['note'].replace(chr(10), '  ·  ')}</span>"
                        for p in pdp["panels"]],
    )
    elevation, azimuth = np.radians(PDP_VIEW)
    eye = 2.1 * np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    customdata = np.c_[np.arange(len(g)), decay]
    for column, panel in enumerate(pdp["panels"], start=1):
        value = "log ŷ %{z:.3f}" if column == 1 else "ŷ %{z:.4g}"
        fig.add_trace(go.Surface(
            x=pdp["g_grid"], y=pdp["p_grid"], z=panel["surface"],
            surfacecolor=np.log10(panel["colour"]), coloraxis="coloraxis", opacity=0.8,
            hovertemplate=f"global %{{x:.2f}}<br>PC1 %{{y:.2f}}<br>{value}<extra>surface</extra>",
        ), 1, column)
        stems = np.full((len(g), 3, 3), np.nan)
        stems[:, :2] = np.stack([np.c_[g, score, panel["under"]],
                                 np.c_[g, score, panel["true"]]], 1)
        stems = stems.reshape(-1, 3)
        fig.add_trace(go.Scatter3d(
            x=stems[:, 0], y=stems[:, 1], z=stems[:, 2], mode="lines", hoverinfo="skip",
            line=dict(color=INK_SOFT, width=1.5), opacity=0.5, showlegend=False,
        ), 1, column)
        true_text = "log y %{z:.3f}" if column == 1 else "y %{z:.4g}"
        fig.add_trace(go.Scatter3d(
            x=g, y=score, z=panel["true"], mode="markers", showlegend=False,
            customdata=np.c_[customdata, panel["under"]],
            marker=dict(size=4, color=np.log10(decay), coloraxis="coloraxis",
                        line=dict(color=INK, width=0.5)),
            hovertemplate=("test graph %{customdata[0]:.0f}<br>global %{x:.3f}<br>"
                           f"PC1 %{{y:.3f}}<br>true {true_text}<br>"
                           "surface under it %{customdata[2]:.4g}<extra></extra>"),
        ), 1, column)
        fig.update_scenes(
            dict(xaxis_title=pdp["xlabel"], yaxis_title=pdp["ylabel"],
                 zaxis_title=panel["zlabel"], aspectratio=dict(x=1, y=1, z=0.75),
                 camera=dict(eye=dict(zip("xyz", eye))),
                 **{f"{a}axis": dict(backgroundcolor=SURFACE, gridcolor=GRID) for a in "xyz"}),
            row=1, col=column,
        )
    fig.update_layout(
        title=dict(text=pdp["title"], x=0.005, font=dict(size=18, color=INK)),
        coloraxis=dict(
            colorscale=[[t, to_hex(PROPERTY(t))] for t in np.linspace(0, 1, 11)],
            cmin=low, cmax=high,
            colorbar=dict(title=dict(text="decay exponent", side="right"),
                          tickvals=np.log10(ticks), ticktext=[str(v) for v in ticks]),
        ),
        height=800, margin=dict(l=10, r=10, t=110, b=40),
    )
    write_html(fig, metrics, out)


def write_html(fig, metrics, out):
    """The plotly counterpart of `footer` then `savefig`: house colours, provenance, one file."""
    fig.add_annotation(text=provenance(metrics), x=0.0, y=0.0, xref="paper", yref="paper",
                       xanchor="left", yanchor="top", yshift=-18, showarrow=False,
                       font=dict(size=10, color=INK_SOFT))
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(color=INK_SOFT))
    fig.write_html(out, include_plotlyjs=True)


def traversal(model, data, scaler, metrics, device, axis, out, spatial=False):
    """Decode along one latent axis from real encodings: test graphs by steps.

    Each row starts from one test graph's posterior mean — the same graphs
    `reconstruction.png` shows — and moves it by t·σ along the axis, holding
    everything else at the encoding, so the centre column is that graph's
    reconstruction. σ is the spread of the axis coordinate over the test
    graphs, a between-graph unit, and the sign is set so +t raises log y
    across them; an eigenvector or a latent dimension has no sign of its own.

    ``gbd`` is the highest-rate global dimension. ``PC1`` and ``PC2`` are the
    sorted-160 components, the relabelling-invariant PCA of `latent_property`.
    A step there moves each graph's sorted values, which are then put back on
    the nodes they came from — rank r of dimension d returns to the node that
    held rank r. Past a few σ values can cross ranks; the assignment is still
    by the original ranks, which is the only inverse the sort has.

    ``spatial`` draws each decoded matrix as a landscape, aligned outward from
    the reconstruction in the centre column.
    """
    mu = data["mu"].numpy()
    mu_global = data["mu_global"].numpy()
    log_y = scaler.inverse(data["y"].numpy())
    n_graphs, n_nodes, d_latent = mu.shape
    rows = min(N_EXAMPLES, n_graphs)
    n_steps = len(TRAVERSAL_STEPS)

    def oriented(scores):
        """Spread of the coordinate, and the sign that makes +t raise log y."""
        r = np.corrcoef(scores, log_y)[0, 1] if scores.std() > 0 and log_y.std() > 0 else 0.0
        return scores.std(), -1.0 if r < 0 else 1.0

    if axis == "gbd":
        d_global = mu_global.shape[1]
        if not d_global:
            return placeholder(metrics, out, "Global traversal — this run has no global latent")
        global_kl = np.asarray(
            metrics.get("global_kl_per_dim") or np.zeros(d_global), dtype=float
        )
        top = int(np.argmax(global_kl))
        sigma, sign = oriented(mu_global[:, top])

        def latents(graph):
            z = np.repeat(mu[graph][None], n_steps, 0)
            z_global = np.repeat(mu_global[graph][None], n_steps, 0)
            z_global[:, top] += sign * sigma * TRAVERSAL_STEPS
            return z, z_global

        title = (f"Global dim {top} traversal: z_g = μ_g + t·σ — "
                 f"{global_kl[top]:.2f} nats, σ = {sigma:.3f}")
    else:
        k = int(axis.removeprefix("PC")) - 1
        n_sorted = n_nodes * d_latent
        if n_graphs <= k + 1:
            return placeholder(metrics, out, f"Sorted-{n_sorted} {axis} traversal — "
                                             f"needs {k + 2}+ test graphs")
        descriptor = sorted_descriptor(mu)
        mean, singular, right = principal_axes(descriptor)
        sigma, sign = oriented((descriptor - mean) @ right[k])
        direction = sign * right[k]
        share = singular[k] ** 2 / max((singular ** 2).sum(), LOG_FLOOR)
        order = np.argsort(mu, axis=1)

        def latents(graph):
            moved = descriptor[graph] + np.outer(sigma * TRAVERSAL_STEPS, direction)
            z = unsort(moved, np.broadcast_to(order[graph], (n_steps, n_nodes, d_latent)))
            return z, np.repeat(mu_global[graph][None], n_steps, 0)

        title = (f"Sorted-{n_sorted} {axis} traversal: z = μ + t·σ·v — "
                 f"σ = {sigma:.3f}, {share:.1%} of latent variance")

    decoded, predicted = zip(*(
        decode_batch(model, device, scaler, *latents(graph)) for graph in range(rows)
    ))
    true = np.exp(log_y[:rows])
    centre = n_steps // 2
    titles = []
    for row in range(rows):
        titles.append([])
        for column, t in enumerate(TRAVERSAL_STEPS):
            value = f"ŷ {predicted[row][column]:.0f}"
            if column == centre:
                value += f"  ·  true {true[row]:.0f}"
            header = "t = 0 (recon)" if t == 0 else f"t = {t:+d}σ"
            titles[row].append(f"{header}\n{value}" if row == 0 else value)

    labels = [f"test example {row}" for row in range(rows)]
    if spatial:
        draw_spatial_grid(decoded, titles, labels, centre, title, metrics, out)
    else:
        draw_matrix_grid(decoded, titles, labels, reconstruction_floor(data, rows),
                         title, metrics, out)


def align_nodes(mu_a, mu_b):
    """The relabelling of B's nodes that puts them nearest A's, as an index array.

    ``mu_b[align_nodes(mu_a, mu_b)]`` is B with node i matched to A's node i,
    minimising the summed squared distance between paired node latents. The
    node latent is a set, so slot i of two graphs are unrelated nodes until
    this pairs them; anything combining two latents needs it first.
    """
    cost = ((mu_a[:, None, :] - mu_b[None, :, :]) ** 2).sum(-1)
    rows, columns = linear_sum_assignment(cost)
    return columns[np.argsort(rows)]


def interpolation(model, data, scaler, metrics, device, out, spatial=False):
    """Straight lines in latent space between test graphs, after node matching.

    Row i runs from test example i to example i + 1, wrapping, so the three
    rows are 0 → 1, 1 → 2 and 2 → 0 — the graphs `reconstruction.png` shows.
    B's node latents are first relabelled onto A's by `align_nodes`; the global
    is graph-level and interpolates as it is. Everything is drawn in A's node
    order, so the right end is B's reconstruction permuted by the match, which
    the decoder's equivariance makes exact.

    ``spatial`` draws each decoded matrix as a landscape, aligned outward from
    A's reconstruction on the left.
    """
    mu = data["mu"].numpy()
    mu_global = data["mu_global"].numpy()
    log_y = scaler.inverse(data["y"].numpy())
    rows = min(N_EXAMPLES, len(mu))
    if rows < 2:
        return placeholder(metrics, out, "Interpolation — needs 2+ test graphs")
    alpha = np.linspace(0.0, 1.0, INTERPOLATION_STEPS)[:, None, None]
    true = np.exp(log_y[:rows])

    decoded, titles, labels = [], [], []
    for row in range(rows):
        a, b = row, (row + 1) % rows
        matched = mu[b][align_nodes(mu[a], mu[b])]
        z = (1 - alpha) * mu[a] + alpha * matched
        z_global = (1 - alpha[:, 0]) * mu_global[a] + alpha[:, 0] * mu_global[b]
        log_T_hat, predicted = decode_batch(model, device, scaler, z, z_global)
        decoded.append(log_T_hat)

        # Mean distance between paired node latents, as labelled and as matched.
        raw = np.linalg.norm(mu[a] - mu[b], axis=-1).mean()
        aligned = np.linalg.norm(mu[a] - matched, axis=-1).mean()
        labels.append(f"test {a} → {b}\nnode dist {raw:.2f} → {aligned:.2f}")

        titles.append([])
        for column, step in enumerate(alpha[:, 0, 0]):
            value = f"ŷ {predicted[column]:.0f}"
            if column in (0, INTERPOLATION_STEPS - 1):
                value += f"  ·  true {true[a if column == 0 else b]:.0f}"
            # The row label names the pair; a header naming it would fit row 0 only.
            header = (f"α = {step:.0f}" if column in (0, INTERPOLATION_STEPS - 1)
                      else f"α = {column}/{INTERPOLATION_STEPS - 1}")
            titles[row].append(f"{header}\n{value}" if row == 0 else value)

    title = ("Interpolation: z = (1 − α)·μ_A + α·P·μ_B, P the node match; "
             "global z_g interpolated unmatched")
    if spatial:
        draw_spatial_grid(decoded, titles, labels, 0, title, metrics, out)
    else:
        draw_matrix_grid(decoded, titles, labels, reconstruction_floor(data, rows),
                         title, metrics, out)


def graph_views(mu, mu_global, log_y, metrics):
    """The graph-level panels of `latent_property`, as projections an iterate can use.

    Each view is ``(title, xlabel, ylabel, project)``, where ``project`` takes
    latents ``(k, n_nodes, d_latent)`` and ``(k, d_global)`` to ``(k, 2)``. The
    axes, PCA means and components are fitted on the test encodings once and
    then applied unchanged, so a latent that is not a graph's encoding lands
    exactly where a graph with that encoding would.

    ``latent_property``'s "sorted + global, unweighted" panel is left out. The
    global there is one column beside n_nodes·d_latent sorted values and barely
    loads on the top two components, so the panel repeats the sorted PCA and
    hides the global, which is where the ascent mostly moves.
    """
    n_graphs, n_nodes, d_latent = mu.shape
    d_global = mu_global.shape[1]
    ranked = latent_axes(mu, log_y, metrics, d_global)
    correlation, kl_per_dim = ranked["correlation"], ranked["kl_per_dim"]
    n_sorted = n_nodes * d_latent

    def two(order):
        """The first two dimensions, padded when fewer are live."""
        order = list(order)
        return (order + [d for d in range(d_latent) if d not in order])[:2]

    def label(dim, basis):
        if basis == "rate":
            return f"graph mean, dim {dim} — {kl_per_dim[dim]:.2f} nats"
        return f"graph mean, dim {dim} — r {correlation[dim]:+.2f}"

    def means(title, order, basis):
        """A graph-mean view on the first two dimensions of `order`.

        At d_latent = 1 there is no second dimension to pad with, so y is held
        at zero and the path runs along a line. `latent_property` jitters
        instead, which works for one scatter but not here: the projection is
        applied to the background and to each iterate separately, so jitter
        would scatter the path.
        """
        dims = two(order)
        if len(dims) == 2:
            return (title, label(dims[0], basis), label(dims[1], basis),
                    lambda z, z_global: z.mean(1)[:, dims])
        return (title, label(dims[0], basis), "held at 0 — one latent dimension",
                lambda z, z_global: np.c_[z.mean(1)[:, dims[0]], np.zeros(len(z))])

    views = [
        means("Graph means, highest-rate axes", ranked["by_rate"], "rate"),
        means("Graph means, property axes", ranked["by_property"], "property"),
    ]

    if n_graphs > 2:
        mean, singular, right = principal_axes(sorted_descriptor(mu))
        captured = (singular[:2] ** 2).sum() / max((singular ** 2).sum(), LOG_FLOOR)

        def sorted_pcs(z, z_global):
            return (sorted_descriptor(z) - mean) @ right[:2].T
    else:
        sorted_pcs = None

    if d_global:
        top = ranked["top_global"]
        g_label = f"global dim {top} — {ranked['global_kl'][top]:.2f} nats"
        for dim, basis, criterion in ((ranked["by_rate"][0], "rate", "highest-rate"),
                                      (ranked["by_property"][0], "property", "top property")):
            views.append((f"Global against the {criterion} node dim", g_label, label(dim, basis),
                          lambda z, z_global, dim=dim: np.c_[z_global[:, top], z.mean(1)[:, dim]]))
        if sorted_pcs is not None:
            views.append((f"Global against sorted-{n_sorted} PC1", g_label,
                          f"sorted-{n_sorted} PC1",
                          lambda z, z_global: np.c_[z_global[:, top], sorted_pcs(z, z_global)[:, 0]]))

    if sorted_pcs is not None:
        views.append((f"All {n_sorted}, sorted invariant — top 2 hold {captured:.1%}",
                      f"sorted-{n_sorted} PC1", f"sorted-{n_sorted} PC2", sorted_pcs))

    if d_global and n_graphs > 2:
        weight = np.sqrt(n_nodes)
        global_mean = mu_global.mean(0)

        def joint(z, z_global):
            return np.c_[sorted_descriptor(z), weight * (z_global - global_mean)]

        joint_mean, _, joint_right = principal_axes(joint(mu, mu_global))
        n_joint = n_sorted + d_global
        views.append((f"Sorted + global ×√{n_nodes}", f"sorted-{n_joint} PC1",
                      f"sorted-{n_joint} PC2",
                      lambda z, z_global: (joint(z, z_global) - joint_mean) @ joint_right[:2].T))
    return views


def path_columns(log_y_hat, n_columns):
    """Iterates to draw: the start, the end, and those nearest evenly spaced log ŷ.

    Even in log ŷ rather than in iteration because BFGS gains most of its ground
    in a few steps; evenly spaced iterations would be mostly the same matrix.
    It serves a descent as well, the targets then running downhill. The
    path need not be monotone in ŷ once the penalty is on, so this is
    nearest-value, not interpolation. Repeats collapse, and the columns they
    free go to the unused iterates, evenly in iteration.
    """
    last = len(log_y_hat) - 1
    targets = np.linspace(log_y_hat[0], log_y_hat[-1], n_columns)[1:-1]
    chosen = {0, last, *(int(np.argmin(np.abs(log_y_hat - t))) for t in targets)}
    unused = np.setdiff1d(np.arange(last + 1), sorted(chosen))
    spare = min(n_columns - len(chosen), len(unused))
    if spare > 0:
        chosen |= set(unused[np.linspace(0, len(unused) - 1, spare).round().astype(int)].tolist())
    return sorted(chosen)


def decay_text(log_y):
    """A decay exponent that may have run off to 10^±1000 without becoming inf or 0."""
    if np.log(1.0) <= log_y < np.log(1e6):
        return f"{np.exp(log_y):.0f}"
    if np.log(1e-3) <= log_y < np.log(1.0):
        return f"{np.exp(log_y):.3g}"
    # Past 10^±10⁴ the exponent itself needs significant figures, not decimals.
    exponent = log_y / np.log(10)
    return f"10^{exponent:.1f}" if abs(exponent) < 1e4 else f"10^({exponent:.3g})"


def optimization(model, data, scaler, metrics, device, graph, lam, sense, out):
    """BFGS from one test graph: its decoded path, then the path in latent space.

    The top row is decoded matrices at selected iterates, drawn like the
    traversals. Below are the graph-level views of `latent_property` with the
    100 test graphs faded behind the path: every accepted iterate, the start
    square, the end a star, arrows in iteration order. The background is
    coloured by the true exponent and the path by the predicted one, on one
    scale fitted to the data, so a path that leaves the data's range clips at
    the top colour, or the bottom one when minimizing, and its final value is
    written on the panel instead. ``sense`` is +1 to ascend, −1 to descend.
    """
    mu = data["mu"].numpy()
    mu_global = data["mu_global"].numpy()
    log_y = scaler.inverse(data["y"].numpy())
    if graph >= len(mu):
        return placeholder(metrics, out, f"Optimization — no test example {graph}")
    decay = np.exp(log_y)

    path = optimize.search(model, scaler, mu[graph], mu_global[graph], lam, sense)
    columns = path_columns(path.log_y_hat, N_OPTIMIZATION_COLUMNS)
    decoded, _ = decode_batch(model, device, scaler, path.z[columns], path.z_global[columns])
    views = graph_views(mu, mu_global, log_y, metrics)

    n_rows = int(np.ceil((len(views) + 1) / 4))
    height = 2.6 + 3.9 * n_rows + 1.6
    fig = plt.figure(figsize=(17, height))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.6, 3.9 * n_rows],
                             left=0.045, right=0.955, top=1 - 0.8 / height,
                             bottom=0.75 / height, hspace=0.9 / (2.6 + 3.9 * n_rows) * 2)

    top = outer[0].subgridspec(1, N_OPTIMIZATION_COLUMNS + 1,
                               width_ratios=[1] * N_OPTIMIZATION_COLUMNS + [0.05], wspace=0.12)
    floor = reconstruction_floor(data, N_EXAMPLES)
    for column, (k, matrix) in enumerate(zip(columns, decoded)):
        ax = fig.add_subplot(top[column])
        image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=floor, vmax=0.0, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        head = "start (recon)" if k == 0 else ("final" if k == columns[-1] else "")
        text = f"iter {k}" + (f" — {head}" if head else "")
        text += f"\nŷ {decay_text(path.log_y_hat[k])}"
        if k == 0:
            text += f"  ·  true {decay[graph]:.0f}"
        ax.set_title(text, fontsize=9)
    bar = fig.colorbar(image, cax=fig.add_subplot(top[N_OPTIMIZATION_COLUMNS]))
    bar.set_label("log₁₀ T̂", fontsize=8, color=INK_SOFT)
    bar.ax.tick_params(labelsize=7)

    lower = outer[1].subgridspec(n_rows, 5, width_ratios=[1, 1, 1, 1, 0.05],
                                 wspace=0.32, hspace=0.42)
    norm = plt.Normalize(decay.min(), decay.max())
    # Colour only; the text on each panel carries the real value past the clip.
    path_colour = np.exp(np.clip(path.log_y_hat, np.log(decay.min()) - 1.0,
                                 np.log(decay.max()) + 1.0))
    for index, (title, xlabel, ylabel, project) in enumerate(views):
        ax = fig.add_subplot(lower[index // 4, index % 4])
        background = project(mu, mu_global)
        ax.scatter(background[:, 0], background[:, 1], c=decay, cmap=PROPERTY,
                   norm=norm, s=30, alpha=0.4, linewidth=0)
        points = project(path.z, path.z_global)
        for k in range(len(points) - 1):
            if not np.allclose(points[k], points[k + 1]):
                ax.annotate("", xy=points[k + 1], xytext=points[k],
                            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8,
                                            mutation_scale=8, shrinkA=2, shrinkB=2))
        for sl, marker, size in ((slice(1, -1), "o", 24), (slice(0, 1), "s", 70),
                                 (slice(-1, None), "*", 190)):
            ax.scatter(points[sl, 0], points[sl, 1], c=path_colour[sl], cmap=PROPERTY,
                       norm=norm, marker=marker, s=size, edgecolor=INK, linewidth=0.8,
                       zorder=3)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # The spare panel: how the run ended, and what the markers are.
    ax = fig.add_subplot(lower[len(views) // 4, len(views) % 4])
    ax.set_axis_off()
    moved = np.linalg.norm(
        np.c_[path.z.reshape(len(path.z), -1), path.z_global][-1]
        - np.r_[path.z[0].ravel(), path.z_global[0]])
    ax.text(0.0, 1.0,
            f"test example {graph}, true decay exponent {decay[graph]:.0f}\n"
            f"ŷ {decay_text(path.log_y_hat[0])} → {decay_text(path.log_y_hat[-1])}\n"
            f"‖x − x₀‖ = {moved:.3g} after {path.nit} iterations,\n"
            f"{path.nfev} evaluations\n"
            + textwrap.fill(f"exit: {path.message}", 44),
            va="top", fontsize=8.5, color=INK, transform=ax.transAxes)
    for marker, size, name in (("s", 70, "start, the graph's encoding"),
                               ("o", 24, "BFGS iterate"), ("*", 190, "final iterate")):
        ax.scatter([], [], marker=marker, s=size, color=SURFACE, edgecolor=INK,
                   linewidth=0.8, label=name)
    ax.scatter([], [], s=30, color=PROPERTY_SOLID, alpha=0.4, linewidth=0,
               label="test graphs, true exponent")
    ax.legend(loc="lower left", fontsize=8)

    # A mappable of its own, so the bar shows the colours at full strength
    # rather than at the background's transparency.
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=PROPERTY),
                       cax=fig.add_subplot(lower[:, 4]), extend="max" if sense > 0 else "min")
    bar.set_label("decay exponent — true for test graphs, predicted on the path",
                  fontsize=8, color=INK_SOFT)
    bar.ax.tick_params(labelsize=7)

    sign, direction = ("−", "ascent") if sense > 0 else ("+", "descent")
    objective = (f"f(x) = {sign}log ŷ(x), unregularized" if lam == 0
                 else f"f(x) = {sign}log ŷ(x) + (λ/2)·‖x − x₀‖², λ = {lam:g}")
    fig.suptitle(f"BFGS latent {direction} from test example {graph}: {objective}",
                 fontsize=12, color=INK, x=0.005, ha="left")
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def decode_batch(model, device, scaler, z, z_global):
    """Latents to ``(log10 T̂, predicted decay exponent)``, one entry per step."""
    with torch.no_grad():
        z, z_global = (torch.as_tensor(a, dtype=torch.float32, device=device)
                       for a in (z, z_global))
        T_hat, _ = model.decode(z, z_global)
        log_y_hat = model.predict(z, z_global).cpu().numpy()
    return log_matrix(T_hat.cpu().numpy()), np.exp(scaler.inverse(log_y_hat))


def reconstruction_floor(data, rows):
    """The reconstruction figure's colour floor, so decoded grids read against it."""
    return float(np.nanpercentile(log_matrix(data["T"][:rows].numpy()), 2))


def draw_matrix_grid(decoded, titles, row_labels, floor, title, metrics, out):
    """Rows of decoded log10 matrices on one shared scale, titled cell by cell."""
    rows, columns = len(decoded), len(decoded[0])
    height = 2.45 * rows + 1.1
    fig = plt.figure(figsize=(16, height))
    grid = fig.add_gridspec(
        rows, columns + 1, width_ratios=[1] * columns + [0.05],
        left=0.045, right=0.95, top=1 - 0.95 / height, bottom=0.3 / height,
        wspace=0.12, hspace=0.32,
    )
    for row in range(rows):
        for column in range(columns):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(decoded[row][column], cmap=SEQUENTIAL, vmin=floor, vmax=0.0,
                              interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(titles[row][column], fontsize=9)
            if column == 0:
                ax.set_ylabel(row_labels[row], fontsize=8)
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, columns]))
    colorbar.set_label("log₁₀ T̂", fontsize=8, color=INK_SOFT)
    colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{title}   |   ŷ is the predicted decay exponent", fontsize=12,
                 color=INK, x=0.005, ha="left")
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def draw_spatial_grid(decoded, titles, row_labels, anchor, title, metrics, out):
    """Rows of decoded matrices as spatial graphs, each row on shared axes.

    A row keeps its node labels from column to column, so each layout is
    aligned by Procrustes onto its already-aligned neighbour, working outward
    from the ``anchor`` column. Chaining keeps a smooth path from flipping where
    a far column would match the anchor poorly. Edge widths share one scale per
    row, set by its heaviest entry.
    """
    rows, columns = len(decoded), len(decoded[0])
    height = 2.5 * rows + 1.1
    fig = plt.figure(figsize=(18, height))
    grid = fig.add_gridspec(rows, columns, left=0.045, right=0.99,
                            top=1 - 0.95 / height, bottom=0.3 / height,
                            wspace=0.08, hspace=0.32)
    for row in range(rows):
        matrices = [np.nan_to_num(10.0 ** m.astype(np.float64)) for m in decoded[row]]
        layouts = [landscape.positions(T)[0] for T in matrices]
        for column in list(range(anchor + 1, columns)) + list(range(anchor - 1, -1, -1)):
            neighbour = column - 1 if column > anchor else column + 1
            layouts[column] = landscape.align(layouts[column], layouts[neighbour], scale=False)
        bounds = landscape.limits(*layouts)
        heaviest = max(T.max() for T in matrices)
        for column in range(columns):
            ax = fig.add_subplot(grid[row, column])
            landscape.draw(ax, matrices[column], layouts[column], heaviest, bounds,
                           width_max=2.2, node_size=45, label_size=4.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(titles[row][column], fontsize=9)
            if column == 0:
                ax.set_ylabel(row_labels[row], fontsize=8)

    fig.suptitle(f"{title}   |   ŷ is the predicted decay exponent, edge width ∝ T̂_ij",
                 fontsize=12, color=INK, x=0.005, ha="left")
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def placeholder(metrics, out, message):
    """A figure that says why there is nothing to draw, so every run has the file."""
    fig = plt.figure(figsize=(8, 2.4))
    fig.text(0.5, 0.55, message, ha="center", va="center", fontsize=11, color=INK_SOFT)
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
    # The partial-dependence grid is a million predictions; the PNG and the
    # page share one computation, whichever asks first.
    pdp = functools.cache(lambda: pdp_surface(model, data, scaler, metrics, device))

    figures = run_dir / paths.FIGURES
    traversals = figures / paths.TRAVERSALS
    optimizations = figures / paths.OPTIMIZATION
    traversals.mkdir(parents=True, exist_ok=True)
    optimizations.mkdir(exist_ok=True)
    diagnostics = {
        "training_curve": lambda p: training_curve(history, metrics, p),
        "predictions": lambda p: predictions(data, scaler, metrics, p),
        "reconstruction": lambda p: reconstruction(data, metrics, p),
        "spatial_graph": lambda p: spatial_graph(data, metrics, p),
        "latent": lambda p: latent(metrics, p),
        "latent_property": lambda p: latent_property(data, scaler, metrics, p),
        "latent_property_3D": lambda p: latent_property_3D(pdp(), metrics, p),
    }
    interactive = {
        "latent_property_3D": lambda p: latent_property_3D_html(pdp(), metrics, p),
    }
    jobs = [(figures / f"{name}.png", diagnostics[name]) for name in paths.DIAGNOSTIC_FIGURES]
    jobs += [(figures / f"{name}.html", interactive[name])
             for name in paths.INTERACTIVE_FIGURES]
    jobs += [
        (traversals / f"traversal_{axis}{view}.png",
         lambda p, a=axis, v=view: traversal(model, data, scaler, metrics, device, a, p,
                                             spatial=v == paths.SPATIAL))
        for axis in paths.TRAVERSAL_AXES for view in paths.PATH_VIEWS
    ]
    interpolations = {
        "interpolation": lambda p, spatial: interpolation(model, data, scaler, metrics,
                                                          device, p, spatial),
    }
    jobs += [(traversals / f"{name}{view}.png",
              lambda p, n=name, v=view: interpolations[n](p, v == paths.SPATIAL))
             for name in paths.INTERPOLATIONS for view in paths.PATH_VIEWS]
    jobs += [
        (optimizations / f"{sense}_g{graph}{variant}.png",
         lambda p, g=graph, lam=OPTIMIZATION_LAMBDA[variant], s=OPTIMIZATION_SENSE[sense]:
             optimization(model, data, scaler, metrics, device, g, lam, s, p))
        for sense in paths.OPTIMIZATION_SENSES
        for graph in paths.OPTIMIZATION_GRAPHS for variant in paths.OPTIMIZATION_VARIANTS
    ]

    written = []
    for path, draw in jobs:
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
