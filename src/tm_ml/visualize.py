"""Diagnostic figures for a trained run.

    pixi run -e ml visualize --run TMVAE_b0p15-bw20-e300-lff0p05-schcosine_Tom1000

PNGs into ``figures/`` in the run directory, each stamped with a provenance
footer — run, checkpoint epoch and score, drop, split seed, git hash, timestamp —
so a figure is self-describing wherever it ends up:

- ``training_curve.png``  the loss terms and the three schedules, per epoch
- ``predictions.png``     the property, predicted against true, on the test split
- ``reconstruction.png``  example matrices, and error against the size of the
                          true entry, which is where forward KL is blind
- ``latent.png``          how many latent dimensions are actually carrying code
- ``latent_property.png`` where graphs sit in the latent, coloured by the
                          property, at node and graph level and in two bases

and into ``figures/interpolation_and_traversals/``:

- ``traversal_gbd.png``   decoded matrices stepping along the global dimension
- ``traversal_PC1.png``   the same along sorted-160 PC1
- ``traversal_PC2.png``   and PC2

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
# Traversal steps, in units of the axis's between-graph σ.
TRAVERSAL_STEPS = np.arange(-3, 4)
# Below this rate a dimension is prior noise, not code.
ACTIVE_DIM_MIN_KL = 0.01
LOG_FLOOR = 1e-12


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


def node_features(T):
    """Interpretable per-node summaries, each equivariant under relabelling.

    Every one reduces over the *other* index, so it is a property of the node
    and not of the labelling — the condition `permutationsandlatent.md` sets out
    for a legitimate node feature. These are what make the latent axes readable:
    a dimension is named by what it correlates with.
    """
    n = T.shape[-1]
    off = off_diagonal(n, torch.device("cpu")).numpy()
    log_T = np.log(np.clip(T, LOG_FLOOR, None))

    column = T / np.clip(T.sum(1, keepdims=True), LOG_FLOOR, None)
    # The diagonal is structurally zero and would win every minimum.
    stationary = np.empty(T.shape[:2])
    for graph, matrix in enumerate(T):
        values, vectors = np.linalg.eig(matrix.T)
        leading = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
        stationary[graph] = leading / leading.sum()

    return {
        "inflow": T.sum(1),
        "out entropy": -(T * log_T).sum(2),
        "in entropy": -(column * np.log(np.clip(column, LOG_FLOOR, None))).sum(1),
        "max outflow": T.max(2),
        "min log outflow": np.where(off, log_T, np.inf).min(2),
        "log stationary π": np.log(np.clip(stationary, LOG_FLOOR, None)),
    }


def latent_property(data, scaler, metrics, out):
    """Where graphs land in the latent, and whether that place tracks the property.

    Two rows, because the latent is per-node while the property is per-graph.
    The top row is every node latent, 20 per graph sharing one colour; the
    bottom is each graph's mean over its 20 nodes, the invariant summary the
    property actually rides on. Two bases, because rate and property-relevance
    are different things — the highest-rate dimension need not be the one the
    target moves along, and on the b0.01 run it is not.

    A run with a global latent gets two more rows. The third puts the global
    dimension against the best node-side axis by each criterion — rate,
    property, sorted PC1 — all as graph means, one point per graph like the
    global itself. The fourth is the sorted PCA with the global appended, both
    raw and weighted by sqrt(n_nodes). Raw, it is one column beside 160 and
    barely moves the components; weighted, it counts as much as one node
    dimension, which fills n_nodes sorted slots. Neither weight is the right
    one, so both are drawn and the global's loading is written on each.
    """
    mu = data["mu"].numpy()
    log_y = scaler.inverse(data["y"].numpy())
    # Colour in the units the property is quoted in; the axis ranking below
    # stays on log y, which is what the head regresses and what R² is measured
    # against, so the two are not the same scale by design.
    decay = np.exp(log_y)
    n_graphs, n_nodes, d_latent = mu.shape
    flat = mu.reshape(-1, d_latent)
    graph_mean = mu.mean(1)
    rng = np.random.default_rng(0)

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

    # Law of total variance over the node axis. The share is small and the
    # information is not: this is the 3% the property rides on.
    between = graph_mean.var(0)[live].sum()
    within = mu.var(1).mean(0)[live].sum()
    between_fraction = between / (between + within)

    mu_global = data["mu_global"].numpy()
    d_global = mu_global.shape[1]
    n_rows = 4 if d_global else 2
    features = node_features(data["T"].numpy())

    # Margins held in inches, so the extra rows add height instead of squeezing.
    height = 4.4 * n_rows
    fig = plt.figure(figsize=(16.5, height))
    grid = fig.add_gridspec(
        n_rows, 4, width_ratios=[1, 1, 1, 0.038],
        left=0.05, right=0.935, top=1 - 0.92 / height, bottom=0.88 / height,
        wspace=0.30, hspace=0.42,
    )
    axes = [[fig.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(n_rows)]
    bar = fig.add_subplot(grid[:, 3])

    # The per-dimension number belongs on the axis it describes, not in the
    # title: four panels of it across one row runs the titles into each other.
    def label(dim, basis):
        if basis == "rate":
            return f"latent dim {dim} — {kl_per_dim[dim]:.2f} nats"
        return f"latent dim {dim} — r {correlation[dim]:+.2f}"

    def scatter(ax, points, dims, basis, colour, small):
        """One panel of points in two latent coordinates, coloured by the property."""
        x = points[:, dims[0]]
        if len(dims) >= 2:
            y, ylabel = points[:, dims[1]], label(dims[1], basis)
        else:
            # d_latent = 1, or a collapsed run with a single surviving unit.
            y, ylabel = rng.uniform(-1, 1, len(x)), "jitter — one live dimension"
        style = dict(s=7, alpha=0.55, linewidth=0) if small else dict(
            s=34, alpha=0.95, edgecolor=SURFACE, linewidth=0.5)
        ax.set_xlabel(label(dims[0], basis))
        ax.set_ylabel(ylabel)
        return ax.scatter(x, y, c=colour, cmap=SEQUENTIAL, **style)

    def graph_scatter(ax, x, y, xlabel, ylabel, title):
        ax.scatter(x, y, c=decay, cmap=SEQUENTIAL, s=34, alpha=0.95,
                   edgecolor=SURFACE, linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    node_colour = np.repeat(decay, n_nodes)
    property_title = (
        "Node latents, property axes" if enough
        else "Node latents — too few graphs to rank by property"
    )

    handle = scatter(axes[0][0], flat, by_rate[:2], "rate", node_colour, True)
    axes[0][0].set_title("Node latents, highest-rate axes")
    scatter(axes[0][1], flat, by_property[:2], "property", node_colour, True)
    axes[0][1].set_title(property_title)
    for ax in axes[0][:2]:
        ax.annotate(
            f"{between_fraction:.1%} of this spread is between graphs;\n"
            f"the rest is nodes differing within one",
            xy=(0.03, 0.97), xycoords="axes fraction", va="top",
            fontsize=7.5, color=INK_SOFT,
        )

    scatter(axes[1][0], graph_mean, by_rate[:2], "rate", decay, False)
    axes[1][0].set_title("Graph means, highest-rate axes")
    scatter(axes[1][1], graph_mean, by_property[:2], "property", decay, False)
    axes[1][1].set_title(
        "Graph means, property axes" if enough
        else "Graph means — too few graphs to rank by property"
    )

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

    sorted_axis = axes[3][0] if d_global else axes[1][2]
    coordinates, _ = pca_panel(sorted_axis, descriptor, f"sorted-{n_sorted}",
                               f"All {n_sorted}, sorted invariant")

    # Whether a dimension means the same thing at every node. It does, and not
    # by luck: the encoder applies one shared pointwise readout to every node,
    # with no node-indexed parameters anywhere, so dim d cannot mean one thing
    # at slot 3 and another at slot 17.
    ax = axes[0][2]
    table = np.array([
        [np.corrcoef(flat[:, d], values.ravel())[0, 1] for d in live]
        for values in features.values()
    ])
    image = ax.imshow(table, cmap=DIVERGING, vmin=-1.0, vmax=1.0,
                      aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(live)), [f"dim {d}" for d in live], fontsize=7.5)
    ax.set_yticks(range(len(features)), list(features), fontsize=7.5)
    ax.grid(False)
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            ax.annotate(f"{table[row, column]:+.2f}", xy=(column, row), ha="center",
                        va="center", fontsize=7, color=INK)
    ax.set_title("What each dimension means — same readout at every node")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    if d_global:
        global_kl = np.asarray(
            metrics.get("global_kl_per_dim") or np.zeros(d_global), dtype=float
        )
        # The highest-rate global dimension. On a collapsed run it is still drawn,
        # and its rate on the axis says it is noise.
        top = int(np.argmax(global_kl))
        g = mu_global[:, top]
        g_r = np.corrcoef(g, log_y)[0, 1] if enough and g.std() > 0 else np.nan
        g_label = f"global dim {top} — {global_kl[top]:.2f} nats, r {g_r:+.2f}"

        def r_with_g(values):
            return np.corrcoef(g, values)[0, 1] if g.std() > 0 and values.std() > 0 else np.nan

        for ax, dim, basis, criterion in (
            (axes[2][0], by_rate[0], "rate", "highest-rate"),
            (axes[2][1], by_property[0], "property", "top property"),
        ):
            values = graph_mean[:, dim]
            graph_scatter(ax, g, values, g_label, label(dim, basis),
                          f"Global against the {criterion} node dim — r {r_with_g(values):+.2f}")
        ax = axes[2][2]
        if coordinates is not None:
            graph_scatter(ax, g, coordinates[:, 0], g_label, f"sorted-{n_sorted} PC1",
                          f"Global against sorted-{n_sorted} PC1 — "
                          f"r {r_with_g(coordinates[:, 0]):+.2f}")
        else:
            ax.set_axis_off()

        # The graph-level counterpart of the node table above: node features
        # reduced over nodes, plus the relaxation time the spectrum sets.
        ax = axes[1][2]
        T64 = data["T"].numpy().astype(np.float64)
        second = np.sort(np.abs(np.linalg.eigvals(T64)), axis=1)[:, -2]
        graph = {f"mean {k}": v.mean(1) for k, v in features.items()}
        graph.update({f"std {k}": v.std(1) for k, v in features.items()})
        graph["log relaxation time"] = np.log(
            -1.0 / np.log(np.clip(second, LOG_FLOOR, 1 - 1e-9)))
        column = np.array([[r_with_g(v)] for v in graph.values()])
        ax.imshow(column, cmap=DIVERGING, vmin=-1.0, vmax=1.0,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks([0], [f"global dim {top}"], fontsize=7.5)
        ax.set_yticks(range(len(graph)), list(graph), fontsize=7)
        ax.grid(False)
        for row, value in enumerate(column[:, 0]):
            ax.annotate(f"{value:+.2f}", xy=(0, row), ha="center", va="center",
                        fontsize=7, color=INK)
        ax.set_title("What the global dim means — graph features")

        n_joint = n_sorted + d_global
        centred_global = mu_global - mu_global.mean(0)
        for ax, weight, name in (
            (axes[3][1], 1.0, "unweighted"),
            (axes[3][2], np.sqrt(n_nodes), f"global ×√{n_nodes}"),
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

    fig.colorbar(handle, cax=bar).set_label(
        "decay exponent", fontsize=8, color=INK_SOFT)
    bar.tick_params(labelsize=7)
    fig.suptitle("Latent space against the property", fontsize=12, color=INK,
                 x=0.005, ha="left")
    footer(fig, metrics)
    fig.savefig(out)
    plt.close(fig)


def traversal(model, data, scaler, metrics, device, axis, out):
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
            moved = moved.reshape(n_steps, n_nodes, d_latent)
            z = np.empty_like(moved)
            np.put_along_axis(z, np.broadcast_to(order[graph], moved.shape), moved, axis=1)
            return z, np.repeat(mu_global[graph][None], n_steps, 0)

        title = (f"Sorted-{n_sorted} {axis} traversal: z = μ + t·σ·v — "
                 f"σ = {sigma:.3f}, {share:.1%} of latent variance")

    decoded, predicted = [], []
    with torch.no_grad():
        for graph in range(rows):
            z, z_global = (torch.as_tensor(a, dtype=torch.float32, device=device)
                           for a in latents(graph))
            T_hat, _ = model.decode(z, z_global)
            decoded.append(log_matrix(T_hat.cpu().numpy()))
            predicted.append(np.exp(scaler.inverse(model.predict(z, z_global).cpu().numpy())))
    true = np.exp(log_y[:rows])
    # The reconstruction figure's scale, so the centre column reads against it.
    floor = float(np.nanpercentile(log_matrix(data["T"][:rows].numpy()), 2))

    fig = plt.figure(figsize=(16, 2.45 * rows + 1.1))
    grid = fig.add_gridspec(
        rows, n_steps + 1, width_ratios=[1] * n_steps + [0.05],
        left=0.03, right=0.95, top=1 - 0.95 / (2.45 * rows + 1.1),
        bottom=0.3 / (2.45 * rows + 1.1), wspace=0.12, hspace=0.32,
    )
    centre = n_steps // 2
    for row in range(rows):
        for column, t in enumerate(TRAVERSAL_STEPS):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(decoded[row][column], cmap=SEQUENTIAL, vmin=floor, vmax=0.0,
                              interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            value = f"ŷ {predicted[row][column]:.0f}"
            if column == centre:
                value += f"  ·  true {true[row]:.0f}"
            header = "t = 0 (recon)" if t == 0 else f"t = {t:+d}σ"
            ax.set_title(f"{header}\n{value}" if row == 0 else value, fontsize=9)
            if column == 0:
                ax.set_ylabel(f"test example {row}", fontsize=8)
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[:, n_steps]))
    colorbar.set_label("log₁₀ T̂", fontsize=8, color=INK_SOFT)
    colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{title}   |   ŷ is the predicted decay exponent", fontsize=12,
                 color=INK, x=0.005, ha="left")
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

    figures = run_dir / paths.FIGURES
    traversals = figures / paths.TRAVERSALS
    traversals.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "training_curve": lambda p: training_curve(history, metrics, p),
        "predictions": lambda p: predictions(data, scaler, metrics, p),
        "reconstruction": lambda p: reconstruction(data, metrics, p),
        "latent": lambda p: latent(metrics, p),
        "latent_property": lambda p: latent_property(data, scaler, metrics, p),
    }
    jobs = [(figures / f"{name}.png", diagnostics[name]) for name in paths.DIAGNOSTIC_FIGURES]
    jobs += [
        (traversals / f"traversal_{axis}.png",
         lambda p, a=axis: traversal(model, data, scaler, metrics, device, a, p))
        for axis in paths.TRAVERSAL_AXES
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
