"""Score a trained run on its held-out test split; write ``metrics.json``.

    pixi run -e ml evaluate --run TMVAE_b0p15-bw20-e300-lff0p05-schcosine_Tom1000

Reads the run directory and nothing else: the checkpoint carries the config, so
the split is reconstructed from the seed it was trained with and the test
examples are the ones the model never saw. Everything runs with ``sample=False``
so a re-evaluation of the same checkpoint reproduces the same numbers.

Four groups of metrics:

- **the property**, which is what the project is for — R^2 and RMSE on log y,
  RMSE and relative error back in decay-exponent units, against a train-fitted
  mean baseline. Only the trivial baseline for now; the strong one is the
  combinatorial edge-editing comparison in `miscSources.md`, and it is a stage
  of its own rather than a line here.
- **the reconstruction**, in KL and in log space, and broken out by the
  magnitude of the true entry. Forward KL weights each term by the true T_ij,
  so it barely sees an entry of 1e-12 missed by nine orders of magnitude; the
  by-magnitude table is where that shows up if it is happening.
- **validity**, the analogue of the sibling's presence metrics: an exactly zero
  diagonal and rows summing to 1 are constraints, not quantities, and an error
  here is a broken model rather than a bad score.
- **the latent**, which decides whether gradient ascent has 160 dimensions to
  work in or a dozen in disguise. This is the block worth reading before
  trusting any optimization result.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from tm_ml import device as device_module
from tm_ml import paths
from tm_ml.datasets import TargetScaler, load_splits
from tm_ml.ingest import git_commit
from tm_ml.models import TMVAE, TMVAEConfig, off_diagonal, tmvae_loss

# A latent dimension counts as used when the spread of its posterior mean across
# examples clears this. Same convention as the sibling, and the same caveat: it
# is a variance across examples, so it needs at least two of them.
ACTIVE_UNIT_THRESHOLD = 0.01

MAGNITUDE_BINS = 10


def load_run(run, device=None):
    """Rebuild the model and its splits from a run directory alone.

    Which card that is comes from the run's own `gpu` unless `device` overrides
    it. A configured card has to reach every stage that opens one, or pinning a
    run away from a card that is full or contended moves only the training and
    leaves the stages after it to pick for themselves. `gpu: auto` is recorded
    as `auto` rather than as whatever it resolved to that day, so the default
    still re-chooses here — the card training landed on may since have filled.

    The checkpoint is read onto the cpu first because the config that names the
    card is inside it; there is nothing to resolve until it has been opened.
    Costs one cpu-side copy of a small state dict, paid once per stage.
    """
    run_dir = paths.resolve(run)
    checkpoint_path = run_dir / paths.CHECKPOINT
    if not checkpoint_path.exists():
        raise SystemExit(f"{checkpoint_path} does not exist; train the run first")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if device is None:
        device = device_module.resolve(cfg["gpu"])

    model = TMVAE(TMVAEConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    splits = load_splits(cfg)
    scaler = TargetScaler(**checkpoint["target_scaler"])
    return run_dir, checkpoint, cfg, model, splits, scaler, device


@torch.no_grad()
def collect(model, dataset, model_cfg, device, batch_size):
    """One deterministic pass, keeping what every metric below is computed from."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    kept = {k: [] for k in ("T", "T_hat", "log_T_hat", "y", "y_hat", "mu", "logvar",
                            "mu_global", "logvar_global")}
    losses = {}
    seen = 0

    for T, y in loader:
        T, y = T.to(device), y.to(device)
        out = model(T, sample=False)
        loss = tmvae_loss(out, T, y, model_cfg)

        for term in ("total", "recon", "kl", "kl_node", "kl_global", "prop", "log_recon"):
            losses[term] = losses.get(term, 0.0) + getattr(loss, term).item() * len(T)
        seen += len(T)

        for name, value in (
            ("T", T), ("T_hat", out.T_hat), ("log_T_hat", out.log_T_hat), ("y", y),
            ("y_hat", out.log_y_hat), ("mu", out.mu), ("logvar", out.logvar),
            ("mu_global", out.mu_global), ("logvar_global", out.logvar_global),
        ):
            kept[name].append(value.cpu())

    collected = {k: torch.cat(v, 0) for k, v in kept.items()}
    collected["losses"] = {k: v / seen for k, v in losses.items()}
    return collected


def fit_line(x, y):
    """Least squares y ~ slope * x + intercept, with the slope's standard error.

    Which way round matters, and the two directions answer different questions.
    Regressing the prediction on the truth gives *resolution*: how much of the
    true spread the model resolves, which is below 1 for any model that is not
    perfect and is not a defect. Regressing the truth on the prediction gives
    *calibration*: whether the truth averages to what was predicted, which is 1
    for an honest conditional mean however weak the model is. Only the second
    one being off is a bug.
    """
    n = len(x)
    centered = x - x.mean()
    sxx = float((centered ** 2).sum())
    if n < 3 or sxx == 0.0:
        return None, None, None

    slope = float((centered * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    residual = y - (slope * x + intercept)
    stderr = float(np.sqrt(float((residual ** 2).sum()) / (n - 2) / sxx))
    return slope, intercept, stderr


def property_metrics(data, scaler):
    """Prediction quality, in the training target and back in real units.

    R^2 is against the test split's own mean, the conventional definition. The
    skill number is against the *training* mean, which is the honest baseline —
    it is what a model with no access to the matrix could have said in advance.
    """
    y, y_hat = data["y"].numpy(), data["y_hat"].numpy()

    residual = y_hat - y
    sse = float((residual ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())

    # The training mean is 0 in the transformed target whenever standardizing is
    # on; inverting makes that explicit rather than relying on it.
    train_mean_transformed = scaler.transform(np.array(scaler.mean))
    baseline = float(np.sqrt(((y - train_mean_transformed) ** 2).mean()))
    rmse = float(np.sqrt((residual ** 2).mean()))

    log_y, log_y_hat = scaler.inverse(y), scaler.inverse(y_hat)
    exponent, exponent_hat = np.exp(log_y), np.exp(log_y_hat)
    relative = np.abs(exponent_hat - exponent) / exponent
    resolution = fit_line(log_y, log_y_hat)
    calibration = fit_line(log_y_hat, log_y)

    return {
        "test_prop": float((residual ** 2).mean()),
        "test_r2": 1.0 - sse / sst if sst else None,
        "test_rmse_transformed": rmse,
        "baseline_rmse_train_mean": baseline,
        "skill_vs_train_mean": 1.0 - rmse / baseline if baseline else None,
        "test_rmse_log": float(np.sqrt(((log_y_hat - log_y) ** 2).mean())),
        "test_mae_log": float(np.abs(log_y_hat - log_y).mean()),
        # Squared error on a log is a lognormal likelihood on the exponent, so
        # relative error is the quantity the loss actually cares about.
        "test_median_relative_error": float(np.median(relative)),
        "test_rmse_exponent": float(np.sqrt(((exponent_hat - exponent) ** 2).mean())),
        "test_exponent_range": [float(exponent.min()), float(exponent.max())],
        # Prediction on truth: the fraction of the true spread the model
        # resolves. An MSE-optimal conditional mean lands on R^2 here, so this
        # sitting below 1 is the model being imperfect, not being wrong.
        "test_resolution_slope": resolution[0],
        "test_resolution_intercept": resolution[1],
        "test_resolution_slope_stderr": resolution[2],
        # Truth on prediction, the other direction: does the truth average to
        # what was predicted. This is the one that should be 1, and the one
        # worth acting on when it is not.
        "test_calibration_slope": calibration[0],
        "test_calibration_intercept": calibration[1],
        "test_calibration_slope_stderr": calibration[2],
        "baseline_note": "trivial only; the strong baseline is combinatorial edge editing",
    }


def reconstruction_metrics(data, log_eps):
    """How well T is reproduced, in KL, in probability space, and in log space."""
    T, T_hat = data["T"], data["T_hat"]
    n = T.shape[-1]
    off = off_diagonal(n, T.device)

    log_T = torch.log(T.clamp_min(log_eps))
    log_T_hat = data["log_T_hat"].masked_fill(~off, 0.0)

    true = T[..., off].flatten()
    predicted = T_hat[..., off].flatten()
    log_error = (log_T - log_T_hat)[..., off].flatten().abs()
    error = (predicted - true).abs()

    return {
        "test_recon_rmse": float(torch.sqrt((error ** 2).mean())),
        "test_recon_max_abs_error": float(error.max()),
        "test_log_recon_mae": float(log_error.mean()),
        "test_log_recon_max": float(log_error.max()),
        "_by_magnitude": magnitude_table(true, error, log_error),
    }


def magnitude_table(true, error, log_error):
    """Reconstruction error bucketed by the size of the true entry.

    The point of the whole table: forward KL weights each term by T_ij, so the
    bottom buckets contribute almost nothing to the loss and can be arbitrarily
    wrong without the headline number moving. 15% of entries sit below 1e-4.
    """
    quantiles = torch.linspace(0, 1, MAGNITUDE_BINS + 1, dtype=true.dtype)
    edges = torch.quantile(true, quantiles)
    edges[0], edges[-1] = -float("inf"), float("inf")

    rows = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (true >= lower) & (true < upper)
        if not mask.any():
            continue
        rows.append({
            "true_min": float(true[mask].min()),
            "true_max": float(true[mask].max()),
            "count": int(mask.sum()),
            "mean_abs_error": float(error[mask].mean()),
            "mean_abs_log_error": float(log_error[mask].mean()),
        })
    return rows


def validity_metrics(data):
    """Constraints, not scores. A failure here means the model is broken."""
    T_hat = data["T_hat"]
    diagonal = torch.diagonal(T_hat, dim1=-2, dim2=-1)
    return {
        "recon_diag_max_abs": float(diagonal.abs().max()),
        "recon_row_sum_max_dev": float((T_hat.sum(-1) - 1).abs().max()),
        "recon_all_finite": bool(torch.isfinite(T_hat).all()),
        "recon_min_entry": float(T_hat.min()),
    }


def latent_metrics(data, model_cfg):
    """Whether the latent is 160 dimensions or a dozen in disguise.

    Node latents are pooled across nodes rather than flattened into one
    160-vector, and that is the difference from a graph-level model. Dimension k
    means the same thing for every node, because the encoder applies the same
    map to each; but node 3 of one matrix has nothing to do with node 3 of
    another, since the labelling is arbitrary. So the sample is a node latent,
    and there are N x 20 of them over d = 8 coordinates.
    """
    mu = data["mu"]
    n_examples, n_nodes, d_latent = mu.shape
    pooled = mu.reshape(-1, d_latent)

    metrics = {
        "latent_dim_per_node": d_latent,
        "latent_dim_total": n_nodes * d_latent,
        "n_node_latents": pooled.shape[0],
        "d_global": model_cfg.d_global,
        **structure(pooled, data["logvar"].reshape(-1, d_latent), prefix="latent"),
    }

    if model_cfg.d_global:
        metrics.update(
            structure(data["mu_global"], data["logvar_global"], prefix="global")
        )
    return metrics


def structure(mu, logvar, prefix):
    """Activity, effective dimensionality and redundancy of one latent block.

    Adapted from the sibling's `latent_structure`. Everything is float64: the
    log-determinants below lose far too much in float32.
    """
    mu = mu.double()
    mean_var = logvar.double().exp().mean(0)
    n, d = mu.shape

    kl_per_dim = (0.5 * (mu.pow(2) + logvar.double().exp() - 1.0 - logvar.double())).mean(0)

    out = {}
    if n > 1:
        activity = mu.var(0, unbiased=True)
        out[f"{prefix}_active_units"] = int((activity > ACTIVE_UNIT_THRESHOLD).sum())
        out[f"{prefix}_active_fraction"] = float((activity > ACTIVE_UNIT_THRESHOLD).float().mean())
    else:
        # A variance across examples needs at least two; None says unmeasurable
        # rather than claiming every dimension is dead.
        out[f"{prefix}_active_units"] = None
        out[f"{prefix}_active_fraction"] = None

    out[f"{prefix}_au_threshold"] = ACTIVE_UNIT_THRESHOLD
    out[f"{prefix}_kl_total"] = float(kl_per_dim.sum())
    out[f"{prefix}_kl_per_dim_mean"] = float(kl_per_dim.mean())
    out[f"{prefix}_kl_per_dim_max"] = float(kl_per_dim.max())
    out[f"{prefix}_kl_participation_ratio"] = participation(kl_per_dim)
    out[f"{prefix}_dims_to_90pct_kl"] = dims_to(kl_per_dim, 0.9)
    out[f"{prefix}_noise_floor_var"] = float(mean_var.mean())

    # PCA by SVD of the centred means: better conditioned than forming a d x d
    # covariance, and the right singular vectors are needed just below.
    centred = mu - mu.mean(0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centred, full_matrices=False)
    eigenvalues = singular ** 2 / (n - 1)
    cumulative = torch.cumsum(eigenvalues, 0) / eigenvalues.sum()
    for pct in (90, 95, 99):
        out[f"{prefix}_pca_components_{pct}"] = int(
            torch.searchsorted(cumulative, pct / 100.0).item()
        ) + 1
    out[f"{prefix}_pca_participation_ratio"] = participation(eigenvalues)

    # A direction carries information only where its spread across examples
    # beats the posterior noise along it. The noise is anisotropic in the
    # coordinate basis and these directions mix coordinates, so the floor is the
    # quadratic form, not a scalar mean.
    out[f"{prefix}_pca_components_above_noise"] = int(
        (eigenvalues > (vh ** 2) @ mean_var).sum()
    )
    out[f"{prefix}_pca_eigenvalues"] = [round(float(v), 8) for v in eigenvalues]
    out[f"{prefix}_kl_per_dim"] = [round(float(v), 8) for v in kl_per_dim]

    # Cov(mu) is rank <= n-1, so the correlation matrix is singular once the
    # sample is no wider than the latent and the determinant says nothing.
    if n > d:
        keep = centred.std(0, unbiased=True) > 0
        sign, logabsdet = torch.linalg.slogdet(torch.corrcoef(centred[:, keep].T))
        out[f"{prefix}_total_correlation_gauss"] = (
            float(-0.5 * logabsdet) if sign > 0 else None
        )
    else:
        out[f"{prefix}_total_correlation_gauss"] = None
    # Independent dimensions land near this from sampling noise alone. Read the
    # difference against it, never the raw value.
    out[f"{prefix}_tc_null_bias"] = d * (d - 1) / (4.0 * n)

    # Cov(z) = Cov(E[z|x]) + E[Cov(z|x)], exactly, since Cov(z|x) is diagonal by
    # construction. Moment matching minimizes KL over Gaussians, so this is a
    # lower bound on KL(q(z)||p(z)) and the mutual information is an UPPER bound
    # (Hoffman & Johnson, ELBO surgery).
    sigma_aggregate = centred.T @ centred / (n - 1) + torch.diag(mean_var)
    mean_mu = mu.mean(0)
    sign, logabsdet = torch.linalg.slogdet(sigma_aggregate)
    if sign > 0:
        aggregate_kl = 0.5 * (
            float(torch.diagonal(sigma_aggregate).sum())
            + float(mean_mu @ mean_mu) - d - float(logabsdet)
        )
        out[f"{prefix}_agg_kl_gauss"] = aggregate_kl
        out[f"{prefix}_mi_upper_bound"] = float(kl_per_dim.sum()) - aggregate_kl
    else:
        out[f"{prefix}_agg_kl_gauss"] = None
        out[f"{prefix}_mi_upper_bound"] = None
    return out


def participation(v):
    """Effective number of contributors: len(v) for a flat vector, 1 for a spike."""
    denominator = float((v ** 2).sum())
    return float(v.sum()) ** 2 / denominator if denominator > 0 else 0.0


def dims_to(v, fraction):
    """How many of the largest entries it takes to reach a fraction of the total."""
    cumulative = torch.cumsum(torch.sort(v, descending=True).values, 0)
    return int(torch.searchsorted(cumulative, fraction * cumulative[-1]).item()) + 1


def evaluate(run, batch_size=None, num_threads=1):
    """Score one run and write its ``metrics.json``."""
    torch.set_num_threads(num_threads)
    run_dir, checkpoint, cfg, model, splits, scaler, device = load_run(run)
    model_cfg = TMVAEConfig(**checkpoint["model_config"])

    data = collect(model, splits.test, model_cfg, device, batch_size or cfg["batch_size"])

    metrics = {
        "run": run,
        "model": checkpoint["model"],
        "drop": cfg["drop"],
        "n_train": splits.sizes["train"],
        "n_val": splits.sizes["val"],
        "n_test": splits.sizes["test"],
        "split_seed": cfg["seed"],
        "best_epoch": checkpoint["epoch"],
        "best_score": checkpoint["score"],
        "best_metric": checkpoint["metric"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device_module.describe(device),
        "target_scaler": scaler.as_dict(),
        "git_commit": git_commit(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # The training terms on the test split, at the configured weights, so
        # they are comparable with the run's own history.
        **{f"test_{k}": v for k, v in data["losses"].items()},
        "test_elbo": data["losses"]["recon"] + data["losses"]["kl"],
        **property_metrics(data, scaler),
        **reconstruction_metrics(data, model_cfg.log_eps),
        **validity_metrics(data),
        **latent_metrics(data, model_cfg),
    }

    out = run_dir / paths.METRICS
    out.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    report(metrics)
    return metrics


def report(m):
    print(f"{m['run']}  (epoch {m['best_epoch']}, {m['n_test']} test examples)")
    print(
        f"  property   R2 {m['test_r2']:.3f}  skill vs train mean {m['skill_vs_train_mean']:+.1%}  "
        f"median relative error {m['test_median_relative_error']:.1%}  "
        f"RMSE {m['test_rmse_exponent']:.1f} in exponent units"
    )
    print(
        f"  range      resolution {m['test_resolution_slope']:.3f} "
        f"+/- {m['test_resolution_slope_stderr']:.3f} (expect R2)  "
        f"calibration {m['test_calibration_slope']:.3f} "
        f"+/- {m['test_calibration_slope_stderr']:.3f} (expect 1)"
    )
    print(
        f"  recon      KL {m['test_recon']:.3f}  rmse {m['test_recon_rmse']:.4f}  "
        f"log-space MAE {m['test_log_recon_mae']:.3f}"
    )
    print(
        f"  validity   diag {m['recon_diag_max_abs']:.1e}  "
        f"row sums off by {m['recon_row_sum_max_dev']:.1e}  finite {m['recon_all_finite']}"
    )
    active = f"{m['latent_active_units']}/{m['latent_dim_per_node']}"
    print(
        f"  latent     KL {m['test_kl']:.2f} (node {m['test_kl_node']:.2f}, "
        f"global {m['test_kl_global']:.2f})  active {active} per node  "
        f"PCA90 {m['latent_pca_components_90']}  above noise "
        f"{m['latent_pca_components_above_noise']}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory name under results/")
    parser.add_argument("--batch_size", type=int, help="override the run's batch size")
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    evaluate(args.run, args.batch_size, args.num_threads)


if __name__ == "__main__":
    main()
