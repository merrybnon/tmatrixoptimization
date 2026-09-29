"""How close the aggregate posterior is to the prior, in shape and not just scale.

    pixi run -e ml python -m tm_ml.latent_diagnostics --run global_latent/TMVAE_...

The prior is p(z) = N(0, I). The aggregate posterior is the mixture of every
example's encoder Gaussian, q(z) = (1/M) Σ_m N(z; μ_m, σ_m²). Sampling for
generation assumes the two agree. The per-example KL in the loss only reaches
that agreement through

    E_x[ KL(q(z|x) ‖ p(z)) ]  =  I(x; z)  +  KL( q(z) ‖ p(z) )

so this reads the right-hand side off directly. `evaluate.structure` already
reports a Gaussian moment-match of q(z), which is a lower bound on
KL(q ‖ p) and is blind to skew, clumps and holes. Here q(z) is the exact
mixture and both KLs are Monte Carlo estimates against it:

- KL(q ‖ p): mass the model puts where the prior does not.
- KL(p ‖ q): prior mass where no example lives. These are the holes that
  sampling from N(0, I) falls into.

Node latents are pooled over examples and nodes, the same as in `evaluate`, so
the node block is the per-node marginal. Correlations between the nodes of one
matrix are not visible to it.
"""

import argparse
import json
import math

import numpy as np
import torch

from tm_ml.evaluate import collect, load_run
from tm_ml.models import TMVAEConfig

N_SAMPLES = 8192
CHUNK = 256
# Density ratio q/p below which a prior sample counts as landing in a hole.
HOLE_RATIO = 0.1
SEED = 0

LOG_2PI = math.log(2 * math.pi)


def log_prior(z):
    return -0.5 * (z.pow(2).sum(1) + z.shape[1] * LOG_2PI)


def log_aggregate(z, mu, logvar):
    """log q(z) under the equal-weight mixture of N(μ_m, σ_m²), chunked over z."""
    inv_var = (-logvar).exp()
    log_norm = -0.5 * (logvar.sum(1) + mu.shape[1] * LOG_2PI)
    out = []
    for block in z.split(CHUNK):
        sq = ((block[:, None, :] - mu[None]) ** 2 * inv_var[None]).sum(2)
        out.append(torch.logsumexp(log_norm[None] - 0.5 * sq, 1))
    return torch.cat(out) - math.log(len(mu))


def mc(values):
    """Mean and its standard error, over Monte Carlo samples."""
    return float(values.mean()), float(values.std() / math.sqrt(len(values)))


def moments(samples):
    """Per-dimension mean, variance, skew and excess kurtosis of the samples."""
    mean = samples.mean(0)
    centred = samples - mean
    var = centred.pow(2).mean(0)
    std = var.sqrt()
    return {
        "mean": mean.tolist(),
        "var": var.tolist(),
        "skew": (centred.pow(3).mean(0) / std**3).tolist(),
        "excess_kurtosis": (centred.pow(4).mean(0) / var**2 - 3.0).tolist(),
    }


def diagnose(mu, logvar, generator):
    """Everything above for one latent block of M encoder Gaussians over d dims."""
    mu, logvar = mu.double(), logvar.double()
    m, d = mu.shape

    # z ~ q: pick a component uniformly, then sample it.
    pick = torch.randint(m, (N_SAMPLES,), generator=generator)
    eps = torch.randn(N_SAMPLES, d, generator=generator, dtype=torch.float64)
    z_q = mu[pick] + eps * (0.5 * logvar[pick]).exp()
    z_p = torch.randn(N_SAMPLES, d, generator=generator, dtype=torch.float64)

    kl_qp = mc(log_aggregate(z_q, mu, logvar) - log_prior(z_q))
    log_ratio_p = log_aggregate(z_p, mu, logvar) - log_prior(z_p)
    kl_pq = mc(-log_ratio_p)

    per_example_kl = float(
        (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)).sum(1).mean()
    )
    return {
        "n_components": m,
        "d": d,
        "per_example_kl": per_example_kl,
        "kl_q_p": kl_qp,
        "kl_p_q": kl_pq,
        "mutual_information": per_example_kl - kl_qp[0],
        "hole_fraction": float((log_ratio_p < math.log(HOLE_RATIO)).double().mean()),
        "median_sigma": (0.5 * logvar).exp().median(0).values.tolist(),
        "aggregate": moments(z_q),
    }


def report(name, r):
    print(f"\n── {name}: {r['n_components']} encoder Gaussians over {r['d']} dims")
    print(f"  per-example KL (the loss term)   {r['per_example_kl']:8.3f} nats")
    print(f"    = mutual information I(x; z)   {r['mutual_information']:8.3f}")
    print(f"    + KL(q(z) ‖ p(z))              {r['kl_q_p'][0]:8.3f} ± {r['kl_q_p'][1]:.3f}")
    print(f"  KL(p(z) ‖ q(z))  (holes)         {r['kl_p_q'][0]:8.3f} ± {r['kl_p_q'][1]:.3f}")
    print(f"  prior samples with q/p < {HOLE_RATIO}     {100 * r['hole_fraction']:7.1f} %")
    a = r["aggregate"]
    print("  dim    mean     var    skew   exkurt   median σ")
    for k in range(r["d"]):
        print(f"  {k:3d} {a['mean'][k]:7.3f} {a['var'][k]:7.3f} {a['skew'][k]:7.3f}"
              f" {a['excess_kurtosis'][k]:8.3f} {r['median_sigma'][k]:10.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    args = parser.parse_args()

    run_dir, checkpoint, cfg, model, splits, _, device = load_run(
        args.run, device=torch.device("cpu")
    )
    model_cfg = TMVAEConfig(**checkpoint["model_config"])
    data = collect(model, getattr(splits, args.split), model_cfg, device, cfg["batch_size"])
    generator = torch.Generator().manual_seed(SEED)

    d_latent = data["mu"].shape[-1]
    results = {
        "run": args.run,
        "split": args.split,
        "node": diagnose(
            data["mu"].reshape(-1, d_latent), data["logvar"].reshape(-1, d_latent), generator
        ),
    }
    if model_cfg.d_global:
        results["global"] = diagnose(data["mu_global"], data["logvar_global"], generator)

    print(f"{args.run}  ({args.split} split)")
    for name in ("node", "global"):
        if name in results:
            report(name, results[name])

    out = run_dir / f"latent_diagnostics_{args.split}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
