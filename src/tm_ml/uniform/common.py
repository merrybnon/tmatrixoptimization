"""The run, its encodings, and the measures every script reports.

Everything runs in float64 on the cpu, as `optimize.py` does: the model is
small, and float32 noise breaks line searches long before an optimum.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tm_ml import evaluate as evaluate_module
from tm_ml.datasets import read_processed, split_indices
from tm_ml.models import off_diagonal
from tm_ml.optimize import float64_copy

RUN = "input_norm/TMVAE_b0p01-bw20-dg1-do0p1-enormfixed-e600-ll3-lff0p05-schcosine-s1_Tom1000"
N = 20
LOG_19 = float(np.log(N - 1))
# The slides' bounds: 20 isolated patches of carrying capacity 100, and one
# well-mixed population of 20 × 100.
A_MIN, A_MAX = 100.0, 2000.0


@dataclass
class Context:
    """The float64 model, its target scaler, and every graph encoded once.

    ``mu`` is ``(1000, 20, d)`` and ``mu_global`` ``(1000, d_global)``, over
    the whole drop in its stored order; ``split`` names each graph's split.
    """

    model: torch.nn.Module
    scaler: object
    T: torch.Tensor
    log_y: np.ndarray
    mu: torch.Tensor
    mu_global: torch.Tensor
    split: np.ndarray
    out_dir: Path


def load(run=RUN):
    # The box has 256 cores; left alone, torch takes them all and every script
    # running beside a training job slows to a crawl.
    torch.set_num_threads(8)
    run_dir, _, cfg, model, _, scaler, _ = evaluate_module.load_run(run, device="cpu")
    model64 = float64_copy(model)
    for p in model64.parameters():
        p.requires_grad_(False)

    matrices, targets = read_processed(cfg["drop"])
    split = np.empty(len(matrices), dtype=object)
    for name, idx in split_indices(len(matrices), cfg["val_frac"], cfg["test_frac"],
                                   cfg["seed"]).items():
        split[idx] = name

    T = torch.from_numpy(matrices).double()
    with torch.no_grad():
        mu, _, mu_global, _ = model64.encode(T)

    out_dir = run_dir / "uniform"
    out_dir.mkdir(exist_ok=True)
    return Context(model64, scaler, T, np.log(targets), mu, mu_global, split, out_dir)


def uniform(batch=()):
    """The well-mixed matrix, 1/19 off the diagonal and 0 on it."""
    U = torch.full((N, N), 1.0 / (N - 1), dtype=torch.float64)
    U.fill_diagonal_(0.0)
    return U.expand(*batch, N, N).clone()


def log_y_hat(ctx, z, z_global):
    """Predicted log y in real units, batched: ``(B, n, d), (B, g) -> (B,)``."""
    return ctx.model.predict(z, z_global) * ctx.scaler.std + ctx.scaler.mean


def encode(ctx, T):
    """Posterior means of ``(B, n, n)`` matrices."""
    mu, _, mu_global, _ = ctx.model.encode(T)
    return mu, mu_global


def cycle_log_y_hat(ctx, T):
    """ŷ of a matrix: encode to the posterior mean, then predict."""
    return log_y_hat(ctx, *encode(ctx, T))


def decode(ctx, z, z_global):
    return ctx.model.decode(z, z_global)[0]


def nonuniformity(T):
    """Per matrix, the mean over rows of KL(T_i ‖ U_i), in nats: ``(B, n, n) -> (B,)``.

    log 19 − H(T_i), over the 19 off-diagonal entries of row i, then the mean
    over the 20 rows. Exactly 0 for the uniform matrix and nowhere else.
    """
    T = torch.as_tensor(T)
    off = off_diagonal(T.shape[-1], T.device)
    plogp = torch.where(off, T * torch.log(T.clamp_min(1e-300)), T.new_zeros(()))
    return (LOG_19 + plogp.sum(-1)).mean(-1)


def log10_spread(T):
    """Per matrix, the std of log₁₀ T_ij over the 380 off-diagonal entries."""
    T = torch.as_tensor(T)
    off = off_diagonal(T.shape[-1], T.device)
    return torch.log10(T[..., off].clamp_min(1e-300)).std(-1)


def describe(ctx, T, label):
    """One line: ŷ through the encoder, and how far T is from uniform."""
    with torch.no_grad():
        y = np.exp(cycle_log_y_hat(ctx, T).numpy())
        kl = nonuniformity(T).numpy()
        spread = log10_spread(T).numpy()
    for k in range(len(y)):
        print(f"  {label}[{k}]  ŷ(enc T) {y[k]:9.1f}   KL to U {kl[k]:.3e}   "
              f"log₁₀ spread {spread[k]:.3f}")
