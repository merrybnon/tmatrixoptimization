"""Optimize the matrix, not the latent: F(T) = log ŷ(enc T).

The latent optimizer can move along directions the decoder ignores, which is
how it scores the one matrix U anywhere from 0.3 to 25,000. Scoring the matrix
through the encoder removes that freedom: F is a function of T alone. The
encoder ends in a LayerNorm, so its output is bounded over every input, and so
is F. The row-stochastic matrices are compact, so F has a maximum, and the
question is only where.

Two parameterizations:

- ``logits``: T = rowsoftmax(L), any row-stochastic matrix with a zero
  diagonal, 380 free logits. U is L = 0.
- ``landscape``: T = generate(points), 20 points in the plane with the drop's
  W_ij = D_ij⁻⁴. These are the only matrices the generator makes. U is not
  among them: it needs 20 mutually equidistant points, which the plane does
  not have room for past 3.

Starts: test graphs, random logits, U itself (logits only).
"""

import argparse

import numpy as np
import torch
from scipy.optimize import minimize

from tm_ml.landscape import DISPERSAL_POWER, positions
from tm_ml.uniform import common
from tm_ml.uniform.hessian import logits_to_T


def generate_torch(points):
    n = points.shape[0]
    diff = points[:, None] - points[None]
    D2 = (diff ** 2).sum(-1) + torch.eye(n, dtype=points.dtype)
    W = D2 ** (-DISPERSAL_POWER / 2) * (1 - torch.eye(n, dtype=points.dtype))
    return W / W.sum(1, keepdim=True)


def run(ctx, to_T, x0, maxiter):
    def f(x):
        x = torch.tensor(x, requires_grad=True)
        v = -common.cycle_log_y_hat(ctx, to_T(x)[None])[0]
        (g,) = torch.autograd.grad(v, x)
        return v.item(), g.numpy()

    res = minimize(f, x0, jac=True, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-6})
    with torch.no_grad():
        T = to_T(torch.from_numpy(res.x))[None]
    return T, float(np.exp(-res.fun)), res


def summarize(ctx, label, T, y, y0):
    kl = common.nonuniformity(T).item()
    spread = common.log10_spread(T).item()
    print(f"  {label:<22} ŷ {y0:8.1f} → {y:8.1f}   KL to U {kl:.3f}   log₁₀ spread {spread:.3f}")
    return kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()

    ctx = common.load()
    rng = np.random.default_rng(0)
    test = np.flatnonzero(ctx.split == "test")[:args.starts]
    off = ~np.eye(common.N, dtype=bool)
    ends = {}

    print("logits: T = rowsoftmax(L)")
    starts = [("U", np.zeros(380))]
    for i in test:
        L = np.log(ctx.T[i].numpy().clip(1e-300))
        starts.append((f"test graph {i}", (L - L[off].reshape(20, 19).mean(1)[:, None])[off]))
    for k in range(args.starts):
        starts.append((f"random {k}", rng.normal(0, 3, 380)))
    for label, x0 in starts:
        with torch.no_grad():
            y0 = float(np.exp(common.cycle_log_y_hat(ctx, logits_to_T(torch.from_numpy(x0))[None])))
        T, y, res = run(ctx, logits_to_T, x0, args.maxiter)
        summarize(ctx, label, T, y, y0)
        ends[f"logits {label}"] = (T[0].numpy(), y)

    print("\nlandscape: T = generate(points)")
    starts = []
    for i in test:
        starts.append((f"test graph {i}", positions(ctx.T[i].numpy())[0].ravel()))
    for k in range(args.starts):
        starts.append((f"random {k}", rng.uniform(0, 1, 40)))
    # Points spread as evenly as the plane allows: a 4 × 5 grid and a circle.
    gx, gy = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 0.75, 4))
    starts.append(("grid 4 × 5", np.stack([gx.ravel(), gy.ravel()], 1).ravel()))
    a = 2 * np.pi * np.arange(20) / 20
    starts.append(("circle", np.stack([np.cos(a), np.sin(a)], 1).ravel()))
    to_T = lambda x: generate_torch(x.reshape(20, 2))
    for label, x0 in starts:
        with torch.no_grad():
            y0 = float(np.exp(common.cycle_log_y_hat(ctx, to_T(torch.from_numpy(x0))[None])))
        T, y, res = run(ctx, to_T, x0, args.maxiter)
        summarize(ctx, label, T, y, y0)
        ends[f"landscape {label}"] = (T[0].numpy(), y)

    np.savez(ctx.out_dir / "matrix_space_ends.npz",
             labels=np.array(list(ends)), T=np.stack([v[0] for v in ends.values()]),
             y=np.array([v[1] for v in ends.values()]))


if __name__ == "__main__":
    main()
