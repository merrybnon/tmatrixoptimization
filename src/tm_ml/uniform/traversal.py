"""The global traversal, by the numbers, on the same matrices in several runs.

    pixi run -e ml python -m tm_ml.uniform.traversal --runs <run> ... [--graphs 613 9 48]

`visualize` draws ``traversal_gbd.png`` from each run's own first test graphs,
which differ between drops whose splits differ. This takes the same matrices,
named by their Tom1000 index and found in each run's drop by content, and
steps the global from the matrix's own encoding by t·σ, σ the spread of the
global over that run's test graphs, as the figure does, and past it. Per step:
ŷ(z), the decoded matrix's mean row KL to U, and ŷ(enc dec z).
"""

import argparse

import numpy as np
import torch

from tm_ml.datasets import read_processed
from tm_ml.uniform import common

T_STEPS = (-10, -6, -3, -2, -1, 0, 1, 2, 3, 6, 10, 30)
# Defaults: the reference's first test graph, a median one, and the highest target.
GRAPHS = (9, 23, 613)


def locate(ctx, T):
    """Index of matrix ``T`` in this run's drop, by exact content."""
    hits = np.flatnonzero((ctx.T.numpy() == T).all((1, 2)))
    if len(hits) != 1:
        raise ValueError(f"matrix found {len(hits)} times in the drop")
    return int(hits[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--graphs", type=int, nargs="+", default=list(GRAPHS),
                        help="Tom1000 indices")
    args = parser.parse_args()
    tom, targets = read_processed("Tom1000")

    for run in args.runs:
        ctx = common.load(run)
        test = ctx.split == "test"
        sigma = float(ctx.mu_global[test, 0].std())
        with torch.no_grad():
            g_U = common.encode(ctx, common.uniform((1,)))[1].item()
        print(f"\n{run}\n  σ of the global over the test graphs {sigma:.3f}; "
              f"global range {ctx.mu_global.min():+.3f} … {ctx.mu_global.max():+.3f}; "
              f"enc(U) global {g_U:+.3f}")
        for k in args.graphs:
            i = locate(ctx, tom[k])
            g0 = ctx.mu_global[i, 0].item()
            gg = torch.tensor([g0 + t * sigma for t in T_STEPS], dtype=torch.float64)[:, None]
            z = ctx.mu[i:i + 1].expand(len(T_STEPS), -1, -1)
            with torch.no_grad():
                y_z = np.exp(np.clip(common.log_y_hat(ctx, z, gg).numpy(), -700, 700))
                T_hat = common.decode(ctx, z, gg)
                kl = common.nonuniformity(T_hat).numpy()
                y_re = np.exp(common.cycle_log_y_hat(ctx, T_hat).numpy())
            print(f"  Tom1000 graph {k} (y {targets[k]:.0f}, {ctx.split[i]}), own global {g0:+.3f}")
            print("    t·σ      " + "".join(f"{t:>9}" for t in T_STEPS))
            print("    ŷ(z)     " + "".join(f"{v:>9.3g}" for v in y_z))
            print("    KL→U     " + "".join(f"{v:>9.3f}" for v in kl))
            print("    ŷ(enc T̂) " + "".join(f"{v:>9.0f}" for v in y_re))


if __name__ == "__main__":
    main()
