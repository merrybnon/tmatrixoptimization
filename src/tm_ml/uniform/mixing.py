"""ŷ along the straight line from each test graph to U.

    pixi run -e ml python -m tm_ml.uniform.mixing --runs <run> ...

T(s) = (1 − s)·T + s·U is row-stochastic with a zero diagonal for every s,
and reversible when T is only if their stationary distributions agree, so it
is a path through general matrices. If U is the maximum in a sense an
optimizer can use, ŷ(enc T(s)) should rise toward s = 1. Per run, over the
100 test graphs: where along s the maximum falls, how many paths end at
their maximum, and how many rise monotonically.
"""

import argparse

import numpy as np
import torch

from tm_ml.uniform import common

S = np.linspace(0, 1, 41)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    U = common.uniform()
    for run in args.runs:
        ctx = common.load(run)
        test = np.flatnonzero(ctx.split == "test")
        with torch.no_grad():
            T = ctx.T[test]
            paths = torch.stack([(1 - s) * T + s * U for s in S], 1)       # B S n n
            y = np.exp(common.cycle_log_y_hat(ctx, paths.reshape(-1, common.N, common.N))
                       .numpy().reshape(len(test), len(S)))
        argmax = S[y.argmax(1)]
        monotone = (np.diff(y, axis=1) >= -1e-9 * y[:, 1:]).all(1)
        print(f"\n{run}")
        print(f"  ŷ(U) {y[0, -1]:.1f}; path max ŷ: median {np.median(y.max(1)):.1f}, "
              f"largest {y.max():.1f}")
        print(f"  s at the path maximum: median {np.median(argmax):.3f}; "
              f"{(argmax == 1).sum()}/{len(test)} paths peak at U, "
              f"{monotone.sum()}/{len(test)} rise monotonically")
        print("  median ŷ over the test graphs at s = " +
              ", ".join(f"{s:.2f}: {np.median(y[:, k]):.0f}" for k, s in enumerate(S)
                        if k % 5 == 0))


if __name__ == "__main__":
    main()
