"""How far from U matrix-space ascent can start and still end at U.

    pixi run -e ml python -m tm_ml.uniform.basin --runs <run> ...

U is a strict local maximum of ŷ(enc T) in every model tried, so it has a
basin. Starting BFGS on log ŷ(enc T), over rowsoftmax logits, from
T(s) = (1 − s)·T + s·U for test graphs T and several s, this reports how many
end within KL 0.01 of U, and where the rest end.
"""

import argparse

import numpy as np
import torch

from tm_ml.uniform import common
from tm_ml.uniform.compare import NEAR_U, bfgs
from tm_ml.uniform.hessian import logits_to_T

S = (0.5, 0.75, 0.9, 0.97)
GRAPHS = 8


def to_logits(T):
    off = ~np.eye(common.N, dtype=bool)
    L = np.log(T.clip(1e-300))
    return (L - L[off].reshape(common.N, common.N - 1).mean(1)[:, None])[off]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    U = common.uniform().numpy()
    for run in args.runs:
        ctx = common.load(run)
        test = np.flatnonzero(ctx.split == "test")[:GRAPHS]
        print(f"\n{run}")
        for s in S:
            ends, kls, starts_kl = [], [], []
            for i in test:
                T0 = (1 - s) * ctx.T[i].numpy() + s * U
                starts_kl.append(float(common.nonuniformity(torch.from_numpy(T0)[None])))
                res = bfgs(lambda x: -common.cycle_log_y_hat(ctx, logits_to_T(x)[None])[0],
                           to_logits(T0))
                with torch.no_grad():
                    kls.append(float(common.nonuniformity(logits_to_T(torch.from_numpy(res.x))[None])))
                ends.append(float(np.exp(-res.fun)))
            print(f"  s {s:.2f} (start KL {np.median(starts_kl):.3f}): "
                  f"{sum(k < NEAR_U for k in kls)}/{len(kls)} end at U; "
                  f"end KL median {np.median(kls):.3f}, end ŷ median {np.median(ends):.0f}, "
                  f"max {np.max(ends):.0f}")


if __name__ == "__main__":
    main()
