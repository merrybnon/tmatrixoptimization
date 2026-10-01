"""Matrix-space ascent with a maximum-entropy tie-break.

    pixi run -e ml python -m tm_ml.uniform.tiebreak --runs <run> ... [--eps 0.01 0.1 1 3]

Maximizes log ŷ(enc T) − ε·KL(T ‖ U) over rowsoftmax logits, KL the mean row
KL from `common.nonuniformity`. With a predictor whose maximum is at U, such as
the uniform_max head, U is the unique maximizer for every ε > 0: it attains the
maximum of both terms and nothing else attains the second. Among matrices the
model rates equally, the tie-break prefers the most mixed. Where ŷ(enc U) sits
below the predictor's maximum, ε has to outweigh the gap before U wins, and
then the entropy term, not the model, chose it.

Starts: test graphs and random logits, as in `compare`.
"""

import argparse

import numpy as np
import torch

from tm_ml.uniform import common
from tm_ml.uniform.basin import to_logits
from tm_ml.uniform.compare import NEAR_U, bfgs
from tm_ml.uniform.hessian import logits_to_T

STARTS = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.1, 1.0, 3.0])
    args = parser.parse_args()
    for run in args.runs:
        ctx = common.load(run)
        rng = np.random.default_rng(0)
        test = np.flatnonzero(ctx.split == "test")[:STARTS]
        x0s = [to_logits(ctx.T[i].numpy()) for i in test]
        x0s += [rng.normal(0, 3, 380) for _ in range(STARTS)]
        with torch.no_grad():
            y_U = float(np.exp(common.cycle_log_y_hat(ctx, common.uniform((1,))).item()))
        print(f"\n{run}\n  ŷ(U) {y_U:.1f}")
        for eps in args.eps:
            ys, kls = [], []
            for x0 in x0s:
                def f(x):
                    T = logits_to_T(x)[None]
                    return -common.cycle_log_y_hat(ctx, T)[0] + eps * common.nonuniformity(T)[0]

                res = bfgs(f, x0, maxiter=1000)
                with torch.no_grad():
                    T = logits_to_T(torch.from_numpy(res.x))[None]
                    ys.append(float(np.exp(common.cycle_log_y_hat(ctx, T).item())))
                    kls.append(float(common.nonuniformity(T)))
            print(f"  ε {eps:<5g} {sum(k < NEAR_U for k in kls):>2}/{len(kls)} end at U; "
                  f"end KL median {np.median(kls):.4f}, max {np.max(kls):.4f}; "
                  f"end ŷ median {np.median(ys):.0f}, max {np.max(ys):.0f}")


if __name__ == "__main__":
    main()
