"""The same diagnostics on several runs, one row each.

    pixi run -e ml python -m tm_ml.uniform.compare --runs <run> <run> ...

Per run:

- ``R²``            test R² on log y, from metrics.json
- ``ŷ(U)``          ŷ(enc U), against the slides' 2000
- ``U local max``   share of 300 random logit directions along which ŷ(enc T)
                    falls from U at a step of 0.1
- ``ring max``      the largest ŷ(enc T) on rings of logit radius 10, 20 and 40
                    around U, random and data-shaped directions
- ``matrix opt``    BFGS on log ŷ(enc T) over rowsoftmax logits from the test
                    graphs and random starts: the median end ŷ, the largest,
                    and how many ended within KL 0.01 of U
- ``reversible opt`` the same over reversible matrices, W symmetric
- ``landscape opt`` the same over 20 points in the plane, T = generate(points)
- ``prior 1``       the latent optimizer with the N(0, I) penalty at λ = 1, from
                    the test graphs: median ŷ(z), and how many decode to U
- ``fibre``         ŷ over identical node latents in the data's box, all of
                    which decode to U: min and max
- ``g = +100``      ŷ(z) with a test graph's node latents and the global at 100
"""

import argparse
import json

import numpy as np
import torch
from scipy.optimize import minimize

from tm_ml import paths
from tm_ml.uniform import common
from tm_ml.uniform.hessian import logits_to_T
from tm_ml.uniform.matrix_space import generate_torch, reversible_T, symmetric_logits
from tm_ml.landscape import positions

NEAR_U = 0.01


def bfgs(f_torch, x0, maxiter=300, bounds=None):
    def f(x):
        x = torch.tensor(x, requires_grad=True)
        v = f_torch(x)
        (g,) = torch.autograd.grad(v, x)
        return v.item(), g.numpy()

    with np.errstate(all="ignore"):
        res = minimize(f, x0, jac=True, method="L-BFGS-B" if bounds else "BFGS",
                       bounds=bounds, options={"maxiter": maxiter})
    return res


def diagnostics(run, starts=8, seed=0):
    ctx = common.load(run)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    out = {}
    metrics = json.loads((paths.resolve(run) / "metrics.json").read_text())
    out["R²"] = metrics["test_r2"]

    U = common.uniform((1,))
    with torch.no_grad():
        out["ŷ(U)"] = float(np.exp(common.cycle_log_y_hat(ctx, U).item()))

        V = torch.randn(300, 380, dtype=torch.float64)
        V /= V.norm(dim=1, keepdim=True)
        T = torch.stack([logits_to_T(0.1 * v) for v in V])
        out["U local max"] = float((common.cycle_log_y_hat(ctx, T).numpy()
                                    < np.log(out["ŷ(U)"])).mean())

        off = ~torch.eye(common.N, dtype=torch.bool)
        L = torch.log(ctx.T.clamp_min(1e-300))
        L = torch.where(off, L, torch.zeros(()))
        L = (L - L.sum(-1, keepdim=True) / (common.N - 1))[:, off]
        # U itself, if the drop holds it, has no direction away from U.
        L = L[L.norm(dim=1) > 0][:150]
        D = torch.cat([L / L.norm(dim=1, keepdim=True), V[:150]])
        ring = 0.0
        for r in (10, 20, 40):
            T = torch.stack([logits_to_T(r * d) for d in D])
            ring = max(ring, float(np.exp(common.cycle_log_y_hat(ctx, T).numpy()).max()))
        out["ring max"] = ring

    test = np.flatnonzero(ctx.split == "test")[:starts]
    offn = off.numpy()

    # Matrix space.
    x0s = []
    for i in test:
        Li = np.log(ctx.T[i].numpy().clip(1e-300))
        x0s.append((Li - Li[offn].reshape(20, 19).mean(1)[:, None])[offn])
    x0s += [rng.normal(0, 3, 380) for _ in range(starts)]
    ys, kls = [], []
    for x0 in x0s:
        res = bfgs(lambda x: -common.cycle_log_y_hat(ctx, logits_to_T(x)[None])[0], x0)
        with torch.no_grad():
            T = logits_to_T(torch.from_numpy(res.x))[None]
        ys.append(float(np.exp(-res.fun)))
        kls.append(float(common.nonuniformity(T)))
    out["matrix opt median"] = float(np.median(ys))
    out["matrix opt max"] = float(np.max(ys))
    out["matrix opt → U"] = f"{sum(k < NEAR_U for k in kls)}/{len(kls)}"
    out["matrix opt min KL"] = float(np.min(kls))

    # Reversible matrices, the family the data and U share.
    ys, kls = [], []
    w0s = [symmetric_logits(ctx.T[i].numpy()) for i in test]
    w0s += [rng.normal(0, 3, 190) for _ in range(starts)]
    for w0 in w0s:
        res = bfgs(lambda w: -common.cycle_log_y_hat(ctx, reversible_T(w)[None])[0], w0)
        with torch.no_grad():
            T = reversible_T(torch.from_numpy(res.x))[None]
        ys.append(float(np.exp(-res.fun)))
        kls.append(float(common.nonuniformity(T)))
    out["reversible opt median"] = float(np.median(ys))
    out["reversible opt max"] = float(np.max(ys))
    out["reversible opt → U"] = f"{sum(k < NEAR_U for k in kls)}/{len(kls)}"
    out["reversible opt min KL"] = float(np.min(kls))

    # Landscapes.
    ys = []
    pts = [positions(ctx.T[i].numpy())[0].ravel() for i in test]
    pts += [rng.uniform(0, 1, 40) for _ in range(starts)]
    for p0 in pts:
        res = bfgs(lambda x: -common.cycle_log_y_hat(ctx, generate_torch(x.reshape(20, 2))[None])[0], p0)
        ys.append(float(np.exp(-res.fun)))
    out["landscape opt median"] = float(np.median(ys))
    out["landscape opt max"] = float(np.max(ys))

    # Latent, prior penalty.
    d = ctx.mu.shape[-1]
    ys, near = [], 0
    for i in test:
        x0 = np.concatenate([ctx.mu[i].numpy().ravel(), ctx.mu_global[i].numpy()])

        def f(x):
            z, g = x[:common.N * d].reshape(1, common.N, d), x[common.N * d:].reshape(1, -1)
            return -common.log_y_hat(ctx, z, g)[0] + 0.5 * (x ** 2).sum()

        res = bfgs(f, x0)
        with torch.no_grad():
            x = torch.from_numpy(res.x)
            z, g = x[:common.N * d].reshape(1, common.N, d), x[common.N * d:].reshape(1, -1)
            ys.append(float(np.exp(common.log_y_hat(ctx, z, g).item())))
            near += float(common.nonuniformity(common.decode(ctx, z, g))) < NEAR_U
    out["prior 1 ŷ(z)"] = float(np.median(ys))
    out["prior 1 → U"] = f"{near}/{len(test)}"

    # The fibre.
    with torch.no_grad():
        flat = ctx.mu.reshape(-1, d)
        c = flat.min(0).values + (flat.max(0).values - flat.min(0).values) * torch.rand(
            2000, d, dtype=torch.float64)
        g = ctx.mu_global[:, :1]
        gg = g.min() + (g.max() - g.min()) * torch.rand(2000, 1, dtype=torch.float64)
        y = np.exp(common.log_y_hat(ctx, c[:, None].expand(-1, common.N, -1), gg).numpy())
        out["fibre min"], out["fibre max"] = float(y.min()), float(y.max())

        z = ctx.mu[test[:1]]
        out["g=+100"] = float(np.exp(np.clip(common.log_y_hat(
            ctx, z, torch.full((1, 1), 100.0, dtype=torch.float64)).item(), -700, 700)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--starts", type=int, default=8)
    args = parser.parse_args()
    rows = {}
    for run in args.runs:
        rows[run] = diagnostics(run, args.starts)
        print(f"\n{run}")
        for k, v in rows[run].items():
            print(f"  {k:<22} {v:.4g}" if isinstance(v, float) else f"  {k:<22} {v}")


if __name__ == "__main__":
    main()
