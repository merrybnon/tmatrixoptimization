"""Many regularizers and constraints on the latent optimization, side by side.

Every method maximizes something over x = (node latents, global), 161 numbers,
from the same starts: the first test graphs' encodings and enc(U). Each
endpoint is reported three ways:

- ŷ(z), what a latent objective climbs
- ŷ(enc dec z), the model's score for the matrix actually decoded, which no
  latent direction the decoder ignores can inflate
- KL(dec z ‖ U), mean over rows, how far the decoded matrix is from uniform

Methods:

- ``free``          −log ŷ(z), unregularized
- ``prior λ``       + (λ/2)‖x‖², the VAE prior
- ``anchor λ``      + (λ/2)‖x − x₀‖², what optimize.py does
- ``box``           confined to each coordinate's range over the data
- ``aggpost λ``     + λ·(−log q(x)), q the aggregate posterior: a Gaussian
                    mixture fitted to the 20000 node latents, per node, and a
                    Gaussian to the 1000 globals
- ``cycle λ``       + (λ/2)‖enc(dec x) − x‖², keep x a point the encoder would
                    give its own decoded matrix
- ``reenc``         −log ŷ(enc dec x), score the decoded matrix, not x
- ``reenc+aggpost`` the two together
- ``fibre``         x confined to identical node latents, where dec x = U
"""

import argparse

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture

from tm_ml.uniform import common


class AggregatePosterior:
    """−log q(x) for the node latents (mean over nodes) and the global, differentiable."""

    def __init__(self, ctx, components=20):
        nodes = ctx.mu.reshape(-1, ctx.mu.shape[-1]).numpy()
        gmm = GaussianMixture(components, covariance_type="full", random_state=0).fit(nodes)
        self.log_w = torch.log(torch.from_numpy(gmm.weights_))
        self.means = torch.from_numpy(gmm.means_)
        self.prec_chol = torch.from_numpy(gmm.precisions_cholesky_)
        d = nodes.shape[1]
        self.const = (-0.5 * d * np.log(2 * np.pi)
                      + torch.log(torch.diagonal(self.prec_chol, dim1=-2, dim2=-1)).sum(-1))
        g = ctx.mu_global.numpy()
        self.g_mean = torch.from_numpy(g.mean(0))
        self.g_std = torch.from_numpy(g.std(0))
        # Reference: the same quantity at the data's own encodings.
        with torch.no_grad():
            ref = self.nll(ctx.mu, ctx.mu_global)
        self.data_median = float(ref.median())
        self.data_max = float(ref.max())

    def nll(self, z, z_global):
        """``(B, n, d), (B, g) -> (B,)``."""
        diff = z[..., None, :] - self.means                      # B n K d
        y = torch.einsum("bnkd,kde->bnke", diff, self.prec_chol)
        log_comp = self.log_w + self.const - 0.5 * (y ** 2).sum(-1)
        node = -torch.logsumexp(log_comp, -1).mean(-1)
        glob = 0.5 * (((z_global - self.g_mean) / self.g_std) ** 2).sum(-1)
        return node + glob


def split(x, d):
    return x[:common.N * d].reshape(1, common.N, d), x[common.N * d:].reshape(1, -1)


def make_methods(ctx, agg):
    d = ctx.mu.shape[-1]

    def latent_y(x):
        return common.log_y_hat(ctx, *split(x, d))[0]

    def reenc_y(x):
        return common.cycle_log_y_hat(ctx, common.decode(ctx, *split(x, d)))[0]

    def cycle_gap(x):
        z, g = split(x, d)
        mu, mg = common.encode(ctx, common.decode(ctx, z, g))
        return ((mu - z) ** 2).sum() + ((mg - g) ** 2).sum()

    def aggpost(x):
        return agg.nll(*split(x, d))[0]

    m = {"free": lambda x, x0: -latent_y(x)}
    for lam in (1.0, 0.1):
        m[f"prior {lam:g}"] = lambda x, x0, lam=lam: -latent_y(x) + 0.5 * lam * (x ** 2).sum()
        m[f"anchor {lam:g}"] = (lambda x, x0, lam=lam:
                                -latent_y(x) + 0.5 * lam * ((x - x0) ** 2).sum())
    m["box"] = lambda x, x0: -latent_y(x)
    for lam in (1.0, 0.1):
        m[f"aggpost {lam:g}"] = lambda x, x0, lam=lam: -latent_y(x) + lam * aggpost(x)
        m[f"cycle {lam:g}"] = lambda x, x0, lam=lam: -latent_y(x) + 0.5 * lam * cycle_gap(x)
    m["reenc"] = lambda x, x0: -reenc_y(x)
    m["reenc+aggpost 0.1"] = lambda x, x0: -reenc_y(x) + 0.1 * aggpost(x)
    return m


def run(ctx, objective, x0, bounds=None, maxiter=300):
    x0_t = torch.from_numpy(x0)

    def f(x):
        x = torch.tensor(x, requires_grad=True)
        v = objective(x, x0_t)
        (g,) = torch.autograd.grad(v, x)
        return v.item(), g.numpy()

    method = "L-BFGS-B" if bounds is not None else "BFGS"
    with np.errstate(all="ignore"):
        res = minimize(f, x0, jac=True, method=method, bounds=bounds,
                       options={"maxiter": maxiter})
    x = res.x if np.isfinite(res.x).all() else x0
    return x, res


def score(ctx, x, agg):
    d = ctx.mu.shape[-1]
    with torch.no_grad():
        z, g = split(torch.from_numpy(x), d)
        T = common.decode(ctx, z, g)
        return dict(
            y=float(np.exp(np.clip(common.log_y_hat(ctx, z, g).item(), -700, 700))),
            y_re=float(np.exp(common.cycle_log_y_hat(ctx, T).item())),
            kl=float(common.nonuniformity(T).item()),
            nll=float(agg.nll(z, g).item()),
            spread=float(z[0].std(0).norm()),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    ctx = common.load()
    agg = AggregatePosterior(ctx)
    d = ctx.mu.shape[-1]
    print(f"aggregate-posterior −log q at the data: median {agg.data_median:.2f}, "
          f"max {agg.data_max:.2f}")

    test = np.flatnonzero(ctx.split == "test")[:args.starts]
    starts = [(f"g{i}", np.concatenate([ctx.mu[i].numpy().ravel(), ctx.mu_global[i].numpy()]))
              for i in test]
    with torch.no_grad():
        mu_U, g_U = common.encode(ctx, common.uniform((1,)))
    starts.append(("U", np.concatenate([mu_U[0].numpy().ravel(), g_U[0].numpy()])))

    flat = torch.cat([ctx.mu.reshape(len(ctx.mu), -1), ctx.mu_global], 1).numpy()
    box = list(zip(np.tile(ctx.mu.reshape(-1, d).min(0).values.numpy(), common.N).tolist()
                   + [float(flat[:, -1].min())],
                   np.tile(ctx.mu.reshape(-1, d).max(0).values.numpy(), common.N).tolist()
                   + [float(flat[:, -1].max())]))

    methods = make_methods(ctx, agg)
    rows = []
    print(f"\n{'method':<18} {'start':<6} {'ŷ(z)':>10} {'ŷ(enc dec z)':>13} {'KL→U':>7} "
          f"{'−log q':>8} {'node spread':>11}")
    for name, objective in methods.items():
        if args.only and name not in args.only:
            continue
        for label, x0 in starts:
            x, res = run(ctx, objective, x0, bounds=box if name == "box" else None)
            s = score(ctx, x, agg)
            rows.append((name, label, s))
            print(f"{name:<18} {label:<6} {s['y']:>10.4g} {s['y_re']:>13.1f} {s['kl']:>7.3f} "
                  f"{s['nll']:>8.2f} {s['spread']:>11.3f}")

    # The fibre, alone: identical node latents c and a global g, 9 numbers.
    if not args.only or "fibre" in args.only:
        def fib_obj(v, v0):
            z = v[:d].expand(1, common.N, d)
            return -common.log_y_hat(ctx, z, v[d:].reshape(1, 1))[0] + 0.1 * agg.nll(
                z, v[d:].reshape(1, 1))[0]
        for label, x0 in starts:
            v0 = np.concatenate([x0[:common.N * d].reshape(common.N, d).mean(0), x0[-1:]])
            v, res = run(ctx, fib_obj, v0)
            x = np.concatenate([np.tile(v[:d], common.N), v[d:]])
            s = score(ctx, x, agg)
            rows.append(("fibre+aggpost 0.1", label, s))
            print(f"{'fibre+aggpost 0.1':<18} {label:<6} {s['y']:>10.4g} {s['y_re']:>13.1f} "
                  f"{s['kl']:>7.3f} {s['nll']:>8.2f} {s['spread']:>11.3f}")

    print(f"\n{'method':<18} {'median ŷ(z)':>12} {'median ŷ(enc dec z)':>20} "
          f"{'median KL→U':>12} {'min KL→U':>9}")
    for name in dict.fromkeys(r[0] for r in rows):
        ss = [r[2] for r in rows if r[0] == name]
        print(f"{name:<18} {np.median([s['y'] for s in ss]):>12.4g} "
              f"{np.median([s['y_re'] for s in ss]):>20.1f} "
              f"{np.median([s['kl'] for s in ss]):>12.3f} {min(s['kl'] for s in ss):>9.3f}")


if __name__ == "__main__":
    main()
