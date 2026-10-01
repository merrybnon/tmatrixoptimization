"""Where the uniform matrix sits, in the data and in the latent, and what ŷ says there.

Three questions:

1. How far is U from the training matrices? Measured by mean row KL to U and
   the log₁₀ spread, against the same over the drop's other graphs.
2. What does the model do with U? Its encoding, ŷ there, its reconstruction,
   and where its global and node latents fall against the data's.
3. The fibre. Identical node latents decode to U whatever their value; how
   much does ŷ vary over them, inside the range the data occupies?
"""

import argparse

import numpy as np
import torch
from scipy.optimize import minimize

from tm_ml.uniform import common


def percentile_of(value, sample):
    return 100.0 * (sample < value).mean()


def main():
    args = common.run_arg(argparse.ArgumentParser()).parse_args()
    torch.manual_seed(0)
    ctx = common.load(args.run)
    U = common.uniform((1,))
    y = np.exp(ctx.log_y)
    train = ctx.split == "train"

    # A drop may hold U itself as a labelled point (1000andUniform); the
    # questions here are about the rest of the data, so exact copies of U are
    # set aside and reported.
    kl_all = common.nonuniformity(ctx.T).numpy()
    is_U = np.abs(kl_all) < 1e-9
    print(f"1. the data against U, over the {(~is_U).sum()} graphs other than U"
          + (f" (U itself: index {np.flatnonzero(is_U).tolist()}, target "
             f"{y[is_U][0]:.0f}, {ctx.split[is_U][0]})" if is_U.any() else ""))
    keep = ~is_U
    kl = np.where(keep, kl_all, np.inf)
    spread = common.log10_spread(ctx.T).numpy()
    order = np.argsort(-np.where(keep, y, -np.inf))
    print(f"  mean row KL to U: min {kl[keep].min():.3f}, median {np.median(kl[keep]):.3f}, "
          f"max {kl[keep].max():.3f}")
    print(f"  log₁₀ spread:     min {spread[keep].min():.3f}, median {np.median(spread[keep]):.3f}")
    print(f"  target: max {y[keep].max():.1f}, and above 1500: {(y[keep] > 1500).sum()} graphs")
    print("  the ten highest targets, with their KL to U and log₁₀ spread:")
    for i in order[:10]:
        print(f"    graph {i:4d} ({ctx.split[i]:>5})  y {y[i]:7.1f}  KL {kl[i]:.3f}  "
              f"spread {spread[i]:.3f}")
    rho = np.corrcoef(np.argsort(np.argsort(kl[keep])),
                      np.argsort(np.argsort(ctx.log_y[keep])))[0, 1]
    print(f"  Spearman ρ(KL to U, y) over those graphs: {rho:+.3f}")

    print("\n2. U through the model")
    with torch.no_grad():
        mu_U, g_U = common.encode(ctx, U)
        y_U = np.exp(common.log_y_hat(ctx, mu_U, g_U).item())
        T_U = common.decode(ctx, mu_U, g_U)
    node_spread_U = mu_U[0].std(0).norm().item()
    node_spread = ctx.mu.std(1).norm(dim=-1).numpy()
    g = ctx.mu_global[:, 0].numpy()
    print(f"  ŷ(enc U) = {y_U:.1f}   (slides: {common.A_MAX:.0f}; largest target {y.max():.1f})")
    print(f"  reconstruction of U: KL to U {common.nonuniformity(T_U).item():.2e}")
    print(f"  global μ of U {g_U.item():+.3f}; data range {g.min():+.3f} … {g.max():+.3f}, "
          f"U at the {percentile_of(g_U.item(), g):.1f}th percentile")
    print(f"  node μ spread of U ‖std over nodes‖ = {node_spread_U:.2e}; "
          f"data {node_spread.min():.3f} … {node_spread.max():.3f}")
    centre = mu_U[0].mean(0)
    graph_means = ctx.mu.mean(1)
    d = (graph_means - centre).norm(dim=-1).numpy()
    print(f"  U's common node latent sits {d.min():.3f} from the nearest graph-mean node "
          f"latent (median {np.median(d):.3f})")
    # What the encoder makes of a matrix nearer and nearer U, from the most
    # uniform training graph.
    i = int(np.argmin(kl))
    print(f"  mixing the most uniform graph ({i}, y {y[i]:.0f}) toward U, (1 − s)·T + s·U:")
    for s in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        T_s = (1 - s) * ctx.T[i:i + 1] + s * U
        with torch.no_grad():
            m, gg = common.encode(ctx, T_s)
            print(f"    s {s:4.2f}  ŷ {np.exp(common.log_y_hat(ctx, m, gg).item()):8.1f}  "
                  f"global {gg.item():+.3f}  node spread {m[0].std(0).norm().item():.3f}")

    print("\n3. the fibre: identical node latents, every one decoding to U")
    lo_c, hi_c = ctx.mu.reshape(-1, ctx.mu.shape[-1]).min(0).values, \
        ctx.mu.reshape(-1, ctx.mu.shape[-1]).max(0).values
    n = 2000
    c = lo_c + (hi_c - lo_c) * torch.rand(n, ctx.mu.shape[-1], dtype=torch.float64)
    gg = torch.empty(n, 1, dtype=torch.float64).uniform_(g.min(), g.max())
    z = c[:, None, :].expand(-1, common.N, -1)
    with torch.no_grad():
        T_fib = common.decode(ctx, z, gg)
        y_fib = np.exp(common.log_y_hat(ctx, z, gg).numpy())
        y_cyc = np.exp(common.cycle_log_y_hat(ctx, T_fib).numpy())
    kl_fib = common.nonuniformity(T_fib).numpy()
    print(f"  {n} random points, c uniform in the node box, g in the global range")
    print(f"  decoded KL to U: max {kl_fib.max():.2e}")
    print(f"  ŷ on the fibre: min {y_fib.min():.1f}, median {np.median(y_fib):.1f}, "
          f"max {y_fib.max():.1f}")
    print(f"  ŷ(enc dec z) on the fibre: min {y_cyc.min():.1f}, max {y_cyc.max():.1f}  "
          f"(one matrix, so one value)")
    r = np.corrcoef(gg[:, 0].numpy(), np.log(y_fib))[0, 1]
    print(f"  corr(g, log ŷ) on the fibre: {r:+.3f}")

    # Extremes of ŷ over the fibre, bounded to the box: a 9-dim search.
    bounds = [(float(a), float(b)) for a, b in zip(lo_c, hi_c)] + [(g.min(), g.max())]
    d_lat = ctx.mu.shape[-1]

    def f(x, sense):
        x = torch.tensor(x, requires_grad=True)
        zz = x[:d_lat].expand(1, common.N, d_lat)
        v = -sense * common.log_y_hat(ctx, zz, x[d_lat:].reshape(1, 1))[0]
        (gr,) = torch.autograd.grad(v, x)
        return v.item(), gr.numpy()

    for sense, name in ((1, "max"), (-1, "min")):
        best = None
        for k in range(20):
            x0 = np.array([np.random.default_rng(k).uniform(a, b) for a, b in bounds])
            res = minimize(f, x0, args=(sense,), jac=True, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        print(f"  {name} ŷ over the fibre in the box: {np.exp(-sense * best.fun):.1f} "
              f"at g {best.x[-1]:+.3f}")


if __name__ == "__main__":
    main()
