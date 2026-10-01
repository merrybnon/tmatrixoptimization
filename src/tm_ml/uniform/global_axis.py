"""What happens far out along the global dimension.

For a few test graphs, hold the node latents at the graph's encoding and set
the global to g over a long range, far past the data's −1.6 … +1.7. At each g:

- ŷ(z), what the latent optimizer climbs
- the decoded T̂ and its mean row KL to U
- ŷ(enc T̂), the prediction for the matrix actually decoded

Then the mechanism. ``from_latent`` is affine in [z_i ; g], so at large |g|
every node's first hidden state is g·w_g plus an O(1) node term, and the
pre-norm decoder layers and the final LayerNorm see only its direction. The
node terms shrink relative to g, which is what makes the nodes alike.
"""

import argparse

import numpy as np
import torch

from tm_ml.uniform import common

G = np.concatenate([-np.logspace(3, -1, 13), [0.0], np.logspace(-1, 3, 13)])
GRAPHS = 4


def main():
    args = common.run_arg(argparse.ArgumentParser()).parse_args()
    ctx = common.load(args.run)
    test = np.flatnonzero(ctx.split == "test")[:GRAPHS]
    g_data = ctx.mu_global[:, 0].numpy()
    print(f"global over the {len(g_data)} graphs: {g_data.min():+.3f} … {g_data.max():+.3f}")

    for i in test:
        print(f"\ngraph {i}, y {np.exp(ctx.log_y[i]):.0f}, own global {g_data[i]:+.3f}")
        print(f"  {'g':>9}  {'ŷ(z)':>10}  {'KL(T̂‖U)':>9}  {'ŷ(enc T̂)':>9}  {'node spread h₀/‖h₀‖':>20}")
        z = ctx.mu[i:i + 1].expand(len(G), -1, -1)
        gg = torch.tensor(G, dtype=torch.float64)[:, None]
        with torch.no_grad():
            y_z = np.exp(common.log_y_hat(ctx, z, gg).numpy())
            T_hat = common.decode(ctx, z, gg)
            kl = common.nonuniformity(T_hat).numpy()
            y_c = np.exp(common.cycle_log_y_hat(ctx, T_hat).numpy())
            h0 = ctx.model.from_latent(ctx.model._broadcast(z, gg))
            h0 = h0 / h0.norm(dim=-1, keepdim=True)
            spread = h0.std(1).norm(dim=-1).numpy()
        for k, g in enumerate(G):
            print(f"  {g:>+9.2f}  {y_z[k]:>10.4g}  {kl[k]:>9.3e}  {y_c[k]:>9.1f}  {spread[k]:>20.3e}")

    # The limit itself: as g → ±∞ the decoded matrix tends to whatever the
    # decoder makes of a single direction ±w_g on every node, the same for
    # every graph, which is a matrix with identical nodes, i.e. U.
    w = ctx.model.from_latent.weight[:, -1]
    print(f"\n‖w_g‖ = {w.norm():.3f}, against ‖W_z‖ per node dim "
          f"{ctx.model.from_latent.weight[:, :-1].norm(dim=0).numpy().round(3)}")


if __name__ == "__main__":
    main()
