"""Is U a maximum of ŷ(enc T) over matrices? The gradient and Hessian there.

T = rowsoftmax(L) with the diagonal masked, L the 380 off-diagonal logits.
Adding a constant to a row of L leaves T alone, so 20 directions are null and
361 are real. F(L) = log ŷ(enc T(L)) is invariant under relabelling, because
the encoder is equivariant and the predictor pools. U is fixed by every
relabelling, so ∇F(U) is too, and the only invariant off-diagonal pattern is a
constant, which a row softmax ignores: ∇F(U) = 0 in every real direction, for
any model built this way. Whether U is a maximum is the Hessian's to say.

The true effective population is invariant under relabelling as well, so the
same holds for it: U is a critical point of the real target, and the slides
put it at the top.
"""

import argparse

import numpy as np
import torch

from tm_ml.models import off_diagonal
from tm_ml.uniform import common


def logits_to_T(flat):
    n = common.N
    off = off_diagonal(n, flat.device)
    L = flat.new_full((n, n), float("-inf"))
    L = L.masked_scatter(off, flat)
    return torch.softmax(L, -1)


def F(ctx, flat):
    return common.cycle_log_y_hat(ctx, logits_to_T(flat)[None])[0]


def row_constant_basis():
    """The 20 directions that add a constant to one row's logits."""
    n = common.N
    off = off_diagonal(n, "cpu")
    B = []
    for i in range(n):
        M = torch.zeros(n, n, dtype=torch.float64)
        M[i] = 1.0
        B.append(M[off])
    B = torch.stack(B, 1)
    return B / B.norm(dim=0)


def main():
    args = common.run_arg(argparse.ArgumentParser()).parse_args()
    ctx = common.load(args.run)
    x = torch.zeros(common.N * (common.N - 1), dtype=torch.float64)  # U

    xg = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(F(ctx, xg), xg)
    print(f"log ŷ(enc U) = {F(ctx, x).item():.4f}, ŷ = {np.exp(F(ctx, x).item()):.1f}")
    print(f"‖∇F(U)‖ over the 380 logits = {g.norm():.3e}")

    H = torch.autograd.functional.hessian(lambda v: F(ctx, v), x)
    # Project out the 20 row-constant directions, which T does not see.
    B = row_constant_basis()
    P = torch.eye(len(x), dtype=torch.float64) - B @ B.T
    Hp = P @ H @ P
    w = np.linalg.eigvalsh(Hp.numpy())
    # 20 exact zeros from the projection; drop the 20 smallest in magnitude.
    real = np.sort(w[np.argsort(np.abs(w))[20:]])
    print(f"Hessian on the 361 real directions: {(real > 0).sum()} positive, "
          f"{(real < 0).sum()} negative")
    vals, counts = np.unique(np.round(real, 5), return_counts=True)
    print("distinct eigenvalues (rounded) and multiplicities:")
    for v, c in zip(vals, counts):
        print(f"  {v:+.5f} × {c}")

    # The ascent direction, if any: the top eigenvector, applied as T(s·v).
    W, V = np.linalg.eigh(Hp.numpy())
    v = torch.from_numpy(V[:, -1])
    print("\nalong the top eigenvector, ŷ(enc T(s·v)):")
    for s in (0.0, 0.5, 1, 2, 4, 8):
        T = logits_to_T(s * v)[None]
        print(f"  s {s:4.1f}  ŷ {np.exp(F(ctx, s * v).item()):8.1f}   KL to U "
              f"{common.nonuniformity(T).item():.3e}")
    M = np.zeros((common.N, common.N))
    M[off_diagonal(common.N, "cpu").numpy()] = V[:, -1]
    print("  top eigenvector as a logit matrix, row sums and its symmetric part's share:")
    print(f"    max |row sum| {np.abs(M.sum(1)).max():.2e}, "
          f"‖sym‖/‖M‖ {np.linalg.norm((M + M.T) / 2) / np.linalg.norm(M):.3f}")

    np.save(ctx.out_dir / "hessian_U_eigenvalues.npy", real)


if __name__ == "__main__":
    main()
