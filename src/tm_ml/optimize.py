"""Climb the predicted decay exponent in latent space, one graph at a time, with BFGS.

    pixi run -e ml python -m tm_ml.optimize --run <run> --lam 0 0.01 0.1 1

Starts from a test graph's posterior mean and minimizes

    f(x) = −log ŷ(x) + (λ/2)·‖x‖²

over x, the 20 × d_latent node latent flattened with the global appended. The
second term is −log N(0, I) up to a constant, the VAE's prior: the decoder and
predictor only ever saw latents the KL term kept near it, so the penalty
charges the optimizer for leaving the region the model knows. λ = 0 is the
unregularized run, where nothing stops the predictor extrapolating.

Only the predictor enters f. The decoder is the caller's, to draw the path.

The CLI prints each path so λ can be chosen by reading it: ŷ, how far the
latent has moved, the gradient, and what BFGS's inverse-Hessian estimate
learned by the end.
"""

import argparse
import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

N_GRAPHS = 3


@dataclass
class Ascent:
    """One BFGS path, every accepted iterate from the start to the exit.

    ``z`` is ``(k, n_nodes, d_latent)`` and ``z_global`` ``(k, d_global)``, with
    k the accepted iterates plus the start; line-search trial points are not
    iterates and are not kept. ``hess_inv`` is BFGS's final estimate of H⁻¹.
    """

    z: np.ndarray
    z_global: np.ndarray
    log_y_hat: np.ndarray
    f: np.ndarray
    grad_norm: np.ndarray
    lam: float
    nit: int
    nfev: int
    message: str
    hess_inv: np.ndarray


def float64_copy(model):
    """The model on the cpu in float64, for a line search that reads small differences.

    BFGS's Wolfe checks compare f at nearby points; float32's 1e-7 relative
    noise makes them fail early with "precision loss" long before the optimum.
    """
    return copy.deepcopy(model).double().cpu().eval()


def objective(model64, scaler, n_nodes, d_latent, lam):
    """``x -> (f, ∇f)`` in float64 numpy, the form ``minimize(jac=True)`` takes.

    The predictor emits standardized log y; undoing the scaler is monotone, so
    it moves no optimum, but it puts f in e-folds of the decay exponent, which
    is what gives λ units.
    """
    split = n_nodes * d_latent

    def f(x):
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        z = x[:split].reshape(1, n_nodes, d_latent)
        z_global = x[split:].reshape(1, -1)
        log_y_hat = model64.predict(z, z_global)[0] * scaler.std + scaler.mean
        value = -log_y_hat + 0.5 * lam * x.pow(2).sum()
        (grad,) = torch.autograd.grad(value, x)
        return value.item(), grad.numpy()

    return f


def ascend(model, scaler, mu, mu_global, lam, maxiter=200, gtol=1e-5):
    """BFGS from one graph's encoding ``(mu, mu_global)``, both numpy, unbatched."""
    n_nodes, d_latent = mu.shape
    model64 = float64_copy(model)
    fun = objective(model64, scaler, n_nodes, d_latent, lam)

    x0 = np.concatenate([mu.ravel(), mu_global.ravel()]).astype(np.float64)
    path = [x0]
    result = minimize(fun, x0, jac=True, method="BFGS", callback=lambda xk: path.append(xk.copy()),
                      options={"maxiter": maxiter, "gtol": gtol})
    path = np.stack(path)

    # f and its gradient again at each kept iterate: BFGS evaluates them but
    # does not hand them back, and one pass over the path costs nothing.
    values, grads = zip(*(fun(x) for x in path))
    values = np.array(values)
    penalty = 0.5 * lam * (path ** 2).sum(1)
    split = n_nodes * d_latent
    return Ascent(
        z=path[:, :split].reshape(-1, n_nodes, d_latent),
        z_global=path[:, split:],
        log_y_hat=penalty - values,
        f=values,
        grad_norm=np.linalg.norm(np.stack(grads), axis=1),
        lam=lam,
        nit=result.nit,
        nfev=result.nfev,
        message=result.message,
        hess_inv=np.asarray(result.hess_inv),
    )


def report(ascent, graph, true_decay):
    """The path as a table, then how it ended and what H⁻¹ looks like."""
    x = np.concatenate([ascent.z.reshape(len(ascent.z), -1), ascent.z_global], axis=1)
    moved = np.linalg.norm(x - x[0], axis=1)
    norm = np.linalg.norm(x, axis=1)
    print(f"\ntest graph {graph}, true decay exponent {true_decay:.1f}, λ = {ascent.lam:g}")
    print(f"  {'iter':>4}  {'ŷ':>10}  {'log ŷ':>7}  {'f':>9}  {'‖x‖':>7}  "
          f"{'‖x − x₀‖':>9}  {'‖∇f‖':>9}")
    for k in range(len(x)):
        print(f"  {k:>4}  {np.exp(ascent.log_y_hat[k]):>10.1f}  {ascent.log_y_hat[k]:>7.3f}  "
              f"{ascent.f[k]:>9.4f}  {norm[k]:>7.3f}  {moved[k]:>9.3f}  "
              f"{ascent.grad_norm[k]:>9.2e}")

    # B0 = I, and each iteration's rank-two update reshapes it along the one
    # new direction it stepped in, so eigenvalues still at 1 are directions
    # BFGS never learned anything about.
    eigenvalues = np.linalg.eigvalsh(ascent.hess_inv)
    learned = int((np.abs(eigenvalues - 1.0) > 1e-3).sum())
    print(f"  exit after {ascent.nit} iterations, {ascent.nfev} evaluations: {ascent.message}")
    print(f"  H⁻¹ estimate: eigenvalues {eigenvalues.min():.3g} .. {eigenvalues.max():.3g}, "
          f"{learned} of {len(eigenvalues)} moved off 1")


def main():
    from tm_ml import evaluate as evaluate_module
    from tm_ml.models import TMVAEConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory name under results/")
    parser.add_argument("--lam", type=float, nargs="+", default=[0.0],
                        help="prior-penalty weights λ; each is run on every graph")
    parser.add_argument("--graphs", type=int, default=N_GRAPHS,
                        help="how many test graphs, from the first")
    parser.add_argument("--maxiter", type=int, default=200)
    args = parser.parse_args()

    _, checkpoint, cfg, model, splits, scaler, device = evaluate_module.load_run(args.run)
    data = evaluate_module.collect(model, splits.test, TMVAEConfig(**checkpoint["model_config"]),
                                   device, cfg["batch_size"])
    mu, mu_global = data["mu"].numpy(), data["mu_global"].numpy()
    true = np.exp(scaler.inverse(data["y"].numpy()))

    for graph in range(min(args.graphs, len(mu))):
        for lam in args.lam:
            report(ascend(model, scaler, mu[graph], mu_global[graph], lam, args.maxiter),
                   graph, true[graph])


if __name__ == "__main__":
    main()
