"""BFGS in latent space: the gradient it is handed, and the path it takes.

A fresh float64 model on a small graph, as in `test_models.py`: the claims are
about the optimizer and its symmetry, not about a trained predictor.
"""

import numpy as np
import pytest
import torch
from scipy.optimize import check_grad

from tm_ml.datasets import TargetScaler
from tm_ml.models import TMVAE, TMVAEConfig, off_diagonal
from tm_ml.optimize import ascend, float64_copy, objective

N = 6
SCALER = TargetScaler(mean=6.0, std=0.7)


@pytest.fixture(scope="module")
def encoded():
    torch.manual_seed(0)
    model = TMVAE(TMVAEConfig(n_nodes=N, d_model=32, n_heads=4, d_ff=64,
                              encoder_layers=2, decoder_layers=1, d_global=2)).double().eval()
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(1, N, N, generator=g, dtype=torch.float64)
    T = torch.softmax(logits.masked_fill(~off_diagonal(N, logits.device), float("-inf")), -1)
    with torch.no_grad():
        mu, _, mu_global, _ = model.encode(T)
    return model, mu[0].numpy(), mu_global[0].numpy()


@pytest.mark.parametrize("lam", [0.0, 0.5])
def test_gradient_matches_finite_differences(encoded, lam):
    model, mu, mu_global = encoded
    fun = objective(float64_copy(model), SCALER, *mu.shape, lam)
    x = np.concatenate([mu.ravel(), mu_global])
    error = check_grad(lambda v: fun(v)[0], lambda v: fun(v)[1], x)
    assert error < 1e-6 * max(1.0, np.linalg.norm(fun(x)[1]))


@pytest.mark.parametrize("lam", [0.0, 0.5])
def test_path_starts_at_the_encoding_and_never_goes_uphill(encoded, lam):
    model, mu, mu_global = encoded
    path = ascend(model, SCALER, mu, mu_global, lam, maxiter=30)
    np.testing.assert_array_equal(path.z[0], mu)
    np.testing.assert_array_equal(path.z_global[0], mu_global)
    assert len(path.z) == path.nit + 1
    assert np.all(np.diff(path.f) <= 1e-12)
    assert path.log_y_hat[-1] > path.log_y_hat[0]


def test_the_prior_penalty_keeps_the_latent_closer_to_the_origin(encoded):
    model, mu, mu_global = encoded

    def final_norm(lam):
        path = ascend(model, SCALER, mu, mu_global, lam, maxiter=30)
        return np.linalg.norm(np.r_[path.z[-1].ravel(), path.z_global[-1]])

    assert final_norm(1.0) < final_norm(0.0)


def test_bfgs_commutes_with_relabelling(encoded):
    """B0 = I and every s, y permute with the start, so B picks up P B Pᵀ."""
    model, mu, mu_global = encoded
    order = np.random.default_rng(1).permutation(N)
    path = ascend(model, SCALER, mu, mu_global, 0.1, maxiter=10)
    relabelled = ascend(model, SCALER, mu[order], mu_global, 0.1, maxiter=10)

    assert relabelled.nit == path.nit
    np.testing.assert_allclose(relabelled.z, path.z[:, order], atol=1e-8)
    np.testing.assert_allclose(relabelled.z_global, path.z_global, atol=1e-8)
