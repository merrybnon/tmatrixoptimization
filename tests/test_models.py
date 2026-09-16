"""Equivariance, validity and numerics for the transition-matrix VAE.

The checks that matter run in float64 on a small graph, since the claims here
are exact algebraic ones and a 1e-15 residual is the whole point. The ones
about real numerics run on the committed fixture, whose entries reach 1.3e-14
against an exactly zero diagonal.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from tm_ml.ingest import load_drop
from tm_ml.models import TMVAE, TMVAEConfig, off_diagonal, tmvae_loss

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.npz"
TOL = 1e-5


@pytest.fixture(scope="module")
def real_T():
    matrices, targets = load_drop(FIXTURE)
    return (
        torch.from_numpy(matrices).float(),
        torch.from_numpy(np.log(targets)).float(),
    )


def make(**kw):
    torch.manual_seed(0)
    model = TMVAE(TMVAEConfig(**kw)).double().eval()
    return model


def random_T(B, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(B, n, n, generator=g, dtype=torch.float64)
    logits = logits.masked_fill(~off_diagonal(n, logits.device), float("-inf"))
    return torch.softmax(logits, dim=-1)


def permutation(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.eye(n, dtype=torch.float64)[torch.randperm(n, generator=g)]


def permute_T(P, T):
    return P @ T @ P.T


CONFIGS = [
    dict(n_nodes=6, d_model=32, n_heads=4, d_ff=64, encoder_layers=3, decoder_layers=2),
    dict(
        n_nodes=6, d_model=32, n_heads=4, d_ff=64, encoder_layers=3, decoder_layers=2,
        d_global=4,
    ),
]
IDS = ["node-only", "hybrid"]


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_shapes_and_validity(cfg):
    model, n = make(**cfg), cfg["n_nodes"]
    out = model(random_T(4, n))

    assert out.T_hat.shape == (4, n, n)
    assert out.mu.shape == (4, n, model.config.d_latent)
    assert out.z_global.shape == (4, model.config.d_global)
    assert out.log_y_hat.shape == (4,)

    diag = torch.diagonal(out.T_hat, dim1=-2, dim2=-1)
    assert (diag == 0).all(), "diagonal must be exactly zero, not merely small"
    assert torch.allclose(out.T_hat.sum(-1), torch.ones(4, n, dtype=torch.float64))


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_encoder_equivariance(cfg):
    model, n = make(**cfg), cfg["n_nodes"]
    T, P = random_T(4, n), permutation(n)

    mu, logvar, mu_g, _ = model.encode(T)
    mu_p, logvar_p, mu_g_p, _ = model.encode(permute_T(P, T))

    assert torch.allclose(mu_p, P @ mu, atol=TOL), (mu_p - P @ mu).abs().max()
    assert torch.allclose(logvar_p, P @ logvar, atol=TOL)
    # the graph-level latent is invariant, not equivariant
    assert torch.allclose(mu_g_p, mu_g, atol=TOL)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_decoder_equivariance(cfg):
    model, n = make(**cfg), cfg["n_nodes"]
    P = permutation(n)
    z = torch.randn(4, n, model.config.d_latent, dtype=torch.float64)
    z_g = torch.randn(4, model.config.d_global, dtype=torch.float64)

    T_hat, _ = model.decode(z, z_g)
    T_hat_p, _ = model.decode(P @ z, z_g)

    assert torch.allclose(T_hat_p, permute_T(P, T_hat), atol=TOL)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_predictor_invariance(cfg):
    model, n = make(**cfg), cfg["n_nodes"]
    P = permutation(n)
    z = torch.randn(4, n, model.config.d_latent, dtype=torch.float64)
    z_g = torch.randn(4, model.config.d_global, dtype=torch.float64)

    assert torch.allclose(model.predict(P @ z, z_g), model.predict(z, z_g), atol=TOL)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_end_to_end_equivariance_and_invariant_loss(cfg):
    model, n = make(**cfg), cfg["n_nodes"]
    T, P = random_T(4, n), permutation(n)
    log_y = torch.randn(4, dtype=torch.float64)

    out = model(T, sample=False)
    out_p = model(permute_T(P, T), sample=False)

    assert torch.allclose(out_p.T_hat, permute_T(P, out.T_hat), atol=TOL)
    assert torch.allclose(out_p.log_y_hat, out.log_y_hat, atol=TOL)

    loss = tmvae_loss(out, T, log_y, model.config)
    loss_p = tmvae_loss(out_p, permute_T(P, T), log_y, model.config)
    for field in ("total", "recon", "kl", "prop", "log_recon"):
        a, b = getattr(loss, field), getattr(loss_p, field)
        assert torch.allclose(a, b, atol=TOL), f"{field}: {a.item()} vs {b.item()}"


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_sampling_breaks_equivariance_pointwise(cfg):
    """Why sample=False is mandatory in the checks above, not a convenience."""
    model, n = make(**cfg), cfg["n_nodes"]
    T, P = random_T(4, n), permutation(n)

    torch.manual_seed(7)
    out = model(T, sample=True)
    torch.manual_seed(7)
    out_p = model(permute_T(P, T), sample=True)

    assert not torch.allclose(out_p.z, P @ out.z, atol=TOL)


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_ascent_commutes_with_relabelling(cfg):
    """The property the optimization rests on: grad y|_(PZ) = P . grad y|_Z."""
    model, n = make(**cfg), cfg["n_nodes"]
    T, P = random_T(4, n), permutation(n)

    def ascend(T0, steps=8, lr=0.1):
        mu, _, mu_g, _ = model.encode(T0)
        z, z_g = mu.detach().clone(), mu_g.detach().clone()
        for _ in range(steps):
            z.requires_grad_(True)
            z_g.requires_grad_(True)
            y = model.predict(z, z_g).sum()
            gz, gzg = torch.autograd.grad(y, (z, z_g), materialize_grads=True)
            z, z_g = (z + lr * gz).detach(), (z_g + lr * gzg).detach()
        return z, model.decode(z, z_g)[0]

    z, T_out = ascend(T)
    z_p, T_out_p = ascend(permute_T(P, T))

    assert torch.allclose(z_p, P @ z, atol=TOL), (z_p - P @ z).abs().max()
    assert torch.allclose(T_out_p, permute_T(P, T_out), atol=TOL)
    # and the ascent actually moved the prediction
    assert model.predict(z, torch.zeros(4, model.config.d_global, dtype=torch.float64)).mean() != 0


def test_masked_softmax_gives_the_diagonal_no_gradient():
    n = 5
    logits = torch.randn(1, n, n, dtype=torch.float64, requires_grad=True)
    masked = logits.masked_fill(~off_diagonal(n, logits.device), float("-inf"))
    torch.softmax(masked, dim=-1).pow(2).sum().backward()

    diag = torch.diagonal(logits.grad, dim1=-2, dim2=-1)
    assert (diag == 0).all()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_real_data_is_finite_and_trains(cfg, real_T):
    """Entries down to 1.3e-14 and an exact zero diagonal, on the real drop."""
    T, log_y = real_T
    model = TMVAE(TMVAEConfig(**{**cfg, "n_nodes": T.shape[-1]}))
    model.config.n_nodes = T.shape[-1]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    first = last = None
    for step in range(30):
        out = model(T)
        loss = tmvae_loss(out, T, log_y, model.config)
        assert torch.isfinite(loss.total), f"step {step}: {loss}"
        assert torch.isfinite(out.T_hat).all()
        assert torch.isfinite(out.mu).all()
        assert torch.isfinite(loss.log_recon)
        opt.zero_grad()
        loss.total.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{name} grad at step {step}"
        opt.step()
        first = first if first is not None else loss.total.item()
        last = loss.total.item()

    assert last < first, f"loss did not fall: {first} -> {last}"
    assert (torch.diagonal(out.T_hat, dim1=-2, dim2=-1) == 0).all()
    assert torch.allclose(out.T_hat.sum(-1), torch.ones_like(out.T_hat.sum(-1)), atol=1e-6)


def test_latent_gradient_is_dense(real_T):
    """Every latent entry gets signal, so ascent has 160 dimensions to use."""
    T, _ = real_T
    model = TMVAE(TMVAEConfig(n_nodes=T.shape[-1]))
    mu, _, mu_g, _ = model.encode(T[:4])
    z = mu.detach().clone().requires_grad_(True)
    model.predict(z, mu_g.detach()).sum().backward()

    assert (z.grad.abs() > 0).all()


def test_recon_is_zero_for_a_perfect_reconstruction():
    """Sanity on the KL: identical distributions cost nothing, despite the zeros."""
    n = 6
    T = random_T(2, n)
    config = TMVAEConfig(n_nodes=n)
    log_T = torch.log(T.clamp_min(config.log_eps)).masked_fill(
        ~off_diagonal(n, T.device), float("-inf")
    )

    class Out:
        log_T_hat = log_T
        mu = torch.zeros(2, n, 8, dtype=torch.float64)
        logvar = torch.zeros(2, n, 8, dtype=torch.float64)
        mu_global = torch.zeros(2, 0, dtype=torch.float64)
        logvar_global = torch.zeros(2, 0, dtype=torch.float64)
        log_y_hat = torch.zeros(2, dtype=torch.float64)

    loss = tmvae_loss(Out(), T, torch.zeros(2, dtype=torch.float64), config)
    assert abs(loss.recon.item()) < 1e-9
    assert loss.kl.item() == 0.0
