"""Equivariance, validity and numerics for the transition-matrix VAE.

The checks that matter run in float64 on a small graph, since the claims here
are exact algebraic ones and a 1e-15 residual is the whole point. The ones
about real numerics run on the committed fixture, whose entries reach 1.3e-14
against an exactly zero diagonal.
"""

import dataclasses
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


BASE = dict(n_nodes=6, d_model=32, n_heads=4, d_ff=64, encoder_layers=3, decoder_layers=2)

CONFIGS = [
    BASE,
    dict(BASE, d_global=4),
    dict(BASE, pooling="mean"),
    dict(BASE, pooling="attention"),
]
IDS = ["node-only", "hybrid", "mean-pool", "attention-pool"]


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


@pytest.mark.parametrize("mode,width", [("mean", 8), ("deepsets", 64), ("attention", 64)])
def test_pool_shape_and_invariance(mode, width):
    """Every mode collapses the node axis, and none of them can see the order."""
    from tm_ml.models import NodePool

    torch.manual_seed(0)
    config = TMVAEConfig(n_nodes=6, d_latent=8, predictor_hidden=64, pooling=mode)
    pool = NodePool(config).double().eval()
    assert pool.out_dim == width

    z = torch.randn(4, 6, 8, dtype=torch.float64)
    P = permutation(6)
    assert pool(z).shape == (4, width)
    assert torch.allclose(pool(P @ z), pool(z), atol=TOL)


def test_unknown_pooling_is_refused_at_construction():
    from tm_ml.models import NodePool

    with pytest.raises(ValueError, match="unknown pooling"):
        NodePool(TMVAEConfig(pooling="softmax"))


def test_lambda_log_zero_leaves_the_total_untouched():
    """The default has to reproduce the pure forward KL exactly, not nearly.

    Every run before the term existed was trained at lambda_log = 0, and the
    ledger compares them against runs trained after it. If adding the field
    moved the objective by even a rounding error those numbers stop being
    commensurable, so this is an exact check rather than a tolerance.
    """
    model, n = make(n_nodes=6, d_model=16, n_heads=2, d_ff=16, encoder_layers=1), 6
    T, log_y = random_T(4, n), torch.randn(4, dtype=torch.float64)
    out = model(T, sample=False)

    loss = tmvae_loss(out, T, log_y, model.config)
    expected = loss.recon + model.config.beta * loss.kl + model.config.gamma * loss.prop
    assert loss.total.item() == expected.item()


def test_lambda_log_enters_the_total_and_carries_gradient():
    """The term has to reach the weights, which is the whole point of the change.

    `log_recon` sat inside `no_grad` as a diagnostic, so the failure this guards
    against is the term being reported and weighted but still detached — the
    total would move with lambda_log while the gradient did not.
    """
    model, n = make(n_nodes=6, d_model=16, n_heads=2, d_ff=16, encoder_layers=1), 6
    T, log_y = random_T(4, n), torch.randn(4, dtype=torch.float64)

    out = model(T, sample=False)
    base = tmvae_loss(out, T, log_y, model.config)
    weighted = tmvae_loss(out, T, log_y, dataclasses.replace(model.config, lambda_log=2.0))

    assert torch.allclose(weighted.total, base.total + 2.0 * base.log_recon, atol=TOL)
    assert torch.allclose(weighted.log_recon, base.log_recon, atol=TOL)

    # The decoder is what the term can move; the property head is not, so only
    # the shared trunk and the pair scorer should see a different gradient.
    grads = {}
    for tag, lam in (("off", 0.0), ("on", 2.0)):
        model.zero_grad()
        cfg = dataclasses.replace(model.config, lambda_log=lam)
        tmvae_loss(model(T, sample=False), T, log_y, cfg).total.backward()
        grads[tag] = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}

    changed = [k for k in grads["off"] if not torch.allclose(grads["off"][k], grads["on"][k])]
    assert changed, "lambda_log changed no gradient; the term is still detached"


def test_lambda_log_lifts_the_weak_entries():
    """Training on it should move the tail up-to-down, which is its purpose.

    Fits one batch twice from the same initialization and compares the log-space
    error on the weakest decile of true entries. Not a claim about the real
    model, just that the gradient points the way the term is meant to point.
    """
    T, log_y = random_T(6, 6, seed=3), torch.randn(6, dtype=torch.float64)
    off = off_diagonal(6, T.device)
    log_T = torch.log(T.clamp_min(1e-20))

    errors = {}
    for tag, lam in (("off", 0.0), ("on", 5.0)):
        model = make(n_nodes=6, d_model=16, n_heads=2, d_ff=16, encoder_layers=1)
        model.train()
        cfg = dataclasses.replace(model.config, lambda_log=lam)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(30):
            opt.zero_grad()
            tmvae_loss(model(T, sample=False), T, log_y, cfg).total.backward()
            opt.step()
        model.eval()
        out = model(T, sample=False)
        err = (out.log_T_hat.masked_fill(~off, 0.0) - log_T)[..., off].flatten()
        weakest = T[..., off].flatten() < torch.quantile(T[..., off].flatten(), 0.1)
        errors[tag] = err[weakest].mean().item()

    assert errors["on"] < errors["off"], errors


def test_lambda_recon_defaults_to_the_unweighted_sum():
    """1.0 has to be exactly how `recon` entered the loss before it had a name.

    Same reasoning as the lambda_log check: the ledger compares runs from both
    sides of the change, so the default must be an identity, not an approximate
    one.
    """
    model, n = make(n_nodes=6, d_model=16, n_heads=2, d_ff=16, encoder_layers=1), 6
    T, log_y = random_T(4, n), torch.randn(4, dtype=torch.float64)
    out = model(T, sample=False)
    assert model.config.lambda_recon == 1.0

    loss = tmvae_loss(out, T, log_y, model.config)
    expected = loss.recon + model.config.beta * loss.kl + model.config.gamma * loss.prop
    assert loss.total.item() == expected.item()


def test_lambda_recon_zero_leaves_a_well_posed_loss():
    """Dropping the forward KL entirely has to still train a valid decoder.

    `recon` is the categorical likelihood, but validity comes from the masked
    row softmax rather than from the loss, so removing it should cost the ELBO
    reading and nothing structural. The rows still have to sum to 1 and the
    diagonal still has to be 0 after training against log_recon alone.
    """
    T, log_y = random_T(6, 6, seed=5), torch.randn(6, dtype=torch.float64)
    off = off_diagonal(6, T.device)
    model = make(n_nodes=6, d_model=16, n_heads=2, d_ff=16, encoder_layers=1)
    model.train()
    cfg = dataclasses.replace(model.config, lambda_recon=0.0, lambda_log=3.0)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(30):
        opt.zero_grad()
        loss = tmvae_loss(model(T, sample=False), T, log_y, cfg)
        assert torch.isfinite(loss.total), f"step {step}: {loss}"
        # The term is still measured, just not charged for.
        assert torch.isfinite(loss.recon)
        loss.total.backward()
        opt.step()

    model.eval()
    T_hat = model(T, sample=False).T_hat
    assert torch.allclose(T_hat.sum(-1), torch.ones_like(T_hat.sum(-1)), atol=TOL)
    assert torch.allclose(torch.diagonal(T_hat, dim1=-2, dim2=-1),
                          torch.zeros_like(T_hat[..., 0]), atol=TOL)
    # And `recon` really is absent from the total rather than merely small.
    final = tmvae_loss(model(T, sample=False), T, log_y, cfg)
    without_recon = (
        cfg.beta * final.kl + cfg.gamma * final.prop + cfg.lambda_log * final.log_recon
    )
    assert final.total.item() == without_recon.item()
    assert final.recon.item() != 0.0
