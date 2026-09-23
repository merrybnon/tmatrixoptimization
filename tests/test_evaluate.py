"""Metric arithmetic, and one train-then-evaluate pass end to end.

The statistical helpers are worth testing directly because they are the ones
whose failure is a plausible-looking number rather than an exception.
"""

import json

import numpy as np
import pytest
import torch

from tm_ml import evaluate, paths, train
from tm_ml.datasets import TargetScaler
from tm_ml.evaluate import (
    fit_line,
    dims_to,
    magnitude_table,
    participation,
    property_metrics,
    structure,
    validity_metrics,
)


def test_participation_counts_contributors():
    assert participation(torch.ones(8)) == pytest.approx(8.0)
    assert participation(torch.tensor([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert participation(torch.zeros(4)) == 0.0


def test_dims_to_reaches_the_fraction():
    spiked = torch.tensor([10.0, 0.1, 0.1, 0.1])
    assert dims_to(spiked, 0.9) == 1
    assert dims_to(torch.ones(10), 0.9) == 9


def test_structure_finds_the_active_dimensions():
    """Two informative coordinates buried in six that carry only prior noise."""
    torch.manual_seed(0)
    mu = torch.zeros(4000, 6)
    # Equal scales, so the participation ratio should land on exactly 2 —
    # unequal ones correctly give something between 1 and 2.
    mu[:, 0] = torch.randn(4000) * 2.0
    mu[:, 3] = torch.randn(4000) * 2.0
    logvar = torch.zeros(4000, 6)

    out = structure(mu, logvar, prefix="latent")

    assert out["latent_active_units"] == 2
    assert out["latent_active_fraction"] == pytest.approx(2 / 6)
    assert out["latent_pca_components_90"] == 2
    assert out["latent_kl_participation_ratio"] == pytest.approx(2.0, abs=0.1)


def test_structure_reports_unmeasurable_rather_than_dead():
    out = structure(torch.zeros(1, 4), torch.zeros(1, 4), prefix="latent")
    assert out["latent_active_units"] is None
    assert out["latent_active_fraction"] is None


def test_total_correlation_is_none_when_the_sample_is_too_narrow():
    out = structure(torch.randn(4, 8), torch.zeros(4, 8), prefix="latent")
    assert out["latent_total_correlation_gauss"] is None


def test_total_correlation_is_zero_for_a_scalar_latent():
    """d_latent = 1 has to evaluate, not raise.

    `corrcoef` of a single column returns a 0-dim scalar where slogdet wants a
    1x1 matrix, which took down every evaluate job of the latent-width sweep at
    d_latent = 1 after the training had already succeeded. Zero is the right
    answer rather than None: one dimension has nothing to be correlated with.
    """
    out = structure(torch.randn(64, 1), torch.zeros(64, 1), prefix="latent")
    assert out["latent_total_correlation_gauss"] == pytest.approx(0.0, abs=1e-9)
    assert out["latent_active_units"] is not None


def test_magnitude_table_buckets_every_entry():
    true = torch.rand(5000)
    error = torch.rand(5000)
    rows = magnitude_table(true, error, error)

    assert sum(r["count"] for r in rows) == 5000
    assert rows[0]["true_max"] <= rows[-1]["true_min"]


def test_property_metrics_on_a_perfect_prediction():
    scaler = TargetScaler(mean=5.7, std=0.7)
    y = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    data = {"y": y, "y_hat": y.clone()}

    out = property_metrics(data, scaler)
    assert out["test_r2"] == pytest.approx(1.0)
    assert out["test_resolution_slope"] == pytest.approx(1.0)
    assert out["test_calibration_slope"] == pytest.approx(1.0)
    assert out["test_rmse_transformed"] == pytest.approx(0.0)
    assert out["test_median_relative_error"] == pytest.approx(0.0, abs=1e-6)


def test_property_metrics_has_no_skill_when_predicting_the_train_mean():
    scaler = TargetScaler(mean=5.7, std=0.7)
    y = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    data = {"y": y, "y_hat": torch.zeros(4)}

    out = property_metrics(data, scaler)
    assert out["skill_vs_train_mean"] == pytest.approx(0.0, abs=1e-6)


def test_fit_line_reads_a_shrunk_range():
    """Predictions squeezed toward their own mean resolve less than the truth."""
    log_y = np.linspace(4.0, 7.0, 64)
    shrunk = log_y.mean() + 0.6 * (log_y - log_y.mean())

    slope, intercept, stderr = fit_line(log_y, shrunk)

    assert slope == pytest.approx(0.6)
    assert intercept == pytest.approx(log_y.mean() * 0.4)
    assert stderr == pytest.approx(0.0, abs=1e-9)

    perfect, _, _ = fit_line(log_y, log_y.copy())
    assert perfect == pytest.approx(1.0)


def test_fit_line_is_not_symmetric_in_its_arguments():
    """The reverse fit undoes a noiseless squeeze; with noise it would not."""
    log_y = np.linspace(4.0, 7.0, 64)
    shrunk = log_y.mean() + 0.6 * (log_y - log_y.mean())

    assert fit_line(shrunk, log_y)[0] == pytest.approx(1 / 0.6)


def test_fit_line_declines_a_constant_x():
    assert fit_line(np.full(8, 5.0), np.arange(8.0)) == (None, None, None)


def test_validity_metrics_see_a_broken_diagonal():
    T_hat = torch.full((2, 4, 4), 0.25)
    out = validity_metrics({"T_hat": T_hat})
    assert out["recon_diag_max_abs"] == pytest.approx(0.25)
    assert out["recon_row_sum_max_dev"] == pytest.approx(0.0, abs=1e-6)


def test_train_then_evaluate_end_to_end(wired, tiny_config):
    cfg = tiny_config
    train.train(cfg)
    metrics = evaluate.evaluate(cfg["run"])

    written = json.loads((paths.resolve(cfg["run"]) / paths.METRICS).read_text())
    assert written["run"] == cfg["run"]
    assert written["n_test"] == 8

    # Constraints, which must hold whatever two epochs did to the weights.
    assert metrics["recon_diag_max_abs"] == 0.0
    assert metrics["recon_row_sum_max_dev"] < 1e-5
    assert metrics["recon_all_finite"]

    # Every group is present and finite.
    for key in ("test_r2", "test_recon", "test_log_recon_mae", "test_kl",
                "latent_active_units", "latent_pca_components_90"):
        assert metrics[key] is not None
    assert len(metrics["_by_magnitude"]) > 0
    assert metrics["latent_dim_per_node"] == cfg["d_latent"]
    assert metrics["test_kl_global"] == 0.0, "no global latent by default"


def test_evaluate_without_a_checkpoint_says_so(wired):
    with pytest.raises(SystemExit, match="train the run first"):
        evaluate.evaluate("TMVAE_Nonexistent_Tiny40")


def test_evaluate_reads_a_checkpoint_from_before_the_loss_weights(wired, tiny_config):
    # Runs trained before lambda_log and lambda_recon existed saved neither in
    # either config; evaluation has to fall back to how they were trained.
    train.train(tiny_config)
    path = paths.resolve(tiny_config["run"]) / paths.CHECKPOINT
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("lambda_log", "lambda_recon"):
        checkpoint["config"].pop(key)
        checkpoint["model_config"].pop(key)
    torch.save(checkpoint, path)

    m = evaluate.evaluate(tiny_config["run"])
    assert (m["lambda_log"], m["lambda_recon"]) == (0.0, 1.0)
