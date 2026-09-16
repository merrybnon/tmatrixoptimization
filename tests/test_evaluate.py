"""Metric arithmetic, and one train-then-evaluate pass end to end.

The statistical helpers are worth testing directly because they are the ones
whose failure is a plausible-looking number rather than an exception.
"""

import json

import numpy as np
import pytest
import torch

from tm_ml import datasets, evaluate, paths, train
from tm_ml.config import defaults_for
from tm_ml.datasets import TargetScaler
from tm_ml.evaluate import (
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
    assert out["test_rmse_transformed"] == pytest.approx(0.0)
    assert out["test_median_relative_error"] == pytest.approx(0.0, abs=1e-6)


def test_property_metrics_has_no_skill_when_predicting_the_train_mean():
    scaler = TargetScaler(mean=5.7, std=0.7)
    y = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    data = {"y": y, "y_hat": torch.zeros(4)}

    out = property_metrics(data, scaler)
    assert out["skill_vs_train_mean"] == pytest.approx(0.0, abs=1e-6)


def test_validity_metrics_see_a_broken_diagonal():
    T_hat = torch.full((2, 4, 4), 0.25)
    out = validity_metrics({"T_hat": T_hat})
    assert out["recon_diag_max_abs"] == pytest.approx(0.25)
    assert out["recon_row_sum_max_dev"] == pytest.approx(0.0, abs=1e-6)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A processed drop and an empty results tree, both redirected to tmp."""
    rng = np.random.default_rng(0)
    eye = np.eye(5, dtype=bool)
    logits = rng.normal(size=(40, 5, 5))
    logits[:, eye] = -np.inf
    matrices = np.exp(logits - logits.max(-1, keepdims=True))
    matrices[:, eye] = 0.0
    matrices /= matrices.sum(-1, keepdims=True)

    processed = tmp_path / "processed"
    (processed / "Tiny40").mkdir(parents=True)
    np.save(processed / "Tiny40" / "matrices.npy", matrices.astype(np.float32))
    np.save(
        processed / "Tiny40" / "targets.npy",
        np.exp(rng.normal(6, 0.7, 40)).astype(np.float32),
    )

    monkeypatch.setattr(datasets, "PROCESSED_DIR", processed)
    monkeypatch.setattr(paths, "RESULTS_ROOT", tmp_path / "results")
    return tmp_path


def test_train_then_evaluate_end_to_end(wired):
    cfg = defaults_for("tmvae") | {
        "drop": "Tiny40", "epochs": 2, "batch_size": 8, "gpu": "cpu",
        "d_model": 32, "n_heads": 4, "d_ff": 32, "encoder_layers": 2,
        "decoder_layers": 1, "val_frac": 0.2, "test_frac": 0.2,
    }
    cfg["run"] = paths.run_name(cfg)
    cfg["config"] = None

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
