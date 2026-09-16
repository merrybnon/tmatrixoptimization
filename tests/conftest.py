"""Shared fixtures for the stages that read and write a run directory."""

import numpy as np
import pytest

from tm_ml import datasets, paths


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A processed drop and an empty results tree, both redirected into tmp.

    Redirecting both roots is what lets train, evaluate and visualize be
    exercised end to end without writing into the repo's own `results/`.
    """
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


@pytest.fixture
def tiny_config():
    """A resolved config small enough to train in a couple of seconds."""
    from tm_ml.config import defaults_for

    cfg = defaults_for("tmvae") | {
        "drop": "Tiny40", "epochs": 2, "batch_size": 8, "gpu": "cpu",
        "d_model": 32, "n_heads": 4, "d_ff": 32, "encoder_layers": 2,
        "decoder_layers": 1, "val_frac": 0.2, "test_frac": 0.2,
    }
    cfg["run"] = paths.run_path(cfg)
    cfg["config"] = None
    return cfg
