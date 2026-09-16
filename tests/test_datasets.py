"""Splits and the target transform.

The leak matters more than it looks: the scaler's constants are fitted on the
training split alone, and fitting them on everything would quietly put the test
split's location and scale into the training objective.
"""

import numpy as np
import pytest
import torch

from tm_ml.config import defaults_for
from tm_ml.datasets import TargetScaler, load_splits, read_processed, split_indices


@pytest.fixture
def drop(tmp_path):
    """A processed drop of 100 matrices with a wide, skewed target."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(100, 6, 6))
    eye = np.eye(6, dtype=bool)
    logits[:, eye] = -np.inf
    matrices = np.exp(logits - logits.max(-1, keepdims=True))
    matrices[:, eye] = 0.0
    matrices /= matrices.sum(-1, keepdims=True)

    directory = tmp_path / "Fake100"
    directory.mkdir()
    np.save(directory / "matrices.npy", matrices.astype(np.float32))
    np.save(directory / "targets.npy", np.exp(rng.normal(6, 0.7, 100)).astype(np.float32))
    return tmp_path


def config(**kw):
    return defaults_for("tmvae") | {"drop": "Fake100"} | kw


def test_missing_drop_says_how_to_make_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="pixi run ingest --drop Nope"):
        read_processed("Nope", tmp_path)


def test_splits_partition_every_example(drop):
    indices = split_indices(100, 0.1, 0.1, seed=0)
    assert sorted(np.concatenate(list(indices.values()))) == list(range(100))
    assert {k: len(v) for k, v in indices.items()} == {"train": 80, "val": 10, "test": 10}


def test_splits_are_a_pure_function_of_the_seed():
    first = split_indices(100, 0.1, 0.1, seed=3)
    assert (split_indices(100, 0.1, 0.1, seed=3)["train"] == first["train"]).all()
    assert not (split_indices(100, 0.1, 0.1, seed=4)["train"] == first["train"]).all()


def test_splits_that_leave_no_training_data_are_refused():
    with pytest.raises(ValueError, match="none left to train on"):
        split_indices(10, 0.5, 0.5, seed=0)


def test_scaler_is_fitted_on_the_training_split_alone(drop):
    splits = load_splits(config(), root=drop)
    _, targets = read_processed("Fake100", drop)
    log_y = np.log(targets)

    train_indices = split_indices(100, 0.1, 0.1, seed=0)["train"]
    assert splits.scaler.mean == pytest.approx(log_y[train_indices].mean(), rel=1e-5)
    assert splits.scaler.mean != pytest.approx(log_y.mean(), rel=1e-9), "fitted on everything"


def test_standardized_targets_are_centred_on_the_training_split(drop):
    splits = load_splits(config(), root=drop)
    train_y = splits.train.tensors[1]

    assert float(train_y.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(train_y.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


def test_disabling_standardizing_leaves_log_targets(drop):
    splits = load_splits(config(standardize_target=False), root=drop)
    assert splits.scaler.as_dict() == {"mean": 0.0, "std": 1.0}
    assert float(splits.train.tensors[1].mean()) > 4.0, "should still be log y"


def test_the_scaler_round_trips():
    scaler = TargetScaler(mean=5.73, std=0.69)
    log_y = np.array([3.9, 5.7, 7.6])
    assert scaler.inverse(scaler.transform(log_y)) == pytest.approx(log_y)


def test_shapes_and_dtypes(drop):
    splits = load_splits(config(), root=drop)
    T, y = splits.train[0]

    assert splits.n_nodes == 6
    assert T.shape == (6, 6) and T.dtype == torch.float32
    assert y.shape == () and y.dtype == torch.float32
    assert torch.allclose(T.sum(-1), torch.ones(6), atol=1e-6)
    assert (torch.diagonal(T) == 0).all()


def test_nonpositive_targets_are_refused(drop, tmp_path):
    targets = np.load(drop / "Fake100" / "targets.npy")
    targets[7] = 0.0
    np.save(drop / "Fake100" / "targets.npy", targets)

    with pytest.raises(ValueError, match="index 7"):
        load_splits(config(), root=drop)
