"""Serve ``(T, target)`` pairs from a processed drop, split and transformed.

Reads only ``data/processed/<drop>/``, never a raw drop — `ingest.py` owns that
boundary. A drop is 1000 20x20 float32 matrices, 1.6 MB, so everything is held
in memory as one tensor and there is no lazy loading to get wrong.

This module owns the target transform, which is not the model's business:

- the **log** is always taken. The decay exponent runs 48.8 to 2041.6 with skew
  1.86, and squared error on the log is a lognormal likelihood on the exponent,
  which is relative error — being off by 50 on a target of 2000 should not cost
  what being off by 50 on a target of 60 does.
- **standardizing** is optional and, when on, is fitted on the training split
  alone. That is why it lives here rather than in the model: the model sees a
  batch and has no idea which examples are training data. The constants ride in
  the checkpoint so evaluation can report in real units.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import TensorDataset

from tm_ml.ingest import PROCESSED_DIR


@dataclass
class TargetScaler:
    """Affine map on log y, and its inverse. Identity when standardizing is off.

    Keeping the disabled case as mean 0 / std 1 rather than as a branch means
    every caller has one code path, and a checkpoint always carries usable
    constants.
    """

    mean: float
    std: float

    @classmethod
    def fit(cls, log_y, enabled):
        if not enabled:
            return cls(mean=0.0, std=1.0)
        std = float(log_y.std())
        if std == 0:
            raise ValueError("every target is identical; nothing to standardize")
        return cls(mean=float(log_y.mean()), std=std)

    def transform(self, log_y):
        return (log_y - self.mean) / self.std

    def inverse(self, scaled):
        """Back to log y. Compose with exp for the decay exponent itself."""
        return scaled * self.std + self.mean

    def as_dict(self):
        return {"mean": self.mean, "std": self.std}


@dataclass
class Splits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    scaler: TargetScaler
    n_nodes: int
    sizes: dict


def read_processed(drop, root=None):
    """The canonical arrays for a drop, as float64 for the checks below."""
    directory = (PROCESSED_DIR if root is None else root) / drop
    matrices = directory / "matrices.npy"
    if not matrices.exists():
        raise FileNotFoundError(
            f"{matrices} does not exist; run `pixi run ingest --drop {drop}` first"
        )
    return np.load(matrices), np.load(directory / "targets.npy")


def split_indices(n, val_frac, test_frac, seed):
    """Deterministic train/val/test indices from one seed.

    Sizes are taken off the end so rounding never leaves the training split
    short of examples, and the whole thing is a pure function of ``seed``, so
    two runs sharing a seed share a split and are comparable.
    """
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    if n_test + n_val >= n:
        raise ValueError(f"val and test take {n_val + n_test} of {n} examples; none left to train on")

    order = np.random.default_rng(seed).permutation(n)
    return {
        "test": order[:n_test],
        "val": order[n_test:n_test + n_val],
        "train": order[n_test + n_val:],
    }


def load_splits(cfg, root=None):
    """Everything a training run needs from disk, in one object."""
    matrices, targets = read_processed(cfg["drop"], root)

    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"expected (N, n, n) matrices, got {matrices.shape}")
    if (targets <= 0).any():
        bad = int(np.argmax(targets <= 0))
        raise ValueError(
            f"targets must be positive to take a log; index {bad} is {targets[bad]!r}"
        )

    indices = split_indices(len(matrices), cfg["val_frac"], cfg["test_frac"], cfg["seed"])
    log_y = np.log(targets)
    # Fitted on train alone. Fitting on everything would leak the test split's
    # location and scale into training.
    scaler = TargetScaler.fit(log_y[indices["train"]], cfg["standardize_target"])

    T = torch.from_numpy(matrices).float()
    y = torch.from_numpy(scaler.transform(log_y)).float()

    parts = {
        name: TensorDataset(T[idx], y[idx]) for name, idx in indices.items()
    }
    return Splits(
        train=parts["train"],
        val=parts["val"],
        test=parts["test"],
        scaler=scaler,
        n_nodes=int(matrices.shape[1]),
        sizes={name: len(idx) for name, idx in indices.items()},
    )
