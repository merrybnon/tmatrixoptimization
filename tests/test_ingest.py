"""Ingest tests, run against the committed fixture so they need no real drop."""

import json
from pathlib import Path

import numpy as np
import pytest

from tm_ml import ingest as ing

FIXTURE = Path(__file__).parent / "fixtures" / "tiny.npz"


def stochastic(n=4, size=3, seed=0):
    """``n`` random row-stochastic matrices of side ``size``, in float64."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, size, size))
    return m / m.sum(axis=2, keepdims=True)


def write_npz(path, matrices, targets):
    """Write arrays in the drop's own object-array layout."""
    np.savez(
        path,
        **{
            ing.MATRIX_KEY: np.asarray(list(matrices), dtype=object),
            ing.TARGET_KEY: np.asarray(list(targets), dtype=object),
        },
    )
    return path


def test_ingest_fixture_writes_arrays_and_meta(tmp_path):
    meta = ing.ingest("tiny", source=FIXTURE, out_dir=tmp_path)

    matrices = np.load(tmp_path / "matrices.npy")
    targets = np.load(tmp_path / "targets.npy")

    assert matrices.shape == (8, 20, 20)
    assert targets.shape == (8,)
    assert matrices.dtype == np.float32
    assert targets.dtype == np.float32

    assert meta["n"] == 8
    assert meta["shape"] == [20, 20]
    assert meta["row_sum_max_dev"] < ing.ROW_SUM_ATOL
    assert json.loads((tmp_path / "meta.json").read_text()) == meta


def test_ingest_preserves_values(tmp_path):
    """The stacked arrays must match the source element for element."""
    with np.load(FIXTURE, allow_pickle=True) as data:
        expected = np.stack([np.asarray(m, float) for m in data[ing.MATRIX_KEY].tolist()])
        expected_targets = np.asarray(data[ing.TARGET_KEY].tolist(), float)

    ing.ingest("tiny", source=FIXTURE, out_dir=tmp_path)

    np.testing.assert_allclose(np.load(tmp_path / "matrices.npy"), expected, rtol=1e-6)
    np.testing.assert_allclose(np.load(tmp_path / "targets.npy"), expected_targets, rtol=1e-6)


def test_targets_are_not_normalized(tmp_path):
    """Targets land exactly as received, so normalization stays a later choice."""
    ing.ingest("tiny", source=FIXTURE, out_dir=tmp_path)
    targets = np.load(tmp_path / "targets.npy")
    assert targets.min() > 1.0
    assert targets.max() > 100.0


def test_contiguous_output(tmp_path):
    """Downstream memory-maps these; a non-contiguous block would be a silent cost."""
    ing.ingest("tiny", source=FIXTURE, out_dir=tmp_path)
    assert np.load(tmp_path / "matrices.npy").flags["C_CONTIGUOUS"]


def test_validate_accepts_row_stochastic():
    m = stochastic()
    assert ing.validate(m, np.arange(len(m), dtype=float)) < ing.ROW_SUM_ATOL


def test_validate_rejects_bad_row_sums():
    m = stochastic()
    m[2, 1, 0] += 0.3
    with pytest.raises(ValueError, match=r"do not sum to 1.*matrix 2 row 1"):
        ing.validate(m, np.arange(len(m), dtype=float))


def test_validate_rejects_column_stochastic():
    """A transposed drop is the realistic mistake; it must not pass silently."""
    m = np.transpose(stochastic(), (0, 2, 1))
    with pytest.raises(ValueError, match="do not sum to 1"):
        ing.validate(m, np.arange(len(m), dtype=float))


def test_validate_rejects_negative_entries():
    m = stochastic()
    m[1, 0, 0] -= 1.0
    m[1, 0, 1] += 1.0  # keep the row sum at 1 so negativity is what trips
    with pytest.raises(ValueError, match=r"negative entries.*matrix 1"):
        ing.validate(m, np.arange(len(m), dtype=float))


def test_validate_rejects_nan():
    m = stochastic()
    m[3, 2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        ing.validate(m, np.arange(len(m), dtype=float))


def test_validate_rejects_nan_target():
    m = stochastic()
    targets = np.arange(len(m), dtype=float)
    targets[0] = np.nan
    with pytest.raises(ValueError, match="targets contain.*non-finite"):
        ing.validate(m, targets)


def test_validate_rejects_length_mismatch():
    m = stochastic(n=4)
    with pytest.raises(ValueError, match="4 matrices but 3 targets"):
        ing.validate(m, np.arange(3, dtype=float))


def test_validate_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        ing.validate(np.empty((0, 3, 3)), np.empty(0))


def test_load_rejects_ragged_shapes(tmp_path):
    src = write_npz(tmp_path / "ragged.npz", [np.eye(3), np.eye(4)], [1.0, 2.0])
    with pytest.raises(ValueError, match="not all the same shape"):
        ing.load_drop(src)


def test_load_rejects_non_square(tmp_path):
    m = np.full((2, 3), 1 / 3)
    src = write_npz(tmp_path / "oblong.npz", [m, m], [1.0, 2.0])
    with pytest.raises(ValueError, match="not square"):
        ing.load_drop(src)


def test_load_rejects_missing_keys(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, something_else=np.eye(3))
    with pytest.raises(KeyError, match=ing.MATRIX_KEY):
        ing.load_drop(path)


def test_find_source_rejects_unknown_drop():
    with pytest.raises(FileNotFoundError, match="no such drop"):
        ing.find_source("definitely-not-a-drop")


def test_find_source_rejects_ambiguous_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "RAW_DIR", tmp_path)
    drop = tmp_path / "two"
    drop.mkdir()
    write_npz(drop / "a.npz", [np.eye(2)], [1.0])
    write_npz(drop / "b.npz", [np.eye(2)], [1.0])
    with pytest.raises(ValueError, match="expected exactly one"):
        ing.find_source("two")


def test_find_source_locates_single_npz(tmp_path, monkeypatch):
    monkeypatch.setattr(ing, "RAW_DIR", tmp_path)
    drop = tmp_path / "one"
    drop.mkdir()
    expected = write_npz(drop / "Analysis_Results.npz", [np.eye(2)], [1.0])
    assert ing.find_source("one") == expected
