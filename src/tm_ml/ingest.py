"""Freeze a raw transition-matrix drop into canonical arrays.

A drop is whatever arrived from outside the repo, committed untouched under
``data/raw/<drop>/``. Ingest is the only stage that reads it: it stacks the
object array into one contiguous block, validates that the matrices are what
they claim to be, and writes ``data/processed/<drop>/`` for everything
downstream.

    pixi run ingest --drop Tom1000

Targets are written exactly as received. Normalization belongs to the dataset
or the model, so changing your mind about it costs nothing here.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

MATRIX_KEY = "matrixlist"
TARGET_KEY = "Exponential_Decay_parameters"

# Row sums are checked in float64, before the cast to float32, so the tolerance
# tests the drop rather than our own rounding.
ROW_SUM_ATOL = 1e-6


def find_source(drop):
    """Locate the single ``.npz`` inside ``data/raw/<drop>/``."""
    drop_dir = RAW_DIR / drop
    if not drop_dir.is_dir():
        raise FileNotFoundError(
            f"no such drop: {drop_dir}. Available: "
            f"{sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir()) or 'none'}"
        )

    candidates = sorted(drop_dir.glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"no .npz found in {drop_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"{drop_dir} holds {len(candidates)} .npz files "
            f"({', '.join(p.name for p in candidates)}); expected exactly one"
        )
    return candidates[0]


def load_drop(source):
    """Read a drop into ``(N, n, n)`` and ``(N,)`` float64 arrays.

    The source stores matrices as an object array of separate 2D arrays;
    stacking them into one contiguous block is most of what ingest is for.
    """
    with np.load(source, allow_pickle=True) as data:
        missing = {MATRIX_KEY, TARGET_KEY} - set(data.files)
        if missing:
            raise KeyError(
                f"{source} is missing {sorted(missing)}; it holds {sorted(data.files)}"
            )
        raw_matrices = data[MATRIX_KEY].tolist()
        raw_targets = data[TARGET_KEY].tolist()

    matrices = [np.asarray(m, dtype=np.float64) for m in raw_matrices]

    shapes = {m.shape for m in matrices}
    if len(shapes) != 1:
        counts = {s: sum(m.shape == s for m in matrices) for s in shapes}
        raise ValueError(f"matrices are not all the same shape: {counts}")

    shape = shapes.pop()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"matrices are not square 2D arrays: shape {shape}")

    return np.stack(matrices), np.asarray(raw_targets, dtype=np.float64)


def validate(matrices, targets):
    """Check the drop is a set of row-stochastic matrices with paired targets.

    Every failure names the offending index, since a bare assertion on 1000
    matrices tells you nothing about which one is wrong.

    Returns the largest observed deviation of a row sum from 1, which lands in
    ``meta.json`` as a cheap fingerprint of the drop.
    """
    if len(matrices) != len(targets):
        raise ValueError(
            f"{len(matrices)} matrices but {len(targets)} targets"
        )
    if len(matrices) == 0:
        raise ValueError("drop is empty")

    for name, arr in (("matrices", matrices), ("targets", targets)):
        bad = np.argwhere(~np.isfinite(arr))
        if bad.size:
            raise ValueError(
                f"{name} contain {len(bad)} non-finite values; first at index {tuple(bad[0])}"
            )

    negative = np.argwhere(matrices < 0)
    if negative.size:
        i, r, c = negative[0]
        raise ValueError(
            f"matrices contain {len(negative)} negative entries; "
            f"first is matrix {i} at [{r}, {c}] = {matrices[i, r, c]!r}"
        )

    deviation = np.abs(matrices.sum(axis=2) - 1.0)
    worst = np.unravel_index(np.argmax(deviation), deviation.shape)
    max_dev = float(deviation[worst])
    if max_dev > ROW_SUM_ATOL:
        n_bad = int((deviation > ROW_SUM_ATOL).any(axis=1).sum())
        raise ValueError(
            f"rows do not sum to 1 within {ROW_SUM_ATOL:g}: {n_bad} matrices affected, "
            f"worst is matrix {worst[0]} row {worst[1]} off by {max_dev:g}"
        )

    return max_dev


def git_commit():
    """HEAD of the repo, or None outside a repo or before the first commit."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def ingest(drop, source=None, out_dir=None):
    """Validate one drop and write its canonical arrays plus ``meta.json``."""
    source = Path(source) if source else find_source(drop)
    out_dir = Path(out_dir) if out_dir else PROCESSED_DIR / drop

    matrices, targets = load_drop(source)
    max_dev = validate(matrices, targets)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "matrices.npy", matrices.astype(np.float32))
    np.save(out_dir / "targets.npy", targets.astype(np.float32))

    meta = {
        "drop": drop,
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "n": int(len(matrices)),
        "shape": list(matrices.shape[1:]),
        "dtype": "float32",
        "row_sum_max_dev": max_dev,
        "git_commit": git_commit(),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drop", required=True, help="drop name under data/raw/")
    parser.add_argument("--source", help="override the .npz path found in the drop")
    parser.add_argument("--out-dir", help="override data/processed/<drop>/")
    args = parser.parse_args()

    meta = ingest(args.drop, source=args.source, out_dir=args.out_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
