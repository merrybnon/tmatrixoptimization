"""Upsert one row per evaluated run into the benchmarks ledger.

    pixi run -e ml benchmarks

The CSV is **curated, not a mirror of disk**. Three rules follow from that, and
they are the whole reason this is a script rather than a glob:

- a row you wrote by hand, with no ``run`` key, passes through untouched;
- a ``notes`` cell you wrote is never overwritten, even for a run that is
  refreshed on every other column;
- a row whose run directory has since been deleted stays, because the number
  was still true when it was measured and the ledger is the record of what was
  tried, not of what currently exists on disk.

Everything else is refreshed from the run's ``metrics.json``, ``train_meta.json``
and ``config.json`` on every call. Runs are scanned across all of ``results/``
rather than only the sweep that triggered this, so points from separate
invocations end up in one table.

Under Snakemake the output must be declared with ``update()``: Snakemake deletes
a job's declared outputs before running it, which would destroy the hand-written
rows before this script could read them.
"""

import argparse
import csv
import json
from pathlib import Path

from tm_ml import paths

# The ledger's own column order. Anything a hand-written row carries that is not
# here is appended on the right rather than dropped.
COLUMNS = (
    "run", "model", "drop",
    # what was varied
    "d_latent", "d_global", "pooling", "beta", "gamma", "beta_warmup_epochs",
    "gamma_warmup_epochs", "lambda_log", "lambda_log_warmup_epochs",
    "dropout", "weight_decay", "lr", "lr_schedule", "epochs", "seed",
    # the property, which is what the project is for
    "test_r2", "skill_vs_train_mean", "test_median_relative_error",
    "test_rmse_exponent", "test_resolution_slope", "test_resolution_slope_stderr",
    "test_calibration_slope", "test_calibration_slope_stderr",
    # reconstruction, in both spaces
    "test_recon", "test_log_recon_mae", "test_recon_rmse",
    # the latent
    "test_kl", "test_kl_global", "latent_active_units", "latent_dim_per_node",
    "latent_pca_components_above_noise", "latent_mi_upper_bound",
    # constraints, which should be zero forever
    "recon_diag_max_abs", "recon_row_sum_max_dev",
    # provenance
    "n_train", "n_test", "n_params", "best_epoch", "epochs_run", "stopped_early",
    "seconds", "device", "git_commit", "evaluated_at",
    # yours, and never touched
    "notes",
)

PRESERVED = ("notes",)


def read_ledger(path):
    """Existing rows, in file order, or nothing if the ledger is new."""
    path = Path(path)
    if not path.exists():
        return [], []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


# What makes a directory a run rather than a container. Recursing for these
# rather than listing one level down is what lets runs be filed under a
# parent_path of any depth.
RUN_MARKERS = (paths.METRICS, paths.TRAIN_META, paths.CHECKPOINT)


def find_runs(results_root):
    """Every run directory under ``results/``, at whatever depth it is filed."""
    root = Path(results_root)
    if not root.exists():
        return []
    found = set()
    for marker in RUN_MARKERS:
        found |= {path.parent for path in root.rglob(marker)}
    return sorted(found)


def scan(results_root):
    """One row per evaluated run under ``results/``, keyed by its path.

    The key is the path relative to ``results/``, so a run filed under a
    parent_path carries that in its identity — ``initial_testing/TMVAE_...``.

    A run with no ``metrics.json`` has not been evaluated, so it has nothing to
    contribute yet; it is reported rather than silently skipped.
    """
    root = Path(results_root)
    rows, unevaluated = {}, []

    for run_dir in find_runs(root):
        key = run_dir.relative_to(root).as_posix()
        metrics_path = run_dir / paths.METRICS
        if not metrics_path.exists():
            unevaluated.append(key)
            continue

        record = json.loads(metrics_path.read_text())
        for name in (paths.TRAIN_META, paths.STORED_CONFIG):
            extra = run_dir / name
            if extra.exists():
                # metrics.json wins: it is the later measurement, and the two
                # overlap on provenance fields like git_commit.
                record = {**json.loads(extra.read_text()), **record}

        record["run"] = key
        rows[key] = {column: format_cell(record.get(column)) for column in COLUMNS}
    return rows, unevaluated


def format_cell(value):
    """A float rounded to something readable; everything else as it came."""
    if isinstance(value, bool) or value is None:
        return "" if value is None else str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def merge(existing, fields, fresh):
    """Upsert, preserving hand-written rows and hand-written notes."""
    columns = list(COLUMNS) + [f for f in fields if f not in COLUMNS]
    merged, seen = [], set()

    for row in existing:
        key = row.get("run", "")
        if not key or key not in fresh:
            # A hand-written row, or a run whose directory is gone. Both stay.
            merged.append(row)
            continue
        merged.append({**row, **fresh[key], **preserved_from(row)})
        seen.add(key)

    for key, row in fresh.items():
        if key not in seen:
            merged.append(row)
    return merged, columns


def preserved_from(row):
    """Cells the script must never overwrite once a person has filled them in."""
    return {k: row[k] for k in PRESERVED if row.get(k)}


def write_ledger(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def update(csv_path=None, results_root=None):
    """Refresh the ledger from disk and return the rows written."""
    csv_path = Path(csv_path) if csv_path else paths.BENCHMARKS_CSV
    results_root = Path(results_root) if results_root else paths.RESULTS_ROOT

    existing, fields = read_ledger(csv_path)
    fresh, unevaluated = scan(results_root)
    rows, columns = merge(existing, fields, fresh)
    write_ledger(csv_path, rows, columns)

    print(f"{csv_path}: {len(rows)} rows ({len(fresh)} from disk)")
    for name in unevaluated:
        print(f"  not evaluated, no row: {name}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help=f"ledger path (default {paths.BENCHMARKS_CSV})")
    parser.add_argument("--results", help="override the results root")
    args = parser.parse_args()

    update(args.csv, args.results)


if __name__ == "__main__":
    main()
