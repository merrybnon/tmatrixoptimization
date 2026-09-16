"""The ledger is curated, not a mirror of disk.

Every test here is one of the three rules that distinguishes the two: what a
person wrote stays, and what the script measured is refreshed.
"""

import csv
import json

from tm_ml import benchmarks, paths


def make_run(root, name, **metrics):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / paths.METRICS).write_text(json.dumps({
        "run": name, "model": "tmvae", "drop": "Tom1000",
        "test_r2": 0.5, "d_latent": 8, **metrics,
    }))
    (run_dir / paths.TRAIN_META).write_text(json.dumps({"epochs_run": 300, "seconds": 120.0}))
    return run_dir


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_new_ledger_is_created(tmp_path):
    results = tmp_path / "results"
    make_run(results, "TMVAE_a_Tom1000")
    csv_path = tmp_path / "docs" / "runs.csv"

    benchmarks.update(csv_path, results)

    rows = read(csv_path)
    assert len(rows) == 1
    assert rows[0]["run"] == "TMVAE_a_Tom1000"
    assert rows[0]["test_r2"] == "0.5"
    assert rows[0]["epochs_run"] == "300", "train_meta.json is folded in"


def test_columns_follow_the_declared_order(tmp_path):
    results = tmp_path / "results"
    make_run(results, "TMVAE_a_Tom1000")
    csv_path = tmp_path / "runs.csv"
    benchmarks.update(csv_path, results)

    with open(csv_path, newline="") as handle:
        header = next(csv.reader(handle))
    assert header[:3] == ["run", "model", "drop"]
    assert header[-1] == "notes"


def test_measurements_are_refreshed(tmp_path):
    results = tmp_path / "results"
    run_dir = make_run(results, "TMVAE_a_Tom1000", test_r2=0.5)
    csv_path = tmp_path / "runs.csv"
    benchmarks.update(csv_path, results)

    record = json.loads((run_dir / paths.METRICS).read_text())
    record["test_r2"] = 0.9
    (run_dir / paths.METRICS).write_text(json.dumps(record))
    benchmarks.update(csv_path, results)

    rows = read(csv_path)
    assert len(rows) == 1, "a refresh must upsert, not append"
    assert rows[0]["test_r2"] == "0.9"


def test_a_handwritten_note_is_never_overwritten(tmp_path):
    results = tmp_path / "results"
    make_run(results, "TMVAE_a_Tom1000")
    csv_path = tmp_path / "runs.csv"
    benchmarks.update(csv_path, results)

    rows = read(csv_path)
    rows[0]["notes"] = "collapsed; beta too high"
    benchmarks.write_ledger(csv_path, rows, list(rows[0]))

    benchmarks.update(csv_path, results)
    assert read(csv_path)[0]["notes"] == "collapsed; beta too high"


def test_a_handwritten_row_passes_through(tmp_path):
    results = tmp_path / "results"
    make_run(results, "TMVAE_a_Tom1000")
    csv_path = tmp_path / "runs.csv"
    benchmarks.update(csv_path, results)

    rows = read(csv_path)
    rows.append({**{k: "" for k in rows[0]}, "model": "literature", "notes": "baseline to beat"})
    benchmarks.write_ledger(csv_path, rows, list(rows[0]))

    benchmarks.update(csv_path, results)
    after = read(csv_path)
    assert len(after) == 2
    assert after[1]["model"] == "literature"
    assert after[1]["notes"] == "baseline to beat"


def test_a_row_survives_its_run_directory(tmp_path):
    """The ledger records what was tried, not what is currently on disk."""
    results = tmp_path / "results"
    run_dir = make_run(results, "TMVAE_a_Tom1000")
    make_run(results, "TMVAE_b_Tom1000")
    csv_path = tmp_path / "runs.csv"
    benchmarks.update(csv_path, results)

    for path in run_dir.iterdir():
        path.unlink()
    run_dir.rmdir()
    benchmarks.update(csv_path, results)

    runs = [r["run"] for r in read(csv_path)]
    assert runs == ["TMVAE_a_Tom1000", "TMVAE_b_Tom1000"]


def test_an_unevaluated_run_gets_no_row(tmp_path, capsys):
    results = tmp_path / "results"
    make_run(results, "TMVAE_a_Tom1000")
    (results / "TMVAE_untrained_Tom1000").mkdir()
    csv_path = tmp_path / "runs.csv"

    benchmarks.update(csv_path, results)

    assert [r["run"] for r in read(csv_path)] == ["TMVAE_a_Tom1000"]
    assert "not evaluated, no row: TMVAE_untrained_Tom1000" in capsys.readouterr().out


def test_extra_handwritten_columns_are_kept(tmp_path):
    results = tmp_path / "results"
    make_run(results, "TMVAE_a_Tom1000")
    csv_path = tmp_path / "runs.csv"
    benchmarks.update(csv_path, results)

    rows = read(csv_path)
    rows[0]["reviewer"] = "checked 2026-09-16"
    benchmarks.write_ledger(csv_path, rows, list(rows[0]) + ["reviewer"])

    benchmarks.update(csv_path, results)
    after = read(csv_path)[0]
    assert after["reviewer"] == "checked 2026-09-16"


def test_floats_are_rounded_for_reading():
    assert benchmarks.format_cell(0.7156249999) == "0.715625"
    assert benchmarks.format_cell(None) == ""
    assert benchmarks.format_cell(True) == "True"
    assert benchmarks.format_cell([1, 2]) == "[1, 2]"
