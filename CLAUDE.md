# CLAUDE.md

## Project Overview

Machine learning on Markov transition matrices: generative models of the matrices themselves and predictive models mapping a matrix to the dynamics it produces. Data comes from outside the repo and is frozen by an ingest stage before any training reads it.

`WORKLOG.md` has session history and findings, `CHANGELOG.md` records what changed when.

## Python Setup

Python dependencies are managed with pixi. Python commands should be run with pixi run python; the ML environment needs `-e ml`.

## Repository Structure

- `config/` — `data.yaml` names the external source and its checksum; `sweeps/` holds one YAML per sweep
- `workflow/` — `Snakefile` chaining ingest → train → evaluate → visualize → benchmarks, with `rules/` and `profiles/`
- `src/tm_ml/` — the package: `ingest.py`, `datasets.py`, `models.py`, `train.py`, `config.py`, `paths.py`, `device.py`
- `data/` — gitignored; `raw/` the untouched external drop, `processed/` the canonical arrays plus `meta.json` provenance
- `results/` — gitignored run directories, named from the determining hyperparameters
- `tests/` — `fixtures/tiny.npz` is committed so the pipeline runs without the real data
- `docs/` — `benchmarks/` (run sizing and results CSV), `notes/`
