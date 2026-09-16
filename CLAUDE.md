# CLAUDE.md

## Project Overview

Machine learning on Markov transition matrices: generative models of the matrices themselves and predictive models mapping a matrix to the dynamics it produces.

`WORKLOG.md` has session history and findings, `CHANGELOG.md` records what changed when.

## Python Setup

Python dependencies are managed with pixi. Python commands should be run with pixi run python. Anything touching pytorch needs the ml environment, so use pixi run -e ml train, pixi run -e ml pipeline, and so on.

## Repository Structure

`docs/notes/repo_structure.md` is the canonical record, with the reasoning behind each choice; the map below is a summary.

- `config/` — `sweeps/` holds one YAML per sweep
- `workflow/` — `Snakefile` chaining ingest → train → evaluate → visualize → benchmarks, with `rules/` and `profiles/`
- `src/tm_ml/` — the package: `ingest.py`, `datasets.py`, `models.py`, `train.py`, `config.py`, `paths.py`, `device.py`
- `data/` — gitignored; `raw/` the untouched external drop, `processed/` the canonical arrays plus `meta.json` provenance
- `results/` — gitignored run directories, named from the determining hyperparameters
- `tests/` — `fixtures/tiny.npz` is committed so the pipeline runs without the real data
- `docs/` — `benchmarks/` (run sizing and results CSV), `notes/`
