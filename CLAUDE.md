# CLAUDE.md

## Project Overview

Machine learning on Markov transition matrices: generative models of the matrices themselves and predictive models mapping a matrix to the dynamics it produces.

## Python Setup

Python dependencies are managed with pixi. Python commands should be run with pixi run python. Anything touching pytorch needs the ml environment, so use pixi run -e ml train, pixi run -e ml pipeline, and so on.

## Repository Structure

- `config/` — `sweeps/` holds one YAML per sweep
- `workflow/` — `Snakefile` chaining ingest → train → evaluate → visualize → benchmarks
- `src/tm_ml/` — the package: `ingest.py`, `datasets.py`, `models.py`, `train.py`, `evaluate.py`, `visualize.py`, `benchmarks.py`, `config.py`, `paths.py`, `device.py`
- `data/` — gitignored; `raw/` the untouched external drop, `processed/` the canonical arrays plus `meta.json` provenance
- `results/` — gitignored run directories, named from the determining hyperparameters, except the example `gamma_sweep/`
- `docs/` — `benchmarks/` (run sizing and results CSV), `notes/`
