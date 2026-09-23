# Repository structure

How this repo is laid out and why. `README.md` and `CLAUDE.md` carry their own short maps for their own audiences — a human arriving at the repo, and the instructions loaded into every session. This note is where the reasoning lives, and it is edited on its own.

## Layout

| path | holds | tracked |
|---|---|---|
| `config/` | `sweeps/`, one YAML per sweep | yes |
| `workflow/` | `Snakefile`, plus `rules/` and `profiles/` | yes |
| `src/tm_ml/` | the importable package the Snakemake rules call | yes |
| `data/raw/<drop>/` | the untouched external drop, exactly as received | **yes** |
| `data/processed/<drop>/` | `matrices.npy`, `targets.npy`, `meta.json`, and the `data_figures.py` PNGs | no |
| `results/` | run directories, named from the determining hyperparameters | no |
| `tests/` | tests plus `fixtures/tiny.npz` | yes |
| `docs/` | `benchmarks/` (run sizing and results CSV), `notes/` | yes |

## The package

`src/tm_ml/` is one importable package, on `PYTHONPATH` through pixi, called by the Snakemake rules and by the pixi tasks. The module split carries over from `hotspotLandscapes/ml/hotspot_ml/`, which is where these responsibilities are already implemented and worth reading before rewriting any of them. `ingest.py` is the one addition here, since data arrives from outside rather than from a simulation living in the same repo. Written so far: `ingest.py` and `models.py`.

| module | responsibility |
|---|---|
| `ingest.py` | the only code that reads a raw drop. Stacks the source's object array into one contiguous `(N, 20, 20)` block, validates fatally with the offending index named, writes `matrices.npy`, `targets.npy` and `meta.json` |
| `config.py` | resolve a sweep YAML into one config dict per run. Defaults live here, not in the YAML, so a config file holds exactly the parameters that sweep varies; any field given a list is a swept axis and several lists give the cartesian product. Imports nothing heavier than yaml and pathlib, so the Snakefile can build its DAG without loading torch |
| `paths.py` | run-directory naming and resolution. A run lives at `results/<run_name>/` and holds everything for one trained model — checkpoint, history, metrics, and figures under `figures/`. The name is a pure function of the resolved config, so the workflow can compute every output path before torch is imported, and every determining field appears in it, or two points of a sweep share a directory and overwrite each other silently |
| `device.py` | pick a GPU on a shared multi-card box. `gpu: auto` takes the card with least memory in use, tie-broken on utilization, resolved to an explicit `cuda:<i>` rather than `CUDA_VISIBLE_DEVICES`, which has no effect once torch has initialized CUDA in-process. Memory alone is not enough when we launch the jobs: two runs started together both read nvidia-smi before either allocates, see the same idle cards and deterministically pick the same one, so each process also stakes an advisory claim file that a dead PID releases |
| `datasets.py` | serve `(T, target)` pairs out of `data/processed/<drop>/` with the train/val/test split. The target transform belongs here — taking the log, and standardizing on statistics fitted from the train split alone, with the constants exposed so evaluation can invert them |
| `models.py` | the VAE, the property predictor and the loss |
| `train.py` | settings resolve in three layers, later winning: the defaults in `config.py`, then a sweep YAML, then any explicitly passed CLI flag. Best checkpoint, history and figures all land in the one run directory |
| `evaluate.py` | score a trained run on its held-out test split and write `metrics.json`, including whatever landscape-blind baseline the metric has to beat |
| `visualize.py` | diagnostic figures into the run's `figures/`, and decoded latent traversals (global dim, sorted-160 PC1 and PC2), interpolations and BFGS optimization paths (`optimization_g#.png`, `optimization_g#_unreg.png`) into `figures/interpolation_and_traversals/`, each stamped with a provenance footer — checkpoint epoch and score, drop, split seed, git hash, timestamp — so a figure is self-describing wherever it ends up |
| `optimize.py` | BFGS on the predicted property in latent space from one graph's encoding, penalized by distance from the start, `(λ/2)·‖x − x₀‖²`, in float64 on the cpu; the path is every accepted iterate. Its CLI prints each path, for reading how the optimizer behaves |
| `benchmarks.py` | upsert one row per evaluated run into `docs/benchmarks/`. A curated ledger rather than a mirror of disk: hand-written rows and hand-written notes survive, and a row whose run directory was deleted stays |
| `data_figures.py` | run by hand, outside the workflow: `target_statistics.png` (histogram and survival of the targets on linear and log axes, with mean, median, maximum and minimum over the drop's matrices) and `example_matrices.png` (log₁₀ matrices with the four highest, four nearest-median and four lowest targets), written beside the arrays in `data/processed/<drop>/` |
| `style.py` | the palette and matplotlib style every figure shares, kept free of torch so `data_figures.py` runs in the default env |

The pixi tasks name `train`, `evaluate`, `visualize` and `benchmarks`, and the workflow chain in `CLAUDE.md` names the same four, so all of them are part of the design even where a shorter list appears elsewhere.

## The decisions behind it

**Raw drops are committed; processed arrays and results are not.** A clone carries the exact bytes a run trained on, which is the only way to reproduce a result when the generator lives outside the repo. Everything derived from those bytes is reproducible from them, so tracking it would only add merge conflicts on binary files.

Gitignoring `data/` and `results/` needs the per-level `!.gitkeep` spelling. A bare `data/` in `.gitignore` makes git skip the directory entirely and never see the negation, so the directories vanish from a fresh clone.

**Ingest is the only code that reads a raw drop.** It validates, stacks the source's object array into one contiguous block, and writes canonical arrays plus provenance. Everything downstream reads `data/processed/` and never touches `raw/`. One stage owns the external format, so a differently-shaped drop breaks in one place with a message naming the offending index.

**There is no `config/data.yaml`.** The scaffold planned one, naming an external source path and its checksum; it was dropped and never written. Committing the raw drop removed its reason to exist — you cannot point at an external file and checksum it when the file is in-tree. Its two jobs went elsewhere:

- *Finding the source* is `ingest.find_source`, which globs `data/raw/<drop>/*.npz` and fails loudly on zero or several, naming the drops that do exist. Convention rather than configuration.
- *Provenance* is `meta.json`, written after the fact: `row_sum_max_dev` as a cheap fingerprint of the drop, plus `git_commit` and `ingested_at`. Recorded from what was actually read, not asserted in advance.

**Drops are discovered, not enumerated.** `workflow/Snakefile` globs `data/raw/` at parse time, so every drop directory is a target and adding one needs no edit. The `.npz` reaches the ingest rule through an input *function* calling `find_source`, which makes it a real dependency with real mtime tracking rather than a hardcoded name; an unknown drop fails at DAG-build time.

**`PYTHONPATH` is set to `$PIXI_PROJECT_ROOT/src` in `pixi.toml`.** `python -m tm_ml.x` resolves from any directory and the Snakefile imports the package directly, instead of the `sys.path.insert` hack the sibling repo needs.

**`tests/fixtures/tiny.npz` is committed** — a 30 KB slice of the real drop in the same object-array layout — so the pipeline and its tests run without access to the full data, exercising the real code path rather than a mock.

## Relationship to `hotspotLandscapes`

Built from scratch rather than forked. The infrastructure patterns carry over — pixi with a CUDA-only `ml` feature, Snakemake chaining the stages, per-run directories under a gitignored output tree, `WORKLOG.md` and `CHANGELOG.md` — and the architecture does not. The differences that matter here: data arrives from outside rather than from a simulation shipped alongside the code, which is why an ingest stage exists at all; and `src/` is one Python package rather than a Julia simulation with a Python `ml/` subtree.
