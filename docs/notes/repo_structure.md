# Repository structure

The canonical record of how this repo is laid out and why. `README.md` and `CLAUDE.md` each carry a short map for their own audience — a human arriving at the repo, and the instructions loaded into every session — but those are summaries. Structural decisions and their reasoning belong here, and this is the file to edit when the layout changes.

## Layout

| path | holds | tracked |
|---|---|---|
| `config/` | `sweeps/`, one YAML per sweep | yes |
| `workflow/` | `Snakefile`, plus `rules/` and `profiles/` | yes |
| `src/tm_ml/` | the importable package the Snakemake rules call | yes |
| `data/raw/<drop>/` | the untouched external drop, exactly as received | **yes** |
| `data/processed/<drop>/` | `matrices.npy`, `targets.npy`, `meta.json` | no |
| `results/` | run directories, named from the determining hyperparameters | no |
| `tests/` | tests plus `fixtures/tiny.npz` | yes |
| `docs/` | `benchmarks/` (run sizing and results CSV), `notes/` | yes |

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
