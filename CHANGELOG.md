# Changelog

All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `docs/notes/permutationsandlatent.md`: the latent-structure decision worked through, covering the group action on each object in the architecture, a per-component equivariant/invariant ledger, why the reconstruction is equivariant while the loss is invariant (which is what makes matching unnecessary), a worked 3-node relabelling example, the orbit answer to whether one latent can represent all 20! labellings, and the hybrid graph-level latent.
- `docs/notes/architecture.md`: ascent equivariance. The predictor is invariant, so ∇ŷ|_(PZ) = P · ∇ŷ|_Z and latent gradient ascent commutes with relabelling — an optimized landscape cannot depend on how its input happened to be labelled. Carries the test procedure, which must run with sampling off, since equivariance of a sampled z holds only in distribution.
- `docs/notes/GNNsources.md`: a `## Reading order` section with two paths — the narrow one for why a transformer encoder and a GNN are the same thing here, and a general ten-item order into the field. Four new entries: Hamilton's *Graph Representation Learning Book* and Kipf & Welling's GCN paper, both absent and both needed for basic-level grounding; Edwards & Storkey's *Towards a Neural Statistician*, the two-level latent for exchangeable sets behind the hybrid; and Battaglia's global attribute `u`, folded into the existing entry as the mechanism that makes a broadcast invariant preserve equivariance.

### Changed

- `docs/notes/GNNsources.md`: the GE-VAE entry (arXiv 1910.08057) was misfiled beside PIGVAE and GraViti as though it were a graph-level method. Its latent is |V| x P, node-level, so it is the family we already chose reached independently. Rewritten with what the model actually is and the three reasons it is ruled out — a Laplacian-eigenmap encoder that reintroduces the canonicalization discontinuity in spectral form, a symmetric z_iᵀz_j that cannot represent our directed edges, and binary undirected topology whose sparse-likelihood contribution is worthless on a dense 20-node digraph.
- `WORKLOG.md`: the next-steps block reformatted for `~/bin/nextsteps`, which parses `/^### Next steps/` and treats a blank line as the terminator. The previous `### Where we left off` heading with a separate `Next steps:` line and an intervening blank matched nothing, so the tool returned empty.


## [0.1.0] - 2026-09-14

### Added

- Repository scaffold: `config/`, `workflow/`, `src/tm_ml/`, `data/`, `results/`, `tests/`, `docs/`. Raw drops under `data/raw/` are committed so a clone carries the exact bytes a run trained on; `data/processed/` and `results/` are derived and gitignored, which needs the per-level `!.gitkeep` spelling in `.gitignore` because a bare `data/` makes git skip the directory entirely and silently ignore the negation.
- `pixi.toml` with a base environment and an `ml` feature on the `linux-64-cuda` platform, which is what makes conda-forge resolve CUDA-enabled pytorch — `pytorch = { build = "cuda*" }` will not solve on plain `linux-64`. Split rather than collapsed so the base environment stays installable on a machine without CUDA. `PYTHONPATH` is set to `$PIXI_PROJECT_ROOT/src` in `[activation.env]`, so `python -m tm_ml.x` resolves from any directory and the Snakefile imports the package directly instead of the `sys.path.insert` hack the sibling repo needs. `bioconda` stays in the channel list because `snakemake-minimal` lives there, not on conda-forge.
- `src/tm_ml/ingest.py`: the only code that reads a raw drop. Stacks the source's object array of separate 20x20 arrays into one contiguous `(N, 20, 20)` block, validates, and writes `matrices.npy`, `targets.npy` and `meta.json` to `data/processed/<drop>/`. Validation is fatal and names the offending index — square and mutually consistent shapes, row sums within 1e-6 of 1, no non-finite or negative entries, matching lengths. Checked in float64 before the float32 cast, so the tolerance tests the drop rather than our own rounding. Targets are written exactly as received; normalization belongs to the dataset or the model, so changing it later costs no re-ingest.
- `workflow/Snakefile` with the ingest rule. The drop's `.npz` reaches the rule through an input *function* calling `ingest.find_source`, so the file is a real dependency with real mtime tracking rather than a hardcoded name, and an unknown drop fails at DAG-build time naming the drops that do exist. Drop directories are globbed at parse time, so adding one makes it a target with no edit here.
- `tests/test_ingest.py`, 18 tests, run with `pixi run test`. Four cover the happy path against `tests/fixtures/tiny.npz` — a committed 30 KB slice of the real drop in the same object-array layout, so the tests exercise the real code path and need no access to the full data. The rest assert on failure *messages*, not just exception types, and include a transposed drop: Tom's own script sorts by column sum while the data is row-stochastic, so both conventions are already in play and a transposed future drop would otherwise train silently on garbage.
- `docs/notes/architecture.md`: the design, ahead of the code. Masked row softmax before normalization, per-row KL reconstruction, permutation equivariance built in rather than canonicalized, node-level latents. Carries the four places the structural zero diagonal produces NaN rather than a violated constraint.
- `docs/notes/GNNsources.md`, `NetworkDiversity.md`, `miscSources.md`: annotated reading list, the domain literature on network structure and genetic diversity, and everything else, each entry noting what it is and why it bears on this project.
- `data/raw/Tom1000/Analysis_Results.npz`: 1000 20x20 row-stochastic matrices with paired decay exponents.
