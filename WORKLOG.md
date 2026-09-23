# Worklog

Session history and findings. This repo learns a continuous representation of Markov transition matrices, predicts a diversity target from that representation, and optimizes in it — stated generally, inverse design of Markov chains for a target dynamical property. Transition-matrix data comes from Tom; the generator lives outside the repo, unlike `hotspotLandscapes`, where the simulation shipped with the code that consumed it.

`CHANGELOG.md` records what changed when. `docs/notes/architecture.md` is the design.

## 2026-09-14 — repo setup, data characterization, architecture design

New repo, built from scratch rather than forked from `hotspotLandscapes`: the infrastructure patterns carry over, the architecture does not. Working code is ingest plus its Snakemake rule; the model is designed and verified but `src/tm_ml/models.py` is still being typed by hand.

### The data

Measured on `data/processed/Tom1000`, 1000 matrices, 20x20. Every design decision below follows from this table.

| property | value | consequence |
|---|---|---|
| diagonal | exactly 0 in all 1000 | no self-transitions; mask it structurally |
| other zeros | none | every off-diagonal entry strictly positive |
| entry range | 1.3e-14 .. 1.0 | 15% of entries below 1e-4; linear features cannot see them |
| asymmetry | mean abs(T - Tᵀ) = 0.034 vs mean abs(T) = 0.050 | strongly directed; edges need both directions |
| column-sum near-ties | 86% of matrices have an adjacent gap < 0.01, 20% < 0.001 | canonical node ordering is discontinuous |
| spectral gap | abs(λ₂) ≈ 0.9931, corr(target, relaxation time) ≈ 0.03 | the target is not a simple function of λ₂ |
| target | 48.8 .. 2041.6, right-skewed | predict log(target) |
| free parameters | 20 rows x (19 positive entries - 1 sum constraint) = 360 | inside a 400-dimensional ambient space |

Float32 costs nothing: max absolute error against the float64 source is 2.98e-08 and row sums drift to 4.46e-08, four orders inside the 1e-6 tolerance.

### Design decisions

- **Masked row softmax** for the row-stochastic constraint: diagonal logits to -inf *before* the softmax, so T̂_ii is exactly 0, rows sum to exactly 1, and the diagonal logit takes exactly zero gradient. Softmaxing over all 20 and zeroing afterwards would leave rows summing to less than 1.
- **Per-row KL** for reconstruction, since rows are distributions and this is the categorical likelihood written out. Gotcha recorded in `architecture.md`: forward KL weights by the true T_ij, so it barely notices T̂ being too large where truth is tiny, which is where the weak links live.
- **Permutation equivariance built in, not canonicalized.** Sorting nodes by column sum is discontinuous, and measurably so: on matrix 519, whose two closest in-flows differ by 1.08e-06, a perturbation of 9.89e-08 to the input moved the canonical form by 7.00e-03 — **70,842x amplification**, with two nodes trading rank. With 86% of matrices holding a near-tie, that is the typical case, not an edge case.
- **Node-level latents** (20 x 8 = 160 numbers) rather than one graph-level vector, which would need the canonical ordering above or graph matching. Cost: the latent is a set, so interpolating between two landscapes mixes unrelated node slots and needs Hungarian alignment first. Gradient ascent from a single encoded matrix, which is the actual goal, is unaffected.

### The notes

Four files under `docs/notes/`, written before the code:

- `architecture.md` — the design and why the data forces it, the loss, and the four places the zero diagonal produces NaN rather than a violated constraint.
- `GNNsources.md` — the method reading list: message passing, attention, graph VAEs, and a **permutation handling** taxonomy placing our design among the five families (graph matching, node-level latents, learned alignment, autoregressive, diffusion). Read Joshi's *Transformers are Graph Neural Networks* first for why a transformer encoder and a GNN are the same thing on a complete graph.
- `NetworkDiversity.md` — the domain question: how network structure shapes genetic diversity. Real empirical literature, entirely graph-theoretic and statistical, no generative modelling. Establishes the question matters without overlapping the method.
- `miscSources.md` — neural networks that output valid transition matrices (deep MSM, VAMPnets), the combinatorial baseline our optimization has to beat, and the search notes: the specific combination we are building did not turn up anywhere, plus the three nodes to citation-chase before claiming novelty.

### Verified

Against a reference implementation of the architecture, not yet the committed `models.py`:

| check | result |
|---|---|
| encoder equivariance, permuted input vs permuted latent | 1.11e-15 |
| decoder equivariance, permuted latent vs permuted output | 3.47e-17 |
| predictor invariance | 1.11e-16 |
| decoded matrices valid (zero diagonal, rows sum to 1) | pass, including along a latent gradient-ascent path |
| latent gradient ascent moves the predicted property | yes, gradient dense across all 160 entries |

**PyTorch gotcha worth the time it cost:** `nn.TransformerEncoderLayer` takes a fused fast path in eval mode that silently returns NaN when given an arbitrary additive float mask — which is exactly what an edge bias is. Training looks healthy, every evaluation is NaN. Hand-rolled the attention block over `F.scaled_dot_product_attention` instead, which handles additive masks correctly in both modes.

### Next steps
- xExplore the latent representation of the graph more.
- xFinish `models.py` and commit it as is for now.
- xFind literature on the problem of going from landscapes to genetic diversity more broadly.

## 2026-09-15 — permutation ledger, the latent-structure decision, GE-VAE, hybrid latents

No code this session. Worked through the latent-representation question the 09-14 entry stopped on, which turned into an audit of exactly what is equivariant and what is invariant at every point in the architecture. The architecture comes out unchanged, which is the useful result — the reasoning behind it is now written down rather than assumed. Full derivations in `docs/notes/permutationsandlatent.md` under "Claude's Plan".

### The equivariance ledger

The design splits cleanly: **every internal representation is equivariant, every loss term is invariant.** Node and edge features, encoder layers, μ and log σ, decoder logits and T̂ are all equivariant; reconstruction KL, prior KL and the predictor are all invariant. Four consequences worth keeping:

- **The reconstruction is equivariant and the loss is invariant.** L_rec(PTPᵀ, PT̂Pᵀ) = L_rec(T, T̂), because the outer sum runs over the same 20 terms in a different order and each inner KL over the same 20 entries in a different order. This is exactly what "nothing needs matching" means: input node i enters at slot i, its latent is at slot i, its reconstructed row leaves at slot i, so the correct alignment is the identity for every input by construction. Invariance of the objective falls out of equivariance of the map, for free.
- **Invariance is not the opposite of equivariance, it is the special case where the output has no node axis.** A graph-level latent does not lose equivariance, it makes the encoder invariant, and the damage is entirely downstream: T and PTPᵀ both encode to the same z so both decode to the same D(z), which at most one can equal. The network is then punished for failing to reproduce information pooling provably destroyed.
- **Equivariance of the sampled z holds only in distribution**, since ε is drawn independently per node. Any equivariance test must run on μ with sampling off. Worth knowing before writing the test rather than after it fails.
- **The prior KL's invariance depends on the prior being factorized and identical per node.** An exchangeable prior is doing quiet work there; a non-factorized prior over the flat 160-vector would break that term.

### Ascent equivariance, and a check to add

The property the optimization actually rests on, one step beyond the above. The predictor is invariant, so ŷ(PZ) = ŷ(Z); differentiating and using that P is orthogonal gives

```
∇ŷ|_(PZ)  =  P · ∇ŷ|_Z
```

so the gradient field is itself equivariant and **latent gradient ascent commutes with relabelling** — ascending from a relabelled start returns the relabelling of the original result. The answer can never depend on how the input happened to be labelled. Derivation and test procedure now in `architecture.md`: encode T, ascend k steps, decode; encode PTPᵀ, ascend k steps, decode; assert the two agree up to P, with sampling off. **Not yet run** — it joins the checks from 09-14 once `models.py` is finished.

### The decoder's pre-layers are load-bearing

Noticed while checking whether attention already provides a global pathway. It does in the encoder, where every h_i depends on all of T after one layer. In the decoder it does **not** come from the pair function: logits_kl = MLP([z_k ; z_l]) does not involve z_i when i is neither k nor l, so z_i touches only row i and column i, 39 of 380 entries. The decoder's global pathway exists entirely because the few equivariant layers mix the latents before pairing. Those layers are structural, not a refinement, and should not be dropped to save parameters.

### Node-level latents, reaffirmed

The 09-14 choice stands, and the trigger for revisiting it is now specific rather than vague: **needing interpolation between landscapes or prior sampling to work well.** Not disappointing reconstruction and not disappointing single-point ascent — those are β, γ and d. The orbit framing is why: the object representing a graph is the set {z_1 … z_20}, equivalently the orbit {PZ}, which is invariant, and the write-down order leaks only into operations combining *two* latents. Single-matrix ascent, the actual goal, never touches it.

Read **GE-VAE** (arXiv 1910.08057) as a candidate alternative. It is not one: its latent is |V| x P, node-level, so it is the same family we already chose, reached independently — useful as corroboration rather than as an option. Ruled out on three counts, most severe first: its Laplacian-eigenmap encoder reintroduces the discontinuity we rejected canonicalization for, since eigenvectors are defined only up to sign and rotate freely inside near-degenerate subspaces; z_iᵀz_j is symmetric, so T̂_ij = T̂_ji is baked in against our mean abs(T − Tᵀ) = 0.034; and it generates binary undirected topology, so its O(|E| + |V|) selling point is worth nothing on a dense 20-node digraph. The `GNNsources.md` entry had it misfiled next to PIGVAE as though it were graph-level, and is corrected.

### Hybrid latent: an invariant z_graph alongside the equivariant Z_node

Asked whether graph-level dimensions could sit alongside the node-level ones. They can, and the general rule is worth stating: **broadcasting an invariant quantity along the node axis preserves equivariance**, because with respect to that axis it is a constant, acting like a learned bias that happens to depend on the graph. This is Battaglia's global attribute `u` promoted to a latent, and as a latent it is the Neural Statistician (Edwards & Storkey, ICLR 2017).

It adds **no representational power** — by Deep Sets an invariant z_graph is a deterministic function of the node set, so given Z_node it carries zero extra information, and the model can already say anything global by writing it into all 20 node slots. What it adds is cheaper encoding of global facts (one global dimension versus 20 redundant copies, and the prior charges per dimension) and, the real one, **a correlated prior**: drawing z_graph then nodes conditioned on it makes them independent given z_graph but correlated marginally, which is the principled fix for 20 iid draws from N(0,I) having no reason to form a coherent landscape. The risk is posterior collapse on the global slot, since ignoring it is the path of least resistance.

Decision: **build the hook, default it off** — `d_global: int = 0` in `TMVAEConfig`, with the pooling head, broadcast concat and second KL group as no-ops at zero, so enabling it later is a config change rather than a refactor of the decoder signature and the loss. Same trigger as PIGVAE, and it should be tried first, being far cheaper for most of the practical benefit.

### Sources

`GNNsources.md` gained a `## Reading order` section — two paths, the narrow one for the transformer/GNN equivalence and a general ten-item order into the field — plus four entries: Hamilton's *Graph Representation Learning Book* and the canonical Kipf & Welling GCN paper, both conspicuous absences for basic-level grounding; the Neural Statistician; and the global attribute `u` folded into the existing Battaglia entry.

### Next steps
- **Direct optimization without the VAE, as a baseline.** Parameterize a matrix by free logits, apply the masked row softmax, train a predictor directly on T, ascend the logits. Expected to find adversarial matrices — the predictor is only accurate near the data, and nothing confines ascent to that region across 360 free dimensions — which is precisely the argument for the latent, and better demonstrated than asserted. If it does *not* go adversarial, that is important information about how easy the problem is. Related but distinct from the combinatorial edge-editing baseline in `miscSources.md` (arXiv 2008.05589), which a reviewer will also ask for.

## 2026-09-16 — the pipeline end to end, and what the first trained models say

`models.py` finished and committed, then the seven modules around it, then the Snakefile that chains them. `ingest → train → evaluate → visualize → benchmarks` now runs as one DAG; the six-point sweep in `config/sweeps/latent.yaml` is 21 jobs in 5m12s. 158 tests.

The first real runs produced two findings that were predicted in the notes and are now measured, plus one that was not predicted at all.

### The package

| module | what it owns |
|---|---|
| `models.py` | the VAE, the predictor, the loss |
| `config.py` | defaults, sweep expansion, validation, the three schedule evaluators |
| `paths.py` | run naming, the frozen name baseline, `config_guard`, shared filenames |
| `datasets.py` | splits, and the target transform — log always, standardizing fitted on train alone |
| `device.py` | `gpu: auto` picks the least-used card; claim files still unwritten |
| `train.py` | three-layer config resolution, the loop, the run directory |
| `evaluate.py` | property, reconstruction, validity and latent-structure metrics |
| `visualize.py` | four figures, each stamped with provenance |
| `benchmarks.py` | the curated ledger at `docs/benchmarks/runs.csv` |

`config.py` and `paths.py` import nothing heavier than yaml and pathlib, which is what lets the Snakefile expand a sweep and compute every output path before torch is loaded — 27 of the tests run in 0.06s for that reason.

### models.py, and the pooling that was missing

The 09-15 design went in unchanged, and the ascent-equivariance check from that entry now runs and passes rather than being asserted. Checks hold in float64 across four configurations — node-only, hybrid, mean pooling, attention pooling — at machine precision.

One real gap surfaced while writing it. The predictor was `ρ(mean(z_i))`, but Deep Sets is `ρ(Σ φ(z_i))` — φ was the identity. **Wagstaff et al. (ICML 2019)** sharpen this: a mean- or sum-decomposable function needs φ's output at least as wide as the set to represent every continuous invariant function on sets that size. At n = 20 against `d_latent = 8`, the original pooling was under the bound before the missing nonlinearity was even counted. Fixed with a `NodePool` carrying three modes behind a `pooling` config field — `mean` as a baseline, `deepsets` (φ at width 64, which clears the bound) as the default, and `attention` (Set Transformer's PMA) as a sweep axis rather than an argument. 233,326 parameters at defaults.

### Naming runs against a frozen baseline

Run names carry only the fields that differ from a baseline, so a sweep's directory listing reads as what it varied: `TMVAE_dg4-pattention_Tom1000`. Diffing against the live *defaults* would have been wrong, and the failure has two halves — one loud, one silent. Raise the default epochs and a new default run claims the directory an old run occupies, which `config_guard` catches but only by locking it out; then ask for the old value again and it gets a second directory holding a byte-identical config. `paths.NAME_BASELINE` is frozen separately from the defaults, so a default can move freely and nothing renames. Both halves are regression tests.

Two registries enforce the invariant: a determining field with no abbreviation, or no baseline entry, makes `run_name` refuse rather than quietly omit it — the omission being exactly how two sweep points come to share a directory. A field added later is baselined at the value reproducing prior behaviour, so adding one renames nothing. That absorbed three new fields today without touching an existing name.

### Schedules, and one bug they exposed

Added `lr_schedule` (cosine, exponential, off by default), `lr_warmup_epochs`, and `gamma_warmup_epochs` alongside the existing `beta_warmup_epochs`, all evaluated by pure functions in `config.py` so the resolved config stays the only description of a run. `lr_final_frac` is inert without a schedule, so `_finalize` coerces it to a frozen value and sweeping it against a constant rate is refused — otherwise two runs that train identically get two directories.

The bug: model selection ranked on `val_total` **at each epoch's own β and γ**. Under warmup those are different objectives, so an early epoch scores well largely because β is small — the first 300-epoch run's checkpoint came from epoch 10 for exactly that reason. Selection now uses a score recomputed at the *configured* weights from terms already measured. During warmup the gap is visible in the history: at epoch 0 the old rule charged 17.7 nats less than an equivalent post-warmup epoch.

### Posterior collapse at β = 1 is the objective's optimum, not an optimization failure

The first GPU run used a 150-epoch β warmup and collapsed anyway — KL squeezed 131 → 84 → 66 → 26 → 0.15, ending bit-identical to a model that ignores its latent. The warmup bought 100 good epochs and then threw them away.

The arithmetic says why. At epoch 20 the informative solution saved **31.8 nats of reconstruction** and cost **66.6 nats of KL**, so it beats collapse only when

```
β  <  31.8 / 66.6  =  0.478
```

At β = 1, refusing to pay 66.6 to save 31.8 is correct. **No warmup schedule fixes this** — warmup controls when you reach the optimum, not what the optimum is. At β = 0.15 the KL parks at ~64 nats for 280 epochs and the model trains properly: test R² 0.716 on log y, median relative error 21.2%, against R² 0.385 for the β = 1 run.

### What the metrics found

**Forward KL is blind to the weak links, and by a lot.** `architecture.md` predicted this; the by-magnitude table measures it. Bottom decile of true entries (1.7e-09 .. 3.4e-05) is off by a mean of **5.18 nats — a factor of 178** — while contributing 0.0015 of probability-space error. Top decile is off by 0.51 (×1.7). The `reconstruction.png` panels localize it further: the pale rows of a true matrix, nodes with weak out-flow, come back as deep red bands, over-predicted by 100–1000×. That is the case for the auxiliary log-space term the note held in reserve.

**The latent is effectively two dimensions per node.** PCA of the pooled node latents: 0.955, 0.889, 0.042, then everything below 1e-3. Two components clear the posterior noise floor, participation ratio 2.03. Node latents are pooled as `(N × 20, d)` rather than flattened per graph — coordinate *k* means the same thing for every node because the encoder applies one map, but node 3 of one matrix has nothing to do with node 3 of another.

### The latent sweep

`d_latent ∈ {4, 8, 16}` × `β ∈ {0.05, 0.15}`, written to separate two readings of that finding: too much capacity, or too high a price.

| d_latent | β | R² | median rel. | recon | log_recon | KL | above noise |
|---|---|---|---|---|---|---|---|
| 4 | 0.05 | 0.689 | 0.234 | 3.214 | 1.351 | 94.26 | 3 |
| 4 | 0.15 | 0.553 | 0.261 | 5.677 | 1.719 | 63.31 | 2 |
| 8 | 0.05 | 0.637 | 0.201 | 3.008 | 1.381 | 92.96 | 3 |
| 8 | 0.15 | **0.716** | 0.212 | 4.601 | 1.897 | 63.63 | 2 |
| 16 | 0.05 | 0.575 | 0.298 | 3.470 | 1.480 | 97.03 | 3 |
| 16 | 0.15 | 0.587 | 0.262 | 5.907 | 1.919 | 61.95 | 2 |

**β sets the effective width; `d_latent` does not.** Every β = 0.05 run uses three components and ~95 nats, every β = 0.15 run uses two and ~63, across a 4× range of nominal width. The model buys the code the price allows, not the code the container permits — so `d_latent = 4` is not a bottleneck and `d_latent = 16` is 128 unused dimensions per graph. The capacity axis is spent.

Reconstruction responds cleanly to β (recon 3.0–3.5 against 4.6–5.9; `log_recon` ~25% better at the lower price). **R² does not follow** — it is non-monotone in both axes and uncorrelated with reconstruction. Single seeds, 800 training examples, and the training curves already show the property term separating train from val, so the honest reading is that reconstruction is measurable here and prediction is noisy.

### Learning-rate decay buys stability, not accuracy

Matched pair at `d_latent = 8, β = 0.15`, same seed, schedule the only difference:

| | best | last-50 mean | last-50 sd | last-50 range |
|---|---|---|---|---|
| cosine → 5% | 14.132 | 14.244 | **0.053** | **0.251** |
| constant | 14.247 | 14.955 | **0.372** | **2.211** |

The headline gap (R² 0.716 vs 0.691) is within what one seed can say. The variance is not: 7× the sd, 9× the range. A constant rate is still oscillating when it stops, so which checkpoint survives depends on where the bounce lands — the same fragility the configured-weight selection score was added to remove. Decay stays on.

### Housekeeping

`docs/notes/repo_structure.md` is new and holds the layout plus its reasoning, including the `data.yaml` decision, which was implemented in code on 09-14 but never written down — `README.md` and `CLAUDE.md` both still advertised a `config/data.yaml` that was decided against and never created. `parent_path` was added so runs file into folders under `results/`; it is non-determining, so moving a run between folders does not lock it out.

### Next steps
- x**More seeds before trusting any property number.** The sweep's R² ordering is non-monotone in both axes and uncorrelated with reconstruction, which on single seeds is not a result. Three seeds at `d_latent = 8` across β ∈ {0.05, 0.10, 0.15} is nine runs and about eight minutes; `d_latent` can be dropped as an axis now that β is known to set the effective width.
- x**The auxiliary log-space reconstruction term.** The by-magnitude table is the evidence `architecture.md` asked for before adding it: the weakest decile is off by ×178 while paying nothing. Weak links plausibly govern the timescale being predicted, so this is the reconstruction change most likely to move the property.
- x**Direct optimization without the VAE, as a baseline.** Carried over from 09-15 and still the thing that justifies the latent. Parameterize a matrix by free logits, apply the masked row softmax, train a predictor directly on T, ascend the logits. Expected to find adversarial matrices across 360 free dimensions; if it does not, that is important information about how easy the problem is. Distinct from the combinatorial edge-editing baseline (arXiv 2008.05589), which now has a placeholder row in the ledger.
- x**`device.py` claim files, before any multi-GPU sweep.** `gpu: auto` picks the least-used card, so concurrent jobs launched together all pick the same one. Harmless at this model size — three runs shared one A100 at 2.5 GB and 10% — but it is the documented gap.

## 2026-09-18 — regularization, and where the property overfit actually lives

Read the 13 runs on the ledger, found the one axis every sweep so far had left at zero, and swept it. `config/sweeps/regularize.yaml`: β ∈ {0.05, 0.15} × dropout ∈ {0, 0.1} × weight_decay ∈ {0, 1e-3} × seed ∈ {0, 1, 2}, a 2³ factorial replicated three times, 600 epochs. 24 runs, 1h32m serialized on one A100.

### The diagnosis that motivated it

Reconstruction tracked β cleanly and nothing else — flat in `d_latent`, `lr_final_frac` and epochs. Property generalization tracked *nothing*: `val_prop` sat in 0.26–0.49 across all 13 runs and the R² ordering was non-monotone in every axis, on single seeds. Meanwhile the property head overfit 4–5× at the selected checkpoint (train_prop 0.05–0.08 against val_prop 0.27–0.32) while reconstruction did not overfit there at all. Every one of those runs had `dropout = 0` and `weight_decay = 0`.

### Dropout works, weight decay does nothing

Main effects, 12 runs per level, ± SE of the difference:

| | R² | recon | log_recon | median rel. |
|---|---|---|---|---|
| dropout 0 → 0.1 | **+0.045 ± 0.022** | −0.481 ± 0.344 | −0.155 ± 0.095 | **−0.018 ± 0.009** |
| weight_decay 0 → 1e-3 | −0.009 ± 0.024 | +0.034 ± 0.358 | −0.001 ± 0.101 | +0.002 ± 0.010 |
| β 0.05 → 0.15 | +0.027 ± 0.023 | **+1.579 ± 0.123** | **+0.438 ± 0.038** | −0.009 ± 0.010 |

Weight decay's largest effect is a quarter of its own standard error. The pooled recon SE is inflated by β dominating that column, so stratify:

```
beta = 0.05        do 0.0  ->  do 0.1
  recon           2.905   ->   2.249     -0.656 ± 0.090
  log_recon       1.365   ->   1.170     -0.195 ± 0.032
  R2              0.692   ->   0.760     +0.068 ± 0.032
```

**Dropout improved reconstruction and prediction together**, which is not the usual trade, and the overfit ratio collapsed: val_prop/train_prop 3.52 → 1.85 at β = 0.05, 1.72 → 0.97 at β = 0.15.

### The overfit was in the encoder, not the predictor

The prediction going in was the opposite: weight decay reaches every parameter including the property head, dropout reaches only `EquivariantLayer` (`models.py:216`, used by the encoder's 4 layers and the decoder's 2) and never the head at `models.py:335-339`. If the head were memorizing, weight decay would have been the lever. It was not. The encoder was memorizing 800 graphs into latents; the three-layer head reading them was never the problem.

The mechanism is visible in the latent metrics: dropout *raised* usage rather than lowering it — active units 3.0 → 4.0 and KL 94.2 → 99.5 at β = 0.05. Noise inside the encoder forces a distributed code instead of a few brittle directions.

### The β tension resolves in favour of the low price

With dropout on, β = 0.05 and β = 0.15 tie on R² (0.760 vs 0.764) while β = 0.05 reconstructs nearly twice as well (2.25 vs 4.00). The earlier apparent preference for β = 0.15 was the model compensating for an unregularized encoder.

Best cell — β = 0.05, dropout 0.1, wd 1e-3, three seeds:

| | now | prior best on record |
|---|---|---|
| R² | 0.766 ± 0.035 | 0.716 (single seed) |
| recon | 2.206 ± 0.058 | 4.60 |
| log_recon | 1.146 ± 0.067 | 1.897 |

Reconstruction better by 2.1×, log_recon by 1.7×, R² up 0.05 and carrying an error bar for the first time.

## 2026-09-21 — the residual tilt is resolution, not miscalibration

Started from a question about `predictions.png`'s second panel, whose title claimed a trend in the residuals meant a miscalibrated range. It does not, and the panel has been retitled.

### What the tilt is

Regressing the prediction on the truth gives 0.787 ± 0.042 on the b0.05 dropout seed-1 run — the model covers 79% of the true dynamic range. Regressing the truth on the prediction gives 0.998 ± 0.053. Both are the same 100 points; least squares is not symmetric in which variable goes on the left.

For any conditional mean, the tower property gives Cov(ŷ, y) = Var(ŷ), so the first slope is forced to Var(ŷ)/Var(y) = R² and the second to exactly 1. Measured here: 0.787 against a variance ratio of 0.789 and R² 0.780, and the two slopes multiply to r². Shrinkage toward the mean is what an MSE head is supposed to do when the input does not determine the target; it is not a fixable defect, and de-shrinking by a val-fitted slope raised test RMSE from 0.3469 to 0.3713.

The practical rule: plot the residual against the prediction, not against the truth. Against the truth the expected slope is `resolution - 1`, negative for every imperfect model. Against the prediction it is `calibration - 1`, zero, so flat is the correct null and a tilt there is real.

### Sampling noise is not the cause

The property head trains on sampled `z` and evaluates on `mu`, which looked like a candidate. Drawing 64 z per test graph: the induced jitter in the prediction is 0.116 log units against a signal spread of 0.660, nearly unbiased (+0.016), and attenuation of that size predicts a slope near 0.97. Predicting from `z` at eval gives 0.780 against 0.787 from `mu`. Ruled out.

The latent shows why the gap is small. Of 8 dimensions, 4 are dead — μ ≈ 0, σ ≈ 1, pure prior noise in training and exactly 0 at eval. The 4 live ones run σ 0.15 to 0.86 against μ spreads of 0.51 to 1.00, and the pool averages independent per-node noise over 20 nodes before the MLP sees it.

### The sweep, both slopes

All 24 regularization runs re-evaluated and redrawn.

| | range | mean |
|---|---|---|
| resolution | 0.707 – 0.906 | 0.804 |
| calibration | 0.874 – 1.040 | 0.940 |

Resolution tracks R² cell by cell, as it must, and neither β nor dropout nor weight decay moves it beyond seed noise. Calibration sits below 1 in 22 of 24 runs, the mild-overfit signature of predictions a few percent too extreme. The three seeds are the three splits and give 0.910, 0.964 and 0.946, so the split-to-split spread is as large as the shortfall — a consistent lean on three quasi-independent measurements, not yet a demonstrated bias.

## 2026-09-21 — the log-space term, and what actually sets reconstruction quality

Started from the over-prediction visible in `reconstruction.png`. It is real, it is signed, and `architecture.md` predicted it before any model trained: forward KL weights each term by the true T_ij, so an entry of 1e-6 contributes nothing and its fitted value is set by the decoder's inductive bias rather than by the data. Measured on the b0.05 dropout seed-1 run, the weakest decile is over-predicted by a mean of +1.26 in log10 with 97% of entries on the high side, the fit is `log₁₀ T̂ = 0.689·log₁₀ T − 0.514`, and the row softmax turns that compression into a 3.0× inflation of the mass held by each row's weakest ten entries.

### The term

`log_recon` existed as a no-grad diagnostic. `lambda_log` makes it a loss term — unweighted mean absolute log error over the 380 off-diagonal entries, L1 rather than squared because log errors span ten decades. Per entry the KL's gradient weight is T_ij and this term's is `lambda_log / 380`, which cross near the median entry at `lambda_log = 1`.

Across λ ∈ {0, 0.3, 1, 3, 10}, three seeds: mean bias +0.312 → −0.004, weakest decile ×14 → ×1.1, fit slope 0.713 → 0.984, row-tail inflation 2.79× → 1.60×. `recon` did not degrade — it improved, 2.206 → 1.740 — and KL rose 99.8 → 139.3.

### It was buying latent capacity, not trading against the KL

`lambda_recon` was added to make the mix an axis. The `lambda_recon = 0` cell collapses: KL 17.6, one active unit of eight, and `log_recon` three times *worse* at 2.007 despite being the only reconstruction term left. The arithmetic is the 09-16 posterior-collapse calculation — an informative latent saves `3·(2.344 − 0.705) = 4.9` nats of log_recon against a `0.05·(115.9 − 17.8) = 4.9` nat KL bill, exactly break-even, so nothing punishes collapse. With `recon` on, the same latent also saves ~63 nats, 13× the bill. `recon` is what makes the code worth its price; `log_recon` is a *mean* over 380 entries and cannot outbid β on its own.

The scale control settles the rest. (λ_recon 2, λ_log 6) has the same ratio as (1, 3) and does not reproduce it — KL 151.4 against 117.1 — landing instead with (2, 3). So the ratio is not the axis; the reconstruction block's weight against β is, and part of what the λ_log ladder measured was β being diluted.

What survives is a decomposition, and the collapsed cell is what proves it: at KL 17.6, the lowest capacity in the sweep, bias is +0.045, near-centred. **λ_log sets where the log-space error sits; latent capacity sets how tight it is.**

### β is the direct knob

At λ_recon 1, λ_log 3, three seeds:

| β | KL | recon | log_recon | bias | scatter | slope | R² | med rel |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 211.1 | 0.785 | 0.421 | −0.008 | 0.273 | 0.992 | 0.790 ± 0.020 | 0.193 |
| 0.02 | 164.8 | 1.098 | 0.477 | −0.012 | 0.317 | 0.992 | 0.771 ± 0.023 | 0.200 |
| 0.05 | 117.1 | 1.848 | 0.650 | +0.005 | 0.404 | 0.957 | 0.762 ± 0.043 | 0.218 |
| 0.10 | 86.3 | 2.736 | 0.908 | +0.094 | 0.526 | 0.895 | 0.766 ± 0.025 | 0.229 |
| 0.15 | 73.2 | 3.422 | 1.031 | +0.117 | 0.584 | 0.878 | 0.759 ± 0.027 | 0.240 |

β spans a wider capacity range than either λ and every reconstruction column tracks it monotonically. β = 0.01 is the best reconstruction on record here. R² stays flat at 0.759–0.790, but median relative error improves monotonically 0.240 → 0.193, so the property does respond weakly on the robust metric while R² does not.

### A scalar latent does not suffice

The `lambda_recon = 0` collapse scoring R² 0.806 on one active unit suggested the target might be near-scalar. It is not, and that inference was confounded: with no reconstruction pressure the encoder is free to spend its one unit entirely on the property. Forcing the width instead, at three seeds:

| d_latent | R² | recon | log_recon | KL | active |
|---|---|---|---|---|---|
| 1 | 0.442 ± 0.129 | 8.357 | 1.669 | 74.7 | 1.0 |
| 2 | 0.781 ± 0.022 | 4.286 | 1.040 | 95.8 | 2.0 |
| 4 | 0.792 ± 0.030 | 1.751 | 0.623 | 118.3 | 4.0 |
| 8 | 0.762 ± 0.043 | 1.848 | 0.650 | 117.1 | 4.0 |

One dimension per node costs R² 0.35 and triples the log-space error. Two recovers the property, four recovers reconstruction, and eight is four wasted dimensions — 4 and 8 agree on KL, active units and every reconstruction column, which is the 09-16 finding that β sets the effective width, now measured from the other side.

## 2026-09-22 — the global latent: one axis, relocated rather than added

`config/sweeps/global_dims.yaml` adds a graph-level latent beside the 20 node latents, at the settled β 0.01, λ_recon 1, λ_log 3, `d_latent` 8, with `d_global` ∈ {1, 2, 3, 4} × three seeds. The hybrid had been in `permutationsandlatent.md` since 09-16; this is the first time it was trained. Twelve runs took 31.5 min at two concurrent jobs, 288–317 s each, against the ledger's 327 s median for 600 epochs.

### The property does not move

Mean ± sd over the three seeds, each seed being its own split; the d_global 0 row is the β 0.01 cell of `beta_capacity`:

| d_global | R² | med rel | log_recon | global KL |
|---|---|---|---|---|
| 0 | 0.790 ± 0.035 | 0.193 | 0.421 | — |
| 1 | 0.782 ± 0.059 | 0.195 | 0.423 | 2.90 ± 0.22 |
| 2 | 0.789 ± 0.061 | 0.203 | 0.405 | 2.89 ± 0.02 |
| 3 | 0.781 ± 0.055 | 0.196 | 0.393 | 1.64 ± 1.38 |
| 4 | 0.802 ± 0.050 | 0.202 | 0.419 | 1.79 ± 1.36 |

Every cell sits within a seed sd of the baseline. The seed is the larger axis: seed 0 scores ≈ 0.72 at every width, seeds 1 and 2 0.79–0.84, which is the split and not the model.

### Exactly one dimension, whatever the width

All twelve runs leave exactly one global dimension active at ≈ 2.9 nats; the extra dimensions sit on the prior. Seed 1 collapsed it entirely at d_global 3 (0.05 nats) and 4 (0.2), and seed 2 half-collapsed it at 3 (2.0). Nothing at d_global 1 collapsed, but three seeds per cell cannot say whether width invites it.

### The information moved; none was added

A linear probe of log y from the graph-mean node latents, 5-fold CV R² over all 1000 graphs, falls from 0.86–0.90 at baseline to 0.62–0.79 once the global is active, and appending the global restores it to 0.86–0.93. The two collapsed runs read 0.89–0.90, indistinguishable from baseline. The between-graph share of node-latent variance drops accordingly, 2.0–2.7% → 1.7–2.0%. The encoder hands the graph-level signal to the global slot and the node latents stop carrying it.

The model depends on it. Clamping the global to its test mean drops test R² from ≈ 0.72–0.83 to 0.20–0.44 and raises log_recon from ≈ 0.40 to 0.48–0.59; a ±2 sd shift moves predicted log y by ≈ ±1.5. The graph-mean node latents predict only 24–73% of its variance, and it is near-uncorrelated with the sorted-160 PC1 (|r| ≲ 0.2), so it is not a rotation of the dominant node-side axis.

### What the axis is

The same thing in every run, correlations over the 1000 graphs with each run's sign aligned to log y:

| graph feature | r |
|---|---|
| std over nodes of min log outflow | −0.77 … −0.88 |
| std over nodes of log stationary π | −0.76 … −0.86 |
| mean over nodes of log stationary π | +0.72 … +0.84 |
| log decay exponent | +0.63 … +0.85 |
| log relaxation time from abs(λ₂) | −0.69 … −0.80 |
| mean inflow | ≈ 0 |

Node heterogeneity: high where the stationary distribution is even and the weakest links are alike across nodes, which goes with faster decay. It is a property of the whole graph that no single node carries — what a global slot is for — and it is learned reproducibly. `d_global = 1` is sufficient.

One trap for reading the figures: a collapsed global still has a posterior mean that varies across graphs and tracks the node latents at r ≈ 0.85, but at 0.06 nats the decoder and predictor ignore it. Correlation in `latent_property.png` does not mean use; read the nats on the axis first.

### Figures

`latent.png` gains a global row beneath the node row. `latent_property.png` gains two rows for global runs: the global against the best graph-mean node axis by rate, by property and by sorted PC1, and the sorted PCA with the global appended raw and at ×√20. Raw, the global is 4–6% of the variance and loads ≈ 0–0.24 on the top two components; at ×√20 it takes PC1 at ≈ 0.99. Neither weighting is privileged, so both are drawn, and the global-against-PC1 panel says the same thing more directly.

### Next steps
- **Latent traversals and interpolation.** Decode along each active dimension, node and global, holding the rest at a graph's encoding, and show the matrix and its predicted property changing. The global is the natural first traversal, being a single graph-level scalar that needs no alignment. Interpolation between two graphs needs the Hungarian alignment of node slots from 09-14 first, since the node latent is a set; the global half interpolates directly.
- **Optimizing the property in latent space.** Gradient ascent on the predicted property from an encoded matrix, the goal the architecture was built for; ascent equivariance is already verified in `architecture.md`. The global axis, r ≈ 0.8 with log y, is a candidate one-dimensional search direction to compare against full ascent. Predicted gains mean nothing until the decoded matrices are scored by the true generator, which lives outside the repo.

## 2026-09-23 — latent traversals

Figures now live in `results/<run>/figures/`, and `visualize` adds three traversals in `figures/interpolation_and_traversals/`: the highest-rate global dimension and sorted-160 PC1 and PC2. Each is 3 test graphs × 7 steps, t ∈ {−3 … +3}·σ from the graph's own encoding, where σ is the spread of the axis coordinate over the 100 test graphs. The centre column is the reconstruction, and every matrix carries its predicted decay exponent. A PC or latent dimension has no natural sign, so each axis is oriented so that +t raises log y across the test graphs.

PC1 and PC2 are sorted-160 rather than raw-160 because node labels are arbitrary. Over exchangeable slots the raw covariance has only two kinds of eigenvector: a uniform shift, which is just the graph-mean PCs broadcast to every node, and slot contrasts, each degenerate 19-fold in expectation, so which one comes out is sampling noise. A step along a sorted PC is decoded by shifting the sorted values and putting each rank back on the node that held it.

First read on the dg1 seed-1 run. The global is strongly one-sided. On test example 2, ŷ goes 649 → 2251 at +2σ but only 649 → 305 at −2σ; on example 0 it barely moves below zero (88 → 92 at −3σ) and climbs to 767 at +3σ. Visually, +t evens out the rows and removes the pale weak-link bands. Sorted PC1 (15.1% of variance) is smooth and near log-linear, moving ŷ about ±30–50% over ±3σ, far less than the global.

Existing runs were migrated: 417 PNGs across 100 runs moved into `figures/`, and the 12 `global_latent` runs re-rendered.

### Interpolation

`interpolation.png` joins the traversals: test 0 → 1, 1 → 2 and 2 → 0, seven evenly spaced steps along the straight line between the two encodings. The node latent is a set, so the right graph's nodes are first Hungarian-matched to the left's on squared distance between node μ; the global interpolates unmatched. Every column is drawn in the left graph's node order, and the right end is exactly the right graph's reconstruction relabelled, which a test checks through the decoder's equivariance.

On the dg1 seed-1 run matching roughly halves the mean paired node distance (2.80 → 1.46, 2.89 → 1.69, 2.88 → 1.68), and ŷ moves smoothly between the endpoints on every row. Along 1 → 2 (235 → 649) it changes slowly at first and faster toward the end, and example 1's pale weak-link bands fade by about α = 3/6.
