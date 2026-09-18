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
- Finish `models.py` and commit it as is for now.
- Find literature on the problem of going from landscapes to genetic diversity more broadly.

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
- **More seeds before trusting any property number.** The sweep's R² ordering is non-monotone in both axes and uncorrelated with reconstruction, which on single seeds is not a result. Three seeds at `d_latent = 8` across β ∈ {0.05, 0.10, 0.15} is nine runs and about eight minutes; `d_latent` can be dropped as an axis now that β is known to set the effective width.
- **The auxiliary log-space reconstruction term.** The by-magnitude table is the evidence `architecture.md` asked for before adding it: the weakest decile is off by ×178 while paying nothing. Weak links plausibly govern the timescale being predicted, so this is the reconstruction change most likely to move the property.
- **Direct optimization without the VAE, as a baseline.** Carried over from 09-15 and still the thing that justifies the latent. Parameterize a matrix by free logits, apply the masked row softmax, train a predictor directly on T, ascend the logits. Expected to find adversarial matrices across 360 free dimensions; if it does not, that is important information about how easy the problem is. Distinct from the combinatorial edge-editing baseline (arXiv 2008.05589), which now has a placeholder row in the ledger.
- **`device.py` claim files, before any multi-GPU sweep.** `gpu: auto` picks the least-used card, so concurrent jobs launched together all pick the same one. Harmless at this model size — three runs shared one A100 at 2.5 GB and 10% — but it is the documented gap.
