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
