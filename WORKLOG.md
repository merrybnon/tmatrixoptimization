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
