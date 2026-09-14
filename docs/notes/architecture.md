# Architecture

Design note for the transition-matrix VAE. Follows the scheme in `project_vision.md`: encode a landscape into a continuous latent space, predict its diversity from that latent jointly with training, then do gradient-based optimization in the latent space and decode the result.

## What the data forces

Measured on `data/processed/Tom1000` (1000 matrices, 20×20):

| property | value | consequence |
|---|---|---|
| diagonal | exactly 0 in all 1000 | no self-transitions; mask it structurally rather than learn it |
| other zeros | none | every off-diagonal entry is strictly positive |
| entry range | 1.3e-14 .. 1.0 | 15% of entries below 1e-4; linear features cannot see them |
| asymmetry | mean abs(T − Tᵀ) = 0.034 vs mean abs(T) = 0.050 | strongly directed; edges need both directions |
| column-sum near-ties | 86% of matrices have an adjacent gap < 0.01, 20% < 0.001 | canonical node ordering is discontinuous — see below |
| spectral gap | abs(λ₂) ≈ 0.9931, corr(target, relaxation time) ≈ 0.03 | the target is not a simple function of λ₂; the predictor has real work to do |
| target | 48.8 .. 2041.6, right-skewed | predict log(target), not target |

Free parameters per matrix: 20 rows × (19 positive entries − 1 sum constraint) = 360, inside a 400-dimensional ambient space.

## Permutation symmetry is structural, not learned

Node labels are arbitrary: relabelling states gives the same landscape and the same decay exponent. Two ways to handle that.

**Canonicalize** — sort nodes by column sum, then use an ordinary MLP or CNN with a single flat latent. Rejected: with 86% of matrices holding two nodes whose in-flows differ by less than 0.01, an arbitrarily small change to T swaps two nodes and permutes whole rows and columns of the canonical form. That is a map with jump discontinuities scattered densely through input space, to be learned from 1000 examples, and it directly undermines the smooth latent space the optimization depends on.

**Build it in** — the design below is equivariant end to end and invariant exactly where it must be:

- encoder: permuting T permutes the node latents identically
- decoder: permuting node latents permutes the output identically
- predictor: permutation *invariant*, since relabelling cannot change a decay exponent

Element-wise reconstruction loss is then well-posed with no canonicalization and no graph matching. One consequence: permutation augmentation becomes a no-op, since a permuted example produces an identically permuted output and the same loss.

## The model

```
T (20×20, zero diagonal)
   │
   ├─ node features  n_i = [col_sum_i, row_entropy_i, col_entropy_i, max_j T_ij, max_j T_ji]
   └─ edge features  e_ij = [T_ij, T_ji, log T_ij, log T_ji]
   │
   ▼
ENCODER — L layers of edge-conditioned attention over the complete digraph
   h_i ← h_i + MLP( h_i , Σ_j α_ij · W[h_j ; e_ij] )
   │
   ▼
   μ_i , log σ_i   →   z_i ∈ ℝ^d          latent = 20 × d
   │
   ├──────────────────────────────┐
   ▼                              ▼
DECODER                        PREDICTOR (invariant)
 few equivariant layers          pool over i: mean ‖ sum ‖ attention
 logits_ij = MLP([z_i ; z_j])    → MLP → log(decay exponent)
 diagonal → −inf
 row softmax
 → T̂, a valid transition matrix by construction
```

Key parameters:
- `d`: per-node latent width, start at 8 for a 160-dim bottleneck against 360 intrinsic dimensions
- `L`: encoder layers, start at 3-4; the graph is complete so every node is reachable in one hop and depth buys refinement, not range
- Node features must be permutation-invariant scalars — feeding row `T[i,:]` as a node feature breaks equivariance, since its entries are ordered by j
- Edge features carry both directions; treating the graph as undirected discards most of the off-diagonal signal
- Logs in the edge features, or everything below 1e-4 is invisible to the network
- The decoder pair function must be asymmetric (concatenation, not a symmetric bilinear form) so that T̂_ij ≠ T̂_ji is representable
- n = 20 means 400 pairs, so full dense attention is cheap and no neighbourhood sampling is needed

## Loss

```
L = Σ_i KL( T_i ‖ T̂_i )  +  β · Σ_i KL( q(z_i|T) ‖ N(0,I) )  +  γ · ( log ŷ − log y )²
```

Per-row KL is the categorical likelihood written out, which is what rows-are-distributions implies; MSE in probability space would be a choice against the grain of the data.

Joint training of the predictor is the mechanism, not an add-on: it is what organizes the latent space so the property varies smoothly and gradient ascent has something to climb. γ trades reconstruction against a useful latent geometry and matters as much as β.

Gotcha: forward KL weights each term by the true T_ij, so it punishes T̂ being too small where truth is large but barely notices T̂ being too large where truth is tiny. Weak links plausibly govern the timescale being predicted, so track a log-space reconstruction error as a separate diagnostic from the start and add an auxiliary log-space term if it diverges from the KL.

Unlike the binary landscapes in hotspotLandscapes, there is no entropy floor on the neg-ELBO here: the data is continuous on a 360-dimensional manifold, and a continuous density can be arbitrarily peaked, so the reconstruction term is unbounded below.

## Sources

The scheme as a whole — VAE plus jointly trained property predictor, then gradient ascent in latent space — is Gómez-Bombarelli et al., *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules*, ACS Central Science 4(2), 2018. Jin et al., *Junction Tree Variational Autoencoder for Molecular Graph Generation*, ICML 2018, is the follow-up that makes decoded objects valid by construction rather than valid by penalty, which is the same motivation as the masked row softmax here.

The encoder update line decomposes into:

| piece | source |
|---|---|
| aggregate-then-update skeleton, messages conditioned on edge features | Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017 |
| edge features determining the transform applied to a neighbour | Simonovsky & Komodakis, *Dynamic Edge-Conditioned Filters*, CVPR 2017 |
| attention coefficients α_ij over neighbours | Veličković et al., *Graph Attention Networks*, ICLR 2018; use the GATv2 form from Brody et al., *How Attentive are Graph Attention Networks?*, ICLR 2022 |
| edge features folded into the attention itself | Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs*, 2021; Ying et al., *Graphormer*, NeurIPS 2021 |
| residual connection | He et al., *Deep Residual Learning*, CVPR 2016; standard in deep GNNs since Li et al., *DeepGCNs*, ICCV 2019 |
| sum aggregation being the expressive choice | Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019 |
| the general framework these are all instances of | Battaglia et al., *Relational inductive biases, deep learning, and graph networks*, 2018 |

Node-level latents with a pairwise decoder is the structure of Kipf & Welling, *Variational Graph Auto-Encoders*, 2016, with a learned asymmetric pair function in place of their symmetric inner product. Simonovsky & Komodakis, *GraphVAE*, ICANN 2018, is the alternative we are avoiding: a graph-level latent plus approximate graph matching to make the reconstruction loss well-posed.

Invariant pooling in the predictor is Zaheer et al., *Deep Sets*, NeurIPS 2017; attention pooling is the PMA block of Lee et al., *Set Transformer*, ICML 2019. The equivariance framing is Maron et al., *Invariant and Equivariant Graph Networks*, ICLR 2019, and Bronstein et al., *Geometric Deep Learning*, 2021.

VAE and β-weighting: Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014; Higgins et al., *β-VAE*, ICLR 2017.

## Open risks

1000 examples against a few hundred thousand parameters is the binding constraint. Equivariance is worth a large effective-data multiplier, but more drops would buy more than any architectural change.

The target does not correlate with the spectral gap, so what it does depend on is currently unknown. Worth understanding before trusting the predictor, since it decides whether the model is learning the mechanism or fitting noise.

Closing the optimization loop needs ground truth for matrices that have never existed, which means running the external generator on demand rather than receiving finished batches. Without that, an optimized T̂ has a predicted exponent and no way to check it.
