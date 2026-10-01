# Does any optimization find the uniform matrix as the maximum?

Branch `claude-optimization`, 2026-09-30. The reference model is
`results/input_norm/TMVAE_b0p01-bw20-dg1-do0p1-enormfixed-e600-ll3-lff0p05-schcosine-s1_Tom1000`
(fixed edge norm, layernorm node norm, d_global 1, β 0.01, seed 1, test R² 0.813).
Scripts are in `src/tm_ml/uniform/`, run as `pixi run -e ml python -m tm_ml.uniform.<script>`.

The slides (`2026-9-11 The ML Problem.pptx`, slide 5) put the well-mixed landscape,
equal migration to every node, at the top of the range: a_max = 20 × 100 = 2000,
with 20 isolated patches at a_min = 100. With nobody staying put, that landscape is
U, with U_ij = 1/19 off the diagonal. Every measure of distance from U below is the
mean over the 20 rows of KL(T_i ‖ U_i) = log 19 − H(T_i), in nats, 0 only at U.

## Short answer

**For the model as it stands, no.** Every optimizer that reaches U does so through a
degeneracy of the decoder, not because the predictor rates U highest. Every optimizer
that scores the matrix honestly, through the encoder, finds matrices it rates
50–500× higher than U, and the model's own score for U is 671, a third of 2000.
Three things cause it, each sufficient alone:

1. **U is far outside the data.** The nearest of the 1000 matrices is 1.40 nats per
   row from U (median 1.94). The data constrain ŷ only out there, and say nothing about
   the region between them and U.
2. **ŷ(z) is not a function of the matrix.** Every latent with identical node latents
   decodes to exactly U, and ŷ over those latents, inside the data's box, runs from
   0.3 to 25,000. Optimizing ŷ(z) can make U score anything.
3. **The predictor is unbounded and extrapolates.** Along the global, log ŷ rises
   linearly forever. Through the encoder it is bounded, since the encoder ends in a
   LayerNorm, but the bound is near 4 × 10⁵.

**With the model changed, yes.** <!-- filled below -->

## 1. Where U sits

`where.py`.

- The data never come near U: mean row KL to U from 1.40 to 2.46 over the 1000 graphs.
  Uniformness does not predict the target inside the data either: Spearman ρ between
  KL to U and y is −0.05 over the 1000 graphs, as `data_figures.py` found on 09-23.
- **ŷ(enc U) = 671.** The encoder reconstructs U exactly, puts its global at the 99.4th
  percentile of the data's, and gives all 20 nodes the same latent, which no data graph
  has (node spread 0 against 1.36–3.12). That common latent is 3.8 from the nearest
  graph mean.
- Mixing the most uniform training graph toward U, (1 − s)·T + s·U, ŷ goes 1020 →
  1446 at s = 0.25 → 671 at s = 1.

## 2. The model's landscape around U is a crater

`hessian.py`, and the directional probes in it.

The encoder is permutation-equivariant and the predictor pools, so the matrix-level
score F(T) = log ŷ(enc T) is invariant under relabelling, and U is fixed by every
relabelling. For a smooth F that makes ∇F(U) a constant on the off-diagonal entries,
which a row softmax does not see: **U is a critical point of any model built this
way**, and of the true effective population too, which is just as invariant.

F is not smooth at U here: two node features are `T.max(-1)` and `T.max(-2)`, and at U
all 19 entries of each row tie. Autograd picks one argmax, which is why the autograd
gradient at U is 5e-3 and not 0 and why the Hessian there is unreliable. One-sided
slopes are well defined: along all 500 random logit directions F falls from U, at
about −0.0008 in log ŷ per unit of logit. U is a strict but very shallow local
maximum, a kink.

Farther out, ŷ(enc T) on rings of logit radius r around U, 300 directions each:

| r | KL to U | random directions: median / max ŷ | data-shaped: median / max ŷ |
|---|---|---|---|
| 2 | 0.005 | 671 / 671 | 671 / 676 |
| 10 | 0.12–0.19 | 694 / 932 | 1127 / 1905 |
| 20 | 0.44–0.78 | 1186 / 1555 | 1446 / 2042 |
| 40 | 1.2–1.8 | 537 / 1319 | 375 / 1500 |

A rim at 1100–2000 between the data and U, and a crater floor of 671 at U. Along the
straight path from each of the 100 test graphs to U (`mixing.py`) the median ŷ goes
251 → 1236 at s = 0.25 → 671 at U; no path peaks at U.

## 3. Many latents decode to U

`where.py`, part 3. The decoder is equivariant, so identical node latents can only
decode to identical rows, which with a zero diagonal is U. That holds for every
common node latent c and every global g, a 9-dimensional fibre (8 node dims + 1
global). Over 2000 random points of it inside the data's box, the decoded KL to U
is below 1e-15, and ŷ(z) runs from 0.7 to 6049; maximizing and minimizing over the
fibre inside the box gives 25,074 and 0.3. ŷ(enc dec z) is 670.9 at every one of them,
since it is one matrix.

Consequences for this problem:

- **A latent optimizer that reaches U reports a ŷ chosen by its penalty, not by U.**
  With the N(0, I) prior at λ = 1 all 11 starts collapse onto the fibre and report
  ŷ = 2493, close to the slides' 2000. That looks like the answer and is not: λ = 0.1
  reaches the same U and reports 867,500, and unregularized BFGS 10³⁰⁴.
- **The fibre is an attractor.** Identical node latents are the cheapest way to shrink
  ‖x‖ (160 of the 161 coordinates), so the prior penalty's optimum lands on it
  whatever ŷ says, as on 09-23.
- **Only a matrix-level score sees U as one point.** ŷ(enc dec z) collapses the fibre,
  and every comparison below uses it.

## 4. Up the global

`global_axis.py`. Holding a test graph's node latents and setting the global to g:

| g | ŷ(z) | KL(dec z ‖ U) | ŷ(enc dec z) |
|---|---|---|---|
| −1000 | 1e-139 | 2e-5 | 671 |
| −100 | 2e-12 | 2e-3 | 674 |
| −10 | 11 | 0.40 | 1222 |
| 0 (data range −1.6 … 1.7) | 153 | 2.04 | 190 |
| +10 | 1.6e6 | 0.87 | 1647 |
| +100 | 5e36 | 6e-4 | 671 |
| +1000 | inf | 2e-6 | 671 |

(graph 9; the other three graphs checked agree.)

**Yes, the decoded matrix tends to U, in both directions, and only in the limit.**
KL to U falls like 1/g². `from_latent` is affine in [z_i ; g], so each node's first
hidden state is g·w_g plus an O(1) node term; at large |g| every node points along
±w_g, and the pre-norm decoder layers and the final LayerNorm see only the direction.
The nodes become identical, which is the fibre again. Nothing about g → +∞ is special:
g → −∞ decodes to the same U while ŷ(z) goes to zero.

**ŷ(z) keeps rising because the predictor is an MLP with GELUs, linear far from the
data**, with a positive slope in g everywhere (09-24). Scored through the encoder,
the same path peaks around 1650 at g ≈ 5–20 and returns to 671 at U.

## 5. Optimizers and penalties on the reference

`methods.py` for the latent, `matrix_space.py` for the matrix. Starts: 10 test graphs
and enc(U); medians are over those 11 starts.

Latent space, maximizing ŷ(z) unless noted:

| method | median ŷ(z) | median ŷ(enc dec z) | median KL to U | reaches U |
|---|---|---|---|---|
| free | 10³⁰⁴ | 673 | 0.003 | yes, by decoder saturation |
| prior λ = 1 | 2493 | 671 | 0 | yes, by node collapse |
| prior λ = 0.1 | 8.7e5 | 671 | 0 | yes, by node collapse |
| anchor λ = 1 | 2121 | 671 | 1.92 | from the U start only |
| anchor λ = 0.1 | 8.0e5 | 1154 | 1.02 | from the U start only |
| box (data range) | 2.4e4 | 1085 | 0.05 | 3 of 11 |
| aggregate posterior λ = 1 | 1014 | 1560 | 0.94 | no |
| aggregate posterior λ = 0.1 | 1.5e5 | 1269 | 0.11 | no |
| cycle ‖enc dec x − x‖², λ = 1 | 3.1e4 | 4733 | 1.58 | no |
| cycle, λ = 0.1 | 1.7e6 | 1486 | 0.96 | no |
| maximize ŷ(enc dec x) | 202 | 13,031 | 1.63 | no |
| the same + aggregate posterior | 244 | 8,324 | 1.71 | no |
| fibre only + aggregate posterior | 1.4e5 | 671 | 0 | by construction |

The aggregate posterior is a 20-component Gaussian mixture on the 20,000 node
latents, per node, and a Gaussian on the global.

Matrix space, maximizing ŷ(enc T) over three families, 8 test graphs and 8 random
starts each:

| family | median end ŷ | largest | ends at U |
|---|---|---|---|
| all row-stochastic, rowsoftmax(L) | 1.9e5 | 3.8e5 | 0 / 16 |
| reversible, diag(W·1)⁻¹W, W symmetric | 1.0e5 | 1.7e5 | 0 / 16 |
| landscapes, 20 points with W = D⁻⁴ | 5,700 | 9,400 | U is not one |

Started at U itself, every matrix-space search stays there, since U is a local
maximum, at 671.

The general and reversible optima keep entries inside the data's range (smallest
1e-12 to 1e-16, data 1.3e-14), so they are not tiny-entry artefacts. The general ones
break detailed balance, asymmetry up to 1 against 0 in the data, and none is planar,
with 50–65% of the MDS spectrum outside the plane. The landscape optima are valid
landscapes, only more even than any in the data: minimum row entropy 0.15–0.84
against a median of 0.044, and a 4 × 5 grid of points scores 6,986. A landscape needs
20 points in the plane at mutually equal distances to be uniform, so the landscape
family cannot contain U at all, and its true maximum is something else.

## 6. β

`config/sweeps/uniform_beta.yaml`, seed 1, otherwise the reference:

| β | test R² | log_recon | global KL | ŷ(U) | matrix-opt max | paths peaking at U | ŷ(z) at g = +100 |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.813 | 0.18 | 3.2 | 671 | 3.8e5 | 0 / 100 | 5e36 |
| 0.1 | 0.763 | 0.37 | 0 | 1443 | 8.0e4 | — | 260 |
| 0.3 | 0.742 | 1.01 | 0 | 1558 | 4.8e4 | 58 / 100 | 318 |
| 1 | 0.000 | 3.81 | 0 | 306 | 306 | — | 257 |

- **The global runaway disappears, because the global collapses.** At β ≥ 0.1 the
  global slot carries 0 nats and the predictor ignores it.
- **U moves up**, 671 → 1443 → 1558, and at β 0.3 58 of 100 paths from test graphs
  peak at U. A stronger prior pulls q(z) toward N(0, I), and U's identical-node
  encoding toward where the data are.
- **U is still not the optimum.** Matrix-space search still finds 5–8 × 10⁴, and R²
  and reconstruction both worsen. At β = 1 the posterior collapses entirely and the
  predictor outputs a constant 306.

β moves the model in the right direction on U and does nothing about the two
structural problems, the fibre and the unbounded predictor; it trades the property
for it.

<!-- anchor, head, star sections to follow -->
