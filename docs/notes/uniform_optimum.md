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

**With the model changed, yes.** A predictor head whose maximum is U by construction,
log ŷ = log 2000 − ‖A·(φ(x) − φ(x_U))‖², trains to the same R² as the reference. Matrix-
space ascent of log ŷ(enc T) − ε·KL(T ‖ U) then ends at U from all 16 starts, 8 test
graphs and 8 random matrices, at ŷ = 2000 (section 8). On the reference the same search
ends 0.3–0.6 nats per row from U at ŷ ≈ 10⁵.

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

## 7. Telling the model what the slides say

Three training options, all off by default, on this branch (`models.py`, `train.py`):

- `anchor_uniform`: weight on (ŷ(enc U) − 2000)², on every step.
- `prop_cap`: a soft ceiling, log ŷ = c − softplus(c − s), with c = log 2200, just
  above the largest target (2041.6).
- `prop_head: uniform_max`: log ŷ(x) = log 2000 − ‖A·(φ(x) − φ(x_U))‖², φ the
  predictor's hidden layers, x_U the pooled encoding of U under the current weights,
  A a learned 16 × 64 projection. ŷ(enc U) = 2000 exactly and nothing scores above it.
- `lambda_star`: a star-shaped constraint around U. For each matrix in the batch and
  as many random row-stochastic matrices, two points s₁ < s₂ on the segment toward U,
  and relu(ŷ(T(s₁)) − ŷ(T(s₂)) + 0.05·(s₂ − s₁)) in standardized log y. No labels.
  If ŷ rises along the segment from every matrix to U, U is the maximum.

`compare.py` (searches from 8 test graphs and 8 random starts; `→ U` counts ends within
KL 0.01 of U) and `mixing.py` (the 100 test graphs' straight paths to U), seed 1:

| run | R² | ŷ(U) | matrix opt max | reversible max | landscape max | → U | paths peaking at U | ŷ(z), g = +100 |
|---|---|---|---|---|---|---|---|---|
| reference | 0.813 | 671 | 3.8e5 | 1.7e5 | 9,365 | 0/16 | 0/100 | 5e36 |
| cap | 0.833 | 1405 | 2,197 | 2,194 | 2,158 | 0/16 | — | 2200 |
| anchor 1 | 0.817 | 2013 | 2,984 | 2,803 | 2,291 | 0/16 | 0/100 | 2e10 |
| anchor 10 | 0.798 | 2003 | 2,473 | 2,393 | 2,202 | 0/16 | — | 4e10 |
| anchor 1 + cap | 0.831 | 2006 | 2,193 | 2,189 | 2,168 | 0/16 | — | 2200 |
| anchor 10 + cap | 0.798 | 2011 | 2,075 | 2,053 | 2,027 | 0/16 | 4/100 | 2200 |
| uniform_max head | 0.793 | 2000 | 2,000 | 2,000 | 2,000 | 0/16 | 100/100 | 7e-138 |
| anchor 10 + star 1 | 0.829 | 2019 | 2,127 | 2,105 | 2,043 | 0/16 | 100/100 | 3e10 |
| anchor 10 + star 10 | 0.835 | 2001 | 2,195 | 2,140 | 2,137 | 0/16 | 100/100 | 7e11 |
| anchor 10 + star 1 + cap | 0.794 | 2008 | 2,130 | 2,114 | 2,051 | 0/16 | 100/100 | — |
| anchor 10 + star 10 + cap | 0.795 | 2042 | 2,111 | 2,088 | 2,072 | 0/16 | 100/100 | 2200 |

None of them costs prediction: test R² stays in 0.79–0.84 against 0.813, and the
uniform_max head over seeds 0, 1, 2 gives 0.705 / 0.793 / 0.829 against the
reference's 0.709 / 0.813 / 0.794. Reading down the table:

- **One anchored point removes the blow-up everywhere.** Pinning ŷ(enc U) = 2000
  drops the matrix-space maximum from 3.8 × 10⁵ to 2,000–3,000 and the landscape
  maximum from 9,400 to 2,200–2,300. Without the anchor the predictor extrapolated
  "more even, higher" without limit; with it the extrapolation has to land on 2000.
- **The cap alone only gives the blow-up a ceiling.** Everything piles at 2,195 and
  U sits below it, at 1405.
- **The head is the only change that bounds the latent as well.** Far from U's
  encoding the quadratic sends ŷ to zero, so at g = +100 it is 7 × 10⁻¹³⁸ where the
  anchored runs give 10¹⁰. That is the fix for ŷ climbing the global without end.
- **The star constraint makes every path from a test graph peak at U**, 98 of 100
  monotone at weight 10 with the cap.
- **No search from the data or from random matrices ends at U, in any run.** The
  model's landscape is now a plateau at about 2000 with bumps 2–10% above ŷ(U),
  0.9–1.5 nats per row from U, and BFGS in 380 dimensions finds the bumps. Under the
  head the maximum is exactly 2000 but it is a whole surface: the gap is zero
  wherever 16 numbers agree with U's, and searches stop on that surface 1.5 nats per
  row from U.

The bumps are far below what the model can resolve: its median relative error on
the test graphs is 18–19%. On these models U is tied for the maximum within the
model's own error, and the data, all at least 1.4 nats per row from U, cannot break
the tie.

### The basin of U

`basin.py`: BFGS from T(s) = (1 − s)·T + s·U for 8 test graphs, ending within KL 0.01
of U:

| run | s 0.5 (KL 0.61) | 0.75 (0.19) | 0.9 (0.04) | 0.97 (0.004) |
|---|---|---|---|---|
| reference | 0/8 | 0/8 | 0/8 | 0/8, escaping to 2 × 10⁵ |
| anchor 10 + cap | 0/8 | 0/8 | 0/8 | 0/8 |
| uniform_max head | 0/8 | 0/8 | 0/8 | 8/8 |
| anchor 10 + star 10 + cap | 0/8 | 0/8 | 4/8 | 6/8 |

U is a local maximum in every model, but its basin reaches at most KL ≈ 0.04, against
1.4 for the nearest data.

### Head and star together

`config/sweeps/uniform_head_star.yaml`, the uniform_max head with star weight 1 or 10,
seeds 0, 1, 2. Test R² 0.752 / 0.798 / 0.796 at weight 1 and 0.714 / 0.784 / 0.815 at
weight 10, against the reference's 0.709 / 0.813 / 0.794. Every path from a test graph
peaks at U. At weight 10 the median path rises slowly, 1479 at s = 0.12 to 1730 at 0.88,
then jumps to 2000 at U: a sharp isolated peak rather than the edge of a plateau. Plain
matrix-space search still ends at U from none of the 16 starts. At weight 10 it stalls
at local maxima around 1650; at weight 1 it climbs to 2000 on the maximizing surface,
1.5 nats per row from U.

## 8. Finding U: the head and a maximum-entropy tie-break

`tiebreak.py`. Maximize log ŷ(enc T) − ε·KL(T ‖ U) over rowsoftmax logits, from 8 test
graphs and 8 random starts, up to 1000 BFGS iterations:

| model, seed 1 | ε 0.01 | ε 0.1 | ε 1 | ε 3 |
|---|---|---|---|---|
| reference | 0/16 | 0/16 | 0/16 | 0/16, ŷ 1.3e5 |
| anchor 10 + star 10 + cap | 0/16 | 0/16 | 9/16 | 15/16 |
| uniform_max head | 0/16 | 2/16 | 14/16 | **16/16** |
| head + star 1 | 0/16 | 0/16 | **16/16** | **16/16** |
| head + star 10 | 0/16 | 2/16 | 12/16 | **16/16** |

Every search that ends at U ends at ŷ = 2000.

**This is how U comes out as the optimum.** Under the uniform_max head, ŷ ≤ 2000 =
ŷ(enc U) everywhere, so U attains the maximum of both terms, and nothing else attains
the second: U is the unique maximizer for every ε > 0. ε sets only how fast BFGS moves
along the flat top. At ε 0.01 and 0.1 the searches were still moving toward U, median
end KL 0.03–0.6, when they ran out of iterations.

The tie-break is not doing the model's job for it, which the reference shows. There
ŷ(enc U) = 671 against 10⁵ elsewhere, and even at ε = 3 every search stays 0.3–0.6
nats per row from U at ŷ 1.3 × 10⁵. The entropy term only chooses among matrices the
model already rates at the maximum, and only a model whose maximum includes U leaves
it anything to choose.

What it does assume is that, among matrices the model cannot tell apart, the most
mixed is the right one. Without the head, that choice is not available; with the
head, nothing else would pick a single point.

## 9. Answers

**Can any optimization find U as the maximum on the model as it stands?** No.
Section 5 tried the unregularized latent ascent, the N(0, I) prior, a distance-from-
start penalty, the data's box, an aggregate-posterior density, a cycle-consistency
penalty, scoring the decoded matrix through the encoder, and direct search over
general, reversible and landscape matrices. The ones that reach U do it through the
decoder's fibre or its saturation, with a ŷ set by the penalty. The ones that score
the matrix honestly find matrices rated 10⁵. Started at U, a matrix search stays at
U, but from 0.004 nats per row away it already escapes to 2 × 10⁵.

**Why not?** U is 1.4 nats per row or more from every training matrix, so nothing in
the data tells the predictor what happens near it. The predictor's extrapolation there
is a crater, 1100–2000 on a rim between the data and U and 671 at U. And ŷ(z) is not a
function of the matrix: the 9-dimensional fibre of identical node latents decodes to U
and scores anything from 0.3 to 25,000.

**Many latents decode to U. What follows?** Optimizing ŷ(z) is ill-posed wherever the
decoder is many-to-one, and U is the extreme case. A latent optimizer can report any ŷ
it likes for U, and the penalties that shrink ‖x‖ are drawn onto the fibre whatever ŷ
says. Score the matrix, ŷ(enc dec z) or ŷ(enc T), which treats the fibre as the one
point it decodes to.

**Does going up the global reach U?** Yes, in the limit, and not because of anything
about the effective population. The decoded matrix tends to U as g → ±∞, KL ∝ 1/g²,
because the global swamps every node's term in the decoder's first layer and the
nodes become identical. g → −∞ reaches the same U. ŷ(z) meanwhile goes to infinity at
+∞ and to zero at −∞, and ŷ of the decoded matrix returns to 671 in both directions.

**How to stop ŷ climbing the global forever.** It is the predictor extrapolating
linearly far from the data. In order of how well each worked here:

1. The uniform_max head. Far from U's encoding ŷ falls, 7 × 10⁻¹³⁸ at g = +100, and it
   never exceeds 2000. It is the only change that bounds the latent and the matrix
   at once, and it costs no R².
2. A cap, log ŷ = c − softplus(c − s). Bounds ŷ at c; everything optimized piles there.
3. Scoring through the encoder, ŷ(enc dec z). Bounded, since the encoder ends in a
   LayerNorm, but only at about 4 × 10⁵.
4. Confining the latent to the data, a box or the aggregate posterior. The optimum is
   then wherever the constraint stops it.
5. β ≥ 0.1. The global collapses, so there is nothing to climb, at a cost in R².

**Would a larger β help?** Partly, and not enough. It moves ŷ(enc U) toward 2000, 671 →
1443 → 1558 at β 0.01 → 0.1 → 0.3, and at 0.3 58 of 100 paths from test graphs peak
at U, because a stronger prior pulls the aggregate posterior, and U's encoding, toward
N(0, I). It does nothing about the fibre, which is the decoder's equivariance, or the
unbounded predictor, and it costs R² (0.813 → 0.763 → 0.742) and reconstruction
(log_recon 0.18 → 0.37 → 1.01). The global collapses at 0.1, which ends the runaway
along it only by removing it. At β = 1 the whole posterior collapses and the predictor
is a constant.

## Caveats

- **Nothing here checks the truth.** Every ŷ is the model's. Whether U has the largest
  effective population, and what the landscape optima in section 5 would really score,
  only Tom's simulation can say.
- **The slides' claim is an assumption.** The head and the star constraint write it
  into the model. In population genetics a strongly subdivided population with equal
  deme sizes can have N_e above N_T (Wright's island model), so "nothing exceeds the
  well-mixed value" is not a theorem. It matches the data, whose largest target is
  2041.6 against a_max = 2000, within the noise the slides describe.
- **U is not a landscape.** It needs 20 points in the plane at equal mutual distances.
  If the question is the best landscape the generator could produce, the search space
  is the landscape family, U is outside it, and its maximum is something else. Under
  the head that search reaches 1977–2000 (`compare.py`).
- Every number is seed 1 except where seeds are listed, and the search counts are over
  16 starts.

## Next steps

- Score the landscape optima from the head models with the simulator. They are
  valid landscapes, and the only optima here that the generator can produce.
- Seeds 0 and 2 through `compare.py` and `tiebreak.py` for the head runs.
- A head whose maximum is unique without a tie-break would need the gap to vanish
  only at U, e.g. a full-rank A on a representation that does not pool U's identity
  away. Untested.
