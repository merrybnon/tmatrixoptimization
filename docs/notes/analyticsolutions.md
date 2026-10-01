# Analytic solutions for T → a

Can the decay exponent be computed from the transition matrix, instead of learned? The metapopulation genetics literature has a general answer. Applying it here needs an input the drop does not carry: the population size of each patch.

## The problem as posed

Restated from `2026-9-11 The ML Problem.pptx`:

- 20 patches are placed uniformly in the unit square, with off-diagonal W_ij = D_ij⁻⁴ and rows normalized to give T. The diagonal is zero, so every individual migrates every year.
- Each patch has carrying capacity K = 100, and there is no selection. Migration happens once per year, by applying T to the vector of patch populations.
- Tom fits the landscape-average heterozygosity to H(t) = H₀·e^(−t/2a), averaged over 20 CDMetaPop replicates. a is the heterozygosity decay exponent.
- 20 isolated patches would give a = K = 100. Full mixing would give a = 20K = 2000. Slide 6 notes that no deterministic prediction is known for random landscapes.

In population genetics terms, a is the **eigenvalue effective size** N_eE of a subdivided population: the effective size of the whole metapopulation once migration and drift have reached equilibrium.

## What the literature gives

### The exact answer: a linear recursion on identity by descent

Let h_ij be the probability that a gene from patch i and a gene from patch j are not identical by descent. Migration followed by Wright–Fisher sampling within each patch makes h evolve linearly, on n² = 400 variables (n² + n for a diploid model):

```
h'_ij = (1 − δ_ij / 2N_i) · Σ_kl B_ik B_jl h_kl          D = diag(1 − δ_ij / 2N_i) · (B ⊗ B)

B_ik = N_k T_ki / Σ_l N_l T_li                           backward migration: where i's parents came from

a = N_eE = 1 / (2 (1 − λ_max(D)))
```

Heterozygosity in every patch eventually decays at the rate set by λ_max(D). The method goes back to Malécot. Hössjer, Olsson, Laikre and Ryman (2014, 2016) give the general version, for any migration matrix and any local census and effective sizes. Their R program `gesp` (Olsson et al. 2017) takes a forward migration matrix plus those sizes and returns N_eE through exactly this eigenvalue. Tufto & Hindar (2003) take the same bottom-up approach for conservation.

### A closed form when migration is strong

When migration is fast compared with drift, the structured coalescent collapses to a single Kingman coalescent (Nagylaki 1980; Nordborg & Krone 2002), with

```
N_e = 1 / Σ_i (u_i² / N_i)          u = stationary distribution of B
```

Migration here is as strong as it gets, since every individual moves every year. The condition can still fail for a landscape with a nearly isolated cluster. Over Tom1000 the median of 1 − λ₂ is about 0.004, which is the same order as the drift rate 1/(2K) = 0.005.

Whitlock & Barton (1997) is the standard reference on why N_e falls below the total population when patches contribute unequally to the migrant pool. That unequal contribution is the main mechanism in these landscapes.

### The catch

Both results need the local sizes N_i, not just T. Slide 5 notes that migration pushes some patches below carrying capacity. So the patch sizes depend on CDMetaPop's birth, density-regulation and migration order, and T → a has no closed form without a model of those.

## Testing it on Tom1000, without fitting

Each guess for N_i below goes through either the full eigenvalue or the strong-migration formula. Every number is over all 1000 matrices, comparing predicted log a with the true log y. R² is computed with no fitted parameters, so it can be negative.

| assumed patch sizes | formula | corr | R², no fit | median true/predicted |
|---|---|---|---|---|
| N_i = K everywhere | eigenvalue | 0.15 | −2.7 | 0.33 |
| N_i = min(K, K·inflow) | eigenvalue | 0.24 | −1.5 | 0.44 |
| N_i = K·π_i / max π | strong migration, which reduces to a = K / max π | **0.80** | **0.56** | **0.99** |
| N_i = K·π_i / max π | eigenvalue | −0.21 | −8.2 | 0.58 |

The third row comes from a simple assumption: patches lose individuals when they overflow and never regrow. Repeating v → min(K, Tᵀv) then converges to N_i = K·π_i / max π, where π is the stationary distribution of T. The most-visited patch sits at capacity and the rest scale with π. Because T is reversible (detailed balance holds to 5e-15, see WORKLOG), B = T and u = π, and the strong-migration formula collapses to

```
a ≈ K / max_i π_i
```

This matches both limits exactly: uniform π gives 20K = 2000, and a single dominant patch gives about K. With no fitted parameters it also gets the typical level right (median ratio 0.99). Its ranking is weaker: R² is 0.56 with no fit and 0.67–0.71 after a two-parameter log-linear calibration on each training split.

The same sizes in the exact recursion give nonsense. Many patches end up with N_i below 1, where the Wright–Fisher approximation breaks down. So the no-regrowth assumption is wrong in detail, which is unsurprising because CDMetaPop has births and depleted patches refill.

### For context: features computed from T

These come from the same splits the VAE uses (seeds 0, 1, 2; 800/100/100), with each feature set fitted on train only. Values are test R² on log y, one per seed.

| features | R² s0 / s1 / s2 |
|---|---|
| log(1 − λ₂), the spectral gap | 0.07 / 0.13 / 0.18 |
| entropy of π, a single number | 0.68 / 0.70 / 0.79 |
| a = K / max π, calibrated | 0.67 / 0.71 / 0.69 |
| eigenvalue gaps + sorted log π, boosted trees | 0.80 / 0.83 / 0.84 |
| VAE, `input_norm` enormfixed | 0.71 / 0.81 / 0.79 |

The spectral gap is a poor predictor, and the target is not a mixing time. Almost all of the signal sits in how uneven π is, which is what the effective-size theory says should matter.

## Conclusions

- **The general solution exists.** a = N_eE = 1 / (2 (1 − λ_max(D))) for the identity-by-descent recursion above, computable with `gesp` or as a 400 × 400 eigenproblem.
- **The missing input is the patch sizes.** No formula in T alone tops R² ≈ 0.7. The bottleneck is how CDMetaPop's demography maps T to patch populations, not the genetics.
- **a ≈ K / max π is a parameter-free baseline worth keeping.** It gets the scale right and explains why π dominates every learned and hand-built predictor.

## Next steps

1. Ask Tom for the time-averaged population of each patch from runs he already has, or have future runs save it. Feed those sizes and T into the recursion, or into `gesp`, and compare with a.
2. If the prediction lands within the 20-replicate noise, the genetics is solved. The ML problem then reduces to learning T → N_i, a deterministic population-dynamics question.
3. Independently, measure the noise ceiling: rerun the same matrices with fresh seeds and take the R² between replicate fits of a.

## References

- Hössjer, Olsson, Laikre & Ryman (2014). A new general analytical approach for modeling patterns of genetic differentiation and effective size of subdivided populations over time. *Mathematical Biosciences* 258:113–133.
- Hössjer et al. (2016). Metapopulation inbreeding dynamics, effective size and subpopulation differentiation — a general analytical approach for diploid organisms. *Theoretical Population Biology*. [link](https://cyberleninka.org/article/n/1236886)
- Olsson, Laikre, Hössjer & Ryman (2017). gesp: a computer program for modelling genetic effective population size, inbreeding and divergence in substructured populations. *Molecular Ecology Resources*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5724513/)
- Metapopulation effective size and conservation genetic goals for the Fennoscandian wolf (2016). *Heredity*. [link](https://www.nature.com/articles/hdy201644)
- Ryman, Laikre & Hössjer (2019). Do estimates of contemporary effective population size tell us what we want to know? *Molecular Ecology*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6850010/)
- Tufto & Hindar (2003). Effective size in management and conservation of subdivided populations. *Journal of Theoretical Biology* 222:273–281.
- Whitlock & Barton (1997). The effective size of a subdivided population. *Genetics* 146:427–441.
- Nagylaki (1980). The strong-migration limit in geographically structured populations. *Journal of Mathematical Biology*.
- Nordborg & Krone (2002). Separation of time scales and convergence to the coalescent in structured populations. In *Modern Developments in Theoretical Population Genetics*, OUP. [link](https://www.oeaw.ac.at/gmi/detail/news/nordborg-m-krone-sm-2002-separation-of-time-scales-and-convergence-to-the-coalescent-in-structured-populations)
- Laporte & Charlesworth (2002). Effective population size and population subdivision in demographically structured populations. *Genetics*. [link](https://academic.oup.com/genetics/article/162/1/501/6050022)
- On the eigenvalue effective size of structured populations. *Journal of Mathematical Biology*. [link](https://link.springer.com/article/10.1007/s00285-014-0832-5) (paywalled, not read)
- A clarification of historical concepts of effective population size and their applicability to subdivided populations (2025). *Theoretical Population Biology*. [link](https://www.sciencedirect.com/science/article/abs/pii/S0040580925000760) (paywalled, not read)
- Allen et al. (2017). Evolutionary dynamics on any population structure. *Nature* 544:227–230. Pairwise coalescence times on an arbitrary graph, from a linear system. [link](https://www.nature.com/articles/nature21723)
