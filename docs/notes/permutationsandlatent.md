These are my thoughts as they currently stand on the problem of graph permutations and the latent space. Starting at the graph level, we state that permutations of the node labeling are equivalent graphs. ie, it doesn't matter if we call this node 1 or 2 or 3, etc, the graph is still the same. So there are 20! equivalent node labelings for each graph. What does this cause in general for the model? For one, it affects how we do loss. We need to have the node labelings aligned for the input and the reconstruction so we can properly compare the two. In a sense, this is an "equivariance" condition, if we put a certain permutation of a graph, we will get a reconstruction that is permuted in the same way. This is slightly different than an invariant model, where putting in a permutation of the graph will lead to the same outcome, which will be the case for the property prediction (through pooling). 

What would invariant reconstruction look like? This is the alignment problem. We can somehow pick a specific permutation of the node labeling, but claude has claimed that that leads to discontinuities. Let's say for example we sort by inflow before doing any sort of analysis (ie at the front of the model). Claude's claim is that we have many nodes that have extremely close in flows. Then, if we move one of them slightly, so they change rank, this would lead to a discontinuous jump in out transition matrix, because we would have two column swaps, and now what was previously labeled node 4 will be node 5, so it looks like a completely different matrix, even though it is extremely similar. This is the problem, it seems, to picking a specific alignment. So, it seems that we would have an equivariant model, but not invariant.

Now, to the problem of the latent space representation. Let's say we're sticking to the equivariant method for now, although this is mostly agnostic to that. The problem is between a structured and unstructured latent space. Ultimately, the latent space is a vector (list) of numbers, this is what we need to do gradient ascent. One method for generating this is to have each node have a number of latent dimensions, resulting in a latent representation that is a total of 20 x d dimensions, but has a hidden structure of being each node specific. Maybe this isn't such a bad thing. This would allow the model to easily keep equivariance, as the node is the level being tracked the whole time. For an unstructured representation, we would destroy equivariance. Ah, this is where the alignment problem comes in. By having an unstructured latent in the middle, we need someway to realign the input and output, so we can properly compare them for loss. This is where the learned alignment would come in, or where some sort of other invariant/alignment method would happen, like ranking by inflow. 

Can we have a single latent representation represent all of the permutation options? I think we would need to.

Side note, we need to make sure that the thing we are turning into the latent representation doesn't easily have a way to do optimization on it without the latent representation. This is something I need to wrap my head around more.

## Claude's Plan

Written 2026-09-15, working through the note above. Short version: the reasoning above is right, the design stays as it is, and the two questions left open at the end both have answers.

### The action, and the two conditions

A relabelling π acts differently on different objects, and most confusion here comes from conflating those actions:

```
transition matrix    T (20x20)    ↦  P T Pᵀ      both axes are node axes
node features        N (20x5)     ↦  P N         one node axis
edge features        E (20x20x4)  ↦  permute axes 0 and 1
hidden states        H (20xh)     ↦  P H
latent               Z (20xd)     ↦  P Z
target               y (scalar)   ↦  y           no node axis, nothing to act on
```

```
equivariant:  f(P·x) = P·f(x)     carries the relabelling through
invariant:    f(P·x) =   f(x)     does not notice it
```

Invariance is the special case where the output has no node axis. So a graph-level latent does not "destroy equivariance" as the note above puts it — it makes the encoder *invariant*, which is the stronger condition, and the damage it does is entirely downstream in the decoder.

### A worked 3-node example

```
T                             T' = P T Pᵀ,  π swaps labels 2 and 3
      to 1    2    3                to 1    2    3
1  [   0    0.7  0.3 ]         1  [   0    0.3  0.7 ]
2  [  0.4    0   0.6 ]         2  [  0.5    0   0.5 ]
3  [  0.5   0.5   0  ]         3  [  0.4   0.6   0  ]
```

The same graph twice. Row 2 of T' is old node 3's outflows rewritten in the new labels; both matrices are row-stochastic with a zero diagonal, and nothing about the object changed, only the names.

Column sums go [0.9, 1.2, 0.9] ↦ [0.9, 0.9, 1.2] — permuted identically, which is equivariance on a node feature, and they hold an exact tie between nodes 1 and 3, which is what canonicalization has to break arbitrarily.

### Where each piece sits

| piece | condition | why |
|---|---|---|
| node features n_i | equivariant | each entry reduces over the *other* index with a sum, entropy or max, all order-agnostic, so n_i is a property of node i alone |
| edge features e_ij | equivariant | a function of the ordered pair (i, j) only |
| encoder layer | equivariant | the Σ_j discards the order of the neighbours while the per-i structure keeps the node axis |
| μ_i, log σ_i | equivariant | computed from h_i pointwise |
| sampled z_i | equivariant *in distribution* | ε is drawn independently per node, so a single sample is not literally permuted |
| decoder logits | equivariant | MLP of an ordered pair; the masked diagonal is a permutation-stable set; softmax commutes with permuting its inputs |
| reconstruction loss | **invariant** | Σ_i KL(T_i ‖ T̂_i) is a sum of the same terms in a different order |
| prior KL | **invariant** | Σ_i over nodes — this needs the factorized, identical-per-node prior, which is doing quiet work |
| predictor | **invariant** | pooling over i |

The forbidden node feature makes the distinction concrete. Feed node i its raw row T[i,:] and for the 3-node example node 1's "feature" is [0, 0.7, 0.3] before relabelling and [0, 0.3, 0.7] after — but node 1 was never touched, it is the same node with the same outflows to the same physical destinations. The vector changed because its *entries* are indexed by j. That makes it a function of the node **and the labelling**, which lets the network read the labels off its inputs and kills equivariance at layer zero.

### Equivariant reconstruction, invariant loss

Invariant reconstruction, which the note above reaches for, cannot exist: the decoder has to emit an actual 20x20 array and every array has some labelling. What can be invariant is the **loss**, and it is, for free:

```
L_rec( P T Pᵀ , P T̂ Pᵀ )  =  L_rec( T , T̂ )
```

because the outer sum runs over the same 20 terms in a different order and each inner KL over the same 20 entries in a different order. This is the whole point of the equivariant route. Input node i enters at slot i, its latent is at slot i, its reconstructed row leaves at slot i — the correct alignment is the identity for every input, by construction, so there is never anything to match.

With a graph-level latent, T and PTPᵀ both encode to the same z and therefore both decode to the same D(z), which at most one of them can equal. The element-wise loss then depends on which labelling went in, and the network is being punished for failing to reproduce information that pooling provably destroyed. The two escapes are the two families in the taxonomy: minimize the loss over permutations (graph matching), or predict the aligning permutation (PIGVAE). The useful way to see PIGVAE is that it does not put the ordering *into* the latent, it routes the ordering *around* it — a separate permuter carries the alignment alongside an invariant z, which is exactly why the latent stays clean, and exactly why the permuter has to be right.

Small correction to the note above: a rank swap under column-sum sorting is a row swap *and* a column swap, since the canonical form is the full conjugation PTPᵀ. Both of the node's outflows and its inflows move, which is why the measured amplification is as large as it is. Also worth noticing that the 3-node example has column sums [0.9, 1.2, 0.9] — an exact tie, where sorting has no defined answer at all.

### "Can a single latent represent all the permutation options?"

Yes, and we already have it — the object that represents the graph is the **set** {z_1, ..., z_20}, equivalently the orbit {PZ : P ∈ S_20}. All 20! labellings of one graph map to the 20! orderings of one set of vectors. The set is invariant and only the write-down order varies.

The array Z is just a choice of how to write that set down, so the real question is whether anything downstream depends on the choice:

- **Single-matrix gradient ascent does not, provably.** ŷ is invariant, so the gradient field is equivariant and ascent commutes with relabelling — see the derivation now in `architecture.md`, along with the check to run. This is the operation the project is for, and it is safe.
- **Anything combining two latents does.** Interpolating Z_A and Z_B element-wise adds z_A,1 to z_B,1, and slot 1 means unrelated nodes in the two encodings, hence Hungarian alignment first. Prior sampling draws 20 iid vectors and so produces a set with no inter-node correlation at all, which is the weak point for the generative half of the goal rather than the optimization half.

So the set-structured latent does represent all permutations correctly, and the ordering leaks only into operations that combine two different graphs.

### The side note, on optimizing without the latent

The worry is well placed, and the answer is that you *can*: parameterize a matrix by free logits, apply the masked row softmax, train a predictor directly on T, ascend the logits. Every constraint holds and no VAE is involved.

What it should do is find adversarial matrices. The predictor is only accurate near the data it saw, and with 360 free dimensions nothing confines ascent to that region, so it will find inputs that maximize the predicted exponent while looking nothing like a real landscape. The latent's job is precisely that confinement — fewer dimensions than the data has, regularized toward a prior the encoder mapped real data into, with the decoder projecting back onto something plausible. That is the actual argument for the Gómez-Bombarelli scheme, and the direct baseline is the cleanest way to demonstrate it rather than assert it. If direct ascent turns out *not* to go adversarial, that is important information about how easy the problem is.

### Hybrid latent: graph-level dimensions alongside node-level ones

The question is whether the latent can be a pair — an invariant z_graph ∈ ℝᵏ alongside the equivariant Z_node ∈ ℝ^(20xd) — and whether that buys anything.

**It works, and equivariance is preserved trivially.** Set z_graph = pool(H), which is invariant, and broadcast it into the decoder:

```
logits_ij = MLP( [ z_i ; z_j ; z_graph ] )
```

Under a relabelling Z_node ↦ P Z_node while z_graph is unchanged, so logits'_ij = logits_(π⁻¹(i), π⁻¹(j)), which is the permuted logit matrix. The general rule: **broadcasting an invariant quantity along the node axis preserves equivariance**, because with respect to that axis it is a constant, acting like a learned bias that happens to depend on the graph. The same argument allows injecting z_graph into the encoder layers or the predictor. Ascent equivariance survives too — the gradient with respect to z_graph is invariant, with respect to Z_node is equivariant, so joint ascent still commutes with relabelling.

This is the **global attribute u** of Battaglia et al., promoted from a hidden feature to a latent variable, and as a latent it is the **Neural Statistician** (Edwards & Storkey, ICLR 2017). Both are in `GNNsources.md`.

**It adds no representational power.** An invariant z_graph is a deterministic function of the node set — by Deep Sets it is ρ(Σ_i φ(z_i)) — so given Z_node it carries zero additional information. Anything it could say, the model can already say by writing the same value into dimension 0 of all 20 node latents. No new functions become representable.

The intuition that attention makes this unnecessary is right about the encoder, where full attention means every h_i depends on all of T after one layer. It is right about the decoder *only because of the few equivariant layers before the pair MLP*: the pair function itself is local, since logits_kl = MLP([z_k ; z_l]) does not involve z_i when i is neither k nor l, so z_i touches only row i and column i, 39 of 380 entries. The decoder's global pathway exists entirely because those pre-layers mix the latents first. A broadcast z_graph would supply that pathway by construction instead of as something depth has to recompute.

**What it does add**, in increasing order of weight:

- *Diagnostics.* A per-graph invariant vector to cluster, project and compare with no Hungarian alignment, and interpolation becomes partially alignment-free — z_graph interpolates cleanly while the node block still needs matching.
- *KL efficiency.* Storing a global fact redundantly across 20 node slots costs roughly 20x the KL of storing it once globally, since the prior charges per dimension. The useful reframing: d = 8 → 9 costs 20 numbers and k = 20 global dims costs 20 numbers, so the question is not whether this adds capacity but whether capacity is better spent globally or per-node. For global facts, globally.
- *A correlated prior.* This is the real one, and it is the fix for the weakness noted above — sampling 20 iid z_i from N(0,I) gives a set of unrelated nodes with no reason to form a coherent landscape. Draw z_graph ~ N(0, I_k) and the nodes conditioned on it, and they are independent given z_graph but correlated marginally. Standard remedy for exchangeable data, and it targets the generative half of `project_vision.md` directly.

**The risk is posterior collapse on the global slot.** Since Z_node can represent everything already, the path of least resistance is to ignore z_graph and let its KL drive it to the prior — leaving the machinery and none of the benefit, looking like "it didn't help" when it was never used. Track the per-group KL as a training diagnostic from day one and reach for free bits or KL warmup on the global group if it flatlines.

Note that the hybrid and PIGVAE are alternative responses to one problem. The hybrid keeps the equivariant design and fixes the prior's independence; PIGVAE replaces the design to get a genuinely single flat vector. The hybrid is far cheaper and gets most of the practical benefit, so it goes first.

### Plan

1. Keep node-level latents. GE-VAE (arXiv 1910.08057) turns out to be the same choice reached independently — see `GNNsources.md`.
2. Finish `models.py` and commit it.
3. Add the ascent-equivariance check alongside the existing equivariance tests, run with sampling off.
4. PIGVAE stays the v2 candidate, and the trigger is specific: needing interpolation between landscapes or prior sampling to work well. Not disappointing reconstruction or disappointing single-point ascent — those are β, γ and d.
5. Build the hybrid-latent hook now and default it off — `d_global: int = 0` in `TMVAEConfig`, with the pooling head, the broadcast concat and the second KL group as no-ops at zero, so turning it on later is a config change rather than a refactor of the decoder signature and the loss. Turn it on when generation and prior sampling become the priority, which is the same trigger as PIGVAE, and try it first.
6. Later, the no-VAE direct-ascent baseline, for the worklog next steps rather than now.
