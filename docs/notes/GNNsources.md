# Sources

Reading list behind `architecture.md`. Titles and venues are from memory — verify before citing in a manuscript.

To understand why a transformer encoder and a graph neural network are the same thing for our case, read Joshi first, then Veličković, then Dwivedi & Bresson, then Graphormer for the bias mechanism specifically. If you read only one, read Joshi.

## The scheme we are copying

**Gómez-Bombarelli et al., *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules*, ACS Central Science 4(2), 2018.** The paper `project_vision.md` is based on. A VAE maps molecules to a continuous latent space, an MLP trained jointly on that latent predicts a property, and gradient ascent on the MLP's output moves through latent space toward better molecules, which are then decoded. Our substitution is transition matrices for molecules and a heterozygosity decay exponent for the chemical property.

**Jin et al., *Junction Tree Variational Autoencoder for Molecular Graph Generation*, ICML 2018.** The follow-up that exists because the 2018 decoder often emitted invalid molecules, so latent optimization drifted into regions that decoded to nothing. Its fix is to make validity structural rather than penalized. Masked row softmax is the same move for us, and the failure mode it avoids is worth knowing about.

**Liu et al., *Constrained Graph Variational Autoencoders for Molecule Design*, NeurIPS 2018.** Same lineage as the above: enforce validity during decoding rather than penalising invalidity afterwards. Read alongside Jin et al. for two different takes on the same principle.

## Message passing

**Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017.** Unifies earlier graph networks into one skeleton: each node collects messages from its neighbours, sums them, and updates its own state from the sum. Every layer in `architecture.md` is an instance of this. The sum is what makes the whole thing permutation-equivariant.

**Battaglia et al., *Relational inductive biases, deep learning, and graph networks*, 2018.** The general framework message passing, attention, and convolution are all special cases of. Useful as a map of the territory rather than a specific method.

**Simonovsky & Komodakis, *Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs*, CVPR 2017.** Makes the message from a neighbour depend on the edge connecting them. Essential here, since a complete graph's adjacency carries no information and all the signal is in the edge weights. Their mechanism is a hypernetwork generating weights from the edge, which has far more parameters than we can afford; we use the cheaper concatenation variant.

**Hu et al., *Strategies for Pre-training Graph Neural Networks*, ICLR 2020.** Source of GINE, the cheap way to fold edge features into message passing. Closer to what we actually implement than the hypernetwork above.

**Xu et al., *How Powerful are Graph Neural Networks?*, ICLR 2019.** Proves sum aggregation is strictly more expressive than mean or max, because averaging discards how many neighbours sent a message. Does not bite for us — every node has exactly 19 neighbours — but explains why sum is the default elsewhere.

**He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.** Residual connections: a layer computes a correction to its input rather than a replacement, so depth stops degrading accuracy and a layer can default to passing information through unchanged. Every `h_i ← h_i + ...` in the architecture.

**Li et al., *DeepGCNs: Can GCNs Go as Deep as CNNs?*, ICCV 2019.** Carries residual connections and normalization over to deep GNNs specifically.

## Attention

**Joshi, *Transformers are Graph Neural Networks*, The Gradient, 2020.** The starting point, and an article rather than a paper. Argues that a transformer is a GNN with attentional aggregation running on a fully connected graph — the sentence is the graph and the words are the nodes, and since any word can attend to any other, the graph is complete. That is our situation with 20 nodes in place of words, which is why a plain transformer encoder is a legitimate implementation of the architecture.

**Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.** The transformer. Each token computes a query, every token offers a key and a value, and a token's output is a weighted average of values with weights from query-key similarity. On a complete graph this is exactly message passing with learned weights, which is why a plain transformer encoder is a legitimate implementation of our architecture.

**Veličković et al., *Graph Attention Networks*, ICLR 2018.** Attention on graphs: instead of averaging neighbours uniformly, learn how much each neighbour matters. More load-bearing for us than usual, since a complete graph offers no sparsity to provide selectivity.

**Brody et al., *How Attentive are Graph Attention Networks?*, ICLR 2022.** Shows GAT's attention is "static" — the ranking of neighbours ends up the same for every query node — and fixes it by moving the nonlinearity. Use this form, not the original.

**Dwivedi & Bresson, *A Generalization of Transformer Networks to Graphs*, 2021.** How to put edge features into transformer attention on a graph.

**Ying et al., *Do Transformers Really Perform Bad for Graph Representation?* (Graphormer), NeurIPS 2021.** Injects graph structure as an additive bias on attention scores. This is the mechanism for getting transition probabilities into a plain transformer encoder, and the basis of the shortcut implementation in `architecture.md`.

## Graph VAEs

**Kipf & Welling, *Variational Graph Auto-Encoders*, 2016.** One latent vector per node, edges reconstructed from pairs of node latents. The structure we use, except with a learned asymmetric pair function instead of their symmetric inner product, since our matrices are directed.

**Simonovsky & Komodakis, *GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders*, ICANN 2018.** The alternative we rejected: a single graph-level latent, which forces approximate graph matching to make reconstruction loss well-posed against an arbitrary node ordering. Worth reading to understand the problem node-level latents sidestep.

**Salha et al., *Gravity-Inspired Graph Autoencoders for Directed Link Prediction*, CIKM 2019.** arXiv 1905.09570, code at github.com/deezer/gravity_graph_autoencoders. The closest published solution to our decoder problem: Kipf & Welling's symmetric inner product cannot represent directed edges, so this gives each node a position plus a learned scalar mass and scores edges with a gravity-like potential, asymmetric by construction. A more structured alternative to `MLP([z_i ; z_j])` with far fewer parameters, which matters on 1000 examples. Worth benchmarking against ours.

**Williams et al., *Scalable Generative Modeling of Weighted Graphs*, 2025.** arXiv 2507.23111. Current state of weighted-graph generation, and confirms that essentially all deep graph generative models handle topology while ignoring edge weights or bolting them on naively. Their hard problem — jointly modelling which edges exist and how strong they are — is not ours, since our topology is fixed at complete-minus-diagonal for every sample and only the weights vary. Worth saying so explicitly in any write-up, or readers from this community will assume we face it. Their model is autoregressive and sparsity-exploiting, so a poor fit besides: autoregressive generation destroys permutation equivariance and leaves no single latent to optimise in.

## Permutation handling

The central design axis of graph generative modelling, and the one our architecture is a position on. Node labels are arbitrary, so a decoder emitting a specific matrix from an order-free latent has no way to know which of the n! orderings to produce, and element-wise reconstruction loss is ill-posed. Five families of answer:

| family | latent | how permutation is handled | when used |
|---|---|---|---|
| graph matching | graph-level vector | align output to input before scoring | tiny graphs, largely abandoned |
| node-level latents | set of node vectors | equivariance makes the loss well-posed | link prediction, reconstruction fidelity — **ours** |
| learned alignment | graph-level vector | a module learns the alignment | when a single vector is required |
| autoregressive | none | fix or learn a node ordering | large sparse graphs |
| diffusion | none | equivariant net, invariant denoising loss | current best sample quality |

**Winter et al., *Permutation-Invariant Variational Autoencoder for Graph-Level Representation Learning* (PIGVAE), NeurIPS 2021.** arXiv 2104.09856. The strongest alternative to our design, and the one to read if the set-structured latent ever obstructs us. Gets a genuine graph-level latent vector without imposing an ordering or running expensive matching: a separate permuter module predicts a permutation matrix aligning output node order to input, trained through a continuous relaxation of argsort plus an entropy regulariser pushing toward a hard permutation. Better fit for the project's stated goal of a single flat vector, at the cost of much more machinery and a learned component that can fail to converge — when the alignment is wrong the reconstruction gradient is meaningless. Candidate for v2, not v1.

**GraViti: Graph-Level Variational Autoencoders with Relaxed Permutation Invariance, 2026.** arXiv 2605.16668. The recent transformer-based descendant of the above, targeting graph-level latents and state-of-the-art reconstruction.

**Graph Embedding VAE: A Permutation Invariant Model of Graph Structure, 2019.** arXiv 1910.08057. Earlier attempt in the same direction.

**You et al., *GraphRNN*, ICML 2018** and **Liao et al., *GRAN*, NeurIPS 2019.** Build graphs node by node, handling permutation by training over BFS orderings rather than all n!. Not for us — autoregressive generation leaves no single latent to optimise in, and our graph is dense and fixed-size, where these are weakest.

**Chen et al., *Order Matters: Probabilistic Modeling of Node Sequence for Graph Generation*, ICML 2021.** First to *learn* the node ordering rather than fixing it, reporting better results than BFS or DFS. The clearest statement of why prefixed orderings are a liability.

**Zhang et al., *Pard: Permutation-Invariant Autoregressive Diffusion for Graph Generation*, NeurIPS 2024.** arXiv 2402.03687. Reconciles autoregressive generation with permutation invariance by routing it through diffusion.

**Vignac et al., *DiGress: Discrete Denoising Diffusion for Graph Generation*, ICLR 2023.** arXiv 2209.14734. Current state of the art for graph generation quality: a graph transformer trained to reverse a discrete noising process, permutation-equivariant with a permutation-invariant loss, so nothing to match and no ordering to pick. Not usable for us despite that — diffusion models have no encoder, so there is no latent space to encode a real landscape into and ascend a gradient in. Revisit only if sample quality rather than optimisation becomes the bottleneck.

**Niu et al., *Permutation Invariant Graph Generation via Score-Based Generative Modeling*, AISTATS 2020** and **Jo et al., *GDSS*, ICML 2022.** The score-based line DiGress improves on.

Note that nearly all of this literature is about generating *topology* — which edges exist. Ours is fixed at complete-minus-diagonal for every sample and only the weights vary, which removes the combinatorial core: no discrete edge decisions, no variable node counts, no sparsity structure. Worth stating explicitly in a write-up, or readers from this field will assume we face the hard version and wonder why we are not using DiGress.

## Invariance and sets

**Zaheer et al., *Deep Sets*, NeurIPS 2017.** Characterizes functions on sets: any permutation-invariant function can be written as a pooling of per-element encodings. Justifies the predictor's pool-then-MLP shape.

**Lee et al., *Set Transformer*, ICML 2019.** Attention-based pooling (PMA), a learned alternative to mean or sum pooling.

**Maron et al., *Invariant and Equivariant Graph Networks*, ICLR 2019.** Characterizes the linear layers that are equivariant to node relabelling. The theory behind why the design is built the way it is.

**Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*, 2021.** Book-length framing of architectures as consequences of the symmetries of their data. The general argument for building permutation symmetry in rather than learning it.

## VAE foundations

**Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014.** The VAE: the reparameterization trick, the ELBO, encoders that output a distribution rather than a point.

**Higgins et al., *beta-VAE*, ICLR 2017.** Weighting the KL term to trade reconstruction quality against a structured latent space. Our beta.
