# Sources

Reading list behind `architecture.md`. Titles and venues are from memory — verify before citing in a manuscript.

To understand why a transformer encoder and a graph neural network are the same thing for our case, read Joshi first, then Veličković, then Dwivedi & Bresson, then Graphormer for the bias mechanism specifically. If you read only one, read Joshi.

## The scheme we are copying

**Gómez-Bombarelli et al., *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules*, ACS Central Science 4(2), 2018.** The paper `project_vision.md` is based on. A VAE maps molecules to a continuous latent space, an MLP trained jointly on that latent predicts a property, and gradient ascent on the MLP's output moves through latent space toward better molecules, which are then decoded. Our substitution is transition matrices for molecules and a heterozygosity decay exponent for the chemical property.

**Jin et al., *Junction Tree Variational Autoencoder for Molecular Graph Generation*, ICML 2018.** The follow-up that exists because the 2018 decoder often emitted invalid molecules, so latent optimization drifted into regions that decoded to nothing. Its fix is to make validity structural rather than penalized. Masked row softmax is the same move for us, and the failure mode it avoids is worth knowing about.

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

## Invariance and sets

**Zaheer et al., *Deep Sets*, NeurIPS 2017.** Characterizes functions on sets: any permutation-invariant function can be written as a pooling of per-element encodings. Justifies the predictor's pool-then-MLP shape.

**Lee et al., *Set Transformer*, ICML 2019.** Attention-based pooling (PMA), a learned alternative to mean or sum pooling.

**Maron et al., *Invariant and Equivariant Graph Networks*, ICLR 2019.** Characterizes the linear layers that are equivariant to node relabelling. The theory behind why the design is built the way it is.

**Bronstein et al., *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*, 2021.** Book-length framing of architectures as consequences of the symmetries of their data. The general argument for building permutation symmetry in rather than learning it.

## VAE foundations

**Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014.** The VAE: the reparameterization trick, the ELBO, encoders that output a distribution rather than a point.

**Higgins et al., *beta-VAE*, ICLR 2017.** Weighting the KL term to trade reconstruction quality against a structured latent space. Our beta.
