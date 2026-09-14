# Other sources

Work that is neither graph-neural-network method (`GNNsources.md`) nor the diversity-and-networks domain (`NetworkDiversity.md`), but bears on the project.

Titles and venues are from memory — verify before citing in a manuscript.

## Framing

The general problem, stated without reference to landscapes: **inverse design of Markov chains for a target dynamical property.** Landscapes are the motivating application, but nothing in the method depends on that reading, and the general framing reaches a much wider audience than landscape genetics alone.

## Neural networks that output valid transition matrices

**Deep learning Markov and Koopman models with physical constraints**, 2019. arXiv 1912.07392, from the deep-MSM line. Specifically about forcing a network's output to be a valid stochastic transition matrix, optionally reversible — the same problem masked row softmax solves for us. Read to check whether their constraint machinery buys anything over ours, particularly if reversibility ever becomes a requirement.

**Mardt et al., *VAMPnets for deep learning of molecular kinetics*, Nature Communications, 2018.** arXiv 1710.06012. Learns transition matrices *from trajectory data*, the opposite direction from us: we take matrices as given and model their distribution. Useful as evidence that networks emitting valid transition matrices is established practice rather than something we are inventing, and as the entry point to the Markov state model literature generally.

## The baseline our optimisation has to beat

**Optimizing Graph Structure for Targeted Diffusion**, 2020. arXiv 2008.05589. Optimises graph structure for a dynamical outcome directly, by discrete combinatorial edge editing, with no learned latent space. This is the natural baseline for the optimisation stage: if direct combinatorial search matches latent-space gradient ascent, the VAE is not earning its place. A reviewer will ask for this comparison.

## Search notes

The specific combination — VAE over transition matrices, jointly trained property predictor on the latent, gradient ascent in that latent to design new matrices for a target dynamical property — did not turn up anywhere. The molecule people do it for molecules, the Markov state model people build transition matrices without generating them, the graph generation people generate topology without optimising for a dynamical property, and the landscape genetics people study the question without machine learning.

This was web search, not a systematic review. Before making a novelty claim in writing, citation-chase on Semantic Scholar from three nodes: Gómez-Bombarelli et al. 2018 (forward citations, looking for non-molecular applications), Salha et al. 2019, and Williams et al. 2025. Anything doing what we are doing should cite at least one.
