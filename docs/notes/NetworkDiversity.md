# Network structure and genetic diversity

Sources on the domain question itself: how the structure of a spatial network shapes the genetic diversity of the population living on it. This literature asks our question and has a real body of results, but it is graph-theoretic and statistical throughout — centrality measures correlated against diversity metrics. Nobody in it is doing generative modelling or inverse design, so it establishes that the question matters without overlapping our method.

Titles and venues are from memory — verify before citing in a manuscript.

**Gonzalez Nuñez et al., *PNAS* 121(34), 2024.** The origin of the sibling project in `hotspotLandscapes`, where diversity is produced by a range expansion across a lattice with hotspots rather than by a transition matrix. Same underlying question, different landscape representation.

**Dispersal behaviour and riverine network connectivity shape the genetic diversity of freshwater amphipod metapopulations**, 2022. Finds a significant imprint of network connectivity on both local and global genetic diversity: allelic richness rises toward more central nodes, and genetic differentiation grows with instream distance. The clearest empirical demonstration that network position predicts diversity, which is the premise our predictor assumes.

**Network-based genetic monitoring of landscape fragmentation**, 2025. Graph-theoretic monitoring of fragmentation effects on genetic structure. Useful for how this community defines and measures the diversity response.

**Metapopulation networks unlock the effects of landscape fragmentation on agricultural pests and natural predators**, 2024. Neural networks used to predict species abundance from patch properties, reported above 80% accuracy. Closest thing here to a learned model, though it predicts abundance from patch features rather than modelling the network itself.

**Wu et al., *GNN-SDM: A Graph Neural Network-Based Framework Integrating Complex Landscape Patterns Into Species Distribution Modelling*, Global Ecology and Biogeography, 2025.** GNNs applied to landscape ecology, predicting habitat suitability from patch-based environmental features. Confirms GNNs have reached this domain, still in a predictive rather than generative role.

**Interpretability of graph neural networks to assess effects of global change drivers on ecological networks**, 2025. On extracting which structural features a GNN is actually using. Relevant later, when the question becomes what the predictor learned rather than whether it works.

## Gap

The question — how does network structure determine diversity — has an empirical literature. The inverse question — what network structure *maximises* diversity — does not appear to have been approached generatively here. That is where this project sits.
