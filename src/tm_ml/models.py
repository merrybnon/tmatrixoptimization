"""VAE over transition matrices, with a property predictor on the latent.

Follows `docs/notes/architecture.md`. The encoder is a transformer over 20 node
tokens with an additive attention bias built from edge features, which is a
graph neural network on a complete graph. The decoder scores node pairs and row
softmaxes, so every output is a valid transition matrix by construction. The
predictor pools node latents, making it permutation invariant.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

N_NODE_FEATURES = 5
N_EDGE_FEATURES = 5

@dataclass
class TMVAEConfig:
    n_nodes: int = 20
    d_model: int = 64
    n_heads: int = 8
    d_ff: int = 128
    encoder_layers: int = 4
    decoder_layers: int = 2
    d_latent: int = 8
    edge_hidden: int = 32
    pair_hidden: int = 128
    predictor_hidden: int = 64
    dropout: float = 0.0
    log_eps: float = 1e-20
    beta: float = 1.0
    gamma: float = 1.0

@dataclass
class TMVAEOutput:
    T_hat: torch.Tensor
    log_y_hat: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor

@dataclass
class TMVAELoss:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    prop: torch.Tensor