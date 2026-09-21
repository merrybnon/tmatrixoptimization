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
    d_global: int = 0
    edge_hidden: int = 32
    pair_hidden: int = 128
    predictor_hidden: int = 64
    pooling: str = "deepsets"
    dropout: float = 0.0
    log_eps: float = 1e-20
    beta: float = 1.0
    gamma: float = 1.0
    # Weight on the log-space reconstruction term. 0 leaves it a pure
    # diagnostic, which is how every run before 2026-09-21 was trained.
    lambda_log: float = 0.0


@dataclass
class TMVAEOutput:
    T_hat: torch.Tensor
    log_T_hat: torch.Tensor
    log_y_hat: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    mu_global: torch.Tensor
    logvar_global: torch.Tensor
    z_global: torch.Tensor


@dataclass
class TMVAELoss:
    total: torch.Tensor
    recon: torch.Tensor
    kl: torch.Tensor
    prop: torch.Tensor
    kl_node: torch.Tensor
    kl_global: torch.Tensor
    log_recon: torch.Tensor


def off_diagonal(n, device):
    """Boolean ``(n, n)`` mask, True everywhere the transition matrix lives.

    The 380 off-diagonal entries are the object; the 20 diagonal slots are an
    artifact of storing it as a square array. Bool rather than a float mask on
    purpose — ``masked_fill`` never evaluates the masked entries, while
    multiplying by 0.0 still gives ``nan`` against an ``inf``.
    """
    return ~torch.eye(n, dtype=torch.bool, device=device)


def node_features(T, log_eps):
    """Five permutation-invariant scalars per node: ``(B, n, 5)``.

    Every feature is a symmetric function of the entries it sums over, so
    relabelling the nodes permutes the rows of the result and nothing else.
    Feeding a row ``T[i]`` directly would break that: its entries are ordered
    by j, so a relabelling would reorder the feature vector itself.

    Deliberately coarse. Five scalars cannot reconstruct a 20-node row and are
    not meant to — they give each node a starting identity, and what the model
    knows about individual entries arrives through the attention bias.

    Both entropies run over off-diagonal entries only. ``T_ii = 0`` makes the
    diagonal term ``0 * log 0``, which is zero by convention and ``nan`` in
    floating point. Two guards, and the order matters: the clamp keeps the log
    finite, the ``where`` discards the term whatever it holds. Neither can be
    applied after the fact, since ``nan * 0`` is still ``nan``.
    """
    off = off_diagonal(T.shape[-1], T.device)
    plogp = torch.where(off, T * torch.log(T.clamp_min(log_eps)), T.new_zeros(()))
    return torch.stack(
        [
            # Column sum, not row sum: rows sum to 1 identically, so the row
            # sum is a constant feature carrying no information. In-flow was
            # also what canonicalization would have sorted by — fine as a
            # continuous feature, discontinuous as an ordering.
            T.sum(-2),
            # How spread the out-flow is: near log 19 = 2.94 if a node hands
            # off uniformly, near 0 if one edge dominates. With the two maxima
            # these give a coarse shape of each node's local distribution.
            -plogp.sum(-1),
            -plogp.sum(-2),
            T.max(-1).values,
            T.max(-2).values,
        ],
        dim=-1,
    )


def edge_features(T, log_eps):
    """``[T_ij, T_ji, log T_ij, log T_ji, is_self]`` per ordered pair: ``(B, n, n, 5)``.

    Both directions, because mean abs(T - Tᵀ) = 0.034 against mean abs(T) =
    0.050: treating the graph as undirected discards most of the signal. Logs,
    because 15% of entries sit below 1e-4 and are invisible to a linear layer.

    ``log T_ii`` is ``log 0``. The clamp keeps it finite and the is-self flag
    tells the network the pair is structurally special, which an epsilon alone
    does not — an epsilon says "probability 1e-20", which is a lie.
    """
    log_T = torch.log(T.clamp_min(log_eps))
    eye = torch.eye(T.shape[-1], device=T.device).expand_as(T)
    return torch.stack(
        [T, T.transpose(-1, -2), log_T, log_T.transpose(-1, -2), eye], dim=-1
    )


class EdgeBias(nn.Module):
    """Edge features to one additive attention bias per head: ``(B, H, n, n)``.

    This is how the edges enter the encoder at all — the node features are five
    coarse scalars, and everything the model knows about an individual T_ij
    arrives through this bias. The MLP acts on one pair at a time, so
    relabelling the nodes permutes the bias matrix and nothing else.

    A compression of the architecture note's ``W[h_j ; e_ij]``, which
    conditions the *value* on the edge, down to n_heads numbers per pair that
    condition only the attention weight. Cheaper, and it keeps
    ``scaled_dot_product_attention`` usable. Revisit if reconstruction stalls.

    The LayerNorm is doing real work: the raw features mix probabilities in
    [0, 1] with logs down at -46.
    """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(N_EDGE_FEATURES),
            nn.Linear(N_EDGE_FEATURES, config.edge_hidden),
            nn.GELU(),
            nn.Linear(config.edge_hidden, config.n_heads),
        )

    def forward(self, e):
        return self.net(e).permute(0, 3, 1, 2)


class EdgeBiasedAttention(nn.Module):
    """Multi-head self-attention with an additive per-head bias.

    Hand-rolled rather than `nn.TransformerEncoderLayer`, which takes a fused
    fast path in eval mode that silently returns nan when handed an arbitrary
    additive float mask — which is exactly what an edge bias is. Training looks
    healthy and every evaluation is nan. `F.scaled_dot_product_attention`
    handles additive masks correctly in both modes.
    """

    def __init__(self, config):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError(
                f"d_model {config.d_model} is not divisible by n_heads {config.n_heads}"
            )
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, h, bias=None):
        B, n, _ = h.shape
        qkv = self.qkv(h).view(B, n, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(a.transpose(1, 2).reshape(B, n, -1))


class EquivariantLayer(nn.Module):
    """Pre-norm attention block: ``h <- h + MLP(h, sum_j a_ij v_j)``.

    The message-passing update of the architecture note, written as a
    transformer layer. On a complete graph the two are the same thing.

    Every operation here is per-node (LayerNorm and the feed-forward network,
    both over the feature axis) or symmetric over the node axis (attention), so
    the whole block is equivariant. A LayerNorm over the *node* axis instead
    would not be.
    """

    def __init__(self, config):
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attn = EdgeBiasedAttention(config)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, h, bias=None):
        h = h + self.drop(self.attn(self.norm_attn(h), bias))
        return h + self.drop(self.ff(self.norm_ff(h)))


class NodePool(nn.Module):
    """Collapse node latents to one invariant vector: ``(B, n, d) -> (B, out_dim)``.

    This is the only invariant step in the predictor, and the reason ascent
    commutes with relabelling — everything after it has no node axis to permute.

    Three modes, in increasing order of machinery:

    - ``mean``: the raw average, phi = identity.
    - ``deepsets``: ``rho(mean(phi(z)))``, Zaheer et al. as actually stated.
    - ``attention``: one learned seed query attending over the nodes, the PMA
      block of Set Transformer, which weights nodes instead of averaging them.

    ``mean`` is here as a baseline rather than as a real option. Wagstaff et al.
    (ICML 2019) show a mean- or sum-decomposable function needs phi's output
    width to be at least the set size to represent every continuous invariant
    function on sets that big; at n = 20 against d_latent = 8, the raw average
    is below that bound before the missing nonlinearity is even counted.
    ``predictor_hidden = 64`` clears it, so ``deepsets`` is not capacity-limited
    and ``attention`` is a claim about inductive bias, to be settled by sweeping
    the two rather than by argument.
    """

    def __init__(self, config):
        super().__init__()
        self.mode = config.pooling
        d_hidden = config.predictor_hidden

        if self.mode == "mean":
            self.phi = nn.Identity()
            self.out_dim = config.d_latent
            return

        if self.mode not in ("deepsets", "attention"):
            raise ValueError(
                f"unknown pooling {self.mode!r}; expected mean, deepsets or attention"
            )

        self.phi = nn.Sequential(
            nn.Linear(config.d_latent, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.out_dim = d_hidden

        if self.mode == "attention":
            # One seed query, so the softmax runs over the 20 nodes and the
            # result is a convex combination of them — invariant, like the mean,
            # but with learned weights.
            self.seed = nn.Parameter(torch.randn(d_hidden) * d_hidden**-0.5)
            self.to_kv = nn.Linear(d_hidden, 2 * d_hidden)

    def forward(self, z):
        h = self.phi(z)
        if self.mode != "attention":
            return h.mean(1)

        k, v = self.to_kv(h).chunk(2, dim=-1)
        q = self.seed.expand(h.shape[0], 1, -1)
        return F.scaled_dot_product_attention(q, k, v).squeeze(1)


class TMVAE(nn.Module):
    """Encoder, decoder and property predictor over a 20-node complete digraph.

    Every internal representation is equivariant and every loss term is
    invariant, which is what makes an element-wise reconstruction loss well
    posed with no graph matching: node i enters at slot i, its latent sits at
    slot i, and its reconstructed row leaves at slot i, so the correct
    alignment is the identity for every input by construction.
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or TMVAEConfig()
        cfg = self.config

        self.node_in = nn.Sequential(
            nn.LayerNorm(N_NODE_FEATURES), nn.Linear(N_NODE_FEATURES, cfg.d_model)
        )
        self.edge_in = EdgeBias(cfg)
        self.encoder = nn.ModuleList(
            EquivariantLayer(cfg) for _ in range(cfg.encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(cfg.d_model)
        self.to_latent = nn.Linear(cfg.d_model, 2 * cfg.d_latent)

        # The hybrid-latent hook, off by default. An invariant z_graph
        # broadcast along the node axis preserves equivariance, since with
        # respect to that axis it is a constant. It adds no representational
        # power — by Deep Sets it is a deterministic function of the node set —
        # but it buys a correlated prior, which is the fix for 20 iid draws
        # from N(0, I) having no reason to form a coherent landscape. Turning
        # it on is a config change: this is the only branch, and everything
        # downstream stays zero-width.
        self.to_global = (
            nn.Linear(cfg.d_model, 2 * cfg.d_global) if cfg.d_global else None
        )

        self.from_latent = nn.Linear(cfg.d_latent + cfg.d_global, cfg.d_model)
        self.decoder = nn.ModuleList(
            EquivariantLayer(cfg) for _ in range(cfg.decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(cfg.d_model)
        self.pair = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.pair_hidden),
            nn.GELU(),
            nn.Linear(cfg.pair_hidden, 1),
        )

        self.pool = NodePool(cfg)
        self.predictor = nn.Sequential(
            nn.Linear(self.pool.out_dim + cfg.d_global, cfg.predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.predictor_hidden, cfg.predictor_hidden),
            nn.GELU(),
            nn.Linear(cfg.predictor_hidden, 1),
        )

    def encode(self, T):
        """``T`` to per-node posteriors, plus the graph-level one if enabled."""
        h = self.node_in(node_features(T, self.config.log_eps))
        bias = self.edge_in(edge_features(T, self.config.log_eps))
        for layer in self.encoder:
            h = layer(h, bias)
        h = self.encoder_norm(h)

        mu, logvar = self.to_latent(h).chunk(2, dim=-1)
        if self.to_global is None:
            empty = h.new_zeros(h.shape[0], 0)
            mu_global, logvar_global = empty, empty
        else:
            # Mean over nodes is the invariant step, and the only one.
            mu_global, logvar_global = self.to_global(h.mean(1)).chunk(2, dim=-1)
        return mu, logvar, mu_global, logvar_global

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _broadcast(self, z, z_global):
        """Concatenate the invariant z_graph onto every node latent."""
        return torch.cat([z, z_global.unsqueeze(1).expand(-1, z.shape[1], -1)], dim=-1)

    def decode(self, z, z_global):
        """Node latents to ``(T_hat, log_T_hat)``, both ``(B, n, n)``.

        The pre-layers are load-bearing, not a refinement. The pair function is
        local — logits_kl = MLP([h_k ; h_l]) does not involve h_i for any other
        i, so z_i alone would touch 39 of 380 entries. The decoder's global
        pathway exists entirely because these layers mix the latents first.
        They take no edge bias: there are no edges yet, which is the point.

        Concatenation rather than a symmetric form like z_i·z_j, so that
        T_ij != T_ji is representable at all.
        """
        h = self.from_latent(self._broadcast(z, z_global))
        for layer in self.decoder:
            h = layer(h)
        h = self.decoder_norm(h)

        n = h.shape[1]
        pairs = torch.cat(
            [h.unsqueeze(2).expand(-1, -1, n, -1), h.unsqueeze(1).expand(-1, n, -1, -1)],
            dim=-1,
        )
        logits = self.pair(pairs).squeeze(-1)

        # Mask, then normalize. Order matters absolutely: -inf on the diagonal
        # before the softmax makes T_ii exactly 0, the other 19 entries sum to
        # exactly 1, and the diagonal logit takes exactly zero gradient.
        # Softmaxing over all 20 and zeroing afterwards leaves rows short.
        logits = logits.masked_fill(~off_diagonal(n, logits.device), float("-inf"))
        return F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)

    def predict(self, z, z_global):
        """Pool over nodes, then an MLP: invariant, so it can predict a scalar.

        `NodePool` is where the node axis dies and the whole predictor becomes
        invariant. Everything here is an ordinary MLP on a vector with no node
        structure, which is the point of pooling first.

        Mean rather than sum inside the pool only for conditioning; n is fixed
        at 20, so the two differ by a constant the next linear layer absorbs.
        Sum being the expressive choice is an argument about multisets of
        varying size.

        Emits log(decay exponent), not the exponent: the target runs 48.8 to
        2041.6 and is right-skewed. The log is the caller's to take.
        """
        return self.predictor(torch.cat([self.pool(z), z_global], dim=-1)).squeeze(-1)

    def forward(self, T, sample=True):
        """Encode, sample, decode and predict.

        Pass ``sample=False`` for any equivariance check. Equivariance of a
        sampled z holds only in distribution, since epsilon is drawn
        independently per node, so a single sample from a relabelled input is
        not the relabelling of a single sample from the original.
        """
        mu, logvar, mu_global, logvar_global = self.encode(T)
        if sample:
            z = self.reparameterize(mu, logvar)
            z_global = self.reparameterize(mu_global, logvar_global)
        else:
            z, z_global = mu, mu_global

        T_hat, log_T_hat = self.decode(z, z_global)
        return TMVAEOutput(
            T_hat=T_hat,
            log_T_hat=log_T_hat,
            log_y_hat=self.predict(z, z_global),
            mu=mu,
            logvar=logvar,
            z=z,
            mu_global=mu_global,
            logvar_global=logvar_global,
            z_global=z_global,
        )


def gaussian_kl(mu, logvar):
    """KL( N(mu, sigma) || N(0, I) ), summed over every axis but the batch."""
    per_entry = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return per_entry.flatten(1).sum(-1)


def tmvae_loss(output, T, log_y, config):
    """``recon + beta * KL(q || p) + gamma * (log y_hat - log y)^2 + lambda_log * log_recon``.

    Per-row KL because rows are distributions and this is the categorical
    likelihood written out. Every term is a sum over the graph and a mean over
    the batch. ``log_y`` arrives already logged — the model emits a log, and
    normalizing the target belongs to the dataset.

    The reconstruction is equivariant and this loss is invariant: the outer sum
    runs over the same 20 terms in a different order, and each inner KL over
    the same 20 entries in a different order.
    """
    n = T.shape[-1]
    off = off_diagonal(n, T.device)

    # Two structural zeros meet here. T_ij log T_ij is 0 * log 0 on the
    # diagonal, and the decoder's masked softmax makes log T_hat_ii exactly
    # -inf, so 0 * -inf is nan. Mask before they multiply, not after: nan * 0
    # is still nan.
    log_T = torch.log(T.clamp_min(config.log_eps))
    log_T_hat = output.log_T_hat.masked_fill(~off, 0.0)
    recon = (T * torch.where(off, log_T - log_T_hat, T.new_zeros(()))).sum((-1, -2))

    # The prior KL is invariant only because the prior is factorized and
    # identical per node. A non-factorized prior over the flat 160-vector would
    # break that. Kept as two groups because the global slot is the one that
    # collapses, and a pooled number would hide it.
    kl_node = gaussian_kl(output.mu, output.logvar)
    kl_global = gaussian_kl(output.mu_global, output.logvar_global)
    prop = (output.log_y_hat - log_y).pow(2)

    recon, kl_node, kl_global, prop = (
        t.mean() for t in (recon, kl_node, kl_global, prop)
    )
    kl = kl_node + kl_global

    # The counterweight to the forward KL, and a pure diagnostic when
    # lambda_log is 0. Forward KL weights each term by the true T_ij, so it
    # punishes T_hat being too small where truth is large and barely notices
    # T_hat being too large where truth is tiny — which is where the weak links
    # live, and they plausibly govern the timescale being predicted. Unweighted,
    # so every entry counts the same.
    #
    # L1 rather than squared: log errors span ten decades, and a single entry
    # near the clamp would otherwise set the gradient for its whole row.
    #
    # A mean over entries, where `recon` is a sum over them, so the two terms
    # are deliberately not on one scale. Per entry the KL's weight is T_ij and
    # this term's is lambda_log / 380; they cross near the median entry, 2.6e-3
    # at lambda_log = 1. Below the crossover this term decides and above it the
    # KL still does, which is the intent — it acts on the tail the KL cannot
    # see, and would not reach the strongest entry in a row until a lambda_log
    # two orders larger than anything worth running. The definition is unchanged
    # from when it was diagnostic-only, so the ledger's log_recon column stays
    # comparable across the change.
    log_recon = (log_T - log_T_hat)[..., off].abs().mean()

    return TMVAELoss(
        total=(
            recon
            + config.beta * kl
            + config.gamma * prop
            + config.lambda_log * log_recon
        ),
        recon=recon,
        kl=kl,
        prop=prop,
        kl_node=kl_node,
        kl_global=kl_global,
        log_recon=log_recon,
    )
