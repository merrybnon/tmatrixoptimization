"""Architecture diagram for a trained run, as Mermaid.

    pixi run -e ml architecture --run global_latent/TMVAE_dg1_Tom1000

Writes into ``figures/`` in the run directory:

- ``architecture.mmd``  the Mermaid source, readable as text and rendered by
                        GitHub and VS Code as-is
- ``architecture.svg``  the same, rendered by mermaid-cli

Drawn from the checkpoint rather than from the sweep, so it shows what was
trained: layer counts, widths, the pooling mode, whether the graph-level latent
exists, the loss weights, and the parameter count of each block, summed from the
state dict itself. The pieces that do not vary between runs — the feature lists,
the masked row softmax — are drawn from what `models.py` does today.

Rendering shells out to mermaid-cli through npx, which fetches it on first use
and drives a headless Chrome. `--no-render` skips that and writes the source
only, which is what the tests do.
"""

import argparse
import subprocess
from collections import Counter

import torch

from tm_ml import paths
from tm_ml.models import N_EDGE_FEATURES, N_NODE_FEATURES, TMVAEConfig

MERMAID_CLI = "@mermaid-js/mermaid-cli@11.17.0"
SOURCE = "architecture.mmd"
RENDERED = "architecture.svg"


def param_counts(state_dict):
    """Parameters per top-level module, keyed by its attribute name on `TMVAE`.

    Norms are folded into the stack they close, so the diagram's encoder and
    decoder boxes carry everything between input and latent, and latent and
    pair function.
    """
    fold = {"encoder_norm": "encoder", "decoder_norm": "decoder"}
    counts = Counter()
    for key, tensor in state_dict.items():
        module = key.split(".")[0]
        counts[fold.get(module, module)] += tensor.numel()
    return counts


def _k(n):
    """12345 -> '12.3k', for a label that has to stay short."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _pool_label(m):
    if m.pooling == "mean":
        return f"mean over nodes<br/>φ = identity · out {m.d_latent}"
    phi = f"φ: {m.d_latent} → {m.predictor_hidden} → {m.predictor_hidden}"
    if m.pooling == "attention":
        return f"PMA: one seed query attends over φ(z_i)<br/>{phi}"
    return f"Deep Sets: mean over nodes of φ(z_i)<br/>{phi}"


def diagram(m, cfg, counts, run):
    """The Mermaid source for model config `m`, run config `cfg`, param `counts`."""
    n, h, g = m.n_nodes, m.d_model, m.d_global
    dropout = f", dropout {m.dropout:g}" if m.dropout else ""
    layer = (f"{m.n_heads} heads × {h // m.n_heads} dims"
             f"<br/>FFN {h} → {m.d_ff} → {h}, pre-norm, residual{dropout}")
    total = sum(counts.values())

    lines = [
        "---",
        f"title: {run} · {total:,} parameters",
        "config:",
        "  flowchart:",
        "    wrappingWidth: 320",
        "---",
        "flowchart LR",
        f'  T[/"T · {n}×{n} transition matrix, zero diagonal"/]',
        f'  TH[/"T̂ · {n}×{n}, row-stochastic, zero diagonal"/]',
        "",
        '  subgraph ENC["Encoder · edge-biased graph transformer"]',
        "    direction TB",
        f'    NF["node features · {n}×{N_NODE_FEATURES}<br/>'
        'col sum, row and col entropy, row and col max"]',
        f'    EF["edge features · {n}×{n}×{N_EDGE_FEATURES}<br/>'
        'T_ij, T_ji, log T_ij, log T_ji, is_self"]',
        f'    NI["LayerNorm → Linear {N_NODE_FEATURES} → {h}'
        f'<br/>{_k(counts["node_in"])} params"]',
        f'    EB["edge bias MLP {N_EDGE_FEATURES} → {m.edge_hidden} → {m.n_heads}'
        f'<br/>one bias b_ij per head · {_k(counts["edge_in"])} params"]',
        f'    EL["{m.encoder_layers} × attention layer<br/>'
        f'softmax_j( q_i·k_j / √d + b_ij )<br/>{layer}<br/>'
        f'{_k(counts["encoder"])} params"]',
        "    NF --> NI --> EL",
        "    EF --> EB",
        '    EB -. "same bias, every layer" .-> EL',
        "  end",
        "",
        # The middle column. Mermaid drops a subgraph's own direction as soon
        # as any node inside it links to a node outside, so everything that
        # crosses between columns is an edge between whole subgraphs. That is
        # what keeps the predictor hanging down off the latent rather than
        # laid out as a fourth column beside the decoder.
        '  subgraph MID[" "]',
        "    direction TB",
        '    subgraph LAT["Latent"]',
        "      direction LR",
        f'      ZN(["z_i · {n}×{m.d_latent} node latents<br/>'
        f'Linear {h} → 2×{m.d_latent}: μ_i, log σ²_i<br/>'
        f'{_k(counts["to_latent"])} params"])',
    ]
    if g:
        lines.append(
            f'      ZG(["z_graph · {g} graph-level<br/>mean over nodes, '
            f'Linear {h} → 2×{g}<br/>{_k(counts["to_global"])} params"])'
        )
    lines += [
        "    end",
        '    subgraph PRED["Predictor · invariant"]',
        "      direction TB",
        f'      PO["{_pool_label(m)}<br/>{_k(counts["pool"])} params"]',
        f'      MP["{"concat z_graph → " if g else ""}MLP '
        f'{(m.d_latent if m.pooling == "mean" else m.predictor_hidden) + g} → '
        f'{m.predictor_hidden} → {m.predictor_hidden} → 1'
        f'<br/>{_k(counts["predictor"])} params"]',
        "      PO --> MP",
        "    end",
        '    Y[/"log ŷ · log decay exponent"/]',
        "    ZN --> PO",
        *(["    ZG --> MP"] if g else []),
        "    MP --> Y",
        "  end",
        "",
        '  subgraph DEC["Decoder · equivariant"]',
        "    direction TB",
        f'    FL["{"concat z_graph onto every z_i → " if g else ""}'
        f'Linear {m.d_latent + g} → {h}<br/>{_k(counts["from_latent"])} params"]',
        f'    DL["{m.decoder_layers} × attention layer, no edge bias<br/>{layer}'
        f'<br/>{_k(counts["decoder"])} params"]',
        f'    PF["pair MLP [h_i ; h_j] · {2 * h} → {m.pair_hidden} → 1'
        f'<br/>asymmetric, logits {n}×{n} · {_k(counts["pair"])} params"]',
        '    SM["diagonal → −∞, then row softmax"]',
        "    FL --> DL --> PF --> SM",
        "  end",
        "",
        f'  L{{{{"loss<br/>{cfg.get("lambda_recon", 1.0):g} · Σ_i KL(T_i ‖ T̂_i)'
        f' + {cfg.get("lambda_log", 0.0):g} · log-space recon<br/>'
        f'+ {cfg["beta"]:g} · KL(q ‖ N(0, I)){" node + graph" if g else ""}'
        f'<br/>+ {cfg["gamma"]:g} · (log ŷ − log y)²"}}}}',
        "",
        "  T --> ENC --> MID --> DEC --> TH",
        "  TH -.-> L",
        "",
        "  style MID fill:none,stroke:none",
        "  classDef io fill:#eef2f7,stroke:#56657a",
        "  classDef latent fill:#fdf1dc,stroke:#b07a1e",
        "  classDef loss fill:#f7e8ea,stroke:#9a4452",
        "  class T,TH,Y io",
        f"  class ZN{',ZG' if g else ''} latent",
        "  class L loss",
    ]
    return "\n".join(lines) + "\n"


def render(source, out):
    """Render `source` to `out` with mermaid-cli, whose format follows the suffix."""
    subprocess.run(
        ["npx", "-y", MERMAID_CLI, "-i", str(source), "-o", str(out),
         "-b", "white", "--quiet"],
        check=True,
    )


def architecture(run, render_svg=True):
    run_dir = paths.resolve(run)
    checkpoint_path = run_dir / paths.CHECKPOINT
    if not checkpoint_path.exists():
        raise SystemExit(f"{checkpoint_path} does not exist; train the run first")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    m = TMVAEConfig(**checkpoint["model_config"])
    counts = param_counts(checkpoint["state_dict"])

    out_dir = run_dir / paths.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    source = out_dir / SOURCE
    source.write_text(diagram(m, checkpoint["config"], counts, run))
    written = [source]
    if render_svg:
        render(source, out_dir / RENDERED)
        written.append(out_dir / RENDERED)
    for path in written:
        print(f"wrote {path}")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory name under results/")
    parser.add_argument("--no-render", action="store_true",
                        help="write the Mermaid source only, skipping mermaid-cli")
    args = parser.parse_args()

    architecture(args.run, render_svg=not args.no_render)


if __name__ == "__main__":
    main()
