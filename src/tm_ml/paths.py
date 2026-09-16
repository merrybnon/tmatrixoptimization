"""Run-directory naming and resolution.

A run lives at ``results/<run_name>/`` and holds everything for one trained
model: the checkpoint, the history, the metrics, the figures and the logs. The
name is a pure function of the resolved config, so the workflow can compute
every output path before torch is imported — which is what makes the sweep a
Snakemake DAG rather than a shell loop.

The name carries the fields that differ from their defaults, so a default run
is ``TMVAE_Tom1000`` and a two-axis sweep reads as what it varied:

    TMVAE_Tom1000
    TMVAE_dg4_Tom1000
    TMVAE_dg4-pattention_Tom1000

That is injective over the determining fields: two configs differing anywhere
differ in the token list, since a field at its default contributes no token and
a field away from it contributes one carrying its value.

Injectivity depends on every determining field having an abbreviation, so
`run_name` refuses to name a config with an unregistered one rather than
quietly omitting it. Forgetting that registration is the failure this design
exists to prevent: two points of a sweep sharing a directory and overwriting
each other with no error anywhere.
"""

import json
from pathlib import Path

from tm_ml.config import NON_DETERMINING, defaults_for

RESULTS_ROOT = Path("results")

CLASS_NAMES = {"tmvae": "TMVAE"}

# Filenames every stage agrees on, so the Snakefile and the scripts cannot
# drift apart over a string.
CHECKPOINT = "best.pt"
HISTORY = "history.csv"
METRICS = "metrics.json"
STORED_CONFIG = "config.json"
RESOLVED_CONFIG = "config.yaml"
TRAIN_META = "train_meta.json"

# Field to name token. `model` and `drop` are excluded because they are named
# separately, at the front and the back.
ABBREV = {
    "seed": "s",
    "val_frac": "v",
    "test_frac": "t",
    "batch_size": "bs",
    "epochs": "e",
    "lr": "lr",
    "weight_decay": "wd",
    "grad_clip": "gc",
    "patience": "pat",
    "standardize_target": "std",
    "d_model": "dm",
    "n_heads": "nh",
    "d_ff": "ff",
    "encoder_layers": "el",
    "decoder_layers": "dl",
    "d_latent": "dz",
    "d_global": "dg",
    "edge_hidden": "eh",
    "pair_hidden": "ph",
    "predictor_hidden": "prh",
    "pooling": "p",
    "dropout": "do",
    "beta": "b",
    "gamma": "g",
    "beta_warmup_epochs": "bw",
}

NAMED_SEPARATELY = frozenset({"model", "drop"})

_duplicates = {a for a in ABBREV.values() if list(ABBREV.values()).count(a) > 1}
if _duplicates:
    raise RuntimeError(f"ABBREV is not injective; reused: {sorted(_duplicates)}")


def _fmt(value):
    """Filename-safe value: 1e-3 stays 1e-3, 0.5 becomes 0p5, True becomes 1."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.01:
            mantissa, exponent = f"{value:e}".split("e")
            return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return str(value)


def determining_fields(cfg):
    """The fields that decide what gets trained, so the ones that name a run."""
    return {k: v for k, v in cfg.items() if k not in NON_DETERMINING}


def _tokens(cfg):
    """One token per determining field that differs from its default."""
    defaults = defaults_for(cfg["model"])
    fields = set(determining_fields(cfg)) - NAMED_SEPARATELY

    unregistered = sorted(fields - set(ABBREV))
    if unregistered:
        raise ValueError(
            f"no run-name abbreviation for {', '.join(unregistered)}. Add one to "
            "paths.ABBREV, or add the field to config.NON_DETERMINING if it does "
            "not change what is trained. Without it two configs could share a "
            "run directory."
        )

    return [
        f"{ABBREV[f]}{_fmt(cfg[f])}"
        for f in sorted(fields)
        if cfg[f] != defaults[f]
    ]


def run_name(cfg):
    """The directory name for a resolved config: ``Model[_tokens]_drop``."""
    parts = [CLASS_NAMES[cfg["model"]], "-".join(_tokens(cfg)), cfg["drop"]]
    return "_".join(p for p in parts if p)


def resolve(run):
    """Path to a run directory: ``results/<run>``."""
    return RESULTS_ROOT / run


def config_guard(run_dir, cfg):
    """Record the config, refusing to reuse a directory trained under another.

    The run name only carries fields that differ from their defaults, so
    changing a default changes what every unnamed run means. Under Snakemake a
    checkpoint newer than its inputs makes the train rule look satisfied, and
    the run would be reported done without retraining. This is the check that
    turns that into an error.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stored_path = run_dir / STORED_CONFIG
    fields = determining_fields(cfg)

    if stored_path.exists():
        stored = json.loads(stored_path.read_text())
        differs = sorted(
            k for k in set(stored) | set(fields)
            if str(stored.get(k)) != str(fields.get(k))
        )
        if differs:
            detail = "\n".join(
                f"  {k}: stored {stored.get(k)!r} vs requested {fields.get(k)!r}"
                for k in differs
            )
            raise SystemExit(
                f"{run_dir} was trained with a different config:\n{detail}\n"
                "Delete the directory, or change a field that reaches the run name."
            )

    stored_path.write_text(json.dumps(dict(sorted(fields.items())), indent=2) + "\n")
