"""Run-directory naming and resolution.

A run lives at ``results/<run_name>/`` and holds everything for one trained
model: the checkpoint, the history, the metrics, the figures and the logs. The
name is a pure function of the resolved config, so the workflow can compute
every output path before torch is imported — which is what makes the sweep a
Snakemake DAG rather than a shell loop.

The name carries the fields that differ from `NAME_BASELINE`, so a baseline run
is ``TMVAE_Tom1000`` and a two-axis sweep reads as what it varied:

    TMVAE_Tom1000
    TMVAE_dg4_Tom1000
    TMVAE_dg4-pattention_Tom1000

That is injective over the determining fields: two configs differing anywhere
differ in the token list, since a field at the baseline contributes no token and
a field away from it contributes one carrying its value.

**The baseline is frozen, and is not the same thing as the defaults.**
`config.py`'s defaults are what you get when a sweep does not say; this is what
names are measured against, and it must not move. Diffing against the live
defaults instead would make the defaults part of the on-disk contract, with two
consequences, one loud and one silent:

- raise the default epochs from 50 to 100 and a new default run claims
  ``TMVAE_Tom1000``, the directory a 50-epoch run already occupies. `config_guard`
  catches that, but only by locking the directory out;
- then ask for 50 again and it is named ``TMVAE_e50_Tom1000`` — a second
  directory holding a byte-identical config, retrained for nothing, and a
  duplicate row in the benchmarks table. Nothing catches that one at all.

Against a frozen baseline both disappear: a name depends only on the config, so
one config is always one directory whatever the defaults happen to be. The cost
is that names lengthen as the defaults drift away from the baseline — a field
whose default moved carries its token on every run, reading as unusual when it
is merely current. Re-baselining fixes that and invalidates every existing name,
so it is a migration to do deliberately, with `results/` cleared.

Injectivity also depends on every determining field appearing in both registries
below, so `run_name` refuses to name a config with a field missing from either
rather than quietly omitting it. That omission is the failure this design exists
to prevent: two points of a sweep sharing a directory and overwriting each other
with no error anywhere.

A field added after the baseline was frozen takes as its baseline value the one
that reproduces the behaviour from before it existed — usually its default — so
adding it renames nothing.
"""

import json
from pathlib import Path

from tm_ml.config import NON_DETERMINING

# Absolute, so a run writes to the repo's results/ whatever the working
# directory is. `ingest.py` computes its own ROOT rather than importing this,
# to stay free of the config/yaml import chain.
ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results"
# Committed, unlike results/: the ledger is the record of what was tried.
BENCHMARKS_CSV = ROOT / "docs" / "benchmarks" / "runs.csv"

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
    "lr_schedule": "sch",
    "lr_final_frac": "lff",
    "lr_warmup_epochs": "lrw",
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
    "gamma_warmup_epochs": "gw",
    "lambda_log": "ll",
    "lambda_log_warmup_epochs": "llw",
    # Not "lr*": that prefix is the learning rate's.
    "lambda_recon": "lrec",
    "lambda_recon_warmup_epochs": "lrecw",
}

NAMED_SEPARATELY = frozenset({"model", "drop"})

# Frozen 2026-09-16. What run names are measured against — NOT the defaults.
# Editing a value here renames every run that used it, so do it only as a
# deliberate migration with results/ cleared. Changing a default in config.py is
# free and renames nothing.
NAME_BASELINE = {
    "tmvae": {
        "seed": 0,
        "val_frac": 0.1,
        "test_frac": 0.1,
        "batch_size": 32,
        "epochs": 50,
        "lr": 1e-3,
        "lr_schedule": "constant",
        "lr_final_frac": 0.01,
        "lr_warmup_epochs": 0,
        "weight_decay": 0.0,
        "grad_clip": 5.0,
        "patience": 0,
        "standardize_target": True,
        "d_model": 64,
        "n_heads": 8,
        "d_ff": 128,
        "encoder_layers": 4,
        "decoder_layers": 2,
        "d_latent": 8,
        "d_global": 0,
        "edge_hidden": 32,
        "pair_hidden": 128,
        "predictor_hidden": 64,
        "pooling": "deepsets",
        "dropout": 0.0,
        "beta": 1.0,
        "gamma": 1.0,
        "beta_warmup_epochs": 0,
        "gamma_warmup_epochs": 0,
        # Added 2026-09-21. 0 is how the loss behaved before the term existed,
        # so every run named before this stays named the same.
        "lambda_log": 0.0,
        "lambda_log_warmup_epochs": 0,
        # 1.0 is the unweighted sum the loss used before the field existed.
        "lambda_recon": 1.0,
        "lambda_recon_warmup_epochs": 0,
    },
}

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
    """One token per determining field that differs from the frozen baseline."""
    baseline = NAME_BASELINE[cfg["model"]]
    fields = set(determining_fields(cfg)) - NAMED_SEPARATELY

    unregistered = sorted(fields - set(ABBREV))
    if unregistered:
        raise ValueError(
            f"no run-name abbreviation for {', '.join(unregistered)}. Add one to "
            "paths.ABBREV, or add the field to config.NON_DETERMINING if it does "
            "not change what is trained. Without it two configs could share a "
            "run directory."
        )

    unbaselined = sorted(fields - set(baseline))
    if unbaselined:
        raise ValueError(
            f"no run-name baseline for {', '.join(unbaselined)}. Add it to "
            f"paths.NAME_BASELINE[{cfg['model']!r}], at the value that reproduces "
            "how runs behaved before the field existed, so adding it renames "
            "nothing. Without it two configs could share a run directory."
        )

    return [
        f"{ABBREV[f]}{_fmt(cfg[f])}"
        for f in sorted(fields)
        if cfg[f] != baseline[f]
    ]


def run_name(cfg):
    """The directory name for a resolved config: ``Model[_tokens]_drop``."""
    parts = [CLASS_NAMES[cfg["model"]], "-".join(_tokens(cfg)), cfg["drop"]]
    return "_".join(p for p in parts if p)


def run_path(cfg):
    """Where a run is filed: ``<parent_path>/<run_name>``, or the bare name.

    The parent is organizational and may nest, so ``initial_testing`` and
    ``initial_testing/initial_sweep`` are both fine. It does not reach the run
    name, so the same config filed in two folders keeps one identity — which is
    also why sweeping it is refused.
    """
    parent = str(cfg.get("parent_path") or "").strip("/")
    name = run_name(cfg)
    return f"{parent}/{name}" if parent else name


def resolve(run):
    """Path to a run directory: ``results/<parent_path>/<run_name>``."""
    return RESULTS_ROOT / run


def config_guard(run_dir, cfg):
    """Record the config, refusing to reuse a directory trained under another.

    The name cannot carry a field the baseline does not know about, and a
    re-baselining changes what every existing name means. Under Snakemake a
    checkpoint newer than its inputs makes the train rule look satisfied, so
    either would otherwise be reported done without retraining. This is the
    check that turns that into an error, and it reads the stored config rather
    than the name, so it holds whatever the name happens to say.
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
