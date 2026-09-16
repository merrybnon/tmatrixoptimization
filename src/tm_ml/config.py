"""Resolve a sweep YAML into one config dict per run.

Defaults live here, not in the YAML: a sweep file holds exactly the parameters
that sweep varies, and every field given a list becomes a swept axis. Several
lists give the cartesian product.

    pixi run -e ml pipeline --configfile config/sweeps/latent.yaml

Imports nothing heavier than yaml and pathlib, so the Snakefile can expand a
200-run sweep and build its DAG without loading torch. `paths.py` depends on
this module for the schema; nothing here depends on `paths.py`.

Two model fields are deliberately absent. ``n_nodes`` is a property of the drop
and is read from the data at construction, so it cannot drift from what was
ingested. ``log_eps`` is a numerical guard against log 0, not a hyperparameter;
sweeping it would mean sweeping the definition of the loss.
"""

import itertools
import math
from pathlib import Path

import yaml

# Changing a default here is free and renames nothing: run names are measured
# against the frozen `paths.NAME_BASELINE`, not against these. A default says
# what you get when a sweep stays silent; the baseline says what a name means.
COMMON_DEFAULTS = {
    "model": "tmvae",
    "drop": "Tom1000",
    # Where the run is filed under results/. May contain slashes to nest. Purely
    # organizational, so it is non-determining: the folder a run sits in does not
    # change what is trained, it stays out of the run name and out of
    # config_guard, and a run can be moved between folders without being locked
    # out of its own directory.
    "parent_path": "",
    "seed": 0,
    "val_frac": 0.1,
    "test_frac": 0.1,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-3,
    # Decay is off by default. `lr_final_frac` is the terminal learning rate as
    # a fraction of `lr`, and means nothing without a schedule, so a constant-lr
    # run is coerced back to INERT_LR_FINAL_FRAC in _finalize — otherwise two
    # runs that train identically would differ in the name and be trained twice.
    "lr_schedule": "constant",
    "lr_final_frac": 0.01,
    # Linear ramp to `lr` over the first N epochs, then the schedule takes the
    # rest. Orthogonal to lr_schedule — a warmup with a constant rate is a
    # perfectly ordinary thing to want — so unlike lr_final_frac it is never
    # inert and always reaches the run name.
    "lr_warmup_epochs": 0,
    "weight_decay": 0.0,
    "grad_clip": 5.0,
    # 0 disables early stopping. Determining, because stopping early changes
    # which checkpoint a run ends up selecting.
    "patience": 0,
    # Fitted on the train split alone and inverted for reporting. The target is
    # always logged; this is whether it is also centred and scaled, which fixes
    # what gamma means — log y sits near 5.7 with a spread under 0.7, so an
    # untrained predictor starts ~33 nats from it for no interesting reason.
    "standardize_target": True,
}

MODEL_DEFAULTS = {
    "tmvae": {
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
        # Linear warmup on beta from 0. The per-group KL collapsing before
        # reconstruction has learned anything is the failure mode to expect,
        # and this is the cheapest remedy.
        "beta_warmup_epochs": 0,
        # The same for gamma, though motivated differently: beta warmup protects
        # the latent from collapsing early, gamma warmup lets reconstruction
        # establish itself before the property term starts pulling the latent
        # around. Worth noting the pull is the mechanism, not a nuisance — it is
        # what organizes the latent for ascent — so delaying it is a trade, not
        # a free fix.
        "gamma_warmup_epochs": 0,
    },
}

# Settings that change how a run is executed but not what is trained. They stay
# out of the run name and out of config_guard, and sweeping one is refused —
# every point would land in the same directory.
RUNTIME_DEFAULTS = {
    "gpu": "auto",
    "num_threads": 1,
    "num_workers": 0,
    "log_every": 1,
}

# Set by the workflow rather than by a person: which sweep file this came from
# and which run of it this is.
WORKFLOW_FIELDS = frozenset({"config", "run"})

# Filing, not science. Kept separate from the runtime fields so the reason each
# one is non-determining stays legible.
ORGANIZING_FIELDS = frozenset({"parent_path"})

NON_DETERMINING = frozenset(RUNTIME_DEFAULTS) | WORKFLOW_FIELDS | ORGANIZING_FIELDS

POOLING_MODES = ("mean", "deepsets", "attention")

LR_SCHEDULES = ("constant", "cosine", "exponential")

# Loss weights that can be warmed up, each with a `<name>_warmup_epochs` field.
LOSS_WEIGHTS = ("beta", "gamma")

# What "no decay" looks like on disk. Frozen rather than read from the defaults:
# a constant-lr run is coerced to this value, so moving it would put a spurious
# token on every constant-lr run and rename them all.
INERT_LR_FINAL_FRAC = 0.01

MODEL_FIELDS = {
    kind: frozenset(COMMON_DEFAULTS) | frozenset(fields) | NON_DETERMINING
    for kind, fields in MODEL_DEFAULTS.items()
}


def defaults_for(kind):
    """Every field for one model kind, at its default value."""
    if kind not in MODEL_DEFAULTS:
        raise ValueError(f"unknown model {kind!r}; expected one of {sorted(MODEL_DEFAULTS)}")
    return {**COMMON_DEFAULTS, **MODEL_DEFAULTS[kind], **RUNTIME_DEFAULTS, "model": kind}


def _flatten(spec):
    """Merge nested sections into one mapping.

    A sweep file may group fields under headings for readability — `model:`,
    `training:`, whatever reads best — and any top-level mapping is treated as
    such a section. Grouping carries no meaning; a flat file is equally valid.
    """
    flat = {}
    for key, value in (spec or {}).items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _candidates(flat, field):
    """Every value a field takes across the sweep, whether or not it is an axis."""
    value = flat.get(field, COMMON_DEFAULTS[field])
    return value if isinstance(value, list) else [value]


def expand_sweep(spec):
    """One resolved config per point of the sweep the spec describes."""
    flat = _flatten(spec)
    kind = flat.get("model", COMMON_DEFAULTS["model"])
    if kind not in MODEL_DEFAULTS:
        raise ValueError(f"unknown model {kind!r}; expected one of {sorted(MODEL_DEFAULTS)}")

    unusable = set(flat) - MODEL_FIELDS[kind]
    if unusable:
        elsewhere = {f: k for k, fs in MODEL_DEFAULTS.items() for f in fs if k != kind}
        detail = "; ".join(
            f"{f} (belongs to {elsewhere[f]})" if f in elsewhere else f
            for f in sorted(unusable)
        )
        raise ValueError(f"model {kind!r} does not accept: {detail}")

    axes = {k: v for k, v in flat.items() if isinstance(v, list)}
    for key, values in axes.items():
        if not values:
            raise ValueError(f"{key!r} is an empty list; nothing to sweep")

    # A run directory is a pure function of the determining fields, so sweeping
    # anything else collapses every point onto one directory — silently, since
    # the workflow keys its runs by name.
    inert = sorted(set(axes) & NON_DETERMINING)
    if inert:
        raise ValueError(
            f"cannot sweep {', '.join(inert)}: these do not reach the run name, so "
            "every point of the sweep would land in the same directory. Give a "
            "single value, or vary a field that names a run."
        )

    # The same hazard one level down: lr_final_frac is inert without a
    # schedule, so sweeping it against a constant lr collapses those points
    # onto one directory.
    if "lr_final_frac" in axes and "constant" in _candidates(flat, "lr_schedule"):
        raise ValueError(
            "cannot sweep lr_final_frac with lr_schedule 'constant': it has no "
            "effect without a schedule, so those points would land in the same "
            "directory. Set lr_schedule to cosine or exponential."
        )

    fixed = {k: v for k, v in flat.items() if k not in axes}

    configs = []
    for point in itertools.product(*axes.values()):
        cfg = defaults_for(kind)
        cfg.update(fixed)
        cfg.update(dict(zip(axes, point)))
        _finalize(cfg)
        configs.append(cfg)
    return configs


def _finalize(cfg):
    """Validate a resolved config in place.

    Everything checkable without the data is checked here, so a malformed sweep
    fails while Snakemake is building the DAG rather than an hour into a job.
    """
    kind = cfg["model"]

    for field in ("epochs", "batch_size", "d_latent"):
        if field in cfg and cfg[field] < 1:
            raise ValueError(f"{field} must be at least 1, got {cfg[field]!r}")

    if not 0 < cfg["val_frac"] + cfg["test_frac"] < 1:
        raise ValueError(
            f"val_frac + test_frac must leave a training split: got "
            f"{cfg['val_frac']} + {cfg['test_frac']}"
        )

    if cfg["lr_schedule"] not in LR_SCHEDULES:
        raise ValueError(
            f"unknown lr_schedule {cfg['lr_schedule']!r}; expected one of {list(LR_SCHEDULES)}"
        )
    if not 0 < cfg["lr_final_frac"] <= 1:
        raise ValueError(
            f"lr_final_frac is a fraction of lr and must be in (0, 1], got "
            f"{cfg['lr_final_frac']!r}"
        )
    if not 0 <= cfg["lr_warmup_epochs"] <= cfg["epochs"]:
        raise ValueError(
            f"lr_warmup_epochs must be between 0 and epochs {cfg['epochs']}, got "
            f"{cfg['lr_warmup_epochs']!r}; the rate would never reach lr"
        )
    # Checked before coercing, so a nonsensical value is still reported rather
    # than quietly discarded.
    if cfg["lr_schedule"] == "constant":
        cfg["lr_final_frac"] = INERT_LR_FINAL_FRAC

    if kind != "tmvae":
        return

    if cfg["d_model"] % cfg["n_heads"]:
        raise ValueError(
            f"d_model {cfg['d_model']} is not divisible by n_heads {cfg['n_heads']}"
        )
    if cfg["pooling"] not in POOLING_MODES:
        raise ValueError(
            f"unknown pooling {cfg['pooling']!r}; expected one of {list(POOLING_MODES)}"
        )
    if cfg["d_global"] < 0:
        raise ValueError(f"d_global must be at least 0, got {cfg['d_global']!r}")
    for weight in LOSS_WEIGHTS:
        field = f"{weight}_warmup_epochs"
        if not 0 <= cfg[field] <= cfg["epochs"]:
            raise ValueError(
                f"{field} must be between 0 and epochs {cfg['epochs']}, got "
                f"{cfg[field]!r}; {weight} would never reach its full value"
            )


def lr_at(cfg, epoch):
    """Learning rate for a 0-indexed epoch: ramp to ``lr``, then decay from it.

    The first ``lr_warmup_epochs`` epochs run linearly from ``lr / warmup`` up
    to ``lr`` — starting at a fraction rather than at zero, so no epoch is spent
    not learning. The schedule then decays from ``lr`` to ``lr * lr_final_frac``
    across whatever epochs remain. Warmup applies under a constant rate too,
    where it is a ramp followed by a plateau.

    Pure arithmetic, kept here so the three schedule fields have exactly one
    definition and it can be tested without torch. `train.py` calls this each
    epoch rather than building a torch scheduler, which keeps the resolved
    config the only description of a run.
    """
    lr, schedule = cfg["lr"], cfg["lr_schedule"]
    epochs, warmup = cfg["epochs"], cfg["lr_warmup_epochs"]
    epoch = min(max(epoch, 0), epochs - 1)

    if epoch < warmup:
        return lr * (epoch + 1) / warmup

    span = epochs - 1 - warmup
    if schedule == "constant" or span < 1:
        return lr

    final = lr * cfg["lr_final_frac"]
    progress = (epoch - warmup) / span
    if schedule == "cosine":
        return final + (lr - final) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr * cfg["lr_final_frac"] ** progress


def loss_weight_at(cfg, name, epoch):
    """`beta` or `gamma` for a 0-indexed epoch, ramping linearly to its value.

    The same shape as `lr_at`'s warmup and for the same reason — one definition
    of what the field means, testable without torch. A weight with no warmup is
    its configured value from the first epoch.
    """
    if name not in LOSS_WEIGHTS:
        raise ValueError(f"unknown loss weight {name!r}; expected one of {list(LOSS_WEIGHTS)}")

    target, warmup = cfg[name], cfg[f"{name}_warmup_epochs"]
    epoch = min(max(epoch, 0), cfg["epochs"] - 1)
    if epoch >= warmup:
        return target
    return target * (epoch + 1) / warmup


def load_sweep(path):
    """Resolved configs from a sweep YAML file."""
    spec = yaml.safe_load(Path(path).read_text()) or {}
    return expand_sweep(spec)


def apply_cli_overrides(cfg, pairs):
    """Apply repeated ``--set key=value`` strings, coerced to the default's type.

    The third and last layer of resolution: defaults here, then the sweep YAML,
    then whatever was passed explicitly.
    """
    for pair in pairs or ():
        key, _, raw = pair.partition("=")
        if not _ or key not in cfg:
            raise ValueError(f"cannot set {pair!r}: expected key=value with a known key")
        cfg[key] = _coerce(raw, cfg[key])
    _finalize(cfg)
    return cfg


def _coerce(raw, template):
    """Read a string as the type of the value it is replacing."""
    if isinstance(template, bool):
        if raw.lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"expected a boolean, got {raw!r}")
        return raw.lower() in ("true", "1")
    if isinstance(template, int):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    return raw


def dump_config(cfg, path):
    """Write a resolved config beside the run it produced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(sorted(cfg.items())), sort_keys=False))
