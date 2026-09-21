"""Train one model and write its run directory.

    pixi run -e ml train                                  # one run at defaults
    pixi run -e ml train --set epochs=5 --set gpu=cpu     # a quick smoke run
    pixi run -e ml train --config config/sweeps/latent.yaml --run TMVAE_dg4_Tom1000

Settings resolve in three layers, later winning: the defaults in `config.py`,
then a sweep YAML, then any explicit ``--set``. The workflow uses the third
form, handing over a sweep file and the name of the point to train, which is how
the Snakefile can declare this run's outputs before torch is imported.

Everything lands in one directory, ``results/<run_name>/``: the best checkpoint,
a per-epoch history, the resolved config both as YAML for reading and as JSON
for `paths.config_guard`, and a provenance record.
"""

import argparse
import csv
import dataclasses
import json
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from tm_ml import device as device_module
from tm_ml import paths
from tm_ml.config import (
    apply_cli_overrides,
    defaults_for,
    dump_config,
    load_sweep,
    loss_weight_at,
    lr_at,
)
from tm_ml.datasets import load_splits
from tm_ml.ingest import git_commit
from tm_ml.models import TMVAE, TMVAEConfig, tmvae_loss

# The columns of history.csv, in order. Named once so the writer and the
# per-epoch record cannot disagree.
LOSS_TERMS = ("total", "recon", "kl", "kl_node", "kl_global", "prop", "log_recon")
HISTORY_FIELDS = (
    ("epoch", "lr", "beta", "gamma", "lambda_log", "lambda_recon")
    + tuple(f"train_{t}" for t in LOSS_TERMS)
    + tuple(f"val_{t}" for t in LOSS_TERMS)
    + ("train_score", "val_score", "seconds")
)


def resolve_config(args):
    """Defaults, then the sweep YAML's named point, then explicit overrides."""
    if args.config is None:
        cfg = defaults_for(args.model)
    else:
        candidates = {paths.run_path(c): c for c in load_sweep(args.config)}
        if args.run is None:
            raise SystemExit("--config needs --run naming which point of the sweep to train")
        if args.run not in candidates:
            listing = "\n  ".join(sorted(candidates))
            raise SystemExit(f"{args.config} has no run {args.run!r}. It expands to:\n  {listing}")
        cfg = candidates[args.run]

    cfg = apply_cli_overrides(cfg, args.set)
    cfg["config"] = str(args.config) if args.config else None
    cfg["run"] = paths.run_path(cfg)
    return cfg


def model_config(cfg, n_nodes):
    """The model's own dataclass, filled from the fields it recognizes.

    ``n_nodes`` comes from the data rather than the config, so it cannot drift
    from what was ingested.
    """
    known = {f.name for f in dataclasses.fields(TMVAEConfig)} - {"n_nodes"}
    return TMVAEConfig(n_nodes=n_nodes, **{k: v for k, v in cfg.items() if k in known})


def run_epoch(model, loader, model_cfg, device, optimizer=None, grad_clip=0.0):
    """One pass. With an optimizer it trains; without one it evaluates.

    Evaluation runs with ``sample=False``, so val numbers are a deterministic
    function of the weights. Sampling would add variance to the quantity used
    for model selection and early stopping, for no benefit — the KL term is
    computed from mu and logvar either way.
    """
    training = optimizer is not None
    model.train(training)

    totals = dict.fromkeys(LOSS_TERMS, 0.0)
    seen = 0

    with torch.set_grad_enabled(training):
        for T, y in loader:
            T, y = T.to(device), y.to(device)
            output = model(T, sample=training)
            loss = tmvae_loss(output, T, y, model_cfg)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Every loss term is already a mean over the batch, so weight by
            # batch size to get a mean over the split rather than a mean of
            # means — the last batch is usually short.
            for term in LOSS_TERMS:
                totals[term] += getattr(loss, term).detach().item() * len(T)
            seen += len(T)

    return {term: value / seen for term, value in totals.items()}


def selection_score(losses, cfg):
    """The objective at the *configured* weights, not the epoch's own.

    Under warmup every weight in `LOSS_WEIGHTS` changes every epoch, so ranking
    epochs on the loss they were trained against compares different objectives —
    an early epoch scores well largely because beta was still small, and the
    checkpoint that wins does so by accident. Recomputing from terms already measured
    costs nothing and makes the epochs commensurable.

    `total` stays in the history as the quantity actually optimized that epoch;
    this is the one that selects and stops.
    """
    return (
        cfg["lambda_recon"] * losses["recon"]
        + cfg["beta"] * losses["kl"]
        + cfg["gamma"] * losses["prop"]
        + cfg["lambda_log"] * losses["log_recon"]
    )


def assert_first_step_is_sane(model, loader, model_cfg, device):
    """Check the constraints hold and the loss is finite before training starts.

    `architecture.md` asks for this explicitly: the failure mode of the zero
    diagonal is silent nan propagation, and a row that sums to less than one is
    a broken constraint rather than a bad number. Cheap once, invaluable when
    something is wrong.
    """
    T, y = next(iter(loader))
    T, y = T.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
        output = model(T, sample=False)
        loss = tmvae_loss(output, T, y, model_cfg)

    diagonal = torch.diagonal(output.T_hat, dim1=-2, dim2=-1)
    if not (diagonal == 0).all():
        raise RuntimeError(f"T_hat has a nonzero diagonal, max {diagonal.abs().max():.3e}")

    row_error = (output.T_hat.sum(-1) - 1).abs().max()
    if row_error > 1e-5:
        raise RuntimeError(f"T_hat rows do not sum to 1, worst off by {row_error:.3e}")

    for term in LOSS_TERMS:
        value = getattr(loss, term)
        if not torch.isfinite(value):
            raise RuntimeError(f"{term} is {value} before the first step")


def train(cfg):
    """Train one run to completion and return its provenance record."""
    run_dir = paths.resolve(cfg["run"])
    paths.config_guard(run_dir, cfg)

    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["num_threads"])
    device = device_module.resolve(cfg["gpu"])

    splits = load_splits(cfg)
    loaders = {
        "train": DataLoader(
            splits.train, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], drop_last=False,
        ),
        "val": DataLoader(splits.val, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]),
    }

    model_cfg = model_config(cfg, splits.n_nodes)
    model = TMVAE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['run']}\n  {device_module.describe(device)}")
    print(f"  {n_params:,} parameters against {splits.sizes['train']} training examples")
    print(f"  splits {splits.sizes}, target scaler {splits.scaler.as_dict()}")

    assert_first_step_is_sane(model, loaders["train"], model_cfg, device)

    history = []
    best = {"score": float("inf"), "epoch": -1}
    since_best = 0
    stopped_early = False
    started = time.time()

    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()

        # The three schedules, all pure functions of the epoch. Beta and gamma
        # reach the loss through a per-epoch copy of the model config, so the
        # loss function stays a function of a config and nothing carries
        # mutable training state.
        lr = lr_at(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        epoch_cfg = dataclasses.replace(
            model_cfg,
            beta=loss_weight_at(cfg, "beta", epoch),
            gamma=loss_weight_at(cfg, "gamma", epoch),
            lambda_log=loss_weight_at(cfg, "lambda_log", epoch),
            lambda_recon=loss_weight_at(cfg, "lambda_recon", epoch),
        )

        train_losses = run_epoch(
            model, loaders["train"], epoch_cfg, device, optimizer, cfg["grad_clip"]
        )
        val_losses = run_epoch(model, loaders["val"], epoch_cfg, device)

        scores = {
            "train_score": selection_score(train_losses, cfg),
            "val_score": selection_score(val_losses, cfg),
        }
        row = {
            "epoch": epoch,
            "lr": lr,
            "beta": epoch_cfg.beta,
            "gamma": epoch_cfg.gamma,
            "lambda_log": epoch_cfg.lambda_log,
            "lambda_recon": epoch_cfg.lambda_recon,
            "seconds": round(time.time() - epoch_start, 3),
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_losses.items()},
            **scores,
        }
        history.append(row)

        # Selection is on the configured-weight score, so warmup cannot make an
        # early epoch look good by charging it less for its KL.
        if scores["val_score"] < best["score"]:
            best = {"score": scores["val_score"], "epoch": epoch}
            since_best = 0
            torch.save(
                {
                    "model": "tmvae",
                    "model_config": dataclasses.asdict(model_cfg),
                    "state_dict": model.state_dict(),
                    "config": cfg,
                    "target_scaler": splits.scaler.as_dict(),
                    "epoch": epoch,
                    "score": best["score"],
                    "metric": "val_score",
                },
                run_dir / paths.CHECKPOINT,
            )
        else:
            since_best += 1

        if epoch % cfg["log_every"] == 0 or epoch == cfg["epochs"] - 1:
            print(
                f"  epoch {epoch:4d}  lr {lr:.2e}  "
                f"train {train_losses['total']:9.3f}  val {val_losses['total']:9.3f}  "
                f"score {scores['val_score']:9.3f}  "
                f"(recon {val_losses['recon']:8.3f}  kl {val_losses['kl']:7.3f}  "
                f"prop {val_losses['prop']:7.4f}  log_recon {val_losses['log_recon']:6.3f})"
                + ("  *" if best["epoch"] == epoch else "")
            )

        if cfg["patience"] and since_best >= cfg["patience"]:
            stopped_early = True
            print(f"  stopping early: {since_best} epochs without improvement")
            break

    write_history(run_dir / paths.HISTORY, history)
    dump_config(cfg, run_dir / paths.RESOLVED_CONFIG)

    meta = {
        "run": cfg["run"],
        "device": device_module.describe(device),
        "n_parameters": n_params,
        "splits": splits.sizes,
        "target_scaler": splits.scaler.as_dict(),
        "best_epoch": best["epoch"],
        "best_val_score": best["score"],
        "selection_metric": "val_score, at the configured beta and gamma",
        "epochs_run": len(history),
        "stopped_early": stopped_early,
        "seconds": round(time.time() - started, 1),
        "torch": torch.__version__,
        "git_commit": git_commit(),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / paths.TRAIN_META).write_text(json.dumps(meta, indent=2) + "\n")

    print(f"  best val_score {best['score']:.4f} at epoch {best['epoch']}, {meta['seconds']}s")
    return meta


def write_history(path, history):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="sweep YAML; needs --run")
    parser.add_argument("--run", help="which point of the sweep to train, by run name")
    parser.add_argument("--model", default="tmvae", help="model kind when no --config is given")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE",
        help="override one field; repeatable",
    )
    args = parser.parse_args()

    train(resolve_config(args))


if __name__ == "__main__":
    main()
