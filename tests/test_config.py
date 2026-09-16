"""Sweep expansion, validation, and run naming.

These run without torch and without data, which is the point of both modules:
the Snakefile calls them to build a DAG before anything heavy is imported.
"""

import json

import pytest

from tm_ml import config as C
from tm_ml.config import (
    apply_cli_overrides, defaults_for, expand_sweep, load_sweep, loss_weight_at, lr_at,
)
from tm_ml.paths import (
    ABBREV, NAME_BASELINE, config_guard, determining_fields, run_name, run_path,
)


def test_empty_sweep_is_one_default_run():
    configs = expand_sweep({})
    assert len(configs) == 1
    assert configs[0] == defaults_for("tmvae")


def test_lists_are_axes_and_multiply():
    configs = expand_sweep({"d_global": [0, 4], "pooling": ["deepsets", "attention"]})
    assert len(configs) == 4
    assert {(c["d_global"], c["pooling"]) for c in configs} == {
        (0, "deepsets"), (0, "attention"), (4, "deepsets"), (4, "attention")
    }


def test_scalars_stay_fixed_across_the_sweep():
    configs = expand_sweep({"d_latent": [8, 16], "beta": 4.0})
    assert [c["beta"] for c in configs] == [4.0, 4.0]


def test_sections_are_merged():
    configs = expand_sweep({"model_params": {"d_latent": 16}, "training": {"epochs": 5}})
    assert configs[0]["d_latent"] == 16
    assert configs[0]["epochs"] == 5


def test_unknown_field_is_named():
    with pytest.raises(ValueError, match="does not accept: hidden_dim"):
        expand_sweep({"hidden_dim": 64})


def test_unknown_model_is_named():
    with pytest.raises(ValueError, match="unknown model 'unet'"):
        expand_sweep({"model": "unet"})


def test_empty_axis_is_refused():
    with pytest.raises(ValueError, match="empty list"):
        expand_sweep({"d_latent": []})


def test_sweeping_a_runtime_field_is_refused():
    with pytest.raises(ValueError, match="cannot sweep gpu"):
        expand_sweep({"gpu": ["cuda:0", "cuda:1"]})


@pytest.mark.parametrize("spec,match", [
    ({"d_model": 64, "n_heads": 7}, "not divisible"),
    ({"pooling": "softmax"}, "unknown pooling"),
    ({"val_frac": 0.6, "test_frac": 0.5}, "leave a training split"),
    ({"beta_warmup_epochs": 50, "epochs": 10}, "beta_warmup_epochs must be between"),
    ({"epochs": 0}, "at least 1"),
])
def test_validation_fails_at_expansion(spec, match):
    with pytest.raises(ValueError, match=match):
        expand_sweep(spec)


def test_cli_overrides_coerce_to_the_default_type():
    cfg = apply_cli_overrides(defaults_for("tmvae"), ["d_latent=16", "beta=2.5"])
    assert cfg["d_latent"] == 16 and isinstance(cfg["d_latent"], int)
    assert cfg["beta"] == 2.5 and isinstance(cfg["beta"], float)


def test_cli_override_of_an_unknown_key_is_refused():
    with pytest.raises(ValueError, match="cannot set"):
        apply_cli_overrides(defaults_for("tmvae"), ["hidden_dim=64"])


def test_cli_overrides_are_validated():
    with pytest.raises(ValueError, match="unknown pooling"):
        apply_cli_overrides(defaults_for("tmvae"), ["pooling=softmax"])


def test_load_sweep_reads_a_file(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text("model: tmvae\nd_global: [0, 4]\n")
    assert len(load_sweep(path)) == 2


# --- naming ---------------------------------------------------------------


def test_baseline_run_name_is_bare():
    cfg = defaults_for("tmvae") | NAME_BASELINE["tmvae"]
    assert run_name(cfg) == "TMVAE_Tom1000"


def test_only_off_baseline_fields_appear():
    cfg = defaults_for("tmvae") | NAME_BASELINE["tmvae"] | {
        "d_global": 4, "pooling": "attention",
    }
    assert run_name(cfg) == "TMVAE_dg4-pattention_Tom1000"


def test_runtime_fields_never_reach_the_name():
    cfg = defaults_for("tmvae") | {"gpu": "cuda:3", "num_threads": 8}
    assert run_name(cfg) == "TMVAE_Tom1000"


def test_every_point_of_a_sweep_gets_its_own_name():
    configs = expand_sweep({
        "d_latent": [8, 16], "beta": [1.0, 4.0], "pooling": ["deepsets", "attention"],
    })
    names = {run_name(c) for c in configs}
    assert len(names) == len(configs) == 8


def test_float_formatting_is_filename_safe():
    cfg = defaults_for("tmvae") | {"lr": 3e-4, "dropout": 0.5, "val_frac": 0.2}
    name = run_name(cfg)
    assert "lr3e-4" in name and "do0p5" in name and "v0p2" in name
    assert "/" not in name and " " not in name


def test_every_determining_field_has_an_abbreviation():
    """The guarantee: a new determining field cannot be silently left unnamed."""
    fields = set(determining_fields(defaults_for("tmvae"))) - {"model", "drop"}
    assert fields <= set(ABBREV), sorted(fields - set(ABBREV))


def test_an_unregistered_determining_field_is_refused():
    cfg = defaults_for("tmvae") | {"brand_new_knob": 3}
    with pytest.raises(ValueError, match="no run-name abbreviation for brand_new_knob"):
        run_name(cfg)


# --- config_guard ---------------------------------------------------------


def test_config_guard_writes_then_accepts_the_same_config(tmp_path):
    cfg = defaults_for("tmvae")
    config_guard(tmp_path / "run", cfg)
    config_guard(tmp_path / "run", cfg)

    stored = json.loads((tmp_path / "run" / "config.json").read_text())
    assert stored["d_latent"] == 8
    assert "gpu" not in stored


def test_config_guard_ignores_runtime_changes(tmp_path):
    config_guard(tmp_path / "run", defaults_for("tmvae"))
    config_guard(tmp_path / "run", defaults_for("tmvae") | {"gpu": "cuda:1"})


def test_config_guard_refuses_a_changed_determining_field(tmp_path):
    config_guard(tmp_path / "run", defaults_for("tmvae"))
    with pytest.raises(SystemExit, match="d_latent: stored 8 vs requested 16"):
        config_guard(tmp_path / "run", defaults_for("tmvae") | {"d_latent": 16})


# --- the frozen baseline --------------------------------------------------


def test_baseline_covers_exactly_the_determining_fields():
    """Drift either way is a silent collision waiting to happen."""
    fields = set(determining_fields(defaults_for("tmvae"))) - {"model", "drop"}
    assert fields == set(NAME_BASELINE["tmvae"])


def test_an_unbaselined_determining_field_is_refused(monkeypatch):
    monkeypatch.setitem(ABBREV, "brand_new_knob", "bnk")
    cfg = defaults_for("tmvae") | {"brand_new_knob": 3}
    with pytest.raises(ValueError, match="no run-name baseline for brand_new_knob"):
        run_name(cfg)


def test_moving_a_default_does_not_rename_an_existing_run(monkeypatch):
    """Case 1: raising the default must not claim the old run's directory."""
    trained = defaults_for("tmvae") | NAME_BASELINE["tmvae"]
    assert run_name(trained) == "TMVAE_Tom1000"

    monkeypatch.setitem(C.COMMON_DEFAULTS, "epochs", 100)
    assert run_name(trained) == "TMVAE_Tom1000", "an existing run was renamed"
    assert run_name(defaults_for("tmvae")) == "TMVAE_e100_Tom1000"


def test_asking_for_the_old_default_reuses_the_old_directory(monkeypatch):
    """Case 2: the same config must not be retrained under a second name."""
    trained = defaults_for("tmvae") | NAME_BASELINE["tmvae"]
    monkeypatch.setitem(C.COMMON_DEFAULTS, "epochs", 100)

    again = defaults_for("tmvae") | {"epochs": 50}
    assert again == trained
    assert run_name(again) == run_name(trained) == "TMVAE_Tom1000"


# --- learning-rate decay --------------------------------------------------


def test_decay_is_off_by_default():
    cfg = defaults_for("tmvae")
    assert cfg["lr_schedule"] == "constant"
    assert [lr_at(cfg, e) for e in (0, 25, 49)] == [cfg["lr"]] * 3


@pytest.mark.parametrize("schedule", ["cosine", "exponential"])
def test_a_schedule_runs_from_lr_to_lr_times_final_frac(schedule):
    cfg = expand_sweep({"lr_schedule": schedule, "lr_final_frac": 0.05, "epochs": 20})[0]
    first, last = lr_at(cfg, 0), lr_at(cfg, 19)

    assert first == pytest.approx(cfg["lr"])
    assert last == pytest.approx(cfg["lr"] * 0.05)
    series = [lr_at(cfg, e) for e in range(20)]
    assert series == sorted(series, reverse=True), "lr must decay monotonically"


def test_lr_at_handles_a_single_epoch():
    cfg = expand_sweep({"lr_schedule": "cosine", "epochs": 1})[0]
    assert lr_at(cfg, 0) == cfg["lr"]


def test_final_frac_is_pinned_without_a_schedule():
    """Inert settings must not reach the name, or identical runs train twice."""
    cfg = expand_sweep({"lr_final_frac": 0.5})[0]
    assert cfg["lr_final_frac"] == C.INERT_LR_FINAL_FRAC
    assert run_name(cfg) == "TMVAE_Tom1000"


def test_schedule_and_final_frac_reach_the_name():
    cfg = expand_sweep({"lr_schedule": "cosine", "lr_final_frac": 0.05})[0]
    assert run_name(cfg) == "TMVAE_lff0p05-schcosine_Tom1000"


def test_sweeping_final_frac_without_a_schedule_is_refused():
    with pytest.raises(ValueError, match="cannot sweep lr_final_frac"):
        expand_sweep({"lr_final_frac": [0.01, 0.1]})

    with pytest.raises(ValueError, match="cannot sweep lr_final_frac"):
        expand_sweep({"lr_schedule": ["constant", "cosine"], "lr_final_frac": [0.01, 0.1]})


def test_sweeping_final_frac_with_a_schedule_is_allowed():
    configs = expand_sweep({"lr_schedule": "cosine", "lr_final_frac": [0.01, 0.1]})
    assert len({run_name(c) for c in configs}) == 2


@pytest.mark.parametrize("spec,match", [
    ({"lr_schedule": "linear"}, "unknown lr_schedule"),
    ({"lr_schedule": "cosine", "lr_final_frac": 0}, r"must be in \(0, 1\]"),
    ({"lr_schedule": "cosine", "lr_final_frac": 2.0}, r"must be in \(0, 1\]"),
    ({"lr_final_frac": -1.0}, r"must be in \(0, 1\]"),
])
def test_bad_schedule_settings_fail_at_expansion(spec, match):
    with pytest.raises(ValueError, match=match):
        expand_sweep(spec)


def test_warmup_is_off_by_default():
    assert defaults_for("tmvae")["lr_warmup_epochs"] == 0


def test_warmup_ramps_to_lr_without_a_dead_epoch():
    cfg = expand_sweep({"lr_warmup_epochs": 4, "epochs": 12})[0]
    ramp = [lr_at(cfg, e) for e in range(4)]

    assert ramp[0] > 0, "the first epoch must still learn"
    assert ramp == sorted(ramp)
    assert ramp[-1] == pytest.approx(cfg["lr"])
    assert ramp == [pytest.approx(cfg["lr"] * f) for f in (0.25, 0.5, 0.75, 1.0)]


def test_warmup_applies_under_a_constant_rate():
    """Warmup is orthogonal to the schedule, so it holds at lr afterwards."""
    cfg = expand_sweep({"lr_warmup_epochs": 3, "epochs": 10})[0]
    assert cfg["lr_schedule"] == "constant"
    assert [lr_at(cfg, e) for e in range(3, 10)] == [cfg["lr"]] * 7


@pytest.mark.parametrize("schedule", ["cosine", "exponential"])
def test_decay_starts_from_lr_after_the_ramp(schedule):
    cfg = expand_sweep({
        "lr_schedule": schedule, "lr_final_frac": 0.05,
        "lr_warmup_epochs": 4, "epochs": 20,
    })[0]

    assert lr_at(cfg, 3) == pytest.approx(cfg["lr"]), "ramp must end at lr"
    assert lr_at(cfg, 4) == pytest.approx(cfg["lr"]), "decay must begin at lr"
    assert lr_at(cfg, 19) == pytest.approx(cfg["lr"] * 0.05)

    tail = [lr_at(cfg, e) for e in range(4, 20)]
    assert tail == sorted(tail, reverse=True)


def test_warmup_reaches_the_name_even_with_a_constant_rate():
    cfg = expand_sweep({"lr_warmup_epochs": 5})[0]
    assert run_name(cfg) == "TMVAE_lrw5_Tom1000"


def test_warmup_longer_than_training_is_refused():
    with pytest.raises(ValueError, match="lr_warmup_epochs must be between 0 and epochs"):
        expand_sweep({"lr_warmup_epochs": 20, "epochs": 10})


def test_warmup_may_span_the_whole_run():
    cfg = expand_sweep({"lr_warmup_epochs": 10, "epochs": 10})[0]
    assert lr_at(cfg, 9) == pytest.approx(cfg["lr"])


# --- loss-weight warmup ---------------------------------------------------


@pytest.mark.parametrize("weight", ["beta", "gamma"])
def test_loss_weight_warmup_is_off_by_default(weight):
    cfg = defaults_for("tmvae")
    assert cfg[f"{weight}_warmup_epochs"] == 0
    assert [loss_weight_at(cfg, weight, e) for e in (0, 25, 49)] == [cfg[weight]] * 3


@pytest.mark.parametrize("weight", ["beta", "gamma"])
def test_loss_weight_ramps_then_holds(weight):
    cfg = expand_sweep({f"{weight}_warmup_epochs": 4, weight: 2.0, "epochs": 10})[0]
    series = [loss_weight_at(cfg, weight, e) for e in range(10)]

    assert series[:4] == [pytest.approx(2.0 * f) for f in (0.25, 0.5, 0.75, 1.0)]
    assert series[4:] == [pytest.approx(2.0)] * 6
    assert series == sorted(series)


def test_the_two_warmups_are_independent():
    cfg = expand_sweep({"beta_warmup_epochs": 8, "gamma_warmup_epochs": 2, "epochs": 10})[0]
    assert loss_weight_at(cfg, "gamma", 4) == pytest.approx(cfg["gamma"])
    assert loss_weight_at(cfg, "beta", 4) < cfg["beta"]


def test_unknown_loss_weight_is_refused():
    with pytest.raises(ValueError, match="unknown loss weight 'delta'"):
        loss_weight_at(defaults_for("tmvae"), "delta", 0)


@pytest.mark.parametrize("weight", ["beta", "gamma"])
def test_loss_weight_warmup_longer_than_training_is_refused(weight):
    with pytest.raises(ValueError, match=f"{weight}_warmup_epochs must be between"):
        expand_sweep({f"{weight}_warmup_epochs": 20, "epochs": 10})


def test_gamma_warmup_reaches_the_name():
    cfg = expand_sweep({"gamma_warmup_epochs": 10})[0]
    assert run_name(cfg) == "TMVAE_gw10_Tom1000"


# --- parent_path ----------------------------------------------------------


def test_parent_path_defaults_to_flat():
    cfg = defaults_for("tmvae")
    assert cfg["parent_path"] == ""
    assert run_path(cfg) == run_name(cfg)


def test_parent_path_files_the_run_under_a_folder():
    cfg = defaults_for("tmvae") | {"parent_path": "initial_testing"}
    assert run_path(cfg) == f"initial_testing/{run_name(cfg)}"


def test_parent_path_may_nest():
    cfg = defaults_for("tmvae") | {"parent_path": "initial_testing/initial_sweep/"}
    assert run_path(cfg) == f"initial_testing/initial_sweep/{run_name(cfg)}"


def test_parent_path_does_not_reach_the_run_name():
    """Filing is not science: the same config keeps one identity."""
    plain = defaults_for("tmvae")
    filed = plain | {"parent_path": "somewhere/else"}
    assert run_name(filed) == run_name(plain)


def test_parent_path_is_not_checked_by_config_guard(tmp_path):
    config_guard(tmp_path / "run", defaults_for("tmvae"))
    config_guard(tmp_path / "run", defaults_for("tmvae") | {"parent_path": "moved"})


def test_sweeping_parent_path_is_refused():
    with pytest.raises(ValueError, match="cannot sweep parent_path"):
        expand_sweep({"parent_path": ["a", "b"]})
