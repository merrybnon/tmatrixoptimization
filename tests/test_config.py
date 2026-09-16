"""Sweep expansion, validation, and run naming.

These run without torch and without data, which is the point of both modules:
the Snakefile calls them to build a DAG before anything heavy is imported.
"""

import json

import pytest

from tm_ml.config import apply_cli_overrides, defaults_for, expand_sweep, load_sweep
from tm_ml.paths import ABBREV, config_guard, determining_fields, run_name


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
    ({"beta_warmup_epochs": 50, "epochs": 10}, "exceeds epochs"),
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


def test_default_run_name_is_bare():
    assert run_name(defaults_for("tmvae")) == "TMVAE_Tom1000"


def test_only_non_default_fields_appear():
    cfg = defaults_for("tmvae") | {"d_global": 4, "pooling": "attention"}
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
