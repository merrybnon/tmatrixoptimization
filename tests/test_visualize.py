"""The figures render, and each one is stamped with where it came from."""

import pytest

from tm_ml import evaluate, paths, train, visualize

FIGURES = tuple(f"figures/{name}.png" for name in (
    "training_curve", "predictions", "reconstruction", "latent", "latent_property"
)) + tuple(f"figures/interpolation_and_traversals/traversal_{axis}.png"
           for axis in ("gbd", "PC1", "PC2"))


@pytest.mark.parametrize("d_global", [0, 1])
def test_visualize_writes_every_figure(wired, tiny_config, d_global):
    # Without a global the gbd traversal is a placeholder; with one it is drawn.
    tiny_config = tiny_config | {"d_global": d_global}
    tiny_config["run"] = paths.run_path(tiny_config)
    train.train(tiny_config)
    evaluate.evaluate(tiny_config["run"])
    written = visualize.visualize(tiny_config["run"])
    run_dir = paths.resolve(tiny_config["run"])

    assert [p.relative_to(run_dir).as_posix() for p in written] == list(FIGURES)
    for path in written:
        assert path.exists()
        # A PNG that renders to nothing still writes a file; the header and a
        # plausible size are what say something was actually drawn.
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size > 10_000


def test_visualize_needs_metrics_first(wired, tiny_config):
    train.train(tiny_config)
    with pytest.raises(SystemExit, match="run evaluate first"):
        visualize.visualize(tiny_config["run"])


def test_history_is_read_back_as_numbers(wired, tiny_config):
    train.train(tiny_config)
    history = visualize.read_history(paths.resolve(tiny_config["run"]))

    assert len(history["epoch"]) == tiny_config["epochs"]
    assert history["val_score"].dtype.kind == "f"
