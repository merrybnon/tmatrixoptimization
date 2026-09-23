"""The figures render, and each one is stamped with where it came from."""

import numpy as np
import pytest
import torch

from tm_ml import evaluate, paths, train, visualize
from tm_ml.models import TMVAEConfig

FIGURES = tuple(f"figures/{name}.png" for name in (
    "training_curve", "predictions", "reconstruction", "latent", "latent_property"
)) + tuple(f"figures/interpolation_and_traversals/traversal_{axis}.png"
           for axis in ("gbd", "PC1", "PC2")) + (
    "figures/interpolation_and_traversals/interpolation.png",) + tuple(
    f"figures/interpolation_and_traversals/optimization_g{graph}{variant}.png"
    for graph in (0, 1, 2) for variant in ("", "_unreg"))


@pytest.mark.parametrize("d_global, d_latent", [(0, 8), (1, 8), (0, 1)])
def test_visualize_writes_every_figure(wired, tiny_config, d_global, d_latent):
    # Without a global the gbd traversal is a placeholder; with one it is drawn.
    # d_latent = 1 leaves the graph-mean views a single axis to plot.
    tiny_config = tiny_config | {"d_global": d_global, "d_latent": d_latent}
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


def test_align_nodes_undoes_a_relabelling():
    rng = np.random.default_rng(0)
    mu_a = rng.normal(size=(20, 8))
    shuffle = rng.permutation(20)
    mu_b = mu_a[shuffle]

    match = visualize.align_nodes(mu_a, mu_b)

    assert np.array_equal(shuffle[match], np.arange(20))
    assert np.allclose(mu_b[match], mu_a)


def test_matched_endpoint_is_the_permuted_reconstruction(wired, tiny_config):
    # Drawing the whole row in A's node order is only right if decoding B's
    # relabelled latents gives B's reconstruction relabelled the same way.
    tiny_config = tiny_config | {"d_global": 1}
    tiny_config["run"] = paths.run_path(tiny_config)
    train.train(tiny_config)
    _, checkpoint, cfg, model, splits, _, device = evaluate.load_run(tiny_config["run"])
    data = evaluate.collect(model, splits.test, TMVAEConfig(**checkpoint["model_config"]),
                            device, cfg["batch_size"])
    mu, mu_global = data["mu"].numpy(), data["mu_global"].numpy()

    match = visualize.align_nodes(mu[0], mu[1])
    with torch.no_grad():
        T_hat, _ = model.decode(torch.as_tensor(mu[1][match][None]),
                                torch.as_tensor(mu_global[1][None]))

    expected = data["T_hat"][1].numpy()[np.ix_(match, match)]
    assert np.allclose(T_hat[0].numpy(), expected, atol=1e-6)
