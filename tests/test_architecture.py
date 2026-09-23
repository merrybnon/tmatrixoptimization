"""The diagram follows the checkpoint, and its parameter counts add up."""

import pytest
import torch

from tm_ml import architecture, paths, train


@pytest.mark.parametrize("d_global", [0, 1])
def test_architecture_writes_source_from_checkpoint(wired, tiny_config, d_global):
    tiny_config = tiny_config | {"d_global": d_global}
    tiny_config["run"] = paths.run_path(tiny_config)
    train.train(tiny_config)
    written = architecture.architecture(tiny_config["run"], render_svg=False)
    run_dir = paths.resolve(tiny_config["run"])

    assert [p.relative_to(run_dir).as_posix() for p in written] == ["figures/architecture.mmd"]
    source = written[0].read_text()
    assert "flowchart TB" in source
    # Blocks link as whole subgraphs; a node-level edge leaving the row would
    # make Mermaid drop its left-to-right direction.
    assert "T --> ENC --> LAT --> DEC --> TH" in source
    assert "ROW --> PRED --> Y" in source
    assert f"{tiny_config['encoder_layers']} × attention layer" in source
    # The graph-level latent is drawn only when the run has one.
    assert ("z_graph" in source) == bool(d_global)

    state = torch.load(run_dir / paths.CHECKPOINT, map_location="cpu",
                       weights_only=False)["state_dict"]
    total = sum(t.numel() for t in state.values())
    assert f"{total:,} parameters" in source
