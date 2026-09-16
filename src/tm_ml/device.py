"""Pick a device on a shared multi-card box.

``gpu: auto`` takes the card with the most free memory, which on this machine
matters — the three A100s are shared and their free memory routinely differs by
an order of magnitude. The choice resolves to an explicit ``cuda:<i>`` rather
than to ``CUDA_VISIBLE_DEVICES``, which has no effect once torch has
initialized CUDA in-process.

Not yet handled: two runs launched together by ``snakemake --resources gpu=2``
both read free memory before either has allocated anything, see the same idle
card and pick it, deterministically. The fix is an advisory claim file per
process, skipped when its PID is gone. Worth writing when the first multi-GPU
sweep is launched, and not before — a single run is unaffected.
"""

import torch


def resolve(spec="auto"):
    """``auto`` | ``cpu`` | ``cuda`` | ``cuda:<i>`` to a concrete `torch.device`."""
    if spec == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if spec not in ("auto", "cpu"):
            raise RuntimeError(f"gpu={spec!r} was requested but CUDA is not available")
        return torch.device("cpu")

    if spec == "auto":
        return torch.device("cuda", _most_free())

    device = torch.device(spec)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"gpu={spec!r} but only {torch.cuda.device_count()} cards are visible"
        )
    return device


def _most_free():
    """Index of the card with the most free memory."""
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return max(range(len(free)), key=free.__getitem__)


def describe(device):
    """One line for the run log and `train_meta.json`."""
    if device.type != "cuda":
        return "cpu"
    free, total = torch.cuda.mem_get_info(device.index)
    return (
        f"{device} {torch.cuda.get_device_name(device.index)} "
        f"({free / 2**30:.1f}/{total / 2**30:.1f} GiB free)"
    )
