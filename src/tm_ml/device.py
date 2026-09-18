"""Pick a device on a shared multi-card box.

``gpu: auto`` takes the card with the most free memory, which on this machine
matters — the three A100s are shared and their free memory routinely differs by
an order of magnitude. The choice resolves to an explicit ``cuda:<i>`` rather
than to ``CUDA_VISIBLE_DEVICES``, which has no effect once torch has
initialized CUDA in-process. Resolving also claims the card as the process's
current device rather than only placing tensors on it — see ``_claim``.

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
        return _claim(torch.device("cpu"))

    if not torch.cuda.is_available():
        if spec not in ("auto", "cpu"):
            raise RuntimeError(f"gpu={spec!r} was requested but CUDA is not available")
        return torch.device("cpu")

    if spec == "auto":
        return _claim(torch.device("cuda", _most_free()))

    device = torch.device(spec)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda" and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"gpu={spec!r} but only {torch.cuda.device_count()} cards are visible"
        )
    return _claim(device)


def _claim(device):
    """Point the process's current device at a usable card, not just the tensors.

    Placing modules and tensors explicitly is not enough. Anything in torch that
    takes no device argument reads the current device, which is 0 whatever this
    module resolved: Adam's capture health check calls
    ``torch.accelerator.current_stream()`` that way on every step. When cuda:0
    is full — routine here, a neighbour's job can hold all 80 GB at 0% util —
    that call tries to initialize a context on a card this run never asked for
    and raises `CUDA error: out of memory` at the first optimizer step, long
    after the log reported a different card with plenty free.

    A cpu run needs the same treatment, which is the part that reads wrong until
    you hit it: ``current_accelerator()`` reports cuda whenever cuda is
    available, wherever the tensors actually are, so that health check queries a
    card on behalf of a run that owns nothing on one. If no card can be claimed
    the current device is left alone rather than raising — a cpu run should not
    be refused for the state of hardware it is not using.
    """
    if not torch.cuda.is_available():
        return device
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
        return device
    try:
        torch.cuda.set_device(_most_free())
    except RuntimeError:
        pass
    return device


def _most_free():
    """Index of the card with the most free memory.

    A card with no memory left does not report zero free — querying it needs a
    context and there is no room to make one, so it raises. Routing around a
    card in that state is the whole point of `auto`, so an unqueryable card
    counts as nothing free rather than taking the sweep down with it.
    """
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            free.append(torch.cuda.mem_get_info(i)[0])
        except RuntimeError:
            free.append(0)
    if not any(free):
        raise RuntimeError(
            f"gpu='auto' but none of the {len(free)} visible cards has memory free"
        )
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
