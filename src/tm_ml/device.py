"""Pick a device on a shared multi-card box.

``gpu: auto`` takes the least contended card with room for a run; ``gpu: <i>``
takes that card by index. Either way the choice becomes a concrete
``cuda:<i>`` rather than ``CUDA_VISIBLE_DEVICES``, which has no effect once
torch has initialized CUDA in-process.

Three things this has to get right, each of them learned here the hard way.

Probing must not need a context. ``torch.cuda.mem_get_info`` initializes one on
the card it is reading, so a card full enough to be worth avoiding is also a
card that raises when asked about it — the query whose whole job is routing
around a full card gets taken down by one. nvidia-smi answers from outside the
process and never has that problem.

The card has to become the process's current device, not just where the tensors
were put. Anything in torch that takes no device argument reads the current
device, which is 0 until something sets it: Adam's capture health check calls
``torch.accelerator.current_stream()`` that way on every step. Placing every
tensor by hand on cuda:1 while the current device is still a full cuda:0 earns
`CUDA error: out of memory` at the first optimizer step, naming a card the run
never asked for. A cpu run needs this too, which reads wrong until you hit it:
``current_accelerator()`` reports cuda whenever cuda is available, wherever the
tensors actually live.

Concurrent runs have to be told apart. Two runs started together by
``snakemake --resources gpu=2`` read the cards at the same millisecond, seconds
before either has allocated, and choose the same one. Staggering them does not
help: the ranking is a stateless argmax and a single run is far too small to
move it — ~0.6 GiB of context against cards that routinely differ by 8 GiB, so
the same card keeps winning for another dozen runs. A running process therefore
stakes a claim file and ``auto`` skips cards a live process holds. The claim
makes a card *ineligible* rather than merely poorer, which is the part that
matters. Claims are advisory and self-healing — one whose PID is gone is stale
and gets taken over, so a killed job never wedges a card — and they bind only
our own runs, on a box shared with people who do not use them.
"""

import atexit
import errno
import os
import subprocess
import tempfile
from pathlib import Path

import torch

QUERY = ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"]

CLAIM_DIR = Path(tempfile.gettempdir()) / "tm_ml_gpu_claims"

# Under this a card cannot host a run and `auto` will not consider it. A context
# alone costs ~600 MiB before a single weight is allocated; the rest is margin
# for the run, small here but not nothing. Without a floor the claim mechanism
# happily hands out a card that is full, because spreading runs across cards and
# checking a card can hold one are different questions.
MIN_FREE_MIB = 2048


def gpu_status():
    """``[(index, free_mib, utilization_pct)]``, empty if nvidia-smi cannot answer."""
    try:
        out = subprocess.run(QUERY, capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in out.stdout.strip().splitlines():
        index, free, util = (field.strip() for field in line.split(",", 2))
        cards.append((int(index), int(free), int(util)))
    return cards


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError as error:
        return error.errno == errno.EPERM  # someone else's process, but running
    return True


def _release(path):
    try:
        path.unlink()
    except OSError:
        pass


def claim_gpu(index):
    """Stake a claim on a card. False if a live process already holds it.

    ``O_CREAT | O_EXCL`` makes create-or-fail atomic, so two processes racing
    for one card cannot both win — that is what closes the seconds-wide window
    between choosing a card and allocating on it. A claim left by a crash is
    recognized as stale through its PID and taken over.
    """
    try:
        CLAIM_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return True  # cannot arbitrate; the ranking alone will have to do

    path = CLAIM_DIR / f"gpu{index}"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                holder = int(path.read_text().strip() or -1)
            except (OSError, ValueError):
                holder = -1
            if holder not in (-1, os.getpid()) and _pid_alive(holder):
                return False
            _release(path)  # stale: the holder is gone, retry the create
            continue
        except OSError:
            return True
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()))
        atexit.register(_release, path)
        return True
    return False


def _viable():
    """Cards with room for a run, least contended first.

    `None` rather than an empty list when nvidia-smi could not be reached at
    all, which is a different situation from every card being full.

    Ranked on utilization ahead of free memory, which is the reverse of what
    this module used to do and the reason it once sent a sweep to a card sitting
    at 74% under a neighbour's job while reporting it as the emptiest one. Free
    memory is a gate, answered by the floor; among cards that clear it the
    question left is who else is already there.
    """
    cards = gpu_status()
    if not cards:
        return None
    return sorted((card for card in cards if card[1] >= MIN_FREE_MIB),
                  key=lambda card: (card[2], -card[1]))


def _index_from(spec):
    """An explicit `gpu` value as a card index: ``1`` or ``"1"``."""
    try:
        return int(spec)
    except (TypeError, ValueError):
        hint = ""
        if isinstance(spec, str) and spec.startswith("cuda:"):
            hint = f"; cards are named by index now, so write it as {spec[5:]}"
        raise RuntimeError(
            f"gpu={spec!r} is not 'auto', 'cpu' or a card index{hint}"
        ) from None


def resolve(spec="auto"):
    """``auto`` | ``cpu`` | a card index, to a concrete `torch.device`."""
    if not torch.cuda.is_available():
        if spec not in ("auto", "cpu"):
            raise RuntimeError(f"gpu={spec!r} was requested but CUDA is not available")
        return torch.device("cpu")

    if spec == "cpu":
        # Pointed at a card but deliberately not claiming one: a cpu run is not
        # occupying it, it only needs the current device to be somewhere that
        # can take the context torch will ask for regardless.
        viable = _viable()
        if viable:
            torch.cuda.set_device(viable[0][0])
        return torch.device("cpu")

    if spec == "auto":
        viable = _viable()
        if viable is None:
            index = torch.cuda.current_device()  # no nvidia-smi; take the default
        elif not viable:
            raise RuntimeError(
                f"gpu='auto' but no visible card has {MIN_FREE_MIB} MiB free"
            )
        else:
            # The first card we can claim, or the least contended if every one
            # is already claimed. The claim spreads our own runs out; it is not
            # a scheduler, and refusing to run would be worse than sharing.
            index = next((c[0] for c in viable if claim_gpu(c[0])), viable[0][0])
    else:
        index = _index_from(spec)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"gpu={spec!r} but only {torch.cuda.device_count()} cards are visible"
            )
        # Claimed so a concurrent `auto` steers around it, but the answer is
        # ignored: pinning a card is a decision, not a request.
        claim_gpu(index)

    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def describe(device):
    """One line for the run log and `train_meta.json`."""
    if device.type != "cuda":
        return "cpu"
    free, total = torch.cuda.mem_get_info(device.index)
    return (
        f"{device} {torch.cuda.get_device_name(device.index)} "
        f"({free / 2**30:.1f}/{total / 2**30:.1f} GiB free)"
    )
