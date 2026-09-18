"""Card selection: the viability floor, the ranking, and the claim files.

None of this needs a GPU. `gpu_status` is the only part that talks to the
hardware, so everything above it is tested against a substituted reading, and
the claim files are plain filesystem work.
"""

import os

import pytest

from tm_ml import device as D


@pytest.fixture(autouse=True)
def claim_dir(tmp_path, monkeypatch):
    """Claims go somewhere private, so a test never disturbs a running sweep."""
    monkeypatch.setattr(D, "CLAIM_DIR", tmp_path / "claims")
    return tmp_path / "claims"


def cards(monkeypatch, reading):
    monkeypatch.setattr(D, "gpu_status", lambda: reading)


def test_a_card_under_the_floor_is_not_viable(monkeypatch):
    # cuda:0 as this box routinely has it: memory gone, nothing running.
    cards(monkeypatch, [(0, 37, 0), (1, 60000, 70), (2, 68000, 0)])
    assert [c[0] for c in D._viable()] == [2, 1]


def test_utilization_outranks_free_memory(monkeypatch):
    """The bug this ordering exists for: the emptiest card was also the busiest."""
    cards(monkeypatch, [(1, 20000, 0), (2, 68000, 74)])
    assert D._viable()[0][0] == 1


def test_free_memory_breaks_a_utilization_tie(monkeypatch):
    cards(monkeypatch, [(1, 20000, 0), (2, 68000, 0)])
    assert D._viable()[0][0] == 2


def test_every_card_full_is_distinct_from_no_nvidia_smi(monkeypatch):
    cards(monkeypatch, [(0, 10, 0), (1, 20, 0)])
    assert D._viable() == []
    cards(monkeypatch, [])
    assert D._viable() is None


def test_a_live_claim_excludes_a_card(claim_dir):
    assert D.claim_gpu(3) is True
    (claim_dir / "gpu3").write_text(str(_a_dead_pid()))
    assert D.claim_gpu(3) is True  # stale, taken over
    assert claim_dir.joinpath("gpu3").read_text() == str(os.getpid())


def test_a_claim_held_by_a_live_process_is_refused(claim_dir, monkeypatch):
    claim_dir.mkdir(parents=True)
    (claim_dir / "gpu2").write_text("1")  # pid 1 is always alive
    monkeypatch.setattr(D, "_pid_alive", lambda pid: pid == 1)
    assert D.claim_gpu(2) is False


def test_reclaiming_our_own_card_succeeds(claim_dir):
    assert D.claim_gpu(1) is True
    assert D.claim_gpu(1) is True


def _a_dead_pid():
    """A PID that has certainly exited."""
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    return pid
