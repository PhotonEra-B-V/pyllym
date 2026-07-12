"""Timestamped runs, completion markers, and skip-if-unchanged for TDG."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from pyllym.tdg import build, read_latest
from pyllym.tdg.runs import DONE_MARKER, LATEST_POINTER, spec_hash

SPEC = """
[meta]
feature_name = "Backoff"
mode = "plan"

[api]
module = "os"
signatures = ["def backoff(attempt: int) -> float"]

[[rules]]
id = "rule_1"
text = "grows with attempt"

[[cases]]
test_name = "test_backoff_grows"
scenario = "grows"
when = "None"
given = ["assert 1 < 2"]
covers = ["rule_1"]
"""

# 'os' is a real module and the spec imports nothing from it, so check_deps
# stays quiet; that keeps these tests about run mechanics, not dependencies.

T1 = datetime(2026, 7, 11, 14, 30, 5, tzinfo=UTC)
T2 = datetime(2026, 7, 11, 15, 45, 0, tzinfo=UTC)


async def test_build_writes_timestamped_run_and_markers(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    await build(SPEC, out, now=T1)

    run_dir = out / "2026-07-11-14-30-05"
    assert run_dir.is_dir()
    assert (run_dir / "test_backoff.py").exists()
    assert (run_dir / DONE_MARKER).exists()
    assert (out / LATEST_POINTER).exists()

    latest = read_latest(out)
    assert latest is not None
    assert latest.run_dir == "2026-07-11-14-30-05"
    assert latest.hashes["backoff"] == spec_hash(SPEC)


async def test_second_build_unchanged_reuses(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    await build(SPEC, out, now=T1)
    results = await build(SPEC, out, now=T2)

    assert results[0].reused is True
    # a fresh run dir still materializes, with the reused artifact copied in
    assert (out / "2026-07-11-15-45-00" / "test_backoff.py").exists()
    # pointer advances to the newest run
    assert read_latest(out).run_dir == "2026-07-11-15-45-00"


async def test_second_build_changed_regenerates(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    await build(SPEC, out, now=T1)
    changed = SPEC.replace("grows with attempt", "grows monotonically with attempt")
    results = await build(changed, out, now=T2)

    assert results[0].reused is False
    assert read_latest(out).hashes["backoff"] == spec_hash(changed)


async def test_incomplete_run_is_not_reused(tmp_path: Path) -> None:
    out = tmp_path / "gen"
    await build(SPEC, out, now=T1)
    # simulate a crashed run: drop the done marker of the latest run
    (out / "2026-07-11-14-30-05" / DONE_MARKER).unlink()

    # read_latest refuses a run with no marker
    assert read_latest(out) is None
    # so the next unchanged build regenerates rather than reusing
    results = await build(SPEC, out, now=T2)
    assert results[0].reused is False


async def test_use_runs_false_writes_flat(tmp_path: Path) -> None:
    out = tmp_path / "flat"
    results = await build(SPEC, out, now=T1, use_runs=False)

    assert (out / "test_backoff.py").exists()  # directly in out, no run dir
    assert not (out / LATEST_POINTER).exists()
    assert not (out / DONE_MARKER).exists()
    assert results[0].reused is False


@pytest.mark.parametrize("stamp", ["2026-07-11-14-30-05"])
def test_read_latest_missing_pointer_is_none(tmp_path: Path, stamp: str) -> None:
    assert read_latest(tmp_path) is None
