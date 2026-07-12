"""Timestamped generation runs with a completion marker and skip-if-unchanged.

Each :func:`~pyllym.tdg.builder.build` call writes into its own timestamped run
directory under ``out_dir`` (``2026-07-11-14-30-05/``). A run is only ever
treated as usable once it finishes: on success the builder drops a
``_DONE.json`` marker inside the run dir *and* updates ``out_dir/latest.json``,
a portable pointer (no symlinks) naming the newest completed run plus the
content hash of every spec it generated.

That pointer powers **skip-if-unchanged**: before generating a spec, the
builder hashes it; if the latest completed run already generated an identical
spec (same hash) and that run's artifacts are still on disk, generation is
skipped and the prior output is reused. A changed or new spec falls through to
a fresh generation in the current run.

Nothing here imports the rest of the pipeline — it is pure run bookkeeping, so
the builder can own the "generate vs. reuse" decision without a cycle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DONE_MARKER = "_DONE.json"
LATEST_POINTER = "latest.json"
RUN_STAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"


def utc_now() -> datetime:
    """The default clock: timezone-aware UTC now.

    Injected into :func:`run_stamp` / the builder so tests can pin a fixed
    time and keep generation deterministic.
    """
    return datetime.now(UTC)


def run_stamp(now: datetime) -> str:
    """Format a run-directory name from a timestamp: ``2026-07-11-14-30-05``."""
    return now.strftime(RUN_STAMP_FORMAT)


def spec_hash(spec_text: str) -> str:
    """A stable content hash of a spec's canonical text (sha256, 16 hex chars).

    Short enough to read in ``latest.json``, wide enough that a collision
    between two distinct specs is not a practical concern for skip decisions.
    """
    return hashlib.sha256(spec_text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class LatestPointer:
    """The parsed ``latest.json``: the newest completed run and its spec hashes.

    ``run_dir`` is stored relative to ``out_dir`` so a whole output tree stays
    movable. ``hashes`` maps a spec slug to the content hash generated in that
    run, which is what skip-if-unchanged consults.
    """

    run_dir: str
    stamp: str
    hashes: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {"run_dir": self.run_dir, "stamp": self.stamp, "hashes": self.hashes}


def read_latest(out_dir: Path) -> LatestPointer | None:
    """Load ``out_dir/latest.json`` if it points at a still-complete run.

    Returns ``None`` when there is no pointer, it is unreadable/corrupt, or the
    run it names is missing its ``_DONE.json`` marker (interrupted or manually
    deleted) — in every one of those cases the caller must regenerate rather
    than trust stale state.
    """
    pointer_path = out_dir / LATEST_POINTER
    try:
        data = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    run_dir = data.get("run_dir")
    if not isinstance(run_dir, str):
        return None
    hashes = data.get("hashes")
    if not isinstance(hashes, dict):
        return None
    if not (out_dir / run_dir / DONE_MARKER).exists():
        return None  # named run is incomplete — do not reuse it
    return LatestPointer(
        run_dir=run_dir,
        stamp=str(data.get("stamp", "")),
        hashes={str(k): str(v) for k, v in hashes.items()},
    )


@dataclass(frozen=True)
class Reuse:
    """A spec whose output can be copied forward from the latest completed run."""

    slug: str
    source_run_dir: Path


def reusable(
    latest: LatestPointer | None,
    out_dir: Path,
    slug: str,
    current_hash: str,
) -> Reuse | None:
    """Decide whether ``slug`` can be reused from the latest completed run.

    Reuse only when the latest pointer recorded the *same* hash for this slug
    and the prior run's rendered test file is still present on disk. Any drift
    (hash changed, new spec, artifact deleted) returns ``None`` → regenerate.
    """
    if latest is None:
        return None
    if latest.hashes.get(slug) != current_hash:
        return None
    source = out_dir / latest.run_dir
    if not (source / f"test_{slug}.py").exists():
        return None
    return Reuse(slug=slug, source_run_dir=source)


def write_done_marker(run_dir: Path, stamp: str, hashes: dict[str, str]) -> None:
    """Write the ``_DONE.json`` completion marker inside a finished run dir."""
    marker = {"stamp": stamp, "hashes": hashes, "completed": True}
    (run_dir / DONE_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def write_latest_pointer(out_dir: Path, run_dir: Path, stamp: str, hashes: dict[str, str]) -> None:
    """Point ``out_dir/latest.json`` at a completed run (relative run dir)."""
    pointer = LatestPointer(run_dir=run_dir.name, stamp=stamp, hashes=hashes)
    (out_dir / LATEST_POINTER).write_text(
        json.dumps(pointer.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
