"""The cross-stack **handoff contract** — the bidirectional twin of the planning
bridge (:mod:`pyllym.reactullm_bridge`).

Where ``reactullm-pyllum.contract.json`` is a fixed interface (reactuLLM tells
pyllym how to fill a ``TestPlan``), the handoff contract is the *product* of one
side's generation: the API surface the OTHER side must conform to. It flows in
whichever direction a given generation runs:

* **backend-first** — pyllym/FastAPI generates first; it WRITES the handoff
  describing its endpoints, and reactuLLM consumes it to build a matching front.
* **frontend-first** — reactuLLM generates first; it writes the handoff
  describing the data the components need, and pyllym consumes it (READS the
  latest handoff and implements a matching backend).

``direction`` + ``producer`` / ``consumer`` record who conforms to whom. A
``runId`` (a sortable ``YYYYMMDDThhmmssZ`` UTC id) makes generations a total
order, so the consumer implements ONLY the latest completed handoff — a stale or
half-written one never wins because a failed build writes no ``HANDOFF_DONE.json``
marker.

This module MIRRORS reactuLLM's ``src/handoff.ts`` (the shape) and
``src/handoffStore.ts`` (the latest-wins bookkeeping) so the two repos
interoperate through one shared directory. The file names, key casing, ``runId``
format, and de-collision behavior MUST match — they are the wire format.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The env var that both switches the connection on and locates the shared dir.
HANDOFF_DIR_ENV = "REACTULLM_PYLLUM_HANDOFF_DIR"

#: The three files the shared directory holds (names must match reactuLLM).
HANDOFF_CONTRACT_FILENAME = "handoff.contract.json"
HANDOFF_DONE_FILENAME = "HANDOFF_DONE.json"
HANDOFF_MANIFEST_FILENAME = ".handoff-runs.json"

#: Handoff contract schema version this side produces/understands.
HANDOFF_VERSION = 1

#: ``YYYYMMDDThhmmssZ`` with an optional ``.NNN`` de-collision suffix — matches
#: the Zod regex in reactuLLM's ``handoff.ts``.
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z(?:\.\d{3})?$")
_SUFFIX_RE = re.compile(r"^(.*Z)(?:\.(\d+))?$")


# --- run ids (mirror reactuLLM's runs.ts toRunId / uniqueRunId) ----------------


def utc_now() -> datetime:
    """The default clock: timezone-aware UTC now (injectable for determinism)."""
    return datetime.now(UTC)


def to_run_id(when: datetime) -> str:
    """Format a datetime as a sortable UTC run id: ``YYYYMMDDThhmmssZ``.

    Uses UTC fields regardless of ``when``'s tzinfo, matching reactuLLM's
    ``toRunId`` (which reads ``getUTC*``).
    """
    u = when.astimezone(UTC)
    return f"{u.year:04d}{u.month:02d}{u.day:02d}T{u.hour:02d}{u.minute:02d}{u.second:02d}Z"


def unique_run_id(run_id: str, existing: list[str]) -> str:
    """Ensure ``run_id`` sorts strictly after every id already recorded.

    Run ids are second-granularity, so two generations in the same second (or a
    clock that went backwards) would collide or sort behind history. When the
    incoming id doesn't already sort strictly after the newest recorded id, we
    disambiguate off the NEWEST id — not the stale incoming one — by appending /
    bumping a ``.NNN`` suffix. Anchoring on the newest guarantees the result
    exceeds it, so the sequence stays a total order and the loop terminates.

    A verbatim port of reactuLLM's ``uniqueRunId`` (``runs.ts``).
    """
    taken = set(existing)
    newest = max(existing, default="")
    if run_id > newest and run_id not in taken:
        return run_id

    base = run_id if run_id > newest else newest
    match = _SUFFIX_RE.match(base)
    stem = match.group(1) if match else base
    n = int(match.group(2)) + 1 if (match and match.group(2)) else 2
    candidate = f"{stem}.{n:03d}"
    while candidate in taken:
        n += 1
        candidate = f"{stem}.{n:03d}"
    return candidate


# --- switch + directory --------------------------------------------------------


def handoff_dir(explicit: str | None = None) -> str | None:
    """The shared handoff directory, or ``None`` when the connection is off.

    An explicit argument wins; otherwise ``REACTULLM_PYLLUM_HANDOFF_DIR`` is
    consulted. An empty / whitespace-only value counts as unset (no error) — the
    same switch discipline as the planning contract.
    """
    raw = explicit if explicit is not None else os.environ.get(HANDOFF_DIR_ENV)
    if raw is None or not raw.strip():
        return None
    return raw


# --- reading the latest committed handoff -------------------------------------


def _read_manifest(dir: str) -> list[dict[str, Any]]:
    """The append-only run history (oldest first), or ``[]`` when absent/corrupt.

    A corrupt manifest must not wedge generation — treat it as empty history,
    matching reactuLLM's ``readManifest``.
    """
    path = Path(dir) / HANDOFF_MANIFEST_FILENAME
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    runs = parsed.get("runs") if isinstance(parsed, dict) else None
    return runs if isinstance(runs, list) else []


def _read_done(dir: str) -> dict[str, Any] | None:
    """The completion marker, or ``None`` when absent/unreadable."""
    path = Path(dir) / HANDOFF_DONE_FILENAME
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def latest_handoff(dir: str) -> dict[str, Any] | None:
    """The newest committed handoff contract, or ``None`` if none.

    Prefers the marker (``HANDOFF_DONE.json``) but falls back to the newest
    manifest entry; when both exist the lexically-greater ``runId`` wins, so a
    stale marker from an interrupted write never shadows a newer recorded run.
    Mirrors reactuLLM's ``latestHandoff``.
    """
    done = _read_done(dir)
    runs = _read_manifest(dir)
    newest = runs[-1] if runs else None
    if done and newest:
        return done if str(done.get("runId", "")) >= str(newest.get("runId", "")) else newest
    return done or newest


def is_newer_than_implemented(dir: str, last_run_id: str | None) -> bool:
    """Has a newer handoff arrived than the one the caller last implemented?

    ``last_run_id`` is the runId the consumer built against previously (``None``
    if it has never implemented one). Returns ``True`` only when the latest
    committed handoff sorts strictly after it — so re-running a caught-up
    consumer is a no-op and stale contracts are never re-implemented. Mirrors
    reactuLLM's ``isNewerThanImplemented``.
    """
    latest = latest_handoff(dir)
    if latest is None:
        return False
    latest_id = str(latest.get("runId", ""))
    return last_run_id is None or latest_id > last_run_id


# --- committing a produced handoff --------------------------------------------

_ALLOWED_DIRECTIONS = frozenset({"backend_first", "frontend_first"})
_ALLOWED_PARTIES = frozenset({"reactullm", "pyllum"})
_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE"})


def _validate_contract(contract: dict[str, Any]) -> None:
    """Mechanically screen a committed contract against reactuLLM's Zod schema.

    Keeps a malformed handoff out of the shared directory — reactuLLM's Zod
    parse would reject it on read, so failing here is a clearer local signal.
    """
    if contract.get("version") != HANDOFF_VERSION:
        got = contract.get("version")
        raise ValueError(f"handoff version must be {HANDOFF_VERSION}, got {got!r}")
    if contract.get("direction") not in _ALLOWED_DIRECTIONS:
        raise ValueError(f"handoff direction must be one of {sorted(_ALLOWED_DIRECTIONS)}")
    if contract.get("producer") not in _ALLOWED_PARTIES:
        raise ValueError(f"handoff producer must be one of {sorted(_ALLOWED_PARTIES)}")
    if contract.get("consumer") not in _ALLOWED_PARTIES:
        raise ValueError(f"handoff consumer must be one of {sorted(_ALLOWED_PARTIES)}")
    if not isinstance(contract.get("feature"), str):
        raise ValueError("handoff feature must be a string")
    surface = contract.get("apiSurface")
    if not isinstance(surface, dict) or not isinstance(surface.get("endpoints"), list):
        raise ValueError("handoff apiSurface must have an 'endpoints' list")
    for ep in surface["endpoints"]:
        if not isinstance(ep, dict) or ep.get("method") not in _ALLOWED_METHODS:
            raise ValueError(f"endpoint method must be one of {sorted(_ALLOWED_METHODS)}")
        if not isinstance(ep.get("path"), str):
            raise ValueError("endpoint path must be a string")


def _normalize_endpoint(ep: dict[str, Any]) -> dict[str, Any]:
    """Apply reactuLLM's Zod defaults (summary='', request=null) to an endpoint."""
    return {
        "method": ep["method"],
        "path": ep["path"],
        "summary": ep.get("summary", ""),
        "request": ep.get("request"),
        "response": ep.get("response", {}),
    }


def commit_handoff(
    dir: str,
    contract: dict[str, Any],
    *,
    now: datetime | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Commit a handoff: assign a de-collided ``runId`` and write all three files.

    Call ONLY after the producing generation has fully succeeded — the marker's
    presence is the "this handoff is real" signal. Writes ``HANDOFF_DONE.json``
    **last**, so a failed build leaves no marker. The ``runId`` is de-collided
    against ``.handoff-runs.json`` (a ``.NNN`` suffix on a same-second /
    backwards clock, exactly like reactuLLM's ``uniqueRunId``).

    ``contract`` supplies everything except ``runId`` / ``completedAt`` (those
    are assigned here). ``now`` is injectable for deterministic tests. Returns
    the committed contract (with the assigned ``runId`` / ``completedAt``).
    """
    at = now if now is not None else utc_now()
    at = at.astimezone(UTC)
    target = Path(dir)
    target.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(dir)
    assigned = unique_run_id(
        run_id or to_run_id(at),
        [str(r.get("runId", "")) for r in manifest],
    )

    surface = contract.get("apiSurface") or {}
    committed: dict[str, Any] = {
        "version": contract.get("version", HANDOFF_VERSION),
        "runId": assigned,
        # Millisecond precision + trailing Z, matching JS Date.toISOString().
        "completedAt": at.strftime("%Y-%m-%dT%H:%M:%S.") + f"{at.microsecond // 1000:03d}Z",
        "direction": contract["direction"],
        "producer": contract["producer"],
        "consumer": contract["consumer"],
        "feature": contract["feature"],
        "apiSurface": {
            "entities": surface.get("entities", {}),
            "endpoints": [_normalize_endpoint(ep) for ep in surface.get("endpoints", [])],
        },
    }
    if not _RUN_ID_RE.match(assigned):  # defensive; unique_run_id keeps the shape
        raise ValueError(f"assigned runId {assigned!r} is malformed")
    _validate_contract(committed)

    def body(v: Any) -> str:
        return json.dumps(v, indent=2) + "\n"

    next_manifest = {"runs": [*manifest, committed]}
    # Order matters: manifest + flat contract first, DONE marker LAST so an
    # interrupted commit never leaves a marker pointing at a half-written run.
    (target / HANDOFF_MANIFEST_FILENAME).write_text(body(next_manifest), encoding="utf-8")
    (target / HANDOFF_CONTRACT_FILENAME).write_text(body(committed), encoding="utf-8")
    (target / HANDOFF_DONE_FILENAME).write_text(body(committed), encoding="utf-8")
    return committed


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m pyllym.reactullm_handoff {produce|consume|latest} ...``.

    Subcommands drive the shared directory (``REACTULLM_PYLLUM_HANDOFF_DIR``,
    overridable with ``--dir``):

    * ``produce`` — commit a backend-first handoff read from a JSON file/stdin.
    * ``consume`` — print the latest handoff a frontend-first run left, but only
      if it is newer than ``--implemented`` (else exit non-zero: caught up).
    * ``latest``  — print the latest committed handoff (or exit non-zero).

    Exits non-zero (without acting) when the connection is disabled.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m pyllym.reactullm_handoff",
        description="Produce/consume the cross-stack handoff via the shared directory.",
    )
    parser.add_argument("--dir", default=None, help=f"shared dir (default: ${HANDOFF_DIR_ENV})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prod = sub.add_parser("produce", help="commit a handoff (backend-first)")
    p_prod.add_argument("contract", help="a JSON file with the handoff contract, or '-' for stdin")

    p_cons = sub.add_parser("consume", help="print the latest handoff if newer than --implemented")
    p_cons.add_argument(
        "--implemented", default=None, help="the runId already implemented on this side"
    )

    sub.add_parser("latest", help="print the latest committed handoff")

    args = parser.parse_args(argv)

    shared = handoff_dir(args.dir)
    if shared is None:
        print(
            f"reactuLLM handoff is disabled: set {HANDOFF_DIR_ENV} to the shared "
            "directory to enable it",
            file=sys.stderr,
        )
        return 2

    if args.cmd == "produce":
        raw = sys.stdin.read() if args.contract == "-" else Path(args.contract).read_text("utf-8")
        committed = commit_handoff(shared, json.loads(raw))
        sys.stdout.write(json.dumps(committed, indent=2) + "\n")
        return 0

    if args.cmd == "consume":
        if not is_newer_than_implemented(shared, args.implemented):
            print("no newer handoff than the one already implemented", file=sys.stderr)
            return 1
        latest = latest_handoff(shared)
        sys.stdout.write(json.dumps(latest, indent=2) + "\n")
        return 0

    # latest
    latest = latest_handoff(shared)
    if latest is None:
        print("no committed handoff found", file=sys.stderr)
        return 1
    sys.stdout.write(json.dumps(latest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
