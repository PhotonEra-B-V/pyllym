"""Tests for the cross-stack handoff store (:mod:`pyllym.reactullm_handoff`).

The file names, key casing, ``runId`` format, and de-collision behavior must
match reactuLLM's ``src/handoffStore.ts`` so the two repos interoperate through
one shared directory. These tests pin exactly that.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from pyllym import reactullm_handoff as ho


def _backend_first() -> dict:
    return {
        "version": 1,
        "direction": "backend_first",
        "producer": "pyllum",
        "consumer": "reactullm",
        "feature": "Job search",
        "apiSurface": {
            "entities": {"Job": {"type": "object"}},
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/api/v1/jobs",
                    "summary": "list jobs",
                    "request": None,
                    "response": {"type": "array"},
                }
            ],
        },
    }


def _frontend_first() -> dict:
    return {
        "version": 1,
        "direction": "frontend_first",
        "producer": "reactullm",
        "consumer": "pyllum",
        "feature": "Job search",
        "apiSurface": {"endpoints": [{"method": "POST", "path": "/api/v1/jobs", "response": {}}]},
    }


def _at(second: int) -> datetime:
    return datetime(2026, 7, 13, 14, 22, second, tzinfo=UTC)


# --- run id format + de-collision ---------------------------------------------


def test_to_run_id_is_sortable_utc():
    assert ho.to_run_id(_at(33)) == "20260713T142233Z"
    assert ho._RUN_ID_RE.match(ho.to_run_id(_at(33)))


def test_unique_run_id_de_collides_on_same_second():
    first = ho.unique_run_id("20260713T142233Z", [])
    second = ho.unique_run_id("20260713T142233Z", [first])
    third = ho.unique_run_id("20260713T142233Z", [first, second])
    assert first == "20260713T142233Z"
    assert second == "20260713T142233Z.002"
    assert third == "20260713T142233Z.003"
    # A suffixed id sorts strictly after the bare one (total order preserved).
    assert second > first and third > second


def test_unique_run_id_anchors_on_newest_when_clock_goes_backwards():
    existing = ["20260713T142233Z"]
    # An older incoming id must still sort after history.
    result = ho.unique_run_id("20260101T000000Z", existing)
    assert result > existing[0]


# --- round trips ---------------------------------------------------------------


def test_backend_first_round_trip(tmp_path):
    committed = ho.commit_handoff(str(tmp_path), _backend_first(), now=_at(10))
    assert committed["direction"] == "backend_first"
    assert committed["producer"] == "pyllum"
    assert committed["consumer"] == "reactullm"
    assert committed["runId"] == "20260713T142210Z"
    assert committed["completedAt"] == "2026-07-13T14:22:10.000Z"

    latest = ho.latest_handoff(str(tmp_path))
    assert latest == committed
    # All three files exist; DONE mirrors the flat contract.
    assert (tmp_path / ho.HANDOFF_CONTRACT_FILENAME).exists()
    assert (tmp_path / ho.HANDOFF_DONE_FILENAME).exists()
    manifest = json.loads((tmp_path / ho.HANDOFF_MANIFEST_FILENAME).read_text())
    assert manifest["runs"][-1]["runId"] == committed["runId"]


def test_frontend_first_round_trip(tmp_path):
    committed = ho.commit_handoff(str(tmp_path), _frontend_first(), now=_at(10))
    assert committed["direction"] == "frontend_first"
    assert committed["producer"] == "reactullm"
    assert committed["consumer"] == "pyllum"
    # Zod defaults applied to the terse endpoint.
    ep = committed["apiSurface"]["endpoints"][0]
    assert ep["summary"] == "" and ep["request"] is None
    assert committed["apiSurface"]["entities"] == {}


def test_latest_is_newest_by_run_id(tmp_path):
    older = ho.commit_handoff(str(tmp_path), _backend_first(), now=_at(10))
    newer = ho.commit_handoff(str(tmp_path), _frontend_first(), now=_at(20))
    latest = ho.latest_handoff(str(tmp_path))
    assert latest["runId"] == newer["runId"] > older["runId"]
    assert latest["direction"] == "frontend_first"


def test_same_second_second_commit_wins_latest(tmp_path):
    first = ho.commit_handoff(str(tmp_path), _backend_first(), now=_at(33))
    second = ho.commit_handoff(str(tmp_path), _frontend_first(), now=_at(33))
    assert first["runId"] == "20260713T142233Z"
    assert second["runId"] == "20260713T142233Z.002"
    assert ho.latest_handoff(str(tmp_path))["runId"] == second["runId"]


# --- is_newer_than_implemented -------------------------------------------------


def test_is_newer_than_implemented(tmp_path):
    d = str(tmp_path)
    # Nothing committed yet.
    assert ho.is_newer_than_implemented(d, None) is False

    first = ho.commit_handoff(d, _backend_first(), now=_at(10))
    # Never implemented -> the first is newer.
    assert ho.is_newer_than_implemented(d, None) is True
    # Caught up -> no-op.
    assert ho.is_newer_than_implemented(d, first["runId"]) is False

    second = ho.commit_handoff(d, _frontend_first(), now=_at(20))
    # A newer runId arrived.
    assert ho.is_newer_than_implemented(d, first["runId"]) is True
    assert ho.is_newer_than_implemented(d, second["runId"]) is False


# --- disabled + validation -----------------------------------------------------


def test_handoff_dir_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv(ho.HANDOFF_DIR_ENV, raising=False)
    assert ho.handoff_dir() is None
    monkeypatch.setenv(ho.HANDOFF_DIR_ENV, "  ")
    assert ho.handoff_dir() is None
    monkeypatch.setenv(ho.HANDOFF_DIR_ENV, "/shared/handoff")
    assert ho.handoff_dir() == "/shared/handoff"


def test_commit_rejects_bad_direction(tmp_path):
    bad = _backend_first()
    bad["direction"] = "sideways"
    with pytest.raises(ValueError, match="direction"):
        ho.commit_handoff(str(tmp_path), bad, now=_at(10))


def test_corrupt_manifest_is_treated_as_empty(tmp_path):
    (tmp_path / ho.HANDOFF_MANIFEST_FILENAME).write_text("{ not json", encoding="utf-8")
    committed = ho.commit_handoff(str(tmp_path), _backend_first(), now=_at(10))
    assert committed["runId"] == "20260713T142210Z"


# --- CLI -----------------------------------------------------------------------


def test_cli_produce_then_consume(tmp_path, monkeypatch, capsys):
    d = tmp_path / "shared"
    monkeypatch.setenv(ho.HANDOFF_DIR_ENV, str(d))
    contract_file = tmp_path / "c.json"
    contract_file.write_text(json.dumps(_backend_first()), encoding="utf-8")

    assert ho.main(["produce", str(contract_file)]) == 0
    produced = json.loads(capsys.readouterr().out)
    assert produced["consumer"] == "reactullm"

    # Consume with nothing implemented -> prints latest, exit 0.
    assert ho.main(["consume"]) == 0
    consumed = json.loads(capsys.readouterr().out)
    assert consumed["runId"] == produced["runId"]

    # Consume when already caught up -> exit 1.
    assert ho.main(["consume", "--implemented", produced["runId"]]) == 1


def test_cli_disabled_exits_nonzero(monkeypatch, capsys):
    monkeypatch.delenv(ho.HANDOFF_DIR_ENV, raising=False)
    code = ho.main(["latest"])
    assert code != 0
    assert "disabled" in capsys.readouterr().err
