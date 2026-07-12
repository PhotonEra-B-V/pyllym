"""Tests for the reactuLLM request-mode planner bridge.

The bridge reads the shared ``reactullm-pyllum.contract.json`` and fills
reactuLLM's ``TestPlan`` via one structured-output turn. Tests drive it through
the mocked aiohttp layer (``mock_http``) — no network, no real keys — the same
pattern the ``bdd`` planner tests use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyllym import reactullm_bridge as bridge

from .conftest import sent_requests


def _validate(plan: dict, schema: dict) -> None:
    """Minimal structural check of a plan against the contract's JSON Schema.

    A dependency-free stand-in for full JSON-Schema validation, sufficient to
    catch the failures these tests care about: missing required keys and
    accidental snake_case drift on the camelCase surface reactuLLM consumes.
    """

    def check(value: object, sch: dict, where: str) -> None:
        typ = sch.get("type")
        if typ == "object":
            assert isinstance(value, dict), f"{where}: expected object"
            for key in sch.get("required", []):
                assert key in value, f"{where}: missing required key {key!r}"
            if sch.get("additionalProperties") is False:
                extra = set(value) - set(sch.get("properties", {}))
                assert not extra, f"{where}: unexpected keys {extra}"
            for key, subschema in sch.get("properties", {}).items():
                if key in value:
                    check(value[key], subschema, f"{where}.{key}")
        elif typ == "array":
            assert isinstance(value, list), f"{where}: expected array"
            for i, item in enumerate(value):
                check(item, sch["items"], f"{where}[{i}]")
        elif typ == "string":
            assert isinstance(value, str), f"{where}: expected string"
        elif typ == "boolean":
            assert isinstance(value, bool), f"{where}: expected boolean"

    check(plan, schema, "$")


# The contract shipped alongside the package (authoritative interface shape).
CONTRACT_FILE = Path(__file__).resolve().parents[1] / "reactullm-pyllum.contract.json"


def _contract() -> dict:
    return json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))


def _good_plan() -> dict:
    """A known-good, schema-valid TestPlan with camelCase keys."""
    return {
        "feature": "Search box",
        "api": {
            "module": "@/components/SearchBox",
            "imports": ["import { SearchBox } from '@/components/SearchBox'"],
            "signatures": [
                "export function SearchBox(props: { onSearch: (q: string) => void }): JSX.Element"
            ],
        },
        "fixtures": [{"name": "makeProps", "body": ["return { onSearch: jest.fn() }"]}],
        "cases": [
            {
                "scenario": "typing calls onSearch",
                "testName": "calls onSearch when the user types",
                "given": [
                    "const onSearch = jest.fn()",
                    "render(<SearchBox onSearch={onSearch} />)",
                ],
                "when": "await userEvent.type(screen.getByRole('textbox'), 'hi')",
                "then": ["expect(onSearch).toHaveBeenCalledWith('hi')"],
                "fixtures": [],
                "covers": ["rule_1"],
                "isAsync": True,
                "edgeCase": False,
                "notes": "",
            }
        ],
        "requiredIds": ["rule_1"],
    }


def _mock_plan_response(mock_http, plan: dict) -> None:
    payload = {
        "id": "cmpl-test",
        "model": "gpt-4o",
        "choices": [
            {"message": {"role": "assistant", "content": json.dumps(plan)}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_http.post("https://api.openai.com/v1/chat/completions", payload=payload, repeat=True)


@pytest.fixture
def enabled(monkeypatch):
    """Point the bridge at the shipped contract."""
    monkeypatch.setenv(bridge.CONTRACT_PATH_ENV, str(CONTRACT_FILE))
    return CONTRACT_FILE


# --- schema target round-trip --------------------------------------------------


def test_schema_target_is_the_contract_schema_verbatim():
    contract = _contract()
    target = bridge.test_plan_target(contract)
    # Fed to with_schema as-is: camelCase keys and defaults survive untouched.
    assert target is contract["test_plan_schema"]
    props = target["properties"]["cases"]["items"]["properties"]
    assert "testName" in props and "isAsync" in props and "edgeCase" in props
    assert target["properties"]["requiredIds"]["default"] == []


def test_good_plan_validates_against_contract_schema():
    """A known-good TestPlan round-trips through the contract's JSON Schema."""
    schema = _contract()["test_plan_schema"]
    _validate(_good_plan(), schema)  # raises on mismatch
    # camelCase keys stay camelCase.
    assert "testName" in _good_plan()["cases"][0]
    assert "requiredIds" in _good_plan()


# --- plan_from_spec ------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_from_spec_returns_schema_valid_plan(mock_http, enabled):
    _mock_plan_response(mock_http, _good_plan())

    plan = await bridge.plan_from_spec("Spec: a SearchBox that calls onSearch", model="gpt-4o")

    assert isinstance(plan, dict)
    _validate(plan, _contract()["test_plan_schema"])
    assert plan["cases"][0]["testName"] == "calls onSearch when the user types"
    assert sent_requests(mock_http)  # the planner was actually called


@pytest.mark.asyncio
async def test_plan_from_spec_rejects_non_object_content(mock_http, enabled):
    """A model that can't honor the schema -> clear error, not malformed JSON."""
    payload = {
        "id": "cmpl-test",
        "model": "gpt-4o",
        "choices": [
            {
                "message": {"role": "assistant", "content": "I could not do it"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    mock_http.post("https://api.openai.com/v1/chat/completions", payload=payload, repeat=True)

    with pytest.raises(ValueError, match="non-object content"):
        await bridge.plan_from_spec("Spec: whatever", model="gpt-4o")


# --- contract version guard ----------------------------------------------------


def test_unknown_contract_version_raises(tmp_path, monkeypatch):
    bad = _contract()
    bad["version"] = 999
    path = tmp_path / "bad.contract.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setenv(bridge.CONTRACT_PATH_ENV, str(path))

    with pytest.raises(bridge.UnsupportedContractVersion):
        bridge.load_contract()


# --- disabled path -------------------------------------------------------------


def test_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv(bridge.CONTRACT_PATH_ENV, raising=False)
    assert bridge.is_enabled() is False
    assert bridge.contract_path() is None
    with pytest.raises(bridge.BridgeDisabled):
        bridge.load_contract()


def test_disabled_treats_whitespace_as_unset(monkeypatch):
    monkeypatch.setenv(bridge.CONTRACT_PATH_ENV, "   ")
    assert bridge.is_enabled() is False


def test_cli_exits_nonzero_when_disabled(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv(bridge.CONTRACT_PATH_ENV, raising=False)
    prompt = tmp_path / "spec.prompt.txt"
    prompt.write_text("anything", encoding="utf-8")

    code = bridge.main([str(prompt)])

    assert code != 0
    assert "disabled" in capsys.readouterr().err


def test_default_model_falls_back(monkeypatch):
    monkeypatch.delenv(bridge.MODEL_ENV, raising=False)
    assert bridge.default_model() == bridge.DEFAULT_MODEL
    monkeypatch.setenv(bridge.MODEL_ENV, "gpt-5.4")
    assert bridge.default_model() == "gpt-5.4"
