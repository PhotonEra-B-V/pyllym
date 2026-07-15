"""Tests for the reactuLLM runtime LLM handler.

The handler serves the runtime request/response envelope a shipped reactuLLM
frontend and this backend both speak. It reads the shared
``reactullm-pyllum.runtime.json``, validates the request, routes on ``task``,
runs one pyllym turn (structured or free-form), and replies with a
``response_schema``-valid envelope. Tests drive it through the mocked aiohttp
layer (``mock_http``) — no network, no real keys — the same pattern the bridge
tests use.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyllym import reactullm_runtime as rt
from pyllym.reactullm_runtime import (
    LLMRequest,
    TaskConfig,
    TaskRegistry,
)

from .conftest import sent_requests

# The runtime contract shipped alongside the package (authoritative shape).
CONTRACT_FILE = Path(__file__).resolve().parents[1] / "reactullm-pyllum.runtime.json"


def _contract() -> dict:
    return json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))


def _validate(value: object, schema: dict, where: str = "$") -> None:
    """Structural check against a JSON Schema subset (dependency-free stand-in).

    Handles union type lists, required keys, and additionalProperties:false —
    enough to catch missing keys and snake_case drift on the camelCase surface.
    """
    types = schema.get("type")
    allowed = types if isinstance(types, list) else [types] if types is not None else []

    def ok(t: str) -> bool:
        return {
            "object": isinstance(value, dict),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "null": value is None,
        }.get(t, True)

    if allowed:
        assert any(ok(t) for t in allowed), f"{where}: expected {allowed}"
    if "object" in allowed and isinstance(value, dict):
        for key in schema.get("required", []):
            assert key in value, f"{where}: missing required {key!r}"
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(schema.get("properties", {}))
            assert not extra, f"{where}: unexpected keys {extra}"
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], sub, f"{where}.{key}")
    elif "array" in allowed and isinstance(value, list):
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{where}[{i}]")


def _registry() -> TaskRegistry:
    return (
        TaskRegistry()
        .register(
            "summarize_job",
            TaskConfig(
                default_prompt="You are a concise job-post summarizer.",
                allowed_variables=frozenset({"locale"}),
            ),
        )
        .register(
            "extract_skills",
            TaskConfig(default_prompt="Extract the candidate's skills as JSON."),
        )
    )


def _mock_text(mock_http, text: str) -> None:
    payload = {
        "id": "cmpl-test",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_http.post("https://api.openai.com/v1/chat/completions", payload=payload, repeat=True)


def _mock_json(mock_http, obj: dict) -> None:
    _mock_text(mock_http, json.dumps(obj))


@pytest.fixture
def enabled(monkeypatch):
    """Point the handler at the shipped runtime contract."""
    monkeypatch.setenv(rt.CONTRACT_PATH_ENV, str(CONTRACT_FILE))
    return CONTRACT_FILE


# --- contract shape ------------------------------------------------------------


def test_contract_exposes_request_and_response_schemas():
    contract = _contract()
    req = rt.request_schema(contract)
    resp = rt.response_schema(contract)
    assert set(req["required"]) == {"task", "input"}
    assert "schema" in req["properties"]  # camelCase-on-wire key, not schema_
    assert set(resp["required"]) == {"task", "ok"}


# --- free-form task ------------------------------------------------------------


@pytest.mark.asyncio
async def test_freeform_task_answers_in_text(mock_http, enabled):
    _mock_text(mock_http, "A short summary.")
    req = LLMRequest(task="summarize_job", input="Senior Python role, remote.")

    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is True
    assert resp.text == "A short summary."
    assert resp.data is None
    assert resp.task == "summarize_job"
    _validate(resp.model_dump(by_alias=True), _contract()["response_schema"])
    assert sent_requests(mock_http)


# --- structured task -----------------------------------------------------------


@pytest.mark.asyncio
async def test_structured_task_answers_in_data(mock_http, enabled):
    _mock_json(mock_http, {"skills": ["python", "async"]})
    schema = {
        "type": "object",
        "properties": {"skills": {"type": "array", "items": {"type": "string"}}},
        "required": ["skills"],
    }
    req = LLMRequest(task="extract_skills", input="I write async Python.", schema=schema)

    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is True
    assert resp.data == {"skills": ["python", "async"]}
    assert resp.text is None
    _validate(resp.model_dump(by_alias=True), _contract()["response_schema"])


@pytest.mark.asyncio
async def test_structured_task_with_prose_is_schema_violation(mock_http, enabled):
    _mock_text(mock_http, "I could not produce JSON.")
    schema = {"type": "object", "properties": {"skills": {"type": "array"}}}
    req = LLMRequest(task="extract_skills", input="...", schema=schema)

    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is False
    assert resp.error is not None
    assert resp.error.code == "schema_violation"
    assert resp.data is None and resp.text is None


# --- routing / authorization ---------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_task_is_rejected(enabled):
    req = LLMRequest(task="delete_everything", input="please")
    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is False
    assert resp.error.code == "unknown_task"
    assert resp.task == "delete_everything"  # echoed even on failure


@pytest.mark.asyncio
async def test_default_prompt_used_when_instructions_empty(mock_http, enabled):
    _mock_text(mock_http, "ok")
    req = LLMRequest(task="summarize_job", input="a job")

    await rt.handle(req, registry=_registry())

    body = sent_requests(mock_http)[-1].kwargs["json"]
    system = next(m for m in body["messages"] if m["role"] in ("system", "developer"))
    assert system["content"] == "You are a concise job-post summarizer."


@pytest.mark.asyncio
async def test_caller_instructions_override_default(mock_http, enabled):
    _mock_text(mock_http, "ok")
    req = LLMRequest(task="summarize_job", input="a job", instructions="Be terse.")

    await rt.handle(req, registry=_registry())

    body = sent_requests(mock_http)[-1].kwargs["json"]
    system = next(m for m in body["messages"] if m["role"] in ("system", "developer"))
    assert system["content"] == "Be terse."


# --- variables allow-list ------------------------------------------------------


@pytest.mark.asyncio
async def test_only_allowlisted_variables_are_interpolated(mock_http, enabled):
    _mock_text(mock_http, "ok")
    # 'locale' is allowed for summarize_job; 'secret' is not and must be dropped.
    req = LLMRequest(
        task="summarize_job",
        input="Summarize in {locale}.",
        variables={"locale": "fi", "secret": "sk-leak"},
    )

    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is True
    body = sent_requests(mock_http)[-1].kwargs["json"]
    user = next(m for m in body["messages"] if m["role"] == "user")
    assert "Summarize in fi." in json.dumps(user)
    assert "sk-leak" not in json.dumps(body)


@pytest.mark.asyncio
async def test_disallowed_variable_is_not_interpolated(mock_http, enabled):
    # A non-allowed key is dropped, so its placeholder stays literal (no leak,
    # no interpolation) — the call still succeeds and the secret never travels.
    _mock_text(mock_http, "ok")
    req = LLMRequest(task="summarize_job", input="Use {secret}.", variables={"secret": "sk-leak"})
    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is True
    body = sent_requests(mock_http)[-1].kwargs["json"]
    assert "sk-leak" not in json.dumps(body)
    assert "Use {secret}." in json.dumps(body)  # placeholder left untouched


@pytest.mark.asyncio
async def test_missing_allowed_variable_fails_cleanly(enabled):
    # An ALLOWED placeholder referenced but not supplied (while another allowed
    # var IS supplied, so interpolation runs) -> interpolation KeyError ->
    # invalid_request, not a crash.
    registry = TaskRegistry().register(
        "summarize_job",
        TaskConfig(
            default_prompt="Summarize.",
            allowed_variables=frozenset({"locale", "tone"}),
        ),
    )
    req = LLMRequest(
        task="summarize_job",
        input="Summarize in {locale} with {tone}.",
        variables={"locale": "fi"},  # 'tone' allowed but absent
    )
    resp = await rt.handle(req, registry=registry)

    assert resp.ok is False
    assert resp.error.code == "invalid_request"


# --- error mapping -------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_error_maps_to_provider_error(mock_http, enabled):
    mock_http.post("https://api.openai.com/v1/chat/completions", status=500, repeat=True)
    req = LLMRequest(task="summarize_job", input="a job")

    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is False
    assert resp.error.code == "provider_error"


@pytest.mark.asyncio
async def test_rate_limit_maps_to_rate_limited(mock_http, enabled):
    mock_http.post("https://api.openai.com/v1/chat/completions", status=429, repeat=True)
    req = LLMRequest(task="summarize_job", input="a job")

    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is False
    assert resp.error.code == "rate_limited"


# --- request validation --------------------------------------------------------


@pytest.mark.asyncio
async def test_max_input_enforced(enabled):
    registry = TaskRegistry().register(
        "summarize_job",
        TaskConfig(default_prompt="Summarize.", max_input=5),
    )
    req = LLMRequest(task="summarize_job", input="way too long input")
    resp = await rt.handle(req, registry=registry)

    assert resp.ok is False
    assert resp.error.code == "invalid_request"


# --- disabled path -------------------------------------------------------------


def test_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv(rt.CONTRACT_PATH_ENV, raising=False)
    assert rt.is_enabled() is False
    assert rt.contract_path() is None
    with pytest.raises(rt.RuntimeDisabled):
        rt.load_contract()


def test_disabled_treats_whitespace_as_unset(monkeypatch):
    monkeypatch.setenv(rt.CONTRACT_PATH_ENV, "   ")
    assert rt.is_enabled() is False


@pytest.mark.asyncio
async def test_handle_when_disabled_is_internal_error(monkeypatch):
    monkeypatch.delenv(rt.CONTRACT_PATH_ENV, raising=False)
    req = LLMRequest(task="summarize_job", input="a job")
    resp = await rt.handle(req, registry=_registry())

    assert resp.ok is False
    assert resp.error.code == "internal"


# --- contract version guard ----------------------------------------------------


def test_unknown_contract_version_raises(tmp_path, monkeypatch):
    bad = _contract()
    bad["version"] = 999
    path = tmp_path / "bad.runtime.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setenv(rt.CONTRACT_PATH_ENV, str(path))

    with pytest.raises(rt.UnsupportedContractVersion):
        rt.load_contract()


# --- envelope aliasing ---------------------------------------------------------


def test_request_schema_key_aliases_to_schema_():
    # camelCase 'schema' on the wire maps to schema_ in Python, both directions.
    req = LLMRequest(task="t", input="i", schema={"type": "object"})
    assert req.schema_ == {"type": "object"}
    dumped = req.model_dump(by_alias=True)
    assert "schema" in dumped and "schema_" not in dumped
