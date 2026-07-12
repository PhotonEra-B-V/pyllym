from __future__ import annotations

import inspect
import json
import types

import aiohttp
import pytest
from aioresponses import aioresponses
from aioresponses import core as aioresponses_core

import pyllym

from .seed_fixtures import DATA_DIR, seed_all


class _CompatClientResponse(aiohttp.ClientResponse):
    """aioresponses 0.7.9 predates the required ``stream_writer`` argument
    aiohttp 3.14 added to ``ClientResponse``; supply a no-op stand-in."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("stream_writer", types.SimpleNamespace(output_size=0))
        super().__init__(*args, **kwargs)


if "stream_writer" in inspect.signature(aiohttp.ClientResponse.__init__).parameters:
    aioresponses_core.ClientResponse = _CompatClientResponse


@pytest.fixture
def mock_http():
    """Mock the aiohttp transport layer for one test."""
    with aioresponses() as m:
        yield m


def sent_requests(m: aioresponses) -> list:
    """All requests recorded by an ``aioresponses`` mock, in a flat list."""
    return [call for calls in m.requests.values() for call in calls]


def sent_json(m: aioresponses) -> str:
    """The last recorded request's JSON payload, serialized for assertions."""
    return json.dumps(sent_requests(m)[-1].kwargs.get("json"))


_KEYS = {
    "openai_api_key": "sk-test",
    "anthropic_api_key": "sk-ant-test",
    "gemini_api_key": "gm-test",
    "deepseek_api_key": "ds-test",
    "mistral_api_key": "ms-test",
    "xai_api_key": "xai-test",
    "perplexity_api_key": "px-test",
    "openrouter_api_key": "or-test",
    "qwen_api_key": "qw-test",
    "zhipu_api_key": "zp-test",
    "moonshot_api_key": "mn-test",
    "minimax_api_key": "mm-test",
    "nvidia_api_key": "nv-test",
    "cerebras_api_key": "cb-test",
    "huggingface_api_key": "hf-test",
    "doubao_api_key": "db-test",
    "ernie_api_key": "er-test",
    "databricks_api_key": "dbx-test",
    "databricks_api_base": "https://ws.cloud.databricks.com/serving-endpoints",
    "bedrock_api_key": "AKIA-test",
    "bedrock_secret_key": "secret-test",
    "bedrock_region": "us-east-1",
}


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch):
    """Strip real provider env vars (OPENAI_API_KEY, ...) so the env-var
    fallback can't leak a developer's credentials into tests."""
    for key in pyllym.Configuration._provider_keys:
        monkeypatch.delenv(key.upper(), raising=False)


@pytest.fixture(autouse=True)
def _configure_keys():
    """Give every provider dummy credentials so construction succeeds."""
    cfg = pyllym.config()
    for key, value in _KEYS.items():
        setattr(cfg, key, value)
    yield


@pytest.fixture(autouse=True)
async def _close_shared_sessions():
    """Close the per-loop shared aiohttp sessions before the test loop dies."""
    yield
    await pyllym.aclose()


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixtures():
    """Regenerate the on-disk fixtures once per session so they stay in sync
    with the factories."""
    seed_all()
    assert DATA_DIR.exists()
    yield
