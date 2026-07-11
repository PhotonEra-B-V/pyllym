"""Ollama (local LLM) provider tests.

Ollama speaks the OpenAI chat/completions wire format, so these mock the local
server with ``aioresponses`` — no running Ollama instance required.
"""

from __future__ import annotations

import pytest
from aioresponses import CallbackResult, aioresponses

import pyllm
from pyllm.providers.ollama import Ollama

from . import factories as f
from .conftest import sent_requests

OLLAMA_BASE = "http://localhost:11434/v1"


@pytest.fixture
def _ollama_config():
    """Point pyllm at a local Ollama endpoint for the duration of a test."""
    cfg = pyllm.config()
    prev_base, prev_key = cfg.ollama_api_base, cfg.ollama_api_key
    cfg.ollama_api_base = OLLAMA_BASE
    cfg.ollama_api_key = None
    yield
    cfg.ollama_api_base, cfg.ollama_api_key = prev_base, prev_key


def _headers(m: aioresponses) -> dict:
    return sent_requests(m)[-1].kwargs.get("headers") or {}


def test_ollama_is_local():
    assert Ollama.is_local() is True


@pytest.mark.asyncio
async def test_ollama_chat(_ollama_config):
    with aioresponses() as m:
        m.post(f"{OLLAMA_BASE}/chat/completions", payload=f.openai_chat("reply from ollama"))
        chat = pyllm.create_chat(model="llama3", provider="ollama", assume_model_exists=True)
        msg = await chat.ask("hi")
        assert sent_requests(m)
        assert msg.content == "reply from ollama"
        assert msg.input_tokens == 10


@pytest.mark.asyncio
async def test_ollama_no_auth_header_without_key(_ollama_config):
    """A bare local Ollama needs no credentials — no Authorization header."""
    with aioresponses() as m:
        m.post(f"{OLLAMA_BASE}/chat/completions", payload=f.openai_chat("ok"))
        chat = pyllm.create_chat(model="llama3", provider="ollama", assume_model_exists=True)
        await chat.ask("hi")
        assert "authorization" not in {k.lower() for k in _headers(m)}


@pytest.mark.asyncio
async def test_ollama_bearer_header_with_key(_ollama_config):
    """A secured Ollama (behind a proxy) sends the configured key as a bearer."""
    pyllm.config().ollama_api_key = "secret-token"
    with aioresponses() as m:
        m.post(f"{OLLAMA_BASE}/chat/completions", payload=f.openai_chat("ok"))
        chat = pyllm.create_chat(model="llama3", provider="ollama", assume_model_exists=True)
        await chat.ask("hi")
        headers = {k.lower(): v for k, v in _headers(m).items()}
        assert headers["authorization"] == "Bearer secret-token"


@pytest.mark.asyncio
async def test_ollama_tool_loop(_ollama_config):
    class Weather(pyllm.Tool):
        description = "weather"

        def execute(self, *, city: str):
            return f"Sunny in {city}"

    with aioresponses() as m:
        calls = {"n": 0}

        def responder(url, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return CallbackResult(
                    payload=f.openai_chat(
                        tool_calls=f.openai_tool_call("weather", {"city": "Rome"})
                    )
                )
            return CallbackResult(payload=f.openai_chat("Rome is sunny."))

        m.post(f"{OLLAMA_BASE}/chat/completions", callback=responder, repeat=True)
        chat = pyllm.create_chat(
            model="llama3", provider="ollama", assume_model_exists=True
        ).with_tool(Weather)
        msg = await chat.ask("weather in Rome?")
        assert msg.content == "Rome is sunny."
        assert [m2.role for m2 in chat.messages] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_ollama_streaming(_ollama_config):
    with aioresponses() as m:
        m.post(
            f"{OLLAMA_BASE}/chat/completions",
            body=f.openai_sse("Hel", "lo", "!"),
            headers={"content-type": "text/event-stream"},
        )
        chat = pyllm.create_chat(model="llama3", provider="ollama", assume_model_exists=True)
        chunks = [chunk.content async for chunk in chat.stream("hi")]
        assert "".join(c for c in chunks if c) == "Hello!"
        assert chat.messages[-1].content == "Hello!"
