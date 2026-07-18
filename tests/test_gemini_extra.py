from __future__ import annotations

import re

import pytest
from aioresponses import CallbackResult

import pyllym

from . import factories as f

GENERATE = re.compile(
    r"https://generativelanguage\.googleapis\.com/v1beta/models/.*:generateContent"
)
STREAM = re.compile(
    r"https://generativelanguage\.googleapis\.com/v1beta/models/.*:streamGenerateContent.*"
)


@pytest.mark.asyncio
async def test_gemini_tool_loop(mock_http):
    class Weather(pyllym.Tool):
        description = "weather"

        def execute(self, *, city: str):
            return f"Sunny in {city}"

    calls = {"n": 0}

    def responder(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return CallbackResult(
                payload=f.gemini_response(
                    None, function_call={"name": "weather", "args": {"city": "Tokyo"}}
                )
            )
        return CallbackResult(payload=f.gemini_response("Tokyo is sunny."))

    mock_http.post(GENERATE, callback=responder, repeat=True)
    chat = pyllym.create_chat(model="gemini-2.5-flash").with_tool(Weather)
    msg = await chat.ask("weather in Tokyo?")
    assert msg.content == "Tokyo is sunny."
    assert [m.role for m in chat.messages] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_gemini_streaming(mock_http):
    mock_http.post(
        STREAM,
        body=f.gemini_sse("Hel", "lo!"),
        headers={"content-type": "text/event-stream"},
    )
    chat = pyllym.create_chat(model="gemini-2.5-flash")
    chunks = [c.content async for c in chat.stream("hi")]
    assert "".join(c for c in chunks if c) == "Hello!"
    assert chat.messages[-1].content == "Hello!"


@pytest.mark.asyncio
async def test_gemini_system_messages_use_system_instruction(mock_http):
    from .conftest import sent_requests

    mock_http.post(GENERATE, payload=f.gemini_response("Oui."))
    chat = pyllym.create_chat(model="gemini-2.5-flash").with_instructions("Be terse")
    await chat.ask("hi")
    body = sent_requests(mock_http)[-1].kwargs["json"]
    assert body["systemInstruction"] == {"parts": [{"text": "Be terse"}]}
    assert [c["role"] for c in body["contents"]] == ["user"]


def test_gemini_multiple_system_messages_concatenate():
    provider = pyllym.Provider.resolve("gemini")(pyllym.config())
    payload = provider.render(
        [
            pyllym.Message(role="system", content="Be terse"),
            pyllym.Message(role="system", content="Answer in French"),
            pyllym.Message(role="user", content="hi"),
        ],
        tools={},
        temperature=None,
        model=pyllym.models.find("gemini-2.5-flash"),
    )
    assert payload["systemInstruction"] == {"parts": [{"text": "Be terse\n\nAnswer in French"}]}
    assert [c["role"] for c in payload["contents"]] == ["user"]
