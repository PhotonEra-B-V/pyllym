from __future__ import annotations

import re

import pytest
from aioresponses import CallbackResult

import pyllm

from . import factories as f

GENERATE = re.compile(
    r"https://generativelanguage\.googleapis\.com/v1beta/models/.*:generateContent"
)
STREAM = re.compile(
    r"https://generativelanguage\.googleapis\.com/v1beta/models/.*:streamGenerateContent.*"
)


@pytest.mark.asyncio
async def test_gemini_tool_loop(mock_http):
    class Weather(pyllm.Tool):
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
    chat = pyllm.create_chat(model="gemini-2.5-flash").with_tool(Weather)
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
    chat = pyllm.create_chat(model="gemini-2.5-flash")
    chunks = [c.content async for c in chat.stream("hi")]
    assert "".join(c for c in chunks if c) == "Hello!"
    assert chat.messages[-1].content == "Hello!"
