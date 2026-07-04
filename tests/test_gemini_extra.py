from __future__ import annotations

import httpx
import pytest
import respx

import pyllm

from . import factories as f

GENERATE = r"https://generativelanguage\.googleapis\.com/v1beta/models/.*:generateContent"
STREAM = r"https://generativelanguage\.googleapis\.com/v1beta/models/.*:streamGenerateContent.*"


@pytest.mark.asyncio
@respx.mock
async def test_gemini_tool_loop():
    class Weather(pyllm.Tool):
        description = "weather"

        def execute(self, *, city: str):
            return f"Sunny in {city}"

    calls = {"n": 0}

    def responder(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json=f.gemini_response(
                    None, function_call={"name": "weather", "args": {"city": "Tokyo"}}
                ),
            )
        return httpx.Response(200, json=f.gemini_response("Tokyo is sunny."))

    respx.post(url__regex=GENERATE).mock(side_effect=responder)
    chat = pyllm.create_chat(model="gemini-2.5-flash").with_tool(Weather)
    msg = await chat.ask("weather in Tokyo?")
    assert msg.content == "Tokyo is sunny."
    assert [m.role for m in chat.messages] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
@respx.mock
async def test_gemini_streaming():
    respx.post(url__regex=STREAM).mock(
        return_value=httpx.Response(
            200, content=f.gemini_sse("Hel", "lo!"), headers={"content-type": "text/event-stream"}
        )
    )
    chat = pyllm.create_chat(model="gemini-2.5-flash")
    chunks = [c.content async for c in chat.stream("hi")]
    assert "".join(c for c in chunks if c) == "Hello!"
    assert chat.messages[-1].content == "Hello!"
