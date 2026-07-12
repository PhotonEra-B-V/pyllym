from __future__ import annotations

import json
import re

import pytest
from aioresponses import CallbackResult

import pyllym
from pyllym import Tool


@pytest.mark.asyncio
async def test_openai_chat(mock_http):
    mock_http.post(
        "https://api.openai.com/v1/chat/completions",
        payload={
            "id": "x",
            "model": "gpt-4o",
            "choices": [
                {"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )
    chat = pyllym.create_chat(model="gpt-4o")
    msg = await chat.ask("Hello")
    assert msg.content == "Hi!"
    assert msg.input_tokens == 10
    assert msg.output_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_chat(mock_http):
    mock_http.post(
        "https://api.anthropic.com/v1/messages",
        payload={
            "id": "msg",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello from Claude"}],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        },
    )
    chat = pyllym.create_chat(model="claude-sonnet-4-6")
    msg = await chat.ask("Hello")
    assert msg.content == "Hello from Claude"
    assert msg.output_tokens == 7


@pytest.mark.asyncio
async def test_tool_loop(mock_http):
    class WeatherTool(Tool):
        description = "Get weather"

        def execute(self, *, city: str):
            return f"Sunny in {city}"

    state = {"n": 0}

    def responder(url, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return CallbackResult(
                payload={
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "weather",
                                            "arguments": json.dumps({"city": "Paris"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            )
        return CallbackResult(
            payload={
                "model": "gpt-4o",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "It is sunny in Paris."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

    mock_http.post("https://api.openai.com/v1/chat/completions", callback=responder, repeat=True)
    chat = pyllym.create_chat(model="gpt-4o").with_tool(WeatherTool)
    msg = await chat.ask("Weather in Paris?")
    assert msg.content == "It is sunny in Paris."
    assert [m.role for m in chat.messages] == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_streaming(mock_http):
    sse = (
        b'data: {"model":"gpt-4o","choices":[{"delta":{"content":"Hel"}}]}\n\n'
        b'data: {"model":"gpt-4o","choices":[{"delta":{"content":"lo!"}}]}\n\n'
        b'data: {"model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2}}\n\n'
        b"data: [DONE]\n\n"
    )
    mock_http.post(
        "https://api.openai.com/v1/chat/completions",
        body=sse,
        headers={"content-type": "text/event-stream"},
    )
    chat = pyllym.create_chat(model="gpt-4o")
    chunks = [c.content async for c in chat.stream("Hi")]
    assert "".join(c for c in chunks if c) == "Hello!"
    assert chat.messages[-1].content == "Hello!"


@pytest.mark.asyncio
async def test_gemini_chat(mock_http):
    mock_http.post(
        re.compile(r"https://generativelanguage\.googleapis\.com/v1beta/models/.*:generateContent"),
        payload={
            "modelVersion": "gemini-2.5-flash",
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Hi from Gemini"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4},
        },
    )
    chat = pyllym.create_chat(model="gemini-2.5-flash")
    msg = await chat.ask("Hello")
    assert msg.content == "Hi from Gemini"
    assert msg.input_tokens == 8


@pytest.mark.asyncio
async def test_anthropic_tool_loop(mock_http):
    class WeatherTool(Tool):
        description = "Get weather"

        def execute(self, *, city: str):
            return f"Sunny in {city}"

    state = {"n": 0}

    def responder(url, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return CallbackResult(
                payload={
                    "id": "m1",
                    "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "text", "text": "checking"},
                        {
                            "type": "tool_use",
                            "id": "tu1",
                            "name": "weather",
                            "input": {"city": "Paris"},
                        },
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )
        return CallbackResult(
            payload={
                "id": "m2",
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "It is sunny in Paris."}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    mock_http.post("https://api.anthropic.com/v1/messages", callback=responder, repeat=True)
    chat = pyllym.create_chat(model="claude-sonnet-4-6").with_tool(WeatherTool)
    msg = await chat.ask("Weather in Paris?")
    assert msg.content == "It is sunny in Paris."
    assert [m.role for m in chat.messages] == ["user", "assistant", "tool", "assistant"]
