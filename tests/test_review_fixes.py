"""Regression tests for the findings fixed in the code review."""

from __future__ import annotations

import httpx
import pytest
import respx

import pyllm
from pyllm.chunk import Chunk
from pyllm.errors import (
    ContextLengthExceededError,
    Error,
    OverloadedError,
    ServiceUnavailableError,
    error_for_status,
)
from pyllm.stream_accumulator import StreamAccumulator


def test_error_mapping_gateway_and_context_length():
    assert error_for_status(502) is ServiceUnavailableError
    assert error_for_status(504) is ServiceUnavailableError
    assert error_for_status(400, "prompt is too long: 210000 tokens") is ContextLengthExceededError
    assert error_for_status(400, "bad param") is not ContextLengthExceededError


@pytest.mark.asyncio
async def test_run_until_done_on_empty_chat_raises_clear_error():
    chat = pyllm.create_chat(model="gpt-4o")
    with pytest.raises(Error, match="Nothing to send"):
        await chat.run_until_done()


def test_calls_true_is_rejected_not_coerced_to_one():
    chat = pyllm.create_chat(model="gpt-4o")
    with pytest.raises(ValueError):
        chat.with_tool(None, calls=True)
    with pytest.raises(ValueError):
        chat.with_tool(None, concurrency="threds")


def test_pending_think_tag_flushed_at_end_of_stream():
    acc = StreamAccumulator()
    acc.add(Chunk(role="assistant", content="answer is 5<"))
    msg = acc.to_message(None)
    assert msg.content == "answer is 5<"


def test_with_context_keeps_registry_model_info():
    chat = pyllm.create_chat(model="gpt-4o")
    ctx = pyllm.context()
    chat.with_context(ctx)
    assert chat.model.pricing.text_tokens.input is not None  # not a stub Info


def test_context_has_animate_delegate():
    assert hasattr(pyllm.Context, "animate")


def test_dict_content_does_not_leak_text_into_attachments():
    msg = pyllm.Message(role="user", content={"text": "hello"})
    assert msg.content == "hello"


def test_config_typo_read_raises_attribute_error():
    cfg = pyllm.Configuration()
    with pytest.raises(AttributeError):
        _ = cfg.opnai_api_key
    assert cfg.openai_api_key is None  # registered option, unset -> None


@pytest.mark.asyncio
@respx.mock
async def test_bare_json_error_body_in_200_stream_raises():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=b'{"error": {"type": "overloaded_error", "message": "Overloaded"}}',
            headers={"content-type": "application/json"},
        )
    )
    chat = pyllm.create_chat(model="gpt-4o")
    with pytest.raises(OverloadedError):
        async for _ in chat.stream("hi"):
            pass


@pytest.mark.asyncio
@respx.mock
async def test_stream_early_break_cancels_producer():
    state = {"requests": 0}

    def responder(request):
        state["requests"] += 1
        sse = (
            b'data: {"model":"gpt-4o","choices":[{"delta":{"content":"one"}}]}\n\n'
            b'data: {"model":"gpt-4o","choices":[{"delta":{"content":"two"}}]}\n\n'
            b'data: {"model":"gpt-4o","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=responder)
    chat = pyllm.create_chat(model="gpt-4o")
    async for _chunk in chat.stream("hi"):
        break  # early exit must not raise and must not hang


def test_fal_queue_base_honors_api_base_override():
    pyllm.configure(lambda c: setattr(c, "fal_api_key", "k"))
    from pyllm.protocols.fal import Fal
    from pyllm.providers.fal import Fal as FalProvider

    cfg = pyllm.config().copy()
    cfg.fal_api_base = "https://gateway.corp/fal"
    protocol = Fal(FalProvider(cfg))
    assert protocol._queue_base() == "https://gateway.corp/fal"
    cfg.fal_queue_base = "https://queue.corp"
    assert protocol._queue_base() == "https://queue.corp"


@pytest.mark.asyncio
@respx.mock
async def test_shared_client_reused_across_chats():
    from pyllm.connection import _CLIENT_CACHE

    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "gpt-4o",
                "choices": [
                    {"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    a = pyllm.create_chat(model="gpt-4o")
    b = pyllm.create_chat(model="gpt-4o")
    await a.ask("x")
    await b.ask("y")
    assert a.provider.connection._client is b.provider.connection._client
    await pyllm.aclose()
    import asyncio

    assert asyncio.get_running_loop() not in _CLIENT_CACHE


@pytest.mark.asyncio
async def test_media_async_blob_helpers_exist():
    from pyllm import Image, Video

    img = Image(data="aGk=")  # base64 "hi"
    assert await img.ato_blob() == b"hi"
    vid = Video(data="aGk=")
    assert await vid.ato_blob() == b"hi"
