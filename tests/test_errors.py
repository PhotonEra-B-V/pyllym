from __future__ import annotations

import httpx
import pytest
import respx

import pyllm
from pyllm.errors import BadRequestError, RateLimitError, ServerError, UnauthorizedError


@pytest.mark.asyncio
@respx.mock
async def test_401_maps_to_unauthorized():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    with pytest.raises(UnauthorizedError) as exc:
        await pyllm.create_chat(model="gpt-4o").ask("hi")
    assert "bad key" in str(exc.value)


@pytest.mark.asyncio
@respx.mock
async def test_400_maps_to_bad_request():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "nope"}})
    )
    with pytest.raises(BadRequestError):
        await pyllm.create_chat(model="gpt-4o").ask("hi")


@pytest.mark.asyncio
@respx.mock
async def test_429_retries_then_raises_rate_limit():
    pyllm.configure(lambda c: setattr(c, "max_retries", 0))  # no backoff waits
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "slow down"}})
    )
    with pytest.raises(RateLimitError):
        await pyllm.create_chat(model="gpt-4o").ask("hi")
    assert route.called
    pyllm.configure(lambda c: setattr(c, "max_retries", 3))


@pytest.mark.asyncio
@respx.mock
async def test_500_maps_to_server_error():
    pyllm.configure(lambda c: setattr(c, "max_retries", 0))
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "boom"}})
    )
    with pytest.raises(ServerError):
        await pyllm.create_chat(model="gpt-4o").ask("hi")
    pyllm.configure(lambda c: setattr(c, "max_retries", 3))
