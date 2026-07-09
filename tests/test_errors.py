from __future__ import annotations

import pytest

import pyllm
from pyllm.errors import BadRequestError, RateLimitError, ServerError, UnauthorizedError

from .conftest import sent_requests

URL = "https://api.openai.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_401_maps_to_unauthorized(mock_http):
    mock_http.post(URL, status=401, payload={"error": {"message": "bad key"}})
    with pytest.raises(UnauthorizedError) as exc:
        await pyllm.create_chat(model="gpt-4o").ask("hi")
    assert "bad key" in str(exc.value)


@pytest.mark.asyncio
async def test_400_maps_to_bad_request(mock_http):
    mock_http.post(URL, status=400, payload={"error": {"message": "nope"}})
    with pytest.raises(BadRequestError):
        await pyllm.create_chat(model="gpt-4o").ask("hi")


@pytest.mark.asyncio
async def test_429_retries_then_raises_rate_limit(mock_http):
    pyllm.configure(lambda c: setattr(c, "max_retries", 0))  # no backoff waits
    mock_http.post(URL, status=429, payload={"error": {"message": "slow down"}})
    with pytest.raises(RateLimitError):
        await pyllm.create_chat(model="gpt-4o").ask("hi")
    assert sent_requests(mock_http)
    pyllm.configure(lambda c: setattr(c, "max_retries", 3))


@pytest.mark.asyncio
async def test_500_maps_to_server_error(mock_http):
    pyllm.configure(lambda c: setattr(c, "max_retries", 0))
    mock_http.post(URL, status=500, payload={"error": {"message": "boom"}})
    with pytest.raises(ServerError):
        await pyllm.create_chat(model="gpt-4o").ask("hi")
    pyllm.configure(lambda c: setattr(c, "max_retries", 3))
