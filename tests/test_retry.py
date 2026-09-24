"""Retry policy: rate limits back off sensibly, server hints are honored,
streams retry before their first byte, and quota errors are not retried.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

import pyllym
from pyllym.connection import Connection, retry_after_seconds
from pyllym.errors import PaymentRequiredError, RateLimitError

URL = "https://api.openai.com/v1/chat/completions"
OK = {"choices": [{"message": {"role": "assistant", "content": "hi"}}], "usage": {}}
SSE = 'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff waits instead of sleeping; make jitter deterministic."""
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(Connection, "_sleep", staticmethod(fake_sleep))
    cfg = pyllym.config()
    saved = {
        k: getattr(cfg, k)
        for k in (
            "openai_api_base",
            "max_retries",
            "retry_interval",
            "rate_limit_retry_interval",
            "retry_interval_randomness",
            "retry_max_interval",
        )
    }
    cfg.openai_api_base = None
    cfg.max_retries = 3
    cfg.retry_interval = 0.1
    cfg.rate_limit_retry_interval = 1.0
    cfg.retry_interval_randomness = 0
    cfg.retry_max_interval = 60.0
    yield recorded
    for k, v in saved.items():
        setattr(cfg, k, v)


# --- header parsing -------------------------------------------------------


def test_retry_after_seconds_parses_delay_seconds():
    assert retry_after_seconds({"Retry-After": "7"}) == 7.0
    assert retry_after_seconds({"retry-after": "2.5"}) == 2.5


def test_retry_after_ms_wins_over_retry_after():
    assert retry_after_seconds({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5


def test_retry_after_seconds_parses_http_date():
    when = datetime.now(UTC) + timedelta(seconds=30)
    seconds = retry_after_seconds({"Retry-After": format_datetime(when, usegmt=True)})
    assert seconds is not None and 27 <= seconds <= 30


def test_retry_after_seconds_past_date_is_zero():
    when = datetime.now(UTC) - timedelta(seconds=30)
    assert retry_after_seconds({"Retry-After": format_datetime(when, usegmt=True)}) == 0.0


def test_retry_after_seconds_reads_openai_reset_windows():
    # Only the exhausted window counts; both reset headers are always sent.
    headers = {
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "250ms",
        "x-ratelimit-remaining-tokens": "39000",
        "x-ratelimit-reset-tokens": "6m0s",
    }
    assert retry_after_seconds(headers) == 0.25
    headers["x-ratelimit-remaining-tokens"] = "0"
    assert retry_after_seconds(headers) == 360.0
    # Neither window exhausted: the 429 is about something else, no hint.
    headers["x-ratelimit-remaining-requests"] = "5"
    headers["x-ratelimit-remaining-tokens"] = "5"
    assert retry_after_seconds(headers) is None
    # No remaining counters at all: the sooner reset.
    assert retry_after_seconds({"x-ratelimit-reset-requests": "1h2m3.5s"}) == 3723.5
    assert (
        retry_after_seconds(
            {"x-ratelimit-reset-requests": "2s", "x-ratelimit-reset-tokens": "6m0s"}
        )
        == 2.0
    )


def test_retry_after_seconds_ignores_garbage():
    assert retry_after_seconds({}) is None
    assert retry_after_seconds({"retry-after": "soon"}) is None
    assert retry_after_seconds({"x-ratelimit-reset-requests": "n/a"}) is None


# --- request retry policy -------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_backs_off_from_rate_limit_interval(sleeps, mock_http):
    for _ in range(3):
        mock_http.post(URL, status=429, payload={"error": {"message": "1302 too frequent"}})
    mock_http.post(URL, payload=OK)
    reply = await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert reply.content == "hi"
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_server_errors_keep_the_short_interval(sleeps, mock_http):
    mock_http.post(URL, status=503, payload={"error": {"message": "down"}})
    mock_http.post(URL, payload=OK)
    await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert sleeps == [pytest.approx(0.1)]


@pytest.mark.asyncio
async def test_retry_after_header_overrides_backoff(sleeps, mock_http):
    mock_http.post(
        URL,
        status=429,
        payload={"error": {"message": "slow down"}},
        headers={"Retry-After": "12"},
    )
    mock_http.post(URL, payload=OK)
    await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert sleeps == [12.0]


@pytest.mark.asyncio
async def test_retry_after_above_cap_gives_up_immediately(sleeps, mock_http):
    mock_http.post(
        URL,
        status=429,
        payload={"error": {"message": "daily limit"}},
        headers={"Retry-After": "3600"},
    )
    with pytest.raises(RateLimitError) as exc:
        await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert sleeps == []
    assert exc.value.retry_after == 3600.0
    assert sum(len(calls) for calls in mock_http.requests.values()) == 1


@pytest.mark.asyncio
async def test_computed_backoff_is_capped(sleeps, mock_http):
    pyllym.config().retry_max_interval = 1.5
    for _ in range(3):
        mock_http.post(URL, status=429, payload={"error": {"message": "busy"}})
    mock_http.post(URL, payload=OK)
    await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert sleeps == [1.0, 1.5, 1.5]


@pytest.mark.asyncio
async def test_exhausted_retries_raise_with_retry_after(sleeps, mock_http):
    for _ in range(4):
        mock_http.post(
            URL,
            status=429,
            payload={"error": {"message": "busy"}},
            headers={"Retry-After": "2"},
        )
    with pytest.raises(RateLimitError) as exc:
        await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert sleeps == [2.0, 2.0, 2.0]
    assert exc.value.retry_after == 2.0


@pytest.mark.asyncio
async def test_quota_exhaustion_is_payment_required_not_retried(sleeps, mock_http):
    mock_http.post(
        URL,
        status=429,
        payload={
            "error": {
                "code": "1113",
                "message": "Insufficient balance or no resource package. Please recharge.",
            }
        },
    )
    with pytest.raises(PaymentRequiredError):
        await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert sleeps == []


# --- streaming -------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_retries_rate_limit_before_first_byte(sleeps, mock_http):
    mock_http.post(URL, status=429, payload={"error": {"message": "1305 rate limit"}})
    mock_http.post(URL, body=SSE, headers={"content-type": "text/event-stream"})
    chunks = [c.content async for c in pyllym.create_chat(model="gpt-4o").stream("hi")]
    assert "".join(c for c in chunks if c) == "hi"
    assert sleeps == [1.0]


@pytest.mark.asyncio
async def test_stream_gives_up_after_max_retries(sleeps, mock_http):
    for _ in range(4):
        mock_http.post(URL, status=429, payload={"error": {"message": "busy"}})
    with pytest.raises(RateLimitError):
        async for _ in pyllym.create_chat(model="gpt-4o").stream("hi"):
            pass
    assert sleeps == [1.0, 2.0, 4.0]
