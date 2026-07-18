"""Transport failures must surface as pyllym errors, never raw aiohttp ones.

These tests use real sockets (a closed port, a server that never responds) so
the aiohttp transport genuinely fails.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

import pyllym
from pyllym.errors import ConnectionFailedError


def _closed_port() -> int:
    """A port that was just bound and released — nothing is listening on it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fast_fail():
    """Point OpenAI at localhost with no retries and a short timeout."""
    cfg = pyllym.config()
    saved = (cfg.openai_api_base, cfg.max_retries, cfg.request_timeout)
    cfg.max_retries = 0
    cfg.request_timeout = 0.5
    yield cfg
    cfg.openai_api_base, cfg.max_retries, cfg.request_timeout = saved


@pytest.mark.asyncio
async def test_closed_port_raises_pyllym_error(fast_fail):
    fast_fail.openai_api_base = f"http://127.0.0.1:{_closed_port()}/v1"
    with pytest.raises(pyllym.Error) as exc:
        await pyllym.create_chat(model="gpt-4o").ask("hi")
    assert isinstance(exc.value, ConnectionFailedError)
    assert exc.value.__cause__ is not None


@pytest.mark.asyncio
async def test_timing_out_socket_raises_pyllym_error(fast_fail):
    async def never_respond(reader, writer):
        await reader.read(1)
        await asyncio.sleep(30)

    server = await asyncio.start_server(never_respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    fast_fail.openai_api_base = f"http://127.0.0.1:{port}/v1"
    try:
        with pytest.raises(pyllym.Error) as exc:
            await pyllym.create_chat(model="gpt-4o").ask("hi")
        assert isinstance(exc.value, ConnectionFailedError)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_streaming_closed_port_raises_pyllym_error(fast_fail):
    fast_fail.openai_api_base = f"http://127.0.0.1:{_closed_port()}/v1"
    with pytest.raises(pyllym.Error) as exc:
        async for _ in pyllym.create_chat(model="gpt-4o").stream("hi"):
            pass
    assert isinstance(exc.value, ConnectionFailedError)


@pytest.mark.asyncio
async def test_retries_still_map_http_statuses(fast_fail, mock_http):
    """HTTP-level failures keep their specific error classes."""
    fast_fail.openai_api_base = None
    mock_http.post(
        "https://api.openai.com/v1/chat/completions",
        status=503,
        payload={"error": {"message": "down"}},
    )
    with pytest.raises(pyllym.ServiceUnavailableError):
        await pyllym.create_chat(model="gpt-4o").ask("hi")
