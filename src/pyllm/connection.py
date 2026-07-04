"""Async HTTP connection over httpx: client config plus an error-raising
response hook and a small retry loop.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import random
import weakref
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from .errors import (
    OverloadedError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
    error_for_status,
)

if TYPE_CHECKING:
    from .configuration import Configuration
    from .provider import Provider

logger = logging.getLogger("pyllm")

# Exceptions/statuses worth retrying (mirrors Connection#retry_exceptions).
_RETRY_STATUSES = {429, 500, 502, 503, 529}
_RETRY_EXC = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)
_RETRY_ERRORS = (RateLimitError, ServerError, ServiceUnavailableError, OverloadedError)


@dataclass(slots=True)
class Response:
    """Minimal response wrapper exposing the parts of the HTTP response we use."""

    status: int
    body: Any
    headers: Mapping[str, str] = field(default_factory=dict)
    content: bytes = b""


_TEXTUAL_TYPES = ("application/json", "text/", "application/xml", "+json")


def _parse_body(resp: httpx.Response) -> Any:
    # Check the content-type before touching resp.text so binary payloads
    # (audio, images) are not needlessly decoded to a throwaway str.
    ctype = resp.headers.get("content-type", "")
    if not any(marker in ctype for marker in _TEXTUAL_TYPES) and resp.content[:1] not in (
        b"{",
        b"[",
    ):
        return resp.content
    text = resp.text
    if "application/json" in ctype or (text[:1] in "{["):
        try:
            return resp.json()
        except Exception:
            return text
    return text


# Shared AsyncClients, keyed per event loop then per (base_url, timeout, proxy).
# Sharing preserves connection pooling/keep-alive across Chat instances instead
# of building (and leaking) a fresh client per facade call; keying by loop
# avoids reusing connections across event loops.
_CLIENT_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], httpx.AsyncClient]] = (
    weakref.WeakKeyDictionary()
)


def _shared_client(base_url: str, timeout: Any, proxy: Any) -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    per_loop = _CLIENT_CACHE.setdefault(loop, {})
    key = (str(base_url), timeout, proxy)
    client = per_loop.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            proxy=proxy,
            follow_redirects=True,
        )
        per_loop[key] = client
    return client


async def aclose() -> None:
    """Close every shared HTTP client owned by the current event loop.

    Call once at application shutdown (exported as ``pyllm.aclose``).
    """
    loop = asyncio.get_running_loop()
    per_loop = _CLIENT_CACHE.pop(loop, {})
    for client in per_loop.values():
        if not client.is_closed:
            await client.aclose()


class Connection:
    """Routes one provider's requests through a shared per-loop AsyncClient."""

    def __init__(self, provider: Provider, config: Configuration) -> None:
        self.provider = provider
        self.config = config

    @property
    def _client(self) -> httpx.AsyncClient:
        # Resolved lazily inside the running loop; api_base is read fresh so
        # providers with dynamic bases (e.g. region changes) stay correct.
        return _shared_client(
            self.provider.api_base, self.config.request_timeout, self.config.http_proxy
        )

    @classmethod
    def basic(cls) -> httpx.AsyncClient:
        """A bare client for ad-hoc requests (models.dev, URL attachments)."""
        return httpx.AsyncClient(follow_redirects=True, timeout=30)

    async def aclose(self) -> None:
        """Close the shared client this connection routes through.

        Note the client may be shared with other connections on the same
        base URL; prefer the module-level :func:`aclose` at shutdown.
        """
        client = self._client
        if not client.is_closed:
            await client.aclose()

    async def post(
        self,
        url: str,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
        multipart: bool = False,
    ) -> Response:
        return await self._request(
            "POST", url, payload=payload, headers=headers, multipart=multipart
        )

    async def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> Response:
        return await self._request("GET", url, headers=headers)

    async def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Any = None,
        headers: Mapping[str, str] | None = None,
        multipart: bool = False,
    ) -> Response:
        request_headers = {**self.provider.headers, **(headers or {})}
        kwargs: dict[str, Any] = {"headers": request_headers}
        if payload is not None:
            if multipart:
                kwargs["data"] = payload.get("data")
                kwargs["files"] = payload.get("files")
            else:
                kwargs["json"] = payload

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self._client.request(method, url, **kwargs)
                wrapped = Response(
                    status=resp.status_code,
                    body=_parse_body(resp),
                    headers=resp.headers,
                    content=resp.content,
                )
                if resp.status_code >= 400:
                    self._raise_for_response(wrapped)
                return wrapped
            except _RETRY_EXC as exc:
                last_exc = exc
            except _RETRY_ERRORS as exc:
                last_exc = exc
            if attempt < self.config.max_retries:
                await asyncio.sleep(self._backoff(attempt))
            else:
                break
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        url: str,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield raw response chunks for SSE streaming.

        Errors (non-200) are raised after reading the error body.
        """
        request_headers = {**self.provider.headers, **(headers or {})}
        async with self._client.stream("POST", url, json=payload, headers=request_headers) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                body = _parse_body(resp)
                self._raise_for_response(
                    Response(status=resp.status_code, body=body, headers=resp.headers)
                )
            async for chunk in resp.aiter_bytes():
                yield chunk

    def _raise_for_response(self, response: Response) -> None:
        message = self.provider.parse_error(response)
        error_cls = error_for_status(response.status, message if isinstance(message, str) else None)
        raise error_cls(response, message)

    def _backoff(self, attempt: int) -> float:
        interval = self.config.retry_interval * (self.config.retry_backoff_factor**attempt)
        jitter = interval * self.config.retry_interval_randomness * random.random()
        return interval + jitter


def parse_error_body(body: Any) -> Any:
    """Best-effort extraction of an error message (mirrors Provider#parse_error)."""
    if isinstance(body, str):
        try:
            body = _json.loads(body)
        except (ValueError, TypeError):
            return body or None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, str):
            return error
        candidates = [
            (error or {}).get("message") if isinstance(error, dict) else None,
            body.get("message"),
            body.get("detail"),
        ]
        return next((c for c in candidates if isinstance(c, str)), None)
    if isinstance(body, list):
        parts = []
        for part in body:
            if not isinstance(part, dict):
                continue
            error = part.get("error")
            parts.append(error if isinstance(error, str) else (error or {}).get("message"))
        return ". ".join(p for p in parts if p)
    return body
