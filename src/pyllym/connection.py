"""Async HTTP connection over aiohttp: session config plus an error-raising
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

import aiohttp

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

logger = logging.getLogger("pyllym")

# Exceptions/statuses worth retrying (mirrors Connection#retry_exceptions).
_RETRY_STATUSES = {429, 500, 502, 503, 529}
_RETRY_EXC = (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError)
_RETRY_ERRORS = (RateLimitError, ServerError, ServiceUnavailableError, OverloadedError)


@dataclass(slots=True)
class Response:
    """Minimal response wrapper exposing the parts of the HTTP response we use."""

    status: int
    body: Any
    headers: Mapping[str, str] = field(default_factory=dict)
    content: bytes = b""


_TEXTUAL_TYPES = ("application/json", "text/", "application/xml", "+json")


def _parse_body(ctype: str, content: bytes) -> Any:
    # Check the content-type before decoding so binary payloads (audio,
    # images) are not needlessly decoded to a throwaway str.
    if not any(marker in ctype for marker in _TEXTUAL_TYPES) and content[:1] not in (
        b"{",
        b"[",
    ):
        return content
    text = content.decode("utf-8", errors="replace")
    if "application/json" in ctype or (text[:1] in "{["):
        try:
            return _json.loads(text)
        except Exception:
            return text
    return text


def _merge_url(base: str, url: str) -> str:
    """Join a provider api_base with a (possibly relative) endpoint URL."""
    if "://" in url:
        return url
    return base.rstrip("/") + "/" + url.lstrip("/")


def _client_timeout(timeout: Any) -> aiohttp.ClientTimeout:
    # Per-operation deadlines rather than a total-duration cap, so long
    # streams are not cut off mid-response.
    return aiohttp.ClientTimeout(
        total=None, connect=timeout, sock_connect=timeout, sock_read=timeout
    )


# Shared ClientSessions, keyed per event loop then per timeout. Sharing
# preserves connection pooling/keep-alive across Chat instances instead of
# building (and leaking) a fresh session per facade call; keying by loop
# avoids reusing connections across event loops.
_CLIENT_CACHE: weakref.WeakKeyDictionary[Any, dict[tuple[Any, ...], aiohttp.ClientSession]] = (
    weakref.WeakKeyDictionary()
)


def _shared_client(timeout: Any) -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    per_loop = _CLIENT_CACHE.setdefault(loop, {})
    key = (timeout,)
    client = per_loop.get(key)
    if client is None or client.closed:
        client = aiohttp.ClientSession(timeout=_client_timeout(timeout))
        per_loop[key] = client
    return client


async def aclose() -> None:
    """Close every shared HTTP session owned by the current event loop.

    Call once at application shutdown (exported as ``pyllym.aclose``).
    """
    loop = asyncio.get_running_loop()
    per_loop = _CLIENT_CACHE.pop(loop, {})
    for client in per_loop.values():
        if not client.closed:
            await client.close()


class Connection:
    """Routes one provider's requests through a shared per-loop ClientSession."""

    def __init__(self, provider: Provider, config: Configuration) -> None:
        self.provider = provider
        self.config = config

    @property
    def _client(self) -> aiohttp.ClientSession:
        # Resolved lazily inside the running loop.
        return _shared_client(self.config.request_timeout)

    def _url(self, url: str) -> str:
        # api_base is read fresh so providers with dynamic bases (e.g.
        # region changes) stay correct.
        return _merge_url(self.provider.api_base, url)

    @classmethod
    def basic(cls) -> aiohttp.ClientSession:
        """A bare session for ad-hoc requests (models.dev, URL attachments)."""
        return aiohttp.ClientSession(timeout=_client_timeout(30))

    async def aclose(self) -> None:
        """Close the shared session this connection routes through.

        Note the session may be shared with other connections; prefer the
        module-level :func:`aclose` at shutdown.
        """
        client = self._client
        if not client.closed:
            await client.close()

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

    def _request_kwargs(
        self,
        payload: Any,
        headers: Mapping[str, str] | None,
        multipart: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"headers": {**self.provider.headers, **(headers or {})}}
        if self.config.http_proxy:
            kwargs["proxy"] = self.config.http_proxy
        if payload is not None:
            if multipart:
                form = aiohttp.FormData()
                for name, value in (payload.get("data") or {}).items():
                    form.add_field(name, str(value))
                for name, (filename, blob) in (payload.get("files") or {}).items():
                    form.add_field(name, blob, filename=filename)
                kwargs["data"] = form
            else:
                kwargs["json"] = payload
        return kwargs

    async def _request(
        self,
        method: str,
        url: str,
        *,
        payload: Any = None,
        headers: Mapping[str, str] | None = None,
        multipart: bool = False,
    ) -> Response:
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            # Multipart bodies are single-use; rebuild the kwargs per attempt.
            kwargs = self._request_kwargs(payload, headers, multipart)
            try:
                async with self._client.request(method, self._url(url), **kwargs) as resp:
                    content = await resp.read()
                    wrapped = Response(
                        status=resp.status,
                        body=_parse_body(resp.headers.get("content-type", ""), content),
                        headers=dict(resp.headers),
                        content=content,
                    )
                if resp.status >= 400:
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
        kwargs = self._request_kwargs(payload, headers, multipart=False)
        async with self._client.post(self._url(url), **kwargs) as resp:
            if resp.status >= 400:
                content = await resp.read()
                body = _parse_body(resp.headers.get("content-type", ""), content)
                self._raise_for_response(
                    Response(status=resp.status, body=body, headers=dict(resp.headers))
                )
            async for chunk in resp.content.iter_any():
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
