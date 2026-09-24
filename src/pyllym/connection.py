"""Async HTTP connection over aiohttp: session config plus an error-raising
response hook and a small retry loop.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import random
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, NoReturn

import aiohttp

from .errors import (
    ConnectionFailedError,
    Error,
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
# Any transport-layer failure; always surfaced as ConnectionFailedError so
# callers only ever need to catch pyllym.errors.Error.
_TRANSPORT_EXC = (TimeoutError, aiohttp.ClientError)
_RETRY_ERRORS = (RateLimitError, ServerError, ServiceUnavailableError, OverloadedError)

# Go-style durations used by OpenAI's ``x-ratelimit-reset-*`` headers
# ("1s", "6m0s", "250ms", "1h2m3.5s").
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


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


def _parse_duration(value: str) -> float | None:
    """Seconds in a Go-style duration string, or None if it is not one."""
    parts = _DURATION_RE.findall(value.strip())
    if not parts or "".join(n + u for n, u in parts) != value.strip():
        return None
    return sum(float(n) * _DURATION_UNITS[u] for n, u in parts)


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """How long the server asked us to wait before retrying, in seconds.

    Understands ``retry-after-ms``, ``retry-after`` (delay-seconds or an
    HTTP-date) and, failing those, OpenAI-style ``x-ratelimit-reset-requests``
    / ``x-ratelimit-reset-tokens`` window resets. Those two arrive on every
    response, so only the window whose ``x-ratelimit-remaining-*`` counter is
    at zero counts (the later one if both are); when the remaining counters
    are absent the sooner reset is used, since an early retry merely costs one
    attempt while a late one wastes the wait. Returns None when the server
    gave no usable hint.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    ms = lowered.get("retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    value = lowered.get("retry-after")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                when = None
            if when is not None:
                if when.tzinfo is None:
                    when = when.replace(tzinfo=UTC)
                return max(0.0, (when - datetime.now(UTC)).total_seconds())
    exhausted: list[float] = []
    resets: list[float] = []
    for kind in ("requests", "tokens"):
        raw = lowered.get(f"x-ratelimit-reset-{kind}")
        seconds = _parse_duration(raw) if raw else None
        if seconds is None:
            continue
        resets.append(seconds)
        remaining = lowered.get(f"x-ratelimit-remaining-{kind}")
        if remaining is not None and remaining.strip() == "0":
            exhausted.append(seconds)
    if exhausted:
        return max(exhausted)
    if resets and not any(f"x-ratelimit-remaining-{k}" in lowered for k in ("requests", "tokens")):
        return min(resets)
    return None


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
#
# The key is id(loop), never the loop object -- not even weakly. A
# ClientSession reaches its loop through its TCPConnector, so a
# WeakKeyDictionary keyed on the loop keeps its own key alive through its
# value: the entry is never evicted, and every asyncio.run() strands a loop,
# a session and the connector's resolver thread. Entries are instead evicted
# by _evict_on_close, which hooks the loop's own close(); since id() is only
# unique among live objects, an entry must never outlive its loop.
_CLIENT_CACHE: dict[int, dict[tuple[Any, ...], aiohttp.ClientSession]] = {}


def _current_event_loop() -> asyncio.AbstractEventLoop | None:
    """The thread's current event loop, or None if it has none set."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        # No loop set for this thread (the usual case under asyncio.run).
        return None


def _evict_on_close(loop: asyncio.AbstractEventLoop) -> None:
    """Close and drop a loop's cached sessions when the loop shuts down.

    Wraps ``loop.close()``, which ``asyncio.run`` calls once the loop has
    stopped but before it is discarded. This is the only workable hook: the
    cache entry keeps the loop alive (a session reaches its loop through its
    connector), so a weakref finalizer on the loop would wait forever on a
    collection the entry itself prevents.

    Sessions are closed here rather than merely dropped, so aiohttp does not
    emit "Unclosed client session" from ``__del__``. Callers that want an
    orderly async shutdown should still await :func:`aclose`.
    """
    loop_id = id(loop)
    original = loop.close

    def close() -> None:
        per_loop = _CLIENT_CACHE.pop(loop_id, {})
        pending = [c for c in per_loop.values() if not c.closed]
        if pending and not loop.is_closed():
            # The loop has stopped but is not yet closed, so it can still run
            # these (non-blocking) close coroutines to completion. asyncio.run
            # has already detached the loop from the thread by now, so it is
            # reinstated for the call -- gather() and friends resolve the
            # current loop implicitly and would fail without it.
            previous = _current_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    asyncio.gather(*(c.close() for c in pending), return_exceptions=True)
                )
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("failed to close pooled HTTP sessions", exc_info=True)
            finally:
                asyncio.set_event_loop(previous)
        original()

    loop.close = close  # type: ignore[method-assign]


def _shared_client(timeout: Any) -> aiohttp.ClientSession:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    per_loop = _CLIENT_CACHE.get(loop_id)
    if per_loop is None:
        per_loop = {}
        _CLIENT_CACHE[loop_id] = per_loop
        _evict_on_close(loop)
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
    per_loop = _CLIENT_CACHE.pop(id(asyncio.get_running_loop()), None)
    for client in (per_loop or {}).values():
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
            except _TRANSPORT_EXC as exc:
                # Non-retryable transport failure (bad URL, TLS, protocol).
                raise ConnectionFailedError.wrap(exc) from exc
            if not await self._wait_before_retry(attempt, last_exc, method, url):
                break
        self._reraise(last_exc)

    async def stream(
        self,
        url: str,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield raw response chunks for SSE streaming.

        Errors (non-200) are raised after reading the error body.

        Retryable failures (429/5xx, connection errors) are retried with the
        same policy as :meth:`post` as long as no body bytes have been yielded
        yet; once the stream has started it cannot be replayed, so a dropped
        stream surfaces as :class:`ConnectionFailedError`.
        """
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            kwargs = self._request_kwargs(payload, headers, multipart=False)
            started = False
            try:
                async with self._client.post(self._url(url), **kwargs) as resp:
                    if resp.status >= 400:
                        content = await resp.read()
                        body = _parse_body(resp.headers.get("content-type", ""), content)
                        self._raise_for_response(
                            Response(status=resp.status, body=body, headers=dict(resp.headers))
                        )
                    async for chunk in resp.content.iter_any():
                        started = True
                        yield chunk
                return
            except _RETRY_EXC as exc:
                if started:
                    raise ConnectionFailedError.wrap(exc) from exc
                last_exc = exc
            except _RETRY_ERRORS as exc:
                last_exc = exc
            except _TRANSPORT_EXC as exc:
                raise ConnectionFailedError.wrap(exc) from exc
            if not await self._wait_before_retry(attempt, last_exc, "POST", url):
                break
        self._reraise(last_exc)

    def _raise_for_response(self, response: Response) -> None:
        message = self.provider.parse_error(response)
        error_cls = error_for_status(response.status, message if isinstance(message, str) else None)
        error = error_cls(response, message)
        error.retry_after = retry_after_seconds(response.headers)
        raise error

    @staticmethod
    def _reraise(last_exc: Exception | None) -> NoReturn:
        assert last_exc is not None
        if isinstance(last_exc, Error):
            raise last_exc
        raise ConnectionFailedError.wrap(last_exc) from last_exc

    async def _wait_before_retry(
        self, attempt: int, exc: Exception | None, method: str, url: str
    ) -> bool:
        """Sleep out the backoff for ``attempt``; False means give up instead."""
        if attempt >= self.config.max_retries:
            return False
        delay = self._retry_delay(attempt, exc)
        if delay is None:
            logger.warning(
                "Not retrying %s %s: server asked to wait %.0fs, above retry_max_interval",
                method,
                url,
                getattr(exc, "retry_after", 0.0),
            )
            return False
        logger.info(
            "Retrying %s %s in %.2fs after %s (attempt %d/%d)",
            method,
            url,
            delay,
            type(exc).__name__,
            attempt + 1,
            self.config.max_retries,
        )
        await self._sleep(delay)
        return True

    @staticmethod
    async def _sleep(seconds: float) -> None:
        await asyncio.sleep(seconds)

    def _retry_delay(self, attempt: int, exc: Exception | None) -> float | None:
        """Seconds to wait before retry number ``attempt + 1``.

        A server-supplied ``Retry-After`` wins over the computed backoff; when
        it exceeds ``retry_max_interval`` the request is not retried at all
        (returns None), since retrying sooner would only be refused again.
        Rate limits without a hint back off from ``rate_limit_retry_interval``
        rather than ``retry_interval``: the latter is tuned for transient
        server blips and is far too short for a throttling window.
        """
        cap = self.config.retry_max_interval
        jitter_ratio = self.config.retry_interval_randomness * random.random()
        hint = getattr(exc, "retry_after", None)
        if hint is not None:
            if cap is not None and hint > cap:
                return None
            # A touch of jitter so a fleet released at once does not stampede.
            return hint + self.config.retry_interval * jitter_ratio
        base = (
            self.config.rate_limit_retry_interval
            if isinstance(exc, RateLimitError)
            else self.config.retry_interval
        )
        interval = base * (self.config.retry_backoff_factor**attempt)
        delay = interval + interval * jitter_ratio
        return min(delay, cap) if cap is not None else delay


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
