"""Streaming support.

An async SSE parser over the aiohttp byte stream. Concrete protocols mix in
:class:`StreamingMixin` and implement ``stream_url`` and ``build_chunk``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from .errors import Error
from .stream_accumulator import StreamAccumulator

if TYPE_CHECKING:
    from .message import Message

logger = logging.getLogger("pyllm")

OnChunk = Callable[[Any], Any]


async def iter_sse(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[tuple[str, str]]:
    """Parse an SSE byte stream into ``(event_type, data)`` pairs.

    Concatenates multi-line ``data:`` fields and flushes on blank lines, the
    same framing ``event_stream_parser`` implements.
    """
    buffer = ""
    event_type = "message"
    data_lines: list[str] = []
    async for raw in byte_iter:
        buffer += raw.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line == "":
                if data_lines:
                    yield event_type, "\n".join(data_lines)
                    data_lines = []
                event_type = "message"
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].lstrip())
            # other fields (id:, retry:) are ignored
    if data_lines:
        yield event_type, "\n".join(data_lines)


class StreamingMixin:
    """Mixin providing the streaming completion flow for a protocol."""

    connection: Any
    config: Any

    async def stream_response(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
        on_chunk: OnChunk | None = None,
    ) -> Message:
        accumulator = StreamAccumulator()
        async for chunk in self._iter_chunks(payload, headers):
            accumulator.add(chunk)
            if on_chunk is not None:
                result = on_chunk(chunk)
                if hasattr(result, "__await__"):
                    await result  # type: ignore[misc]
        message = accumulator.to_message(None)
        logger.debug("Stream completed: %s", message.content)
        return message

    async def iter_stream(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> AsyncIterator[Any]:
        """Yield each parsed chunk (for ``async for chunk in ...``)."""
        async for chunk in self._iter_chunks(payload, headers):
            yield chunk

    async def _iter_chunks(
        self,
        payload: dict[str, Any],
        headers: dict[str, str] | None,
    ) -> AsyncIterator[Any]:
        raw_parts: list[bytes] = []

        async def capture(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
            async for part in source:
                raw_parts.append(part)
                yield part

        byte_iter = self.connection.stream(self.stream_url(), payload, headers=headers)
        saw_event = False
        async for event_type, data in iter_sse(capture(byte_iter)):
            saw_event = True
            if event_type == "error":
                self._raise_stream_error(data)
                continue
            if data == "[DONE]":
                continue
            parsed = self._safe_json(data)
            if parsed is None:
                continue
            if isinstance(parsed, dict) and "error" in parsed:
                self._raise_stream_error(data)
                continue
            yield self.build_chunk(parsed)
        if not saw_event:
            # A 200 response with a bare (non-SSE) JSON error body yields zero
            # SSE events; surface it instead of returning an empty message.
            body = b"".join(raw_parts).decode("utf-8", errors="replace").strip()
            if body.startswith("{") and '"error"' in body:
                self._raise_stream_error(body)

    @staticmethod
    def _safe_json(data: str) -> Any:
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            logger.debug("Failed to parse data chunk: %s", exc)
            return None

    def _raise_stream_error(self, data: str) -> None:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = {"message": data}
        message = None
        error_type = ""
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict):
                message = err.get("message")
                error_type = str(err.get("type") or err.get("code") or "")
            elif isinstance(err, str):
                message = err
            message = message or parsed.get("message")
        raise self._stream_error_class(error_type)(None, message or f"Streaming error: {data}")

    @staticmethod
    def _stream_error_class(error_type: str) -> type[Error]:
        # Map in-stream error types to the same taxonomy as HTTP statuses.
        from .errors import OverloadedError, RateLimitError, ServerError

        key = error_type.lower()
        if "overloaded" in key:
            return OverloadedError
        if "rate_limit" in key or "insufficient_quota" in key:
            return RateLimitError
        if "server_error" in key:
            return ServerError
        return Error

    # Implemented by concrete protocols:
    def stream_url(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def build_chunk(self, data: dict[str, Any]) -> Any:  # pragma: no cover - interface
        raise NotImplementedError
