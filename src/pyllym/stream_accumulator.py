"""Assembles streaming chunks into a complete message."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from .citation import Citation
from .thinking import Thinking
from .tokens import Tokens
from .tool_call import ToolCall

logger = logging.getLogger("pyllym")


class StreamAccumulator:
    def __init__(self) -> None:
        self.content = ""
        self._citations: list[Citation] = []
        self._thinking_text = ""
        self._thinking_signature: str | None = None
        self.tool_calls: dict[Any, ToolCall] = {}
        self.model_id: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._cached_tokens: int | None = None
        self._cache_creation_tokens: int | None = None
        self._thinking_tokens: int | None = None
        self._finish_reason: Any = None
        self._inside_think_tag = False
        self._pending_think_tag = ""
        self._latest_tool_call_id: Any = None
        self._tool_call_ids_by_index: dict[Any, Any] = {}

    def add(self, chunk: Any) -> None:
        if self.model_id is None:
            self.model_id = chunk.model_id
        self._handle_chunk_content(chunk)
        self._accumulate_citations(chunk.citations)
        self._append_thinking_from_chunk(chunk)
        if chunk.finish_reason:
            self._finish_reason = chunk.finish_reason
        self._count_tokens(chunk)

    def to_message(self, response: Any) -> Any:
        from .message import Message

        self._flush_pending_think_tag()
        return Message(
            role="assistant",
            content=self.content or None,
            citations=self._resolved_citations(),
            thinking=Thinking.build(
                text=self._thinking_text or None,
                signature=self._thinking_signature,
            ),
            tokens=Tokens.build(
                input=self._input_tokens,
                output=self._output_tokens,
                cached=self._cached_tokens,
                cache_creation=self._cache_creation_tokens,
                thinking=self._thinking_tokens,
            ),
            finish_reason=self._finish_reason,
            model_id=self.model_id,
            tool_calls=self._tool_calls_from_stream() or None,
            raw=response,
        )

    def _flush_pending_think_tag(self) -> None:
        # A trailing prefix of '<think>'/'</think>' held back at end of stream
        # is real text, not a tag — flush it so content isn't silently dropped.
        pending, self._pending_think_tag = self._pending_think_tag, ""
        if not pending:
            return
        if self._inside_think_tag:
            self._thinking_text += pending
        else:
            self.content += pending

    # --- citations -------------------------------------------------------------
    def _accumulate_citations(self, new_citations: list[Citation]) -> None:
        for citation in new_citations or []:
            if citation not in self._citations:
                self._citations.append(citation)

    def _resolved_citations(self) -> list[Citation]:
        return [self._resolve_citation_text(c) for c in self._citations]

    def _resolve_citation_text(self, citation: Citation) -> Citation:
        if citation.text or citation.start_index is None or citation.end_index is None:
            return citation
        span = self.content[citation.start_index : citation.end_index]
        if span:
            return Citation.from_dict({**citation.to_dict(), "text": span})
        return citation

    # --- tool calls ------------------------------------------------------------
    def _tool_calls_from_stream(self) -> dict[Any, ToolCall]:
        result: dict[Any, ToolCall] = {}
        for key, tc in self.tool_calls.items():
            args = tc.arguments
            if isinstance(args, str) and args:
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(
                        "Discarding malformed streamed tool-call arguments for %s: %r",
                        tc.name,
                        args[:200],
                    )
                    args = {}
            elif isinstance(args, str):
                args = {}
            result[key] = ToolCall(
                id=tc.id, name=tc.name, arguments=args, thought_signature=tc.thought_signature
            )
        return result

    def _accumulate_tool_calls(self, new_tool_calls: dict[Any, ToolCall]) -> None:
        for stream_key, tool_call in new_tool_calls.items():
            if tool_call.id:
                self._start_tool_call(stream_key, tool_call)
            else:
                self._append_tool_call_fragment(stream_key, tool_call)

    def _start_tool_call(self, stream_key: Any, tool_call: ToolCall) -> None:
        tool_call_id = tool_call.id or str(uuid.uuid4())
        tool_call_key = tool_call.id
        self.tool_calls[tool_call_key] = ToolCall(
            id=tool_call_id,
            name=tool_call.name,
            arguments=self._initial_tool_call_arguments(tool_call),
            thought_signature=tool_call.thought_signature,
        )
        if stream_key is not None:
            self._tool_call_ids_by_index[stream_key] = tool_call_key
        self._latest_tool_call_id = tool_call_key

    @staticmethod
    def _initial_tool_call_arguments(tool_call: ToolCall) -> Any:
        args = tool_call.arguments
        if args is None or (hasattr(args, "__len__") and len(args) == 0):
            return ""
        return args

    def _append_tool_call_fragment(self, stream_key: Any, tool_call: ToolCall) -> None:
        existing = self._find_tool_call(stream_key)
        if existing is None:
            return
        fragment = tool_call.arguments or ""
        if isinstance(existing.arguments, str):
            existing.arguments += fragment
        if tool_call.thought_signature and existing.thought_signature is None:
            existing.thought_signature = tool_call.thought_signature

    def _find_tool_call(self, stream_key: Any) -> ToolCall | None:
        if stream_key is None:
            return self.tool_calls.get(self._latest_tool_call_id)
        mapped = self._tool_call_ids_by_index.get(stream_key)
        return self.tool_calls.get(mapped) or self.tool_calls.get(stream_key)

    # --- tokens & content ------------------------------------------------------
    def _count_tokens(self, chunk: Any) -> None:
        if chunk.input_tokens:
            self._input_tokens = chunk.input_tokens
        if chunk.output_tokens:
            self._output_tokens = chunk.output_tokens
        if chunk.cached_tokens:
            self._cached_tokens = chunk.cached_tokens
        if chunk.cache_creation_tokens:
            self._cache_creation_tokens = chunk.cache_creation_tokens
        if chunk.thinking_tokens:
            self._thinking_tokens = chunk.thinking_tokens

    def _handle_chunk_content(self, chunk: Any) -> None:
        if chunk.is_tool_call():
            self._accumulate_tool_calls(chunk.tool_calls)
            return
        content_text = chunk.content or ""
        if isinstance(content_text, str):
            self._append_text_with_thinking(content_text)
        else:
            self.content += str(content_text)

    def _append_text_with_thinking(self, text: str) -> None:
        content_chunk, thinking_chunk = self._extract_think_tags(text)
        self.content += content_chunk
        if thinking_chunk:
            self._thinking_text += thinking_chunk

    def _append_thinking_from_chunk(self, chunk: Any) -> None:
        thinking = chunk.thinking
        if not thinking:
            return
        if thinking.text:
            self._thinking_text += str(thinking.text)
        if self._thinking_signature is None:
            self._thinking_signature = thinking.signature

    def _extract_think_tags(self, text: str) -> tuple[str, str | None]:
        start_tag = "<think>"
        end_tag = "</think>"
        remaining = self._pending_think_tag + text
        self._pending_think_tag = ""
        output = ""
        thinking = ""
        while remaining:
            if self._inside_think_tag:
                remaining, thinking = self._consume_think_content(remaining, end_tag, thinking)
            else:
                remaining, output = self._consume_non_think_content(remaining, start_tag, output)
        return output, (thinking or None)

    def _consume_think_content(
        self, remaining: str, end_tag: str, thinking: str
    ) -> tuple[str, str]:
        end_index = remaining.find(end_tag)
        if end_index != -1:
            thinking += remaining[:end_index]
            self._inside_think_tag = False
            return remaining[end_index + len(end_tag) :], thinking
        suffix_len = self._longest_suffix_prefix(remaining, end_tag)
        thinking += remaining[: len(remaining) - suffix_len]
        self._pending_think_tag = remaining[len(remaining) - suffix_len :] if suffix_len else ""
        return "", thinking

    def _consume_non_think_content(
        self, remaining: str, start_tag: str, output: str
    ) -> tuple[str, str]:
        start_index = remaining.find(start_tag)
        if start_index != -1:
            output += remaining[:start_index]
            self._inside_think_tag = True
            return remaining[start_index + len(start_tag) :], output
        suffix_len = self._longest_suffix_prefix(remaining, start_tag)
        output += remaining[: len(remaining) - suffix_len]
        self._pending_think_tag = remaining[len(remaining) - suffix_len :] if suffix_len else ""
        return "", output

    @staticmethod
    def _longest_suffix_prefix(text: str, tag: str) -> int:
        max_len = min(len(text), len(tag) - 1)
        for length in range(max_len, 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0
