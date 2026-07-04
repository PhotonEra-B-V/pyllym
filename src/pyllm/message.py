"""A single message in a conversation."""

from __future__ import annotations

import enum
from typing import Any

from . import utils
from .citation import Citation
from .content import Content
from .errors import InvalidRoleError, ModelNotFoundError
from .thinking import Thinking
from .tokens import Tokens


class Role(enum.StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLES = frozenset(r.value for r in Role)

_STOPPED = {"stop", "end_turn", "stop_sequence"}
_MAX_TOKENS = {"length", "max_tokens", "max_output_tokens", "model_context_window_exceeded"}
_TOOL_CALL = {"tool_calls", "tool_use", "function_call"}
_CONTENT_FILTERED = {
    "blocklist",
    "content_filter",
    "content_filtered",
    "guardrail_intervened",
    "image_recitation",
    "image_safety",
    "model_armor",
    "prohibited_content",
    "recitation",
    "safety",
    "spii",
}


class Message:
    def __init__(self, options: dict[str, Any] | None = None, **kwargs: Any) -> None:
        opts = {**(options or {}), **kwargs}
        if "role" not in opts:
            raise KeyError("role")
        self.role = str(opts["role"])
        self.tool_calls: dict[str, Any] | None = opts.get("tool_calls")
        self._content = self._normalize_content(
            opts.get("content"), role=self.role, tool_calls=self.tool_calls
        )
        self.model_id: str | None = opts.get("model_id")
        self.tool_call_id: str | None = opts.get("tool_call_id")
        self.tokens: Tokens | None = opts.get("tokens") or Tokens.build(
            input=opts.get("input_tokens"),
            output=opts.get("output_tokens"),
            cached=opts.get("cached_tokens"),
            cache_creation=opts.get("cache_creation_tokens"),
            thinking=opts.get("thinking_tokens"),
        )
        self.raw = opts.get("raw")
        self.thinking: Thinking | None = opts.get("thinking")
        self.citations: list[Citation] = utils.to_safe_array(opts.get("citations"))
        self.finish_reason = opts.get("finish_reason")
        self._model_info: Any = None
        self._model_info_loaded = False
        self._ensure_valid_role()

    @property
    def content(self) -> Any:
        return self._content.format() if isinstance(self._content, Content) else self._content

    @content.setter
    def content(self, value: Any) -> None:
        self._content = value

    def is_tool_call(self) -> bool:
        return bool(self.tool_calls)

    def is_tool_result(self) -> bool:
        return bool(self.tool_call_id)

    @property
    def tool_results(self) -> Any:
        return self.content if self.is_tool_result() else None

    def is_stopped(self) -> bool:
        return self._finish_reason_in(_STOPPED)

    def is_max_tokens(self) -> bool:
        return self._finish_reason_in(_MAX_TOKENS)

    def is_tool_call_stop(self) -> bool:
        return self._finish_reason_in(_TOOL_CALL)

    def is_content_filtered(self) -> bool:
        return self._finish_reason_in(_CONTENT_FILTERED)

    @property
    def input_tokens(self) -> int | None:
        return self.tokens.input if self.tokens else None

    @property
    def output_tokens(self) -> int | None:
        return self.tokens.output if self.tokens else None

    @property
    def cached_tokens(self) -> int | None:
        return self.tokens.cached if self.tokens else None

    @property
    def cache_creation_tokens(self) -> int | None:
        return self.tokens.cache_creation if self.tokens else None

    @property
    def cache_read_tokens(self) -> int | None:
        return self.tokens.cache_read if self.tokens else None

    @property
    def cache_write_tokens(self) -> int | None:
        return self.tokens.cache_write if self.tokens else None

    @property
    def thinking_tokens(self) -> int | None:
        return self.tokens.thinking if self.tokens else None

    def cost(self, model: Any = None) -> Any:
        from .cost import Cost

        return Cost(tokens=self.tokens, model=model or self.model_info)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "model_id": self.model_id,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "thinking": self.thinking.text if self.thinking else None,
            "thinking_signature": self.thinking.signature if self.thinking else None,
            "citations": [c.to_dict() for c in self.citations] if self.citations else None,
            "finish_reason": self.finish_reason,
        }
        if self.tokens:
            data.update(self.tokens.to_dict())
        return {k: v for k, v in data.items() if v is not None}

    @property
    def model_info(self) -> Any:
        if not self.model_id:
            return None
        if not self._model_info_loaded:
            self._model_info_loaded = True
            try:
                from . import models as _models

                self._model_info = _models.find(self.model_id)
            except ModelNotFoundError:
                self._model_info = None
        return self._model_info

    # --- internals -------------------------------------------------------------
    def _finish_reason_in(self, reasons: set[str]) -> bool:
        return self._finish_reason_key() in reasons

    def _finish_reason_key(self) -> str:
        return str(self.finish_reason or "").lower().replace("-", "_")

    @staticmethod
    def _normalize_content(content: Any, *, role: str, tool_calls: Any) -> Any:
        if role == "assistant" and content is None and tool_calls:
            return ""
        if isinstance(content, str):
            return Content(content)
        if isinstance(content, dict):
            # The 'text' entry is the message text, not an attachment source.
            attachments = {k: v for k, v in content.items() if k != "text"}
            return Content(content.get("text"), attachments or None)
        return content

    def _ensure_valid_role(self) -> None:
        if self.role not in ROLES:
            raise InvalidRoleError(f"Expected role to be one of: {', '.join(sorted(ROLES))}")
