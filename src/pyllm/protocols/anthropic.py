"""The Anthropic Messages API."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .. import utils
from ..chunk import Chunk
from ..citation import Citation
from ..content import Content, RawContent
from ..errors import Error, UnsupportedAttachmentError
from ..message import Message
from ..model.info import Info
from ..protocol import Protocol
from ..search_results import SearchResults
from ..thinking import Thinking
from ..tool_call import ToolCall

if TYPE_CHECKING:
    from ..tool import Tool

DEFAULT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
    "strict": True,
}


class Anthropic(Protocol):
    # --- endpoints -------------------------------------------------------------
    def completion_url(self) -> str:
        return "v1/messages"

    def stream_url(self) -> str:
        return self.completion_url()

    def models_url(self) -> str:
        return "v1/models"

    def embedding_url(self, *, model: str | None = None) -> str:  # pragma: no cover
        raise Error("Anthropic doesn't support embeddings")

    async def embed(self, text: Any, *, model: str, dimensions: int | None) -> Any:
        raise Error("Anthropic doesn't support embeddings")

    # --- render ----------------------------------------------------------------
    def render_payload(
        self,
        messages: list[Message],
        *,
        tools: dict[str, Tool],
        temperature: float | None,
        model: Info,
        stream: bool = False,
        schema: Any = None,
        thinking: Any = None,
        citations: bool = False,
        tool_prefs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool_prefs = tool_prefs or {}
        system_messages = [m for m in messages if m.role == "system"]
        chat_messages = [m for m in messages if m.role != "system"]
        system_content = self._build_system_content(system_messages)
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": [
                self._format_message(m, thinking=thinking, citations=citations)
                for m in chat_messages
            ],
            "stream": stream,
            "max_tokens": model.max_tokens or 4096,
        }
        self._add_thinking_fields(payload, thinking, model)
        if tools:
            payload["tools"] = [self._function_for(t) for t in tools.values()]
            if tool_prefs.get("choice") is not None or tool_prefs.get("calls") is not None:
                payload["tool_choice"] = self._build_tool_choice(tool_prefs)
        if system_content:
            payload["system"] = system_content
        if temperature is not None:
            payload["temperature"] = temperature
        if schema:
            payload.setdefault("output_config", {}).update(self._build_output_config(schema))
        return payload

    def _build_system_content(self, system_messages: list[Message]) -> list[Any]:
        out: list[Any] = []
        for msg in system_messages:
            content = msg.content
            if isinstance(content, RawContent):
                out.extend(utils.to_safe_array(content.value))
            else:
                formatted = self._format_content(content)
                out.extend(formatted if isinstance(formatted, list) else [formatted])
        return out

    def _build_output_config(self, schema: dict[str, Any]) -> dict[str, Any]:
        normalized = utils.deep_dup(schema["schema"])
        if isinstance(normalized, dict):
            normalized.pop("strict", None)
        return {"format": {"type": "json_schema", "schema": normalized}}

    # --- parse -----------------------------------------------------------------
    def parse_completion_response(self, response: Any) -> Message:
        data = response.body
        blocks = data.get("content") or []
        text_content, citations = self._extract_text_and_citations(blocks)
        thinking_content = self._extract_thinking_content(blocks)
        thinking_signature = self._extract_thinking_signature(blocks)
        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        usage = data.get("usage") or {}
        thinking_tokens = (
            _dig(usage, "output_tokens_details", "thinking_tokens")
            or _dig(usage, "output_tokens_details", "reasoning_tokens")
            or usage.get("thinking_tokens")
            or usage.get("reasoning_tokens")
        )
        return Message(
            role="assistant",
            content=text_content,
            citations=citations,
            thinking=Thinking.build(text=thinking_content, signature=thinking_signature),
            tool_calls=self._parse_tool_calls(tool_uses),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_tokens=_extract_cached_tokens(data),
            cache_creation_tokens=_extract_cache_creation_tokens(data),
            thinking_tokens=thinking_tokens,
            finish_reason=data.get("stop_reason"),
            model_id=data.get("model"),
            raw=response,
        )

    def _extract_text_and_citations(
        self, blocks: list[dict[str, Any]]
    ) -> tuple[str, list[Citation]]:
        text = ""
        citations: list[Citation] = []
        for block in blocks:
            if block.get("type") != "text":
                continue
            block_text = str(block.get("text") or "")
            for citation in utils.to_safe_array(block.get("citations")):
                citations.append(
                    _parse_citation(
                        citation,
                        text=block_text,
                        start_index=len(text),
                        end_index=len(text) + len(block_text),
                    )
                )
            text += block_text
        return text, citations

    def _extract_thinking_content(self, blocks: list[dict[str, Any]]) -> str | None:
        thoughts = "".join(
            b.get("thinking") or b.get("text") or "" for b in blocks if b.get("type") == "thinking"
        )
        return thoughts or None

    def _extract_thinking_signature(self, blocks: list[dict[str, Any]]) -> str | None:
        block = next((b for b in blocks if b.get("type") == "thinking"), None) or next(
            (b for b in blocks if b.get("type") == "redacted_thinking"), None
        )
        if not block:
            return None
        return block.get("signature") or block.get("data")

    # --- message formatting ----------------------------------------------------
    def _format_message(
        self, msg: Message, *, thinking: Any = None, citations: bool = False
    ) -> dict[str, Any]:
        thinking_enabled = bool(thinking and getattr(thinking, "enabled", False))
        if msg.is_tool_call():
            return self._format_tool_call(msg, thinking_enabled)
        if msg.is_tool_result():
            return self._format_tool_result(msg)
        return self._format_basic_message(msg, thinking_enabled, citations=citations)

    def _format_basic_message(
        self, msg: Message, thinking_enabled: bool, *, citations: bool
    ) -> dict[str, Any]:
        blocks: list[Any] = []
        if msg.role == "assistant" and thinking_enabled:
            block = _build_thinking_block(msg.thinking)
            if block:
                blocks.append(block)
        self._append_formatted_content(blocks, msg.content, citations=citations)
        return {"role": _convert_role(msg.role), "content": blocks}

    def _format_tool_call(self, msg: Message, thinking_enabled: bool) -> dict[str, Any]:
        if isinstance(msg._content, RawContent):
            value = msg._content.value
            blocks = value if isinstance(value, list) else [value]
            blocks = _prepend_thinking(list(blocks), msg, thinking_enabled)
            return {"role": "assistant", "content": blocks}
        blocks = _prepend_thinking([], msg, thinking_enabled)
        if msg.content is not None and msg.content != "":
            self._append_formatted_content(blocks, msg.content)
        for tool_call in (msg.tool_calls or {}).values():
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
            )
        return {"role": "assistant", "content": blocks}

    def _format_tool_result(self, msg: Message) -> dict[str, Any]:
        if isinstance(msg._content, RawContent):
            return {"role": "user", "content": msg._content.value}
        content = msg.content
        if content is None or (hasattr(content, "__len__") and len(content) == 0):
            content = "(no output)"
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": self._format_tool_result_content(content),
                }
            ],
        }

    def _format_tool_result_content(self, content: Any) -> Any:
        if isinstance(content, SearchResults):
            return [
                {
                    "type": "search_result",
                    "source": r.get("url") or r.get("title"),
                    "title": r.get("title"),
                    "content": [{"type": "text", "text": r.get("text")}],
                    "citations": {"enabled": True},
                }
                for r in content.results
            ]
        return self._format_content(content)

    def _append_formatted_content(
        self, blocks: list[Any], content: Any, *, citations: bool = False
    ) -> None:
        formatted = self._format_content(content, citations=citations)
        if isinstance(formatted, list):
            blocks.extend(formatted)
        else:
            blocks.append(formatted)

    # --- media -----------------------------------------------------------------
    def _format_content(self, content: Any, *, citations: bool = False) -> Any:
        if isinstance(content, RawContent):
            return content.value
        if isinstance(content, (dict, list)):
            return [{"type": "text", "text": json.dumps(content)}]
        if not isinstance(content, Content):
            return [{"type": "text", "text": content}]
        parts: list[dict[str, Any]] = []
        if content.text:
            parts.append({"type": "text", "text": content.text})
        for attachment in content.attachments:
            if attachment.is_provider_file():
                parts.append(self._format_provider_file(attachment, citations=citations))
                continue
            kind = attachment.type
            if kind == "image":
                parts.append(self._format_image(attachment))
            elif kind == "pdf":
                parts.append(self._format_pdf(attachment, citations=citations))
            elif kind == "text":
                parts.append(
                    self._format_text_document(attachment)
                    if citations
                    else {"type": "text", "text": attachment.for_llm()}
                )
            else:
                raise UnsupportedAttachmentError(attachment.mime_type)
        return parts

    def _format_image(self, image: Any) -> dict[str, Any]:
        if image.is_url():
            return {"type": "image", "source": {"type": "url", "url": str(image.source)}}
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": image.mime_type, "data": image.encoded},
        }

    def _format_provider_file(self, file: Any, *, citations: bool = False) -> dict[str, Any]:
        if file.is_image():
            return {"type": "image", "source": {"type": "file", "file_id": file.provider_file_id}}
        document = {
            "type": "document",
            "source": {"type": "file", "file_id": file.provider_file_id},
        }
        if citations:
            _enable_citations(document, file)
        return document

    def _format_pdf(self, pdf: Any, *, citations: bool = False) -> dict[str, Any]:
        if pdf.is_url():
            source = {"type": "url", "url": str(pdf.source)}
        else:
            source = {"type": "base64", "media_type": pdf.mime_type, "data": pdf.encoded}
        document = {"type": "document", "source": source}
        if citations:
            _enable_citations(document, pdf)
        return document

    def _format_text_document(self, text_file: Any) -> dict[str, Any]:
        document = {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text_file.content},
        }
        _enable_citations(document, text_file)
        return document

    # --- tools -----------------------------------------------------------------
    def _function_for(self, tool: Tool) -> dict[str, Any]:
        declaration = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.params_schema or DEFAULT_INPUT_SCHEMA,
        }
        if not tool.provider_params:
            return declaration
        return utils.deep_merge(declaration, tool.provider_params)

    def _parse_tool_calls(self, content_blocks: Any) -> dict[str, ToolCall] | None:
        if content_blocks is None:
            return None
        if not isinstance(content_blocks, list):
            content_blocks = [content_blocks]
        tool_calls: dict[str, ToolCall] = {}
        for block in content_blocks:
            if not block or block.get("type") != "tool_use":
                continue
            tool_calls[block["id"]] = ToolCall(
                id=block["id"], name=block.get("name"), arguments=block.get("input")
            )
        return tool_calls or None

    def _build_tool_choice(self, tool_prefs: dict[str, Any]) -> dict[str, Any]:
        choice = tool_prefs.get("choice") or "auto"
        calls = tool_prefs.get("calls")
        if choice in ("auto", "none"):
            type_ = choice
        elif choice == "required":
            type_ = "any"
        else:
            type_ = "tool"
        result: dict[str, Any] = {"type": type_}
        if type_ == "tool":
            result["name"] = choice
        if type_ != "none" and calls is not None:
            result["disable_parallel_tool_use"] = calls == "one"
        return result

    # --- thinking --------------------------------------------------------------
    def _add_thinking_fields(self, payload: dict[str, Any], thinking: Any, model: Info) -> None:
        thinking_payload = self._build_thinking_payload(thinking, model)
        if not thinking_payload:
            return
        if thinking_payload.get("thinking"):
            payload["thinking"] = thinking_payload["thinking"]
        if thinking_payload.get("output_config"):
            payload.setdefault("output_config", {}).update(thinking_payload["output_config"])

    def _build_thinking_payload(self, thinking: Any, model: Info) -> dict[str, Any] | None:
        if not (thinking and getattr(thinking, "enabled", False)):
            return None
        effort = str(thinking.effort) if thinking.effort else None
        if effort == "none":
            return None
        budget = thinking.budget
        if budget:
            if model.reasoning_option("budget_tokens"):
                return {"thinking": {"type": "enabled", "budget_tokens": budget}}
            raise ValueError(f"Anthropic thinking budget is not supported for {model.id}")
        if effort is None:
            raise ValueError("Anthropic adaptive thinking requires an effort")
        if model.reasoning_option("effort"):
            return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
        raise ValueError(f"Anthropic thinking effort is not supported for {model.id}")

    # --- streaming -------------------------------------------------------------
    def build_chunk(self, data: dict[str, Any]) -> Chunk:
        delta_type = _dig(data, "delta", "type")
        return Chunk(
            role="assistant",
            model_id=_dig(data, "message", "model"),
            content=_dig(data, "delta", "text") if delta_type == "text_delta" else None,
            citations=self._extract_citations_delta(data, delta_type),
            thinking=Thinking.build(
                text=_dig(data, "delta", "thinking") if delta_type == "thinking_delta" else None,
                signature=_dig(data, "delta", "signature")
                if delta_type == "signature_delta"
                else None,
            ),
            input_tokens=_dig(data, "message", "usage", "input_tokens"),
            output_tokens=_dig(data, "message", "usage", "output_tokens")
            or _dig(data, "usage", "output_tokens"),
            cached_tokens=_extract_cached_tokens(data),
            cache_creation_tokens=_extract_cache_creation_tokens(data),
            tool_calls=self._extract_tool_calls_stream(data),
            finish_reason=_dig(data, "delta", "stop_reason"),
        )

    def _extract_citations_delta(
        self, data: dict[str, Any], delta_type: str | None
    ) -> list[Citation] | None:
        if delta_type != "citations_delta":
            return None
        citation = _dig(data, "delta", "citation")
        return [_parse_citation(citation)] if citation else None

    def _extract_tool_calls_stream(self, data: dict[str, Any]) -> dict[Any, ToolCall] | None:
        if (
            data.get("type") == "content_block_delta"
            and _dig(data, "delta", "type") == "input_json_delta"
        ):
            return {
                data["index"]: ToolCall(
                    id=None, name=None, arguments=_dig(data, "delta", "partial_json")
                )
            }  # type: ignore[arg-type]
        if data.get("type") == "content_block_start":
            tool_calls = self._parse_tool_calls(data.get("content_block"))
            if tool_calls is None or data.get("index") is None:
                return tool_calls
            return {data["index"]: next(iter(tool_calls.values()))}
        return self._parse_tool_calls(data.get("content_block"))

    # --- models ----------------------------------------------------------------
    def parse_list_models_response(self, response: Any, slug: str, capabilities: Any) -> list[Info]:
        out = []
        for model_data in response.body.get("data") or []:
            model_id = model_data["id"]
            out.append(
                Info(
                    {
                        "id": model_id,
                        "name": model_data.get("display_name") or model_id,
                        "provider": slug,
                        "created_at": model_data.get("created_at"),
                        "capabilities": capabilities.critical_capabilities_for(model_id)
                        if capabilities
                        else [],
                        "metadata": {},
                    }
                )
            )
        return out


# --- helpers -------------------------------------------------------------------
_dig = utils.dig


def _convert_role(role: str) -> str:
    return "user" if role in ("tool", "user") else "assistant"


def _build_thinking_block(thinking: Any) -> dict[str, Any] | None:
    if not thinking:
        return None
    if thinking.text:
        block = {"type": "thinking", "thinking": thinking.text, "signature": thinking.signature}
        return {k: v for k, v in block.items() if v is not None}
    if thinking.signature:
        return {"type": "redacted_thinking", "data": thinking.signature}
    return None


def _prepend_thinking(blocks: list[Any], msg: Message, thinking_enabled: bool) -> list[Any]:
    if not thinking_enabled:
        return blocks
    block = _build_thinking_block(msg.thinking)
    if block:
        blocks.insert(0, block)
    return blocks


def _enable_citations(document: dict[str, Any], attachment: Any) -> None:
    if attachment.filename:
        document["title"] = attachment.filename
    document["citations"] = {"enabled": True}


def _parse_citation(
    data: dict[str, Any],
    *,
    text: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
) -> Citation:
    end_page = data.get("end_page_number")
    url = data.get("url") or data.get("source")
    if not (isinstance(url, str) and url.startswith(("http://", "https://"))):
        url = None
    return Citation(
        url=url,
        title=data.get("document_title") or data.get("title"),
        cited_text=data.get("cited_text"),
        text=text,
        start_index=start_index,
        end_index=end_index,
        source_index=data.get("document_index") or data.get("search_result_index"),
        start_page=data.get("start_page_number"),
        end_page=(end_page - 1) if end_page else None,
    )


def _extract_cached_tokens(data: dict[str, Any]) -> int | None:
    return _dig(data, "message", "usage", "cache_read_input_tokens") or _dig(
        data, "usage", "cache_read_input_tokens"
    )


def _extract_cache_creation_tokens(data: dict[str, Any]) -> int | None:
    direct = _dig(data, "message", "usage", "cache_creation_input_tokens") or _dig(
        data, "usage", "cache_creation_input_tokens"
    )
    if direct:
        return direct
    breakdown = _dig(data, "message", "usage", "cache_creation") or _dig(
        data, "usage", "cache_creation"
    )
    if isinstance(breakdown, dict):
        return sum(v for v in breakdown.values() if v)
    return None
