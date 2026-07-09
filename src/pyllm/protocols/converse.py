"""The AWS Bedrock Converse API., composed here into one cohesive protocol class.

Requests are SigV4-signed by :class:`pyllm.providers.bedrock.Bedrock` via its
``sign_headers`` method, applied in :meth:`Converse.sync_response`. Streaming
uses the AWS event-stream binary framing; see :meth:`Converse.stream_url` /
:meth:`Converse.build_chunk` and the provider's streaming notes.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from .. import utils
from ..chunk import Chunk
from ..content import Content, RawContent
from ..errors import Error, UnsupportedAttachmentError
from ..message import Message
from ..model.info import Info
from ..protocol import Protocol
from ..thinking import Thinking
from ..tool_call import ToolCall

if TYPE_CHECKING:
    from ..tool import Tool

logger = __import__("logging").getLogger("pyllm")

SUPPORTED_DOCUMENT_FORMATS = frozenset(
    {"pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"}
)

DEFAULT_INPUT_SCHEMA = {"type": "object", "properties": {}, "required": []}


class Converse(Protocol):
    @staticmethod
    def reasoning_embedded(model: Info) -> bool:
        metadata = model.metadata or {}
        converse = metadata.get("converse") or {}
        reasoning_supported = converse.get("reasoningSupported") or {}
        return bool(reasoning_supported.get("embedded"))

    # --- endpoints -------------------------------------------------------------
    def completion_url(self) -> str:
        assert self.model is not None
        return f"/model/{self.model.id}/converse"

    def stream_url(self) -> str:
        assert self.model is not None
        return f"/model/{self.model.id}/converse-stream"

    # --- signed sync response --------------------------------------------------
    async def sync_response(
        self, payload: dict[str, Any], additional_headers: dict[str, str] | None = None
    ) -> Message:
        body = json.dumps(payload)
        url = self.completion_url()
        signed = self.provider.sign_headers("POST", url, body)
        headers = {**signed, **(additional_headers or {})}
        response = await self.connection.post(url, payload, headers=headers)
        return self.parse_completion_response(response)

    # --- render ----------------------------------------------------------------
    def render_payload(
        self,
        messages: list[Message],
        *,
        tools: dict[str, Tool] | None = None,
        temperature: float | None = None,
        model: Info | None = None,
        stream: bool = False,
        schema: Any = None,
        thinking: Any = None,
        citations: bool = False,
        tool_prefs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert model is not None
        tools = tools or {}
        if citations:
            self._warn_unsupported_citations(model)
        tool_prefs = tool_prefs or {}
        self._used_document_names: dict[str, int] = {}
        system_messages = [m for m in messages if m.role == "system"]
        chat_messages = [m for m in messages if m.role != "system"]
        payload: dict[str, Any] = {"messages": self._format_messages(chat_messages)}
        system_blocks = self._format_system(system_messages)
        if system_blocks:
            payload["system"] = system_blocks
        payload["inferenceConfig"] = self._format_inference_config(model, temperature)
        tool_config = self._format_tool_config(tools, tool_prefs)
        if tool_config:
            payload["toolConfig"] = tool_config
        additional_fields = self._format_additional_model_request_fields(thinking)
        if additional_fields:
            payload["additionalModelRequestFields"] = additional_fields
        output_config = self._build_output_config(schema)
        if output_config:
            payload["outputConfig"] = output_config
        return payload

    def _warn_unsupported_citations(self, model: Info) -> None:
        logger.warning(
            "pyllm does not support citations on Bedrock yet. Ignoring with_citations for %s.",
            model.id,
        )

    # --- parse -----------------------------------------------------------------
    def parse_completion_response(self, response: Any) -> Message:
        message = self._parse_completion_body(response.body, raw=response)
        if message is None:
            raise Error(response, "Empty or unparseable completion response")
        return message

    def _parse_completion_body(self, data: Any, *, raw: Any) -> Message | None:
        if not data:
            return None
        content_blocks = _dig(data, "output", "message", "content") or []
        usage = data.get("usage") or {}
        thinking_text, thinking_signature = self._parse_thinking(content_blocks)
        return Message(
            role="assistant",
            content=self._parse_text_content(content_blocks),
            thinking=Thinking.build(text=thinking_text, signature=thinking_signature),
            tool_calls=self._parse_tool_calls(content_blocks),
            input_tokens=_input_tokens(usage),
            output_tokens=usage.get("outputTokens"),
            cached_tokens=usage.get("cacheReadInputTokens"),
            cache_creation_tokens=usage.get("cacheWriteInputTokens"),
            thinking_tokens=_reasoning_tokens(usage),
            finish_reason=data.get("stopReason"),
            model_id=data.get("modelId"),
            raw=raw,
        )

    def _parse_text_content(self, content_blocks: list[dict[str, Any]]) -> str | None:
        text = "".join(b["text"] for b in content_blocks if isinstance(b.get("text"), str))
        return text or None

    def _parse_thinking(
        self, content_blocks: list[dict[str, Any]]
    ) -> tuple[str | None, str | None]:
        text = ""
        signature: str | None = None
        for block in content_blocks:
            chunk_text, chunk_signature = self._parse_reasoning_content_block(block)
            if chunk_text:
                text += chunk_text
            if signature is None:
                signature = chunk_signature
        return (text or None), signature

    def _parse_reasoning_content_block(
        self, block: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        reasoning_content = block.get("reasoningContent")
        if not isinstance(reasoning_content, dict):
            return None, None
        reasoning_text = reasoning_content.get("reasoningText") or {}
        text = reasoning_text.get("text") if isinstance(reasoning_text.get("text"), str) else None
        signature = (
            reasoning_text.get("signature")
            if isinstance(reasoning_text.get("signature"), str)
            else None
        )
        if signature is None and isinstance(reasoning_content.get("redactedContent"), str):
            signature = reasoning_content["redactedContent"]
        return text, signature

    def _parse_tool_calls(self, content_blocks: list[dict[str, Any]]) -> dict[str, ToolCall] | None:
        tool_calls: dict[str, ToolCall] = {}
        for block in content_blocks:
            tool_use = block.get("toolUse")
            if not tool_use:
                continue
            tool_call_id = tool_use.get("toolUseId")
            tool_calls[tool_call_id] = ToolCall(
                id=tool_call_id,
                name=tool_use.get("name"),
                arguments=tool_use.get("input") or {},
            )
        return tool_calls or None

    # --- message formatting ----------------------------------------------------
    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        tool_result_blocks: list[dict[str, Any]] = []
        for msg in messages:
            if msg.is_tool_result():
                tool_result_blocks.append(self._format_tool_result_block(msg))
                continue
            if tool_result_blocks:
                rendered.append({"role": "user", "content": tool_result_blocks})
                tool_result_blocks = []
            message = self._format_non_tool_message(msg)
            if message:
                rendered.append(message)
        if tool_result_blocks:
            rendered.append({"role": "user", "content": tool_result_blocks})
        return rendered

    def _format_non_tool_message(self, msg: Message) -> dict[str, Any] | None:
        content = self._format_message_content(msg)
        if not content:
            return None
        return {"role": self._format_role(msg.role), "content": content}

    def _format_message_content(self, msg: Message) -> list[dict[str, Any]]:
        if isinstance(msg._content, RawContent):
            raw = self._format_raw_content(msg._content)
            if msg.role == "assistant":
                return raw
            return self._sanitize_non_assistant_raw_blocks(raw)
        blocks: list[dict[str, Any]] = []
        thinking_block = self._format_thinking_block(msg.thinking)
        if msg.role == "assistant" and thinking_block:
            blocks.append(thinking_block)
        text_and_media = self._format_content(msg.content)
        if text_and_media:
            blocks.extend(text_and_media)
        if msg.is_tool_call():
            for tool_call in (msg.tool_calls or {}).values():
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments,
                        }
                    }
                )
        return blocks

    def _format_raw_content(self, content: RawContent) -> list[Any]:
        value = content.value
        return value if isinstance(value, list) else [value]

    def _sanitize_non_assistant_raw_blocks(self, blocks: list[Any]) -> list[Any]:
        return [
            block for block in blocks if isinstance(block, dict) and "reasoningContent" not in block
        ]

    def _format_role(self, role: str) -> str:
        return "assistant" if role == "assistant" else "user"

    def _format_system(self, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            out.extend(self._format_content(msg.content))
        return out

    def _format_inference_config(self, model: Info, temperature: float | None) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if temperature is not None:
            config["temperature"] = temperature
        return config

    # --- tool result formatting ------------------------------------------------
    def _format_tool_result_block(self, msg: Message) -> dict[str, Any]:
        return {
            "toolResult": {
                "toolUseId": msg.tool_call_id,
                "content": self._format_tool_result_content(msg._content),
            }
        }

    def _format_tool_result_content(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, RawContent):
            return self._format_raw_tool_result_content(content.value)
        if isinstance(content, (dict, list)):
            return [{"json": content}]
        if isinstance(content, Content):
            return self._format_content_tool_result_content(content)
        return [_text_tool_result_block(content)]

    def _format_content_tool_result_content(self, content: Content) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if content.text:
            blocks.append(_text_tool_result_block(content.text))
        for attachment in content.attachments:
            blocks.append(_text_tool_result_block(attachment.for_llm()))
        return blocks or [_text_tool_result_block(None)]

    def _format_raw_tool_result_content(self, raw_value: Any) -> list[dict[str, Any]]:
        blocks = raw_value if isinstance(raw_value, list) else [raw_value]
        normalized = [b for b in (self._normalize_tool_result_block(x) for x in blocks) if b]
        return normalized or [{"text": str(raw_value)}]

    def _normalize_tool_result_block(self, block: Any) -> dict[str, Any] | None:
        if not isinstance(block, dict):
            return None
        if any(key in block for key in ("text", "json", "document", "image")):
            return block
        return None

    # --- media -----------------------------------------------------------------
    def _format_content(self, content: Any) -> list[dict[str, Any]]:
        if _empty_content(content):
            return []
        if isinstance(content, RawContent):
            return self._format_raw_content(content)
        if isinstance(content, (dict, list)):
            return [{"text": json.dumps(content)}]
        if not isinstance(content, Content):
            return [{"text": content}]
        return self._format_content_object(content)

    def _format_content_object(self, content: Content) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if content.text:
            blocks.append({"text": content.text})
        for attachment in content.attachments:
            blocks.append(self._format_attachment(attachment))
        return blocks

    def _format_attachment(self, attachment: Any) -> dict[str, Any]:
        if attachment.is_provider_file():
            return self._format_provider_file_attachment(attachment)
        kind = attachment.type
        if kind == "image":
            return self._format_image_attachment(attachment)
        if kind in ("pdf", "document"):
            return self._format_document_attachment(attachment)
        if kind == "text":
            return {"text": attachment.for_llm()}
        raise UnsupportedAttachmentError(attachment.mime_type)

    def _format_image_attachment(self, attachment: Any) -> dict[str, Any]:
        return {
            "image": {
                "format": attachment.format,
                "source": {"bytes": attachment.encoded},
            }
        }

    def _format_document_attachment(self, attachment: Any) -> dict[str, Any]:
        if not _supported_document_format(attachment):
            raise UnsupportedAttachmentError(attachment.mime_type)
        document_name = self._unique_document_name(_sanitize_document_name(attachment.filename))
        return {
            "document": {
                "format": _document_format(attachment),
                "name": document_name,
                "source": {"bytes": attachment.encoded},
            }
        }

    def _format_provider_file_attachment(self, attachment: Any) -> dict[str, Any]:
        if not _supported_document_format(attachment):
            raise UnsupportedAttachmentError(attachment.mime_type)
        document_name = self._unique_document_name(_sanitize_document_name(attachment.filename))
        return {
            "document": {
                "format": _document_format(attachment),
                "name": document_name,
                "source": {
                    "s3Location": {
                        "uri": attachment.provider_file_uri or attachment.provider_file_id
                    }
                },
            }
        }

    def _unique_document_name(self, base_name: str) -> str:
        used = getattr(self, "_used_document_names", {})
        count = used.get(base_name, 0)
        used[base_name] = count + 1
        if count == 0:
            return base_name
        return f"{base_name}_{count + 1}"

    # --- tools -----------------------------------------------------------------
    def _format_tool_config(
        self, tools: dict[str, Tool], tool_prefs: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not tools:
            return None
        config: dict[str, Any] = {"tools": [self._format_tool(t) for t in tools.values()]}
        if tool_prefs.get("choice") is None:
            return config
        tool_choice = self._format_tool_choice(tool_prefs["choice"])
        if tool_choice:
            config["toolChoice"] = tool_choice
        return config

    def _format_tool_choice(self, choice: str) -> dict[str, Any] | None:
        if choice == "auto":
            return {"auto": {}}
        if choice == "none":
            return None
        if choice == "required":
            return {"any": {}}
        return {"tool": {"name": str(choice)}}

    def _format_tool(self, tool: Tool) -> dict[str, Any]:
        input_schema = tool.params_schema or DEFAULT_INPUT_SCHEMA
        tool_spec: dict[str, Any] = {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": input_schema},
            }
        }
        if not tool.provider_params:
            return tool_spec
        return utils.deep_merge(tool_spec, tool.provider_params)

    # --- thinking --------------------------------------------------------------
    def _format_additional_model_request_fields(self, thinking: Any) -> dict[str, Any] | None:
        reasoning_fields = self._format_reasoning_fields(thinking)
        return reasoning_fields or None

    def _format_reasoning_fields(self, thinking: Any) -> dict[str, Any] | None:
        if not (thinking and getattr(thinking, "enabled", False)):
            return None
        effort_config = self._effort_reasoning_config(thinking)
        if effort_config:
            return effort_config
        return self._budget_reasoning_config(thinking)

    def _effort_reasoning_config(self, thinking: Any) -> dict[str, Any] | None:
        effort = str(thinking.effort or "")
        if effort == "" or effort == "none":
            return None
        assert self.model is not None
        if self.reasoning_embedded(self.model):
            return {"reasoning_config": {"type": "enabled", "reasoning_effort": effort}}
        return {"reasoning_effort": effort}

    def _budget_reasoning_config(self, thinking: Any) -> dict[str, Any] | None:
        budget = thinking.budget
        if not isinstance(budget, int):
            return None
        return {"reasoning_config": {"type": "enabled", "budget_tokens": budget}}

    def _format_thinking_block(self, thinking: Any) -> dict[str, Any] | None:
        if not thinking:
            return None
        if thinking.text:
            reasoning_text = utils.compact({"text": thinking.text, "signature": thinking.signature})
            return {"reasoningContent": {"reasoningText": reasoning_text}}
        if thinking.signature:
            return {"reasoningContent": {"redactedContent": thinking.signature}}
        return None

    # --- structured output -----------------------------------------------------
    def _build_output_config(self, schema: dict[str, Any] | None) -> dict[str, Any] | None:
        if not schema:
            return None
        cleaned = utils.deep_dup(schema["schema"])
        if isinstance(cleaned, dict):
            cleaned.pop("strict", None)
        return {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {"schema": json.dumps(cleaned), "name": schema.get("name")}
                },
            }
        }

    # --- streaming -------------------------------------------------------------
    def build_chunk(self, event: dict[str, Any]) -> Chunk:
        if _stream_error_event(event):
            _raise_stream_error(event)
        metadata_usage = _dig(event, "metadata", "usage") or {}
        usage = event.get("usage") or {}
        return Chunk(
            role="assistant",
            model_id=event.get("modelId") or (self.model.id if self.model else None),
            content=self._extract_content_delta(event),
            thinking=Thinking.build(
                text=self._extract_thinking_delta(event),
                signature=self._extract_thinking_signature(event),
            ),
            tool_calls=self._extract_tool_calls_stream(event),
            input_tokens=self._extract_input_tokens(metadata_usage, usage),
            output_tokens=metadata_usage.get("outputTokens") or usage.get("outputTokens"),
            cached_tokens=metadata_usage.get("cacheReadInputTokens")
            or usage.get("cacheReadInputTokens"),
            cache_creation_tokens=metadata_usage.get("cacheWriteInputTokens")
            or usage.get("cacheWriteInputTokens"),
            thinking_tokens=_reasoning_tokens(metadata_usage) or _reasoning_tokens(usage),
            finish_reason=_dig(event, "messageStop", "stopReason") or event.get("stopReason"),
        )

    def _extract_input_tokens(
        self, metadata_usage: dict[str, Any], usage: dict[str, Any]
    ) -> int | None:
        bedrock_usage = metadata_usage if metadata_usage.get("inputTokens") else usage
        if bedrock_usage.get("inputTokens"):
            return _input_tokens(bedrock_usage)
        return None

    def _extract_content_delta(self, event: dict[str, Any]) -> Any:
        return self._normalized_delta(event).get("text")

    def _extract_thinking_delta(self, event: dict[str, Any]) -> Any:
        reasoning_content = self._normalized_delta(event).get("reasoningContent") or {}
        reasoning_text = reasoning_content.get("reasoningText") or {}
        return reasoning_text.get("text") or reasoning_content.get("text")

    def _extract_thinking_signature(self, event: dict[str, Any]) -> str | None:
        reasoning_content = self._normalized_delta(event).get("reasoningContent") or {}
        reasoning_text = reasoning_content.get("reasoningText") or {}
        signature = reasoning_text.get("signature") or reasoning_content.get("signature")
        if signature:
            return signature
        start = _dig(event, "contentBlockStart", "start", "reasoningContent")
        if not start:
            return None
        start_reasoning_text = start.get("reasoningText") or {}
        return start_reasoning_text.get("signature") or start.get("redactedContent")

    def _extract_tool_calls_stream(self, event: dict[str, Any]) -> dict[Any, ToolCall] | None:
        if event.get("contentBlockStart") or event.get("start"):
            return self._extract_tool_call_start(event)
        if event.get("contentBlockDelta") or _dig(event, "delta", "toolUse"):
            return self._extract_tool_call_delta(event)
        return None

    def _extract_tool_call_start(self, event: dict[str, Any]) -> dict[Any, ToolCall] | None:
        tool_use = _dig(event, "contentBlockStart", "start", "toolUse") or _dig(
            event, "start", "toolUse"
        )
        if not tool_use:
            return None
        tool_use_id = tool_use.get("toolUseId")
        return {
            tool_use_id: ToolCall(
                id=tool_use_id, name=tool_use.get("name"), arguments=tool_use.get("input") or {}
            )
        }

    def _extract_tool_call_delta(self, event: dict[str, Any]) -> dict[Any, ToolCall] | None:
        input_ = _dig(self._normalized_delta(event), "toolUse", "input")
        if input_ is None:
            return None
        return {None: ToolCall(id=None, name=None, arguments=input_)}

    def _normalized_delta(self, event: dict[str, Any]) -> dict[str, Any]:
        delta = _dig(event, "contentBlockDelta", "delta") or event.get("delta") or {}
        if isinstance(delta, dict):
            return delta
        if isinstance(delta, str) and delta:
            try:
                return json.loads(delta)
            except json.JSONDecodeError:
                return {}
        return {}


# --- module helpers ------------------------------------------------------------
_dig = utils.dig


def _input_tokens(usage: dict[str, Any]) -> int | None:
    input_tokens = usage.get("inputTokens")
    if input_tokens is None:
        return None
    cache_read = usage.get("cacheReadInputTokens") or 0
    cache_write = usage.get("cacheWriteInputTokens") or 0
    return max(int(input_tokens) - int(cache_read) - int(cache_write), 0)


def _reasoning_tokens(usage: dict[str, Any]) -> int | None:
    return usage.get("reasoningTokens") or _dig(usage, "outputTokensDetails", "reasoningTokens")


def _empty_content(content: Any) -> bool:
    if content is None:
        return True
    return hasattr(content, "__len__") and len(content) == 0


def _text_tool_result_block(text: Any) -> dict[str, str]:
    text = "" if text is None else str(text)
    if text == "":
        text = "(no output)"
    return {"text": text}


def _document_format(attachment: Any) -> str:
    return attachment.extension or attachment.format


def _supported_document_format(attachment: Any) -> bool:
    return _document_format(attachment) in SUPPORTED_DOCUMENT_FORMATS


def _sanitize_document_name(filename: Any) -> str:
    base = os.path.splitext(os.path.basename(str(filename or "")))[0]
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in base)
    return safe or "document"


def _stream_error_event(event: dict[str, Any]) -> bool:
    return any(key.endswith("Exception") for key in event) or event.get("type") == "error"


def _raise_stream_error(event: dict[str, Any]) -> None:
    from ..errors import error_for_status

    if event.get("type") == "error":
        message = _dig(event, "error", "message") or "Bedrock streaming error"
        raise error_for_status(500)(None, message)
    key = next((k for k in event if k.endswith("Exception")), None)
    payload = event.get(key) or {} if key else {}
    message = payload.get("message") or key or "Bedrock streaming error"
    status = {
        "throttlingException": 429,
        "validationException": 400,
        "accessDeniedException": 401,
        "unrecognizedClientException": 401,
        "serviceUnavailableException": 503,
    }.get(key or "", 500)
    raise error_for_status(status)(None, message)
