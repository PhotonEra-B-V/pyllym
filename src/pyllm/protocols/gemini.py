"""The Google Gemini ``generateContent`` API., composed here into one
cohesive protocol class.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from .. import utils
from ..attachment import Attachment
from ..chunk import Chunk
from ..citation import Citation
from ..content import Content, RawContent
from ..embedding import Embedding
from ..errors import UnsupportedAttachmentError
from ..message import Message
from ..model.info import Info
from ..protocol import Protocol
from ..thinking import Thinking
from ..tool_call import ToolCall

if TYPE_CHECKING:
    from ..tool import Tool

logger = logging.getLogger("pyllm")


class Gemini(Protocol):
    # --- endpoints -------------------------------------------------------------
    def completion_url(self) -> str:
        assert self.model is not None
        return f"models/{self.model.id}:generateContent"

    def stream_url(self) -> str:
        assert self.model is not None
        return f"models/{self.model.id}:streamGenerateContent?alt=sse"

    def models_url(self) -> str:
        return "models"

    def embedding_url(self, *, model: str | None = None) -> str:
        return f"models/{model}:batchEmbedContents"

    # --- chat render -----------------------------------------------------------
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
        if citations and not model.is_citations():
            self._warn_unsupported_citations(model)
        tool_prefs = tool_prefs or {}
        payload: dict[str, Any] = {
            "contents": self._format_messages(messages),
            "generationConfig": {},
        }
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        if schema:
            payload["generationConfig"].update(self._structured_output_config(schema, model))
        if thinking and getattr(thinking, "enabled", False):
            payload["generationConfig"]["thinkingConfig"] = self._build_thinking_config(
                model, thinking
            )
        if tools:
            payload["tools"] = self._format_tools(tools)
            # Gemini doesn't support controlling parallel tool calls.
            if tool_prefs.get("choice") is not None:
                payload["toolConfig"] = self._build_tool_config(tool_prefs["choice"])
        return payload

    def _warn_unsupported_citations(self, model: Info) -> None:
        logger.warning(
            "%s does not support citations according to the model registry. "
            "Gemini citations come from Google Search grounding: "
            "with_params(tools=[{'google_search': {}}]).",
            model.id,
        )

    def _build_thinking_config(self, model: Info, thinking: Any) -> dict[str, Any]:
        config: dict[str, Any] = {"includeThoughts": True}
        if thinking.effort:
            config["thinkingLevel"] = thinking.effort
        if isinstance(thinking.budget, int):
            config["thinkingBudget"] = thinking.budget
        return config

    # --- chat parse ------------------------------------------------------------
    def parse_completion_response(self, response: Any) -> Message:
        return self._parse_completion_body(response.body, raw=response)

    def _parse_completion_body(self, data: Any, *, raw: Any) -> Message:
        parts = _dig(data, "candidates", 0, "content", "parts") or []
        tool_calls = self._extract_tool_calls(data)
        content = self._parse_content(data)
        return Message(
            role="assistant",
            content=content,
            citations=self._extract_citations(data, content),
            thinking=Thinking.build(
                text=self._extract_thought_parts(parts),
                signature=self._extract_thought_signature(parts),
            ),
            tool_calls=tool_calls,
            input_tokens=self._input_tokens(data),
            output_tokens=self._calculate_output_tokens(data),
            cached_tokens=_dig(data, "usageMetadata", "cachedContentTokenCount"),
            thinking_tokens=_dig(data, "usageMetadata", "thoughtsTokenCount"),
            finish_reason=_dig(data, "candidates", 0, "finishReason"),
            model_id=data.get("modelVersion") or (self.model.id if self.model else None),
            raw=raw,
        )

    def _input_tokens(self, data: dict[str, Any]) -> int | None:
        prompt_tokens = _dig(data, "usageMetadata", "promptTokenCount")
        if prompt_tokens is None:
            return None
        cached = _dig(data, "usageMetadata", "cachedContentTokenCount") or 0
        return max(int(prompt_tokens) - int(cached), 0)

    def _calculate_output_tokens(self, data: dict[str, Any]) -> int:
        candidates = _dig(data, "usageMetadata", "candidatesTokenCount") or 0
        thoughts = _dig(data, "usageMetadata", "thoughtsTokenCount") or 0
        return int(candidates) + int(thoughts)

    def _parse_content(self, data: dict[str, Any]) -> Any:
        candidate = _dig(data, "candidates", 0)
        if not candidate:
            return ""
        if self._function_call(candidate):
            return ""
        parts = _dig(candidate, "content", "parts")
        if not parts:
            return ""
        non_thought_parts = [p for p in parts if not p.get("thought")]
        if not non_thought_parts:
            return ""
        return self._build_response_content(non_thought_parts)

    def _function_call(self, candidate: dict[str, Any]) -> bool:
        parts = _dig(candidate, "content", "parts")
        return bool(parts) and any(p.get("functionCall") for p in parts)

    # --- grounding citations ---------------------------------------------------
    def _extract_citations(self, data: dict[str, Any], content: Any) -> list[Citation]:
        metadata = _dig(data, "candidates", 0, "groundingMetadata")
        if not metadata:
            return []
        chunks = metadata.get("groundingChunks") or []
        supports = metadata.get("groundingSupports") or []
        if not supports:
            return self._chunk_citations(chunks)
        out: list[Citation] = []
        for support in supports:
            out.extend(self._support_citations(support, chunks, content))
        return out

    def _support_citations(
        self, support: dict[str, Any], chunks: list[Any], content: Any
    ) -> list[Citation]:
        segment = support.get("segment") or {}
        end_index = segment.get("endIndex")
        start_index = segment.get("startIndex")
        if start_index is None and end_index is not None:
            start_index = 0
        out: list[Citation] = []
        for index in utils.to_safe_array(support.get("groundingChunkIndices")):
            source = self._chunk_source(chunks[index] if 0 <= index < len(chunks) else None)
            if not source:
                continue
            out.append(
                Citation(
                    url=source.get("uri"),
                    title=source.get("title"),
                    text=segment.get("text"),
                    start_index=_byte_to_char_index(content, start_index),
                    end_index=_byte_to_char_index(content, end_index),
                    source_index=index,
                )
            )
        return out

    def _chunk_citations(self, chunks: list[Any]) -> list[Citation]:
        out: list[Citation] = []
        for index, chunk in enumerate(chunks):
            source = self._chunk_source(chunk)
            if not source:
                continue
            out.append(
                Citation(url=source.get("uri"), title=source.get("title"), source_index=index)
            )
        return out

    def _chunk_source(self, chunk: Any) -> dict[str, Any] | None:
        if not isinstance(chunk, dict):
            return None
        return chunk.get("web") or chunk.get("retrievedContext")

    # --- thinking extraction ---------------------------------------------------
    def _extract_thought_parts(self, parts: list[dict[str, Any]]) -> str | None:
        thoughts = "".join(p["text"] for p in parts if p.get("thought") and p.get("text"))
        return thoughts or None

    def _extract_thought_signature(self, parts: list[dict[str, Any]]) -> str | None:
        for part in parts:
            signature = (
                part.get("thoughtSignature")
                or part.get("thought_signature")
                or _dig(part, "functionCall", "thoughtSignature")
                or _dig(part, "functionCall", "thought_signature")
            )
            if signature:
                return signature
        return None

    # --- messages --------------------------------------------------------------
    def _format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        return _MessageFormatter(self, messages).format()

    def _format_role(self, role: str) -> str:
        if role == "assistant":
            return "model"
        if role in ("system", "tool"):
            return "user"
        return role

    def _format_parts(self, msg: Message) -> list[dict[str, Any]]:
        if msg.is_tool_call():
            return self._format_tool_call(msg)
        if msg.is_tool_result():
            return self._format_tool_result(msg)
        return self._format_message_parts(msg)

    def _format_message_parts(self, msg: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if msg.role == "assistant" and msg.thinking:
            parts.append(self._build_thought_part(msg.thinking))
        content_parts = self._format_content(msg.content)
        parts.extend(content_parts if isinstance(content_parts, list) else [content_parts])
        return parts

    def _build_thought_part(self, thinking: Any) -> dict[str, Any]:
        part: dict[str, Any] = {"thought": True}
        if thinking.text:
            part["text"] = thinking.text
        if thinking.signature:
            part["thoughtSignature"] = thinking.signature
        return part

    # --- media -----------------------------------------------------------------
    def _format_content(self, content: Any) -> Any:
        if isinstance(content, RawContent):
            return content.value
        if isinstance(content, (dict, list)):
            return [{"text": json.dumps(content)}]
        if not isinstance(content, Content):
            return [{"text": content}]
        parts: list[dict[str, Any]] = []
        if content.text:
            parts.append({"text": content.text})
        for attachment in content.attachments:
            parts.append(self._format_content_attachment(attachment))
        return parts

    def _format_content_attachment(self, attachment: Any) -> dict[str, Any]:
        kind = attachment.type
        if kind == "text":
            return {"text": attachment.for_llm()}
        if kind in ("document", "unknown"):
            raise UnsupportedAttachmentError(attachment.mime_type)
        return self._format_attachment(attachment)

    def _format_attachment(self, attachment: Any) -> dict[str, Any]:
        if attachment.is_provider_file():
            return self._format_file_data(attachment)
        return {"inline_data": {"mime_type": attachment.mime_type, "data": attachment.encoded}}

    def _format_file_data(self, attachment: Any) -> dict[str, Any]:
        uri = attachment.provider_file_uri or attachment.provider_file_id
        return {"file_data": {"mime_type": attachment.mime_type, "file_uri": uri}}

    def _build_response_content(self, parts: list[dict[str, Any]]) -> Any:
        text: list[str] = []
        attachments: list[Attachment] = []
        for index, part in enumerate(parts):
            if part.get("text"):
                text.append(part["text"])
            elif part.get("inlineData"):
                attachment = self._build_inline_attachment(part["inlineData"], index)
                if attachment:
                    attachments.append(attachment)
            elif part.get("fileData"):
                attachment = self._build_file_attachment(part["fileData"], index)
                if attachment:
                    attachments.append(attachment)
        joined = "".join(text) or None
        if not attachments:
            return joined
        return Content(joined, attachments)

    def _build_inline_attachment(
        self, inline_data: dict[str, Any], index: int
    ) -> Attachment | None:
        encoded = inline_data.get("data")
        if not encoded:
            return None
        mime_type = inline_data.get("mimeType")
        try:
            decoded = base64.b64decode(encoded)
        except (ValueError, TypeError) as exc:
            logger.warning("Failed to decode Gemini inline data attachment: %s", exc)
            return None
        filename = _attachment_filename(mime_type, index)
        return Attachment(decoded, filename=filename)

    def _build_file_attachment(self, file_data: dict[str, Any], index: int) -> Attachment | None:
        uri = file_data.get("fileUri")
        if not uri:
            return None
        filename = file_data.get("filename") or _attachment_filename(
            file_data.get("mimeType"), index
        )
        return Attachment(uri, filename=filename)

    # --- tools -----------------------------------------------------------------
    def _format_tools(self, tools: dict[str, Tool]) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {"functionDeclarations": [self._function_declaration_for(t) for t in tools.values()]}
        ]

    def _format_tool_call(self, msg: Message) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if msg.content is not None and not (
            hasattr(msg.content, "__len__") and len(msg.content) == 0
        ):
            formatted = self._format_content(msg.content)
            parts.extend(formatted if isinstance(formatted, list) else [formatted])
        fallback_signature = msg.thinking.signature if msg.thinking else None
        used_fallback = False
        for tool_call in (msg.tool_calls or {}).values():
            part: dict[str, Any] = {
                "functionCall": {"name": tool_call.name, "args": tool_call.arguments}
            }
            signature = tool_call.thought_signature
            if signature is None and fallback_signature and not used_fallback:
                signature = fallback_signature
                used_fallback = True
            if signature:
                part["thoughtSignature"] = signature
            parts.append(part)
        return parts

    def _format_tool_result(
        self, msg: Message, function_name: str | None = None
    ) -> list[dict[str, Any]]:
        function_name = function_name or msg.tool_call_id
        content = msg.content
        if content is None or (hasattr(content, "__len__") and len(content) == 0):
            content = "(no output)"
        return [
            {
                "functionResponse": {
                    "name": function_name,
                    "response": {
                        "name": function_name,
                        "content": self._format_content(content),
                    },
                }
            }
        ]

    def _extract_tool_calls(self, data: Any) -> dict[str, ToolCall] | None:
        if not data:
            return None
        candidate = _dig(data, "candidates", 0) if isinstance(data, dict) else None
        if not candidate:
            return None
        parts = _dig(candidate, "content", "parts")
        if not isinstance(parts, list):
            return None
        tool_calls: dict[str, ToolCall] = {}
        for part in parts:
            function_data = part.get("functionCall")
            if not function_data:
                continue
            call_id = str(uuid.uuid4())
            thought_signature = part.get("thoughtSignature") or part.get("thought_signature")
            tool_calls[call_id] = ToolCall(
                id=call_id,
                name=function_data.get("name"),
                arguments=function_data.get("args") or {},
                thought_signature=thought_signature,
            )
        return tool_calls or None

    def _function_declaration_for(self, tool: Tool) -> dict[str, Any]:
        parameters_schema = tool.params_schema
        declaration: dict[str, Any] = {"name": tool.name, "description": tool.description}
        if parameters_schema:
            declaration["parameters"] = self._convert_tool_schema_to_gemini(parameters_schema)
        if not tool.provider_params:
            return declaration
        return utils.deep_merge(declaration, tool.provider_params)

    def _convert_tool_schema_to_gemini(
        self, schema: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not schema:
            return None
        schema = _deep_stringify_keys(schema)
        if schema.get("type") != "object":
            raise ValueError("Gemini tool parameters must be objects")
        return {
            "type": "OBJECT",
            "properties": {
                k: self._convert_property(v) for k, v in (schema.get("properties") or {}).items()
            },
            "required": [str(r) for r in (schema.get("required") or [])],
        }

    def _convert_property(self, property_schema: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_any_of_schema(property_schema)
        working = normalized or property_schema
        type_ = _param_type_for_gemini(working.get("type"))
        prop: dict[str, Any] = {"type": type_}
        _copy_common_attributes(prop, property_schema)
        _copy_common_attributes(prop, working)
        if type_ == "ARRAY":
            items_schema = (
                working.get("items") or property_schema.get("items") or {"type": "string"}
            )
            prop["items"] = self._convert_property(items_schema)
            _copy_attributes(prop, working, ("minItems", "maxItems"))
            _copy_attributes(prop, property_schema, ("minItems", "maxItems"))
        elif type_ == "OBJECT":
            nested = working.get("properties") or {}
            prop["properties"] = {k: self._convert_property(v) for k, v in nested.items()}
            required = working.get("required") or property_schema.get("required")
            if required is not None:
                prop["required"] = [str(r) for r in required]
        return prop

    def _normalize_any_of_schema(self, schema: dict[str, Any]) -> dict[str, Any] | None:
        any_of = schema.get("anyOf")
        if not isinstance(any_of, list) or not any_of:
            return None
        null_entries = [e for e in any_of if str(_schema_type(e)).lower() == "null"]
        non_null = [e for e in any_of if str(_schema_type(e)).lower() != "null"]
        if len(non_null) == 1 and null_entries:
            normalized = utils.deep_dup(non_null[0])
            normalized["nullable"] = True
            return normalized
        if non_null:
            return utils.deep_dup(non_null[0])
        return {"type": "string", "nullable": True}

    def _build_tool_config(self, tool_choice: str) -> dict[str, Any]:
        config: dict[str, Any] = {
            "mode": "any" if _forced_tool_choice(tool_choice) else tool_choice
        }
        if _specific_tool_choice(tool_choice):
            config["allowedFunctionNames"] = [tool_choice]
        return {"functionCallingConfig": config}

    # --- structured output -----------------------------------------------------
    def _structured_output_config(self, schema: dict[str, Any], model: Info) -> dict[str, Any]:
        config: dict[str, Any] = {"responseMimeType": "application/json"}
        if _response_json_schema_supported(model):
            config["responseJsonSchema"] = self._build_json_schema(schema)
        else:
            config["responseSchema"] = _GeminiSchema(schema.get("schema") or schema).to_dict()
        return config

    def _build_json_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        normalized = utils.deep_dup(schema["schema"])
        if isinstance(normalized, dict):
            normalized.pop("strict", None)
        return _deep_stringify_keys(normalized)

    # --- streaming -------------------------------------------------------------
    def build_chunk(self, data: dict[str, Any]) -> Chunk:
        parts = _dig(data, "candidates", 0, "content", "parts") or []
        return Chunk(
            role="assistant",
            model_id=data.get("modelVersion"),
            content=self._extract_text_content(parts),
            citations=self._extract_citations(data, None),
            thinking=Thinking.build(
                text=self._extract_thought_parts(parts),
                signature=self._extract_thought_signature(parts),
            ),
            input_tokens=self._input_tokens(data),
            output_tokens=self._extract_output_tokens(data),
            cached_tokens=_dig(data, "usageMetadata", "cachedContentTokenCount"),
            thinking_tokens=_dig(data, "usageMetadata", "thoughtsTokenCount"),
            finish_reason=_dig(data, "candidates", 0, "finishReason"),
            tool_calls=self._extract_tool_calls(data),
        )

    def _extract_text_content(self, parts: list[dict[str, Any]]) -> str | None:
        text = "".join(p["text"] for p in parts if not p.get("thought") and p.get("text"))
        return text or None

    def _extract_output_tokens(self, data: dict[str, Any]) -> int | None:
        candidates = _dig(data, "usageMetadata", "candidatesTokenCount") or 0
        thoughts = _dig(data, "usageMetadata", "thoughtsTokenCount") or 0
        total = int(candidates) + int(thoughts)
        return total if total > 0 else None

    # --- embeddings ------------------------------------------------------------
    def render_embedding_payload(
        self, text: Any, *, model: str, dimensions: int | None
    ) -> dict[str, Any]:
        texts = text if isinstance(text, list) else [text]
        return {
            "requests": [
                self._single_embedding_payload(t, model=model, dimensions=dimensions) for t in texts
            ]
        }

    def _single_embedding_payload(
        self, text: Any, *, model: str, dimensions: int | None
    ) -> dict[str, Any]:
        return utils.compact(
            {
                "model": f"models/{model}",
                "content": {"parts": [{"text": str(text)}]},
                "outputDimensionality": dimensions,
            }
        )

    def parse_embedding_response(self, response: Any, *, model: str, text: Any) -> Embedding:
        embeddings = response.body.get("embeddings") or []
        vectors: Any = [e.get("values") for e in embeddings]
        if len(vectors) == 1 and not isinstance(text, list):
            vectors = vectors[0]
        return Embedding(vectors=vectors, model=model, input_tokens=0)

    # --- models ----------------------------------------------------------------
    def parse_list_models_response(self, response: Any, slug: str, capabilities: Any) -> list[Info]:
        out: list[Info] = []
        for model_data in utils.to_safe_array(response.body.get("models")):
            model_id = str(model_data["name"]).replace("models/", "")
            out.append(
                Info(
                    {
                        "id": model_id,
                        "name": model_data.get("displayName") or model_id,
                        "provider": slug,
                        "created_at": None,
                        "context_window": model_data.get("inputTokenLimit")
                        or (capabilities.context_window_for(model_id) if capabilities else None),
                        "max_output_tokens": model_data.get("outputTokenLimit")
                        or (capabilities.max_tokens_for(model_id) if capabilities else None),
                        "capabilities": capabilities.critical_capabilities_for(model_id)
                        if capabilities
                        else [],
                        "pricing": capabilities.pricing_for(model_id) if capabilities else {},
                        "metadata": {
                            "version": model_data.get("version"),
                            "description": model_data.get("description"),
                            "supported_generation_methods": model_data.get(
                                "supportedGenerationMethods"
                            ),
                        },
                    }
                )
            )
        return out


# --- message formatting helper -------------------------------------------------
class _MessageFormatter:
    """Groups consecutive tool-result messages into a single ``user`` turn."""

    def __init__(self, protocol: Gemini, messages: list[Message]) -> None:
        self.protocol = protocol
        self.messages = messages
        self.index = 0
        self.tool_call_names: dict[str, str] = {}

    def format(self) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        while (current := self._current_message()) is not None:
            if self._is_tool_message(current):
                tool_parts, next_index = self._collect_tool_parts()
                formatted.append({"role": "user", "parts": tool_parts})
                self.index = next_index
            else:
                if current.is_tool_call():
                    self._remember_tool_calls(current)
                formatted.append(self._build_standard_message(current))
                self.index += 1
        return formatted

    def _current_message(self) -> Message | None:
        if 0 <= self.index < len(self.messages):
            return self.messages[self.index]
        return None

    def _is_tool_message(self, message: Message | None) -> bool:
        return message is not None and message.role == "tool"

    def _collect_tool_parts(self) -> tuple[list[dict[str, Any]], int]:
        parts: list[dict[str, Any]] = []
        index = self.index
        while index < len(self.messages) and self._is_tool_message(self.messages[index]):
            tool_message = self.messages[index]
            tool_name = (
                self.tool_call_names.pop(tool_message.tool_call_id, None)
                if tool_message.tool_call_id is not None
                else None
            )
            parts.extend(self.protocol._format_tool_result(tool_message, tool_name))
            index += 1
        return parts, index

    def _remember_tool_calls(self, message: Message) -> None:
        for tool_call_id, tool_call in (message.tool_calls or {}).items():
            self.tool_call_names[tool_call_id] = tool_call.name

    def _build_standard_message(self, message: Message) -> dict[str, Any]:
        return {
            "role": self.protocol._format_role(message.role),
            "parts": self.protocol._format_parts(message),
        }


# --- JSON Schema → Gemini schema converter -------------------------------------
class _GeminiSchema:
    """Converts a JSON Schema into Gemini's ``responseSchema`` dialect."""

    def __init__(self, schema: Any) -> None:
        self.raw_schema = utils.deep_dup(schema)
        self.definitions: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any] | None:
        if not self.raw_schema:
            return None
        extracted = self._extract_definitions(self.raw_schema)
        return self._convert(extracted, set())

    def _extract_definitions(self, value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, val in value.items():
                if key in ("$defs", "definitions"):
                    self._merge_definitions(val)
                else:
                    out[key] = self._extract_definitions(val)
            return out
        if isinstance(value, list):
            return [self._extract_definitions(item) for item in value]
        return value

    def _merge_definitions(self, raw_defs: Any) -> None:
        if not raw_defs:
            return
        extracted = self._extract_definitions(raw_defs)
        if not self.definitions:
            self.definitions = extracted
        else:
            self.definitions = utils.deep_merge(self.definitions, extracted)

    def _convert(self, schema: Any, visited_refs: set[str]) -> dict[str, Any]:
        if not isinstance(schema, dict):
            return {"type": "STRING"}
        schema = self._strip_unsupported_keys(schema)
        if schema.get("$ref"):
            resolved = self._resolve_reference(schema, visited_refs)
            if resolved is not None:
                return resolved
        schema = self._normalize_any_of(schema)
        type_ = str(schema.get("type") or "")
        if type_ == "object":
            result = self._build_object(schema, visited_refs)
        elif type_ == "array":
            result = self._build_array(schema, visited_refs)
        elif type_ == "number":
            result = self._build_scalar(
                "NUMBER", schema, ("format", "minimum", "maximum", "enum", "nullable", "multipleOf")
            )
        elif type_ == "integer":
            result = self._build_scalar(
                "INTEGER",
                schema,
                ("format", "minimum", "maximum", "enum", "nullable", "multipleOf"),
            )
        elif type_ == "boolean":
            result = self._build_scalar("BOOLEAN", schema, ("nullable",))
        else:
            result = self._build_scalar("STRING", schema, ("enum", "format", "nullable"))
        if schema.get("description"):
            result["description"] = schema["description"]
        return result

    def _strip_unsupported_keys(self, schema: dict[str, Any]) -> dict[str, Any]:
        copy = dict(schema)
        copy.pop("strict", None)
        copy.pop("additionalProperties", None)
        return copy

    def _resolve_reference(
        self, schema: dict[str, Any], visited_refs: set[str]
    ) -> dict[str, Any] | None:
        ref = schema.get("$ref")
        if not ref or ref in visited_refs:
            return None
        referenced = self._lookup_definition(ref)
        if referenced is None:
            return None
        overrides = {k: v for k, v in schema.items() if k != "$ref"}
        visited_refs.add(ref)
        try:
            merged = utils.deep_merge(referenced, overrides)
            return self._convert(merged, visited_refs)
        finally:
            visited_refs.discard(ref)

    def _lookup_definition(self, ref: str) -> Any:
        segments = [s for s in str(ref).split("/") if s]
        if not segments:
            return None
        if segments and segments[0] == "#":
            segments = segments[1:]
        if segments and segments[0] in ("$defs", "definitions"):
            segments = segments[1:]
        current: Any = self.definitions
        for segment in segments:
            if not isinstance(current, dict):
                return None
            current = current.get(segment)
        return utils.deep_dup(current) if current else None

    def _normalize_any_of(self, schema: dict[str, Any]) -> dict[str, Any]:
        any_of = schema.get("anyOf")
        if not any_of:
            return schema
        options = list(any_of)
        nullables = [o for o in options if _schema_type_lower(o) == "null"]
        non_null = [o for o in options if _schema_type_lower(o) != "null"]
        base = dict(non_null[0]) if non_null else {"type": "string"}
        if nullables:
            base["nullable"] = True
        without_any_of = {k: v for k, v in schema.items() if k != "anyOf"}
        without_any_of.update(base)
        return without_any_of

    def _build_object(self, schema: dict[str, Any], visited_refs: set[str]) -> dict[str, Any]:
        properties = {
            k: self._convert(v, visited_refs) for k, v in (schema.get("properties") or {}).items()
        }
        obj: dict[str, Any] = {"type": "OBJECT", "properties": properties}
        required = list(dict.fromkeys(str(r) for r in utils.to_safe_array(schema.get("required"))))
        if required:
            obj["required"] = required
        if schema.get("propertyOrdering"):
            obj["propertyOrdering"] = schema["propertyOrdering"]
        if "nullable" in schema:
            obj["nullable"] = schema["nullable"]
        return obj

    def _build_array(self, schema: dict[str, Any], visited_refs: set[str]) -> dict[str, Any]:
        items_schema = (
            self._convert(schema["items"], visited_refs)
            if schema.get("items")
            else {"type": "STRING"}
        )
        array: dict[str, Any] = {"type": "ARRAY", "items": items_schema}
        for key in ("minItems", "maxItems", "nullable"):
            if key in schema:
                array[key] = schema[key]
        return array

    def _build_scalar(
        self, type_: str, schema: dict[str, Any], allowed_keys: tuple[str, ...]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"type": type_}
        for key in allowed_keys:
            if key in schema:
                result[key] = schema[key]
        return result


# --- module helpers ------------------------------------------------------------
_dig = utils.dig


def _byte_to_char_index(content: Any, byte_index: int | None) -> int | None:
    if not isinstance(content, str) or byte_index is None:
        return None
    return len(content.encode("utf-8")[:byte_index].decode("utf-8", errors="ignore"))


def _attachment_filename(mime_type: str | None, index: int) -> str:
    if not mime_type:
        return f"gemini_attachment_{index + 1}"
    extension = str(mime_type.split("/")[-1])
    if extension == "jpeg":
        extension = "jpg"
    elif extension == "plain":
        extension = "txt"
    extension = extension.replace("+", ".")
    return f"gemini_attachment_{index + 1}.{extension}"


def _param_type_for_gemini(type_: Any) -> str:
    value = str(type_ or "").lower()
    return {
        "integer": "INTEGER",
        "number": "NUMBER",
        "float": "NUMBER",
        "double": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }.get(value, "STRING")


def _schema_type(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return None
    return schema.get("type")


def _schema_type_lower(option: Any) -> str:
    return str(_schema_type(option) or "").lower()


_COMMON_ATTRS = ("description", "enum", "format", "nullable", "maximum", "minimum", "multipleOf")
_ATTR_ALIASES = {
    "multipleOf": ("multipleOf", "multiple_of"),
    "minItems": ("minItems", "min_items"),
    "maxItems": ("maxItems", "max_items"),
}


def _schema_value(source: dict[str, Any], attribute: str) -> Any:
    for key in _ATTR_ALIASES.get(attribute, (attribute,)):
        if source.get(key) is not None:
            return source[key]
    return None


def _copy_attributes(
    target: dict[str, Any], source: dict[str, Any], attributes: tuple[str, ...]
) -> None:
    for attribute in attributes:
        value = _schema_value(source, attribute)
        if value is not None:
            target[attribute] = value


def _copy_common_attributes(target: dict[str, Any], source: dict[str, Any]) -> None:
    _copy_attributes(target, source, _COMMON_ATTRS)


def _forced_tool_choice(tool_choice: str) -> bool:
    return tool_choice == "required" or _specific_tool_choice(tool_choice)


def _specific_tool_choice(tool_choice: str) -> bool:
    return tool_choice not in ("auto", "none", "required")


def _response_json_schema_supported(model: Info | None) -> bool:
    version = _gemini_version(model)
    return version is not None and version >= (2, 5)


def _gemini_version(model: Info | None) -> tuple[int, int] | None:
    if model is None:
        return None
    metadata = model.metadata or {}
    candidates = [
        model.id,
        model.family,
        metadata.get("version"),
        metadata.get("description"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        version = _extract_version(str(candidate))
        if version is not None:
            return version
    return None


def _extract_version(text: str) -> tuple[int, int] | None:
    match = re.search(r"(\d+)\.(\d+)|\d+", text)
    if not match:
        return None
    if match.group(1) is not None:
        return (int(match.group(1)), int(match.group(2)))
    return (int(match.group(0)), 0)


def _deep_stringify_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _deep_stringify_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_stringify_keys(item) for item in value]
    return value
