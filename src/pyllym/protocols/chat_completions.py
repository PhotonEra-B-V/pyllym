"""OpenAI Chat Completions — the lingua franca of LLM APIs., composed here into
one cohesive protocol class.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import utils
from ..chunk import Chunk
from ..citation import Citation
from ..content import Content, RawContent
from ..embedding import Embedding
from ..errors import Error, UnsupportedAttachmentError
from ..image import Image
from ..message import Message
from ..model.info import Info
from ..moderation import Moderation
from ..protocol import Protocol
from ..speech import Speech
from ..thinking import Thinking
from ..tool_call import ToolCall
from ..transcription import Transcription

if TYPE_CHECKING:
    from ..tool import Tool

EMPTY_PARAMETERS_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
    "strict": True,
}


class ChatCompletions(Protocol):
    # --- endpoints -------------------------------------------------------------
    def completion_url(self) -> str:
        return "chat/completions"

    def stream_url(self) -> str:
        return self.completion_url()

    def models_url(self) -> str:
        return "models"

    def embedding_url(self, *, model: str | None = None) -> str:
        return "embeddings"

    def moderation_url(self) -> str:
        return "moderations"

    def images_url(self, *, with_: Any = None, mask: Any = None) -> str:
        return "images/generations"

    def speech_url(self, *, model: str | None = None) -> str:
        return "audio/speech"

    def transcription_url(self) -> str:
        return "audio/transcriptions"

    def maybe_normalize_temperature(
        self, temperature: float | None, model: Info | None
    ) -> float | None:
        if model is None:
            return temperature
        return normalize_temperature(temperature, model.id)

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
        tool_prefs = tool_prefs or {}
        payload: dict[str, Any] = {
            "model": model.id,
            "messages": self.format_messages(messages),
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [self.tool_for(tool) for tool in tools.values()]
            if tool_prefs.get("choice") is not None:
                payload["tool_choice"] = self.build_tool_choice(tool_prefs["choice"])
            if tool_prefs.get("calls") is not None:
                payload["parallel_tool_calls"] = tool_prefs["calls"] == "many"
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": schema.get("strict"),
                },
            }
        effort = _resolve_effort(thinking)
        if effort:
            payload["reasoning_effort"] = effort
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    # --- chat parse ------------------------------------------------------------
    def parse_completion_response(self, response: Any) -> Message:
        message = self._parse_completion_body(response.body, raw=response)
        if message is None:
            raise Error(response, "Empty or unparseable completion response")
        return message

    def _parse_completion_body(self, data: Any, *, raw: Any) -> Message | None:
        if not data:
            return None
        if isinstance(data, dict) and (data.get("error") or {}).get("message"):
            raise Error(raw, data["error"]["message"])
        message_data = _dig(data, "choices", 0, "message")
        if not message_data:
            return None
        usage = data.get("usage") or {}
        content, thinking_from_blocks = self._extract_content_and_thinking(
            message_data.get("content")
        )
        thinking_text = thinking_from_blocks or _extract_thinking_text(message_data)
        return Message(
            role="assistant",
            content=content,
            citations=self._extract_citations(message_data, data, content),
            thinking=Thinking.build(
                text=thinking_text, signature=_extract_thinking_signature(message_data)
            ),
            tool_calls=self.parse_tool_calls(message_data.get("tool_calls")),
            input_tokens=_input_tokens(usage),
            output_tokens=_output_tokens(usage),
            cached_tokens=_cache_read_tokens(usage),
            cache_creation_tokens=_cache_write_tokens(usage),
            thinking_tokens=_thinking_tokens(usage),
            finish_reason=_dig(data, "choices", 0, "finish_reason"),
            model_id=data.get("model"),
            raw=raw,
        )

    # --- streaming -------------------------------------------------------------
    def build_chunk(self, data: dict[str, Any]) -> Chunk:
        usage = data.get("usage") or {}
        delta = _dig(data, "choices", 0, "delta") or {}
        content_source = delta.get("content") or _dig(data, "choices", 0, "message", "content")
        content, thinking_from_blocks = self._extract_content_and_thinking(content_source)
        return Chunk(
            role="assistant",
            model_id=data.get("model"),
            content=content,
            citations=self._extract_chunk_citations(delta, data),
            thinking=Thinking.build(
                text=thinking_from_blocks
                or delta.get("reasoning_content")
                or delta.get("reasoning"),
                signature=delta.get("reasoning_signature"),
            ),
            tool_calls=self.parse_tool_calls(delta.get("tool_calls"), parse_arguments=False),
            input_tokens=_input_tokens(usage),
            output_tokens=_output_tokens(usage),
            cached_tokens=_cache_read_tokens(usage),
            cache_creation_tokens=_cache_write_tokens(usage),
            thinking_tokens=_thinking_tokens(usage),
            finish_reason=_dig(data, "choices", 0, "finish_reason"),
        )

    def _extract_chunk_citations(
        self, delta: dict[str, Any], data: dict[str, Any]
    ) -> list[Citation]:
        annotations = _parse_annotations(delta.get("annotations"), None)
        if annotations:
            return annotations
        return _parse_root_citations(data)

    # --- messages / media ------------------------------------------------------
    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out = []
        for msg in messages:
            entry = {
                "role": self.format_role(msg.role),
                "content": self._format_message_content(msg),
                "tool_calls": self.format_tool_calls(msg.tool_calls),
                "tool_call_id": msg.tool_call_id,
            }
            entry = {k: v for k, v in entry.items() if v is not None}
            entry.update(_format_thinking(msg))
            out.append(entry)
        return out

    def _format_message_content(self, msg: Message) -> Any:
        content = self.format_content(msg.content)
        if content is None and msg.role == "assistant" and msg.thinking and not msg.is_tool_call():
            return ""
        return content

    def format_role(self, role: str) -> str:
        if role == "system":
            return "system" if self.config.openai_use_system_role else "developer"
        return role

    def format_content(
        self,
        content: Any,
        *,
        document_attachments: str = "pdf",
        image_attachments: bool = True,
        audio_attachments: bool = True,
    ) -> Any:
        if isinstance(content, RawContent):
            value = content.value
            return json.dumps(value) if isinstance(value, dict) else value
        if isinstance(content, (dict, list)):
            return json.dumps(content)
        if not isinstance(content, Content):
            return content
        parts: list[dict[str, Any]] = []
        if content.text:
            parts.append({"type": "text", "text": content.text})
        for attachment in content.attachments:
            parts.append(
                self._format_attachment(
                    attachment,
                    document_attachments=document_attachments,
                    image_attachments=image_attachments,
                    audio_attachments=audio_attachments,
                )
            )
        return parts

    def _format_attachment(
        self,
        attachment: Any,
        *,
        document_attachments: str,
        image_attachments: bool,
        audio_attachments: bool,
    ) -> dict[str, Any]:
        if attachment.is_provider_file():
            if document_attachments == "none":
                raise UnsupportedAttachmentError(attachment.mime_type)
            return {"type": "file", "file": {"file_id": attachment.provider_file_id}}
        kind = attachment.type
        if kind == "image":
            if not image_attachments:
                raise UnsupportedAttachmentError(attachment.mime_type)
            url = str(attachment.source) if attachment.is_url() else attachment.for_llm()
            return {"type": "image_url", "image_url": {"url": url}}
        if kind == "audio":
            if not audio_attachments:
                raise UnsupportedAttachmentError(attachment.mime_type)
            return {
                "type": "input_audio",
                "input_audio": {"data": attachment.encoded, "format": attachment.format},
            }
        if kind in ("pdf", "document"):
            if document_attachments == "all" or (
                document_attachments == "pdf" and attachment.is_pdf()
            ):
                return {
                    "type": "file",
                    "file": {"filename": attachment.filename, "file_data": attachment.for_llm()},
                }
            raise UnsupportedAttachmentError(attachment.mime_type)
        if kind == "text":
            return {"type": "text", "text": attachment.for_llm()}
        raise UnsupportedAttachmentError(attachment.mime_type)

    # --- tools -----------------------------------------------------------------
    def tool_for(self, tool: Tool) -> dict[str, Any]:
        schema = tool.params_schema or EMPTY_PARAMETERS_SCHEMA
        definition = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            },
        }
        if not tool.provider_params:
            return definition
        return utils.deep_merge(definition, tool.provider_params)

    def format_tool_calls(
        self, tool_calls: dict[str, ToolCall] | None
    ) -> list[dict[str, Any]] | None:
        if not tool_calls:
            return None
        out = []
        for tc in tool_calls.values():
            call: dict[str, Any] = {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            if tc.thought_signature:
                call["extra_content"] = {"google": {"thought_signature": tc.thought_signature}}
            out.append(call)
        return out

    def parse_tool_calls(
        self, tool_calls: list[dict[str, Any]] | None, *, parse_arguments: bool = True
    ) -> dict[str, ToolCall] | None:
        if not tool_calls:
            return None
        result: dict[str, ToolCall] = {}
        for tc in tool_calls:
            args_raw = _dig(tc, "function", "arguments")
            if parse_arguments:
                arguments = json.loads(args_raw) if args_raw else {}
            else:
                arguments = args_raw
            result[tc["id"]] = ToolCall(
                id=tc["id"],
                name=_dig(tc, "function", "name"),
                arguments=arguments,
                thought_signature=_dig(tc, "extra_content", "google", "thought_signature"),
            )
        return result

    def build_tool_choice(self, tool_choice: str) -> Any:
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        return {"type": "function", "function": {"name": tool_choice}}

    # --- content/thinking extraction ------------------------------------------
    def _extract_content_and_thinking(self, content: Any) -> tuple[Any, str | None]:
        if isinstance(content, str):
            return _extract_think_tag_content(content)
        if not isinstance(content, list):
            return content, None
        text = "".join(
            b.get("text", "")
            for b in content
            if b.get("type") == "text" and isinstance(b.get("text"), str)
        )
        thinking = "".join(
            _thinking_text_from_block(b) for b in content if b.get("type") == "thinking"
        )
        return (text or None), (thinking or None)

    def _extract_citations(
        self, message_data: dict[str, Any], data: dict[str, Any], content: Any
    ) -> list[Citation]:
        annotations = _parse_annotations(message_data.get("annotations"), content)
        if annotations:
            return annotations
        return _parse_root_citations(data)

    # --- embeddings ------------------------------------------------------------
    def render_embedding_payload(
        self, text: Any, *, model: str, dimensions: int | None
    ) -> dict[str, Any]:
        return utils.compact({"model": model, "input": text, "dimensions": dimensions})

    def parse_embedding_response(self, response: Any, *, model: str, text: Any) -> Embedding:
        data = response.body
        input_tokens = _dig(data, "usage", "prompt_tokens") or 0
        vectors = [d["embedding"] for d in data["data"]]
        if len(vectors) == 1 and not isinstance(text, list):
            vectors = vectors[0]
        return Embedding(vectors=vectors, model=model, input_tokens=input_tokens)

    # --- moderation ------------------------------------------------------------
    def render_moderation_payload(self, input: Any, *, model: str) -> dict[str, Any]:
        return {"model": model, "input": input}

    def parse_moderation_response(self, response: Any, *, model: str) -> Moderation:
        data = response.body
        if (data.get("error") or {}).get("message"):
            raise Error(response, data["error"]["message"])
        return Moderation(id=data.get("id"), model=model, results=data.get("results") or [])

    # --- models ----------------------------------------------------------------
    def parse_list_models_response(self, response: Any, slug: str, capabilities: Any) -> list[Info]:
        out = []
        for model_data in response.body.get("data") or []:
            model_id = model_data["id"]
            created = model_data.get("created")
            out.append(
                Info(
                    {
                        "id": model_id,
                        "name": model_id,
                        "provider": slug,
                        "created_at": created,
                        "context_window": capabilities.context_window_for(model_id)
                        if capabilities
                        else None,
                        "max_output_tokens": capabilities.max_tokens_for(model_id)
                        if capabilities
                        else None,
                        "capabilities": capabilities.critical_capabilities_for(model_id)
                        if capabilities
                        else [],
                        "pricing": capabilities.pricing_for(model_id) if capabilities else {},
                        "metadata": {
                            "object": model_data.get("object"),
                            "owned_by": model_data.get("owned_by"),
                        },
                    }
                )
            )
        return out

    # --- images ----------------------------------------------------------------
    def render_image_payload(
        self,
        prompt: str,
        *,
        model: str,
        size: str,
        with_: Any = None,
        mask: Any = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"model": model, "prompt": prompt, "n": 1, "size": size, **(params or {})}

    def parse_image_response(self, response: Any, *, model: str) -> Image:
        data = response.body
        image_data = (data.get("data") or [None])[0]
        if not image_data:
            raise Error(None, "Unexpected response format from image API")
        return Image(
            url=image_data.get("url"),
            mime_type="image/png",
            revised_prompt=image_data.get("revised_prompt"),
            model_id=model,
            data=image_data.get("b64_json"),
            usage=data.get("usage") or {},
        )

    # --- speech ----------------------------------------------------------------
    def render_speech_payload(
        self,
        input: str,
        *,
        model: str,
        voice: str | None,
        format: str | None,
        params: dict[str, Any] | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        payload = utils.compact(
            {
                "model": model,
                "input": input,
                "voice": voice or "alloy",
                "response_format": format,
                "instructions": options.get("instructions"),
                "speed": options.get("speed"),
            }
        )
        payload.update(params or {})
        return payload

    def parse_speech_response(
        self, response: Any, *, model: str, voice: str | None, format: str | None
    ) -> Speech:
        body = response.content or response.body
        if isinstance(body, str):
            body = body.encode("latin-1", "ignore")
        return Speech(data=body, model=model, voice=voice or "alloy", format=str(format or "mp3"))

    # --- transcription ---------------------------------------------------------
    def render_transcription_payload(
        self, audio_file: str, *, model: str, language: str | None, **options: Any
    ) -> dict[str, Any]:
        path = Path(audio_file).expanduser()
        files = {"file": (path.name, path.read_bytes())}
        data = utils.compact(
            {
                "model": model,
                "language": language,
                "response_format": options.get("response_format"),
                "prompt": options.get("prompt"),
                "temperature": options.get("temperature"),
            }
        )
        return {"data": data, "files": files}

    def parse_transcription_response(self, response: Any, *, model: str) -> Transcription:
        data = response.body
        if isinstance(data, str):
            return Transcription(text=data, model=model)
        usage = data.get("usage") or {}
        return Transcription(
            text=data.get("text"),
            model=model,
            language=data.get("language"),
            duration=data.get("duration"),
            segments=data.get("segments"),
            words=data.get("words"),
            input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
            output_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
        )


# --- module helpers ------------------------------------------------------------
_dig = utils.dig


def _resolve_effort(thinking: Any) -> Any:
    if not thinking:
        return None
    return getattr(thinking, "effort", thinking)


def _format_thinking(msg: Message) -> dict[str, Any]:
    if msg.role != "assistant" or not msg.thinking:
        return {}
    payload: dict[str, Any] = {}
    if msg.thinking.text:
        payload["reasoning"] = msg.thinking.text
        payload["reasoning_content"] = msg.thinking.text
    if msg.thinking.signature:
        payload["reasoning_signature"] = msg.thinking.signature
    return payload


def _extract_thinking_text(message_data: dict[str, Any]) -> str | None:
    candidate = (
        message_data.get("reasoning_content")
        or message_data.get("reasoning")
        or message_data.get("thinking")
    )
    return candidate if isinstance(candidate, str) else None


def _extract_thinking_signature(message_data: dict[str, Any]) -> str | None:
    candidate = message_data.get("reasoning_signature") or message_data.get("signature")
    return candidate if isinstance(candidate, str) else None


def _thinking_text_from_block(block: dict[str, Any]) -> str:
    thinking_block = block.get("thinking")
    if isinstance(thinking_block, str):
        return thinking_block
    if isinstance(thinking_block, list):
        return "".join(i.get("text", "") for i in thinking_block if i.get("type") == "text")
    return block.get("text", "") if isinstance(block.get("text"), str) else ""


def _extract_think_tag_content(text: str) -> tuple[Any, str | None]:
    if "<think>" not in text:
        return text, None
    thinking = "".join(re.findall(r"<think>(.*?)</think>", text, re.DOTALL))
    content = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return (content or None), (thinking or None)


def _parse_annotations(annotations: Any, content: Any) -> list[Citation]:
    out = []
    for annotation in utils.to_safe_array(annotations):
        details = annotation.get("url_citation")
        if not isinstance(details, dict):
            continue
        start = details.get("start_index")
        end = details.get("end_index")
        text = (
            content[start:end]
            if isinstance(content, str) and start is not None and end is not None
            else None
        )
        out.append(
            Citation(
                url=details.get("url"),
                title=details.get("title"),
                text=text,
                start_index=start,
                end_index=end,
            )
        )
    return out


def _parse_root_citations(data: dict[str, Any]) -> list[Citation]:
    search_results = data.get("search_results")
    if isinstance(search_results, list) and search_results:
        out = []
        for index, result in enumerate(search_results):
            if not isinstance(result, dict):
                continue
            out.append(
                Citation(
                    url=result.get("url"),
                    title=result.get("title"),
                    cited_text=result.get("snippet"),
                    source_index=index,
                )
            )
        return out
    return [
        Citation(url=url, source_index=index)
        for index, url in enumerate(utils.to_safe_array(data.get("citations")))
        if isinstance(url, str)
    ]


def _input_tokens(usage: dict[str, Any]) -> int | None:
    if usage.get("prompt_cache_miss_tokens"):
        return usage["prompt_cache_miss_tokens"]
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        return None
    return max(
        int(prompt_tokens)
        - int(_cache_read_tokens(usage) or 0)
        - int(_cache_write_tokens(usage) or 0),
        0,
    )


def _output_tokens(usage: dict[str, Any]) -> int | None:
    completion_tokens = usage.get("completion_tokens")
    if completion_tokens is None:
        return None
    completion_tokens = int(completion_tokens)
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is not None and total_tokens is not None:
        generated = max(int(total_tokens) - int(prompt_tokens), 0)
        if generated > completion_tokens:
            return generated
    return completion_tokens


def _cache_read_tokens(usage: dict[str, Any]) -> int | None:
    return _dig(usage, "prompt_tokens_details", "cached_tokens") or usage.get(
        "prompt_cache_hit_tokens"
    )


def _cache_write_tokens(usage: dict[str, Any]) -> int:
    return _dig(usage, "prompt_tokens_details", "cache_write_tokens") or 0


def _thinking_tokens(usage: dict[str, Any]) -> int | None:
    return _dig(usage, "completion_tokens_details", "reasoning_tokens") or usage.get(
        "reasoning_tokens"
    )


# --- temperature normalization (Temperature concern) ---------------------------
def normalize_temperature(temperature: float | None, model_id: str) -> float | None:
    if (
        _forced_temperature_model(model_id)
        and temperature is not None
        and not _close_to_one(temperature)
    ):
        return 1.0
    if "-search" in model_id:
        return None
    return temperature


def _close_to_one(temperature: float) -> bool:
    return abs(float(temperature) - 1.0) <= 1e-9


def _forced_temperature_model(model_id: str) -> bool:
    return bool(
        re.match(r"^o\d", model_id)
        or re.match(r"^gpt-5(\.\d+)?(-\d{4})?$", model_id)
        or re.match(r"^gpt-5(\.\d+)?-pro", model_id)
    )
