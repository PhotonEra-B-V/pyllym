"""Base wire protocol.

A protocol knows how to talk to a family of APIs: rendering payloads, parsing
responses, streaming chunks, and the endpoints involved. Providers know *where*
to talk and *who* they are. Concrete protocols live under
:mod:`pyllm.protocols` and compose per-concern mixins.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from . import utils
from .errors import UnsupportedAttachmentError
from .streaming import OnChunk, StreamingMixin

if TYPE_CHECKING:
    from .message import Message
    from .model.info import Info
    from .provider import Provider


class Protocol(StreamingMixin):
    def __init__(self, provider: Provider, model: Info | None = None) -> None:
        self.provider = provider
        self.config = provider.config
        self.connection = provider.connection
        self.model = model

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: dict[str, Any],
        temperature: float | None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        schema: Any = None,
        thinking: Any = None,
        citations: bool = False,
        tool_prefs: dict[str, Any] | None = None,
        on_chunk: OnChunk | None = None,
    ) -> Message:
        # Rendering may lazily fetch URL-attachment content (a blocking read);
        # run it in a worker thread so the event loop is never stalled.
        payload = await asyncio.to_thread(
            self.render,
            messages,
            tools=tools,
            tool_prefs=tool_prefs,
            temperature=temperature,
            params=params or {},
            schema=schema,
            thinking=thinking,
            citations=citations,
            stream=on_chunk is not None,
        )
        if on_chunk is not None:
            return await self.stream_response(payload, headers, on_chunk)
        return await self.sync_response(payload, headers)

    def render(
        self,
        messages: list[Message],
        *,
        tools: dict[str, Any],
        temperature: float | None,
        params: dict[str, Any] | None = None,
        schema: Any = None,
        thinking: Any = None,
        citations: bool = False,
        tool_prefs: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        base = self.render_payload(
            messages,
            tools=tools,
            tool_prefs=tool_prefs,
            temperature=self.maybe_normalize_temperature(temperature, self.model),
            model=self.model,
            stream=stream,
            schema=schema,
            thinking=thinking,
            citations=citations,
        )
        return utils.deep_merge(base, params or {})

    async def list_models(self) -> list[Info]:
        response = await self.connection.get(self.models_url())
        return self.parse_list_models_response(
            response, self.provider.slug, self.provider.capabilities
        )

    async def embed(self, text: Any, *, model: str, dimensions: int | None) -> Any:
        payload = self.render_embedding_payload(text, model=model, dimensions=dimensions)
        response = await self.connection.post(self.embedding_url(model=model), payload)
        return self.parse_embedding_response(response, model=model, text=text)

    async def moderate(self, input: Any, *, model: str) -> Any:
        payload = self.render_moderation_payload(input, model=model)
        response = await self.connection.post(self.moderation_url(), payload)
        return self.parse_moderation_response(response, model=model)

    async def paint(
        self,
        prompt: str,
        *,
        model: str,
        size: str,
        with_: Any = None,
        mask: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._validate_paint_inputs(with_=with_, mask=mask)
        payload = self.render_image_payload(
            prompt, model=model, size=size, with_=with_, mask=mask, params=params or {}
        )
        response = await self.connection.post(self.images_url(with_=with_, mask=mask), payload)
        return self.parse_image_response(response, model=model)

    async def speak(
        self,
        input: str,
        *,
        model: str,
        voice: str | None,
        format: str | None,
        params: dict[str, Any] | None = None,
        **options: Any,
    ) -> Any:
        payload = self.render_speech_payload(
            input, model=model, voice=voice, format=format, params=params or {}, **options
        )
        response = await self.connection.post(self.speech_url(model=model), payload)
        return self.parse_speech_response(response, model=model, voice=voice, format=format)

    async def transcribe(
        self, audio_file: str, *, model: str, language: str | None, **options: Any
    ) -> Any:
        payload = self.render_transcription_payload(
            audio_file, model=model, language=language, **options
        )
        response = await self.connection.post(self.transcription_url(), payload, multipart=True)
        return self.parse_transcription_response(response, model=model)

    async def animate(
        self, prompt: str, *, model: str, params: dict[str, Any] | None = None
    ) -> Any:
        raise NotImplementedError("This provider does not support video generation")

    def maybe_normalize_temperature(
        self, temperature: float | None, model: Info | None
    ) -> float | None:
        return temperature

    def parse_error(self, response: Any) -> Any:
        return self.provider.parse_error(response)

    def preprocess_message(self, message: Message) -> Message:
        # Large-file auto-upload is handled per-protocol where supported.
        return message

    async def sync_response(
        self, payload: dict[str, Any], additional_headers: dict[str, str] | None = None
    ) -> Message:
        response = await self.connection.post(
            self.completion_url(), payload, headers=additional_headers
        )
        return self.parse_completion_response(response)

    def _validate_paint_inputs(self, *, with_: Any, mask: Any) -> None:
        if with_ is None and mask is None:
            return
        raise UnsupportedAttachmentError("image reference")

    # --- interface methods concrete protocols implement ------------------------
    def render_payload(self, messages: list[Message], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def parse_completion_response(self, response: Any) -> Message:
        raise NotImplementedError

    def completion_url(self) -> str:
        raise NotImplementedError

    def models_url(self) -> str:
        raise NotImplementedError

    def parse_list_models_response(self, response: Any, slug: str, capabilities: Any) -> list[Info]:
        raise NotImplementedError

    # --- embeddings ---
    def render_embedding_payload(
        self, text: Any, *, model: str, dimensions: int | None
    ) -> dict[str, Any]:
        raise NotImplementedError

    def embedding_url(self, *, model: str | None = None) -> str:
        raise NotImplementedError

    def parse_embedding_response(self, response: Any, *, model: str, text: Any) -> Any:
        raise NotImplementedError

    # --- moderation ---
    def render_moderation_payload(self, input: Any, *, model: str) -> dict[str, Any]:
        raise NotImplementedError

    def moderation_url(self) -> str:
        raise NotImplementedError

    def parse_moderation_response(self, response: Any, *, model: str) -> Any:
        raise NotImplementedError

    # --- images ---
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
        raise NotImplementedError

    def images_url(self, *, with_: Any = None, mask: Any = None) -> str:
        raise NotImplementedError

    def parse_image_response(self, response: Any, *, model: str) -> Any:
        raise NotImplementedError

    # --- speech ---
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
        raise NotImplementedError

    def speech_url(self, *, model: str | None = None) -> str:
        raise NotImplementedError

    def parse_speech_response(
        self, response: Any, *, model: str, voice: str | None, format: str | None
    ) -> Any:
        raise NotImplementedError

    # --- transcription ---
    def render_transcription_payload(
        self, audio_file: str, *, model: str, language: str | None, **options: Any
    ) -> dict[str, Any]:
        raise NotImplementedError

    def transcription_url(self) -> str:
        raise NotImplementedError

    def parse_transcription_response(self, response: Any, *, model: str) -> Any:
        raise NotImplementedError
