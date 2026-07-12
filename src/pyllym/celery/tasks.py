"""Celery task factory bridging pyllym's async API into synchronous workers."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from celery import Celery

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a pyllym coroutine to completion on a fresh event loop.

    pyllym's shared HTTP pools are keyed per event loop, so they are closed
    before the loop shuts down — otherwise each task run would leak clients
    bound to a dead loop.
    """

    async def runner() -> T:
        from ..connection import aclose

        try:
            return await coro
        finally:
            await aclose()

    return asyncio.run(runner())


@dataclass(frozen=True)
class Tasks:
    """The Celery tasks registered by :func:`create_tasks`."""

    ask: Any
    embed: Any
    paint: Any
    speak: Any
    transcribe: Any
    moderate: Any


def create_tasks(app: Celery, *, name_prefix: str = "pyllym", **task_options: Any) -> Tasks:
    """Register pyllym tasks on ``app`` and return them as a :class:`Tasks`.

    Tasks are named ``{name_prefix}.ask``, ``..._embed``, etc. Extra
    ``task_options`` (``queue``, ``max_retries``, ...) are passed through to
    ``app.task``. Arguments and results are broker-friendly: plain JSON
    values in, plain dicts out.
    """

    def task(fn: Any) -> Any:
        return app.task(name=f"{name_prefix}.{fn.__name__}", **task_options)(fn)

    @task
    def ask(
        prompt: Any = None,
        *,
        model: str | None = None,
        provider: str | None = None,
        instructions: str | None = None,
        temperature: float | None = None,
        schema: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        messages: list[dict[str, Any]] | None = None,
        with_: Any = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import pyllym

        chat = pyllym.create_chat(model=model, provider=provider)
        if instructions is not None:
            chat.with_instructions(instructions)
        if temperature is not None:
            chat.with_temperature(temperature)
        if schema is not None:
            chat.with_schema(schema)
        if thinking is not None:
            chat.with_thinking(**thinking)
        if params:
            chat.with_params(**params)
        for message in messages or []:
            chat.add_message(message)
        return run_async(chat.ask(prompt, with_=with_)).to_dict()

    @task
    def embed(
        text: Any,
        *,
        model: str | None = None,
        provider: str | None = None,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        import pyllym

        result = run_async(pyllym.embed(text, model=model, provider=provider, dimensions=dimensions))
        return {
            "vectors": result.vectors,
            "model": result.model,
            "input_tokens": result.input_tokens,
        }

    @task
    def paint(
        prompt: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        size: str = "1024x1024",
        save_path: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import pyllym

        async def painter() -> Any:
            image = await pyllym.paint(
                prompt, model=model, provider=provider, size=size, params=params
            )
            if save_path is not None:
                await image.asave(save_path)
            return image

        image = run_async(painter())
        data: dict[str, Any] = {
            "url": image.url,
            "data": None if save_path else image.data,
            "mime_type": image.mime_type,
            "revised_prompt": image.revised_prompt,
            "model_id": image.model_id,
            "path": save_path,
        }
        return {k: v for k, v in data.items() if v is not None}

    @task
    def speak(
        input: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        voice: str | None = None,
        format: str | None = None,
        save_path: str | None = None,
    ) -> dict[str, Any]:
        import pyllym

        speech = run_async(
            pyllym.speak(input, model=model, provider=provider, voice=voice, format=format)
        )
        if save_path is not None:
            speech.save(save_path)
        data: dict[str, Any] = {
            "data_base64": None if save_path else base64.b64encode(speech.data).decode(),
            "model": speech.model,
            "voice": speech.voice,
            "format": speech.format,
            "mime_type": speech.mime_type,
            "path": save_path,
        }
        return {k: v for k, v in data.items() if v is not None}

    @task
    def transcribe(
        audio_file: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        import pyllym

        result = run_async(
            pyllym.transcribe(audio_file, model=model, provider=provider, language=language)
        )
        data = {
            "text": result.text,
            "model": result.model,
            "language": result.language,
            "duration": result.duration,
            "segments": result.segments,
            "words": result.words,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        }
        return {k: v for k, v in data.items() if v is not None}

    @task
    def moderate(
        input: Any,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        import pyllym

        result = run_async(pyllym.moderate(input, model=model, provider=provider))
        return {
            "id": result.id,
            "model": result.model,
            "results": result.results,
            "flagged": result.is_flagged(),
        }

    return Tasks(
        ask=ask,
        embed=embed,
        paint=paint,
        speak=speak,
        transcribe=transcribe,
        moderate=moderate,
    )
