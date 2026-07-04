"""Speech-to-text."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import models as _models

if TYPE_CHECKING:
    from .context import Context


class Transcription:
    def __init__(self, *, text: str, model: str, **attributes: Any) -> None:
        self.text = text
        self.model = model
        self.language = attributes.get("language")
        self.duration = attributes.get("duration")
        self.segments = attributes.get("segments")
        self.words = attributes.get("words")
        self.input_tokens = attributes.get("input_tokens")
        self.output_tokens = attributes.get("output_tokens")


async def transcribe(
    audio_file: str,
    *,
    model: str | None = None,
    language: str | None = None,
    provider: str | None = None,
    assume_model_exists: bool = False,
    context: Context | None = None,
    **options: Any,
) -> Transcription:
    from . import config as _config

    cfg = context.config if context else _config()
    model = model or cfg.default_transcription_model
    model_info, provider_instance = _models.resolve(
        model, provider=provider, assume_exists=assume_model_exists, config=cfg
    )
    return await provider_instance.transcribe(
        audio_file, model=model_info.id, language=language, **options
    )
