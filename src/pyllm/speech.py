"""Text-to-speech."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import models as _models

if TYPE_CHECKING:
    from .context import Context

MIME_TYPES = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "wav": "audio/wav",
}


class Speech:
    def __init__(
        self,
        *,
        data: bytes,
        model: str,
        voice: str | None = None,
        format: str = "mp3",
        mime_type: str | None = None,
    ) -> None:
        self.data = data
        self.model = model
        self.voice = voice
        self.format = str(format or "mp3")
        self.mime_type = mime_type or MIME_TYPES.get(self.format, f"audio/{self.format}")

    def to_blob(self) -> bytes:
        return self.data

    def save(self, path: str | Path) -> str | Path:
        Path(path).expanduser().write_bytes(self.to_blob())
        return path


async def speak(
    input: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    assume_model_exists: bool = False,
    voice: str | None = None,
    format: str | None = None,
    context: Context | None = None,
    params: dict[str, Any] | None = None,
    **options: Any,
) -> Speech:
    from . import config as _config

    cfg = context.config if context else _config()
    model = model or cfg.default_speech_model
    model_info, provider_instance = _models.resolve(
        model, provider=provider, assume_exists=assume_model_exists, config=cfg
    )
    return await provider_instance.speak(
        input,
        model=model_info.id,
        voice=voice,
        format=format,
        params=params or {},
        **options,
    )
