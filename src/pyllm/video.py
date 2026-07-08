"""Generated video — the video-generation counterpart of :mod:`pyllm.image`.

A pyllm capability for
text-to-video / image-to-video models served by providers such as fal.ai
(LTX, Wan, HunyuanVideo, ...).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import models as _models

if TYPE_CHECKING:
    from .context import Context


class Video:
    def __init__(
        self,
        *,
        url: str | None = None,
        data: str | None = None,
        mime_type: str | None = None,
        model_id: str | None = None,
        duration: float | None = None,
        raw: Any = None,
    ) -> None:
        self.url = url
        self.data = data
        self.mime_type = mime_type or "video/mp4"
        self.model_id = model_id
        self.duration = duration
        self.raw = raw

    def is_base64(self) -> bool:
        return self.data is not None

    def to_blob(self) -> bytes:
        """Fetch the video bytes (blocking; use :meth:`ato_blob` inside async code)."""
        if self.is_base64():
            return base64.b64decode(self.data or "")
        from urllib.request import urlopen

        with urlopen(self.url, timeout=300) as resp:  # type: ignore[arg-type]
            return resp.read()

    async def ato_blob(self) -> bytes:
        """Async variant of :meth:`to_blob` — does not block the event loop."""
        if self.is_base64():
            return base64.b64decode(self.data or "")
        from .connection import Connection

        async with Connection.basic() as client, client.get(self.url) as resp:  # type: ignore[arg-type]
            resp.raise_for_status()
            return await resp.read()

    def save(self, path: str | Path) -> str | Path:
        Path(path).expanduser().write_bytes(self.to_blob())
        return path

    async def asave(self, path: str | Path) -> str | Path:
        Path(path).expanduser().write_bytes(await self.ato_blob())
        return path


async def animate(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = "fal",
    assume_model_exists: bool = False,
    context: Context | None = None,
    params: dict[str, Any] | None = None,
) -> Video:
    from . import config as _config

    cfg = context.config if context else _config()
    model = model or cfg.default_video_model
    if not model:
        raise ValueError("animate requires a model, e.g. model='fal-ai/ltx-video-13b-distilled'")
    model_info, provider_instance = _models.resolve(
        model, provider=provider, assume_exists=assume_model_exists, config=cfg
    )
    return await provider_instance.animate(prompt, model=model_info.id, params=params or {})
