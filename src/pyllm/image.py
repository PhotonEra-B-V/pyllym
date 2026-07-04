"""Generated images."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import models as _models

if TYPE_CHECKING:
    from .context import Context


class Image:
    def __init__(
        self,
        *,
        url: str | None = None,
        data: str | None = None,
        mime_type: str | None = None,
        revised_prompt: str | None = None,
        model_id: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        self.url = url
        self.data = data
        self.mime_type = mime_type
        self.revised_prompt = revised_prompt
        self.model_id = model_id
        self.usage = usage or {}

    def is_base64(self) -> bool:
        return self.data is not None

    def to_blob(self) -> bytes:
        """Fetch the image bytes (blocking; use :meth:`ato_blob` inside async code)."""
        if self.is_base64():
            return base64.b64decode(self.data or "")
        import httpx

        resp = httpx.get(self.url, follow_redirects=True, timeout=60)  # type: ignore[arg-type]
        resp.raise_for_status()
        return resp.content

    async def ato_blob(self) -> bytes:
        """Async variant of :meth:`to_blob` — does not block the event loop."""
        if self.is_base64():
            return base64.b64decode(self.data or "")
        from .connection import Connection

        async with Connection.basic() as client:
            resp = await client.get(self.url, timeout=60)  # type: ignore[arg-type]
            resp.raise_for_status()
            return resp.content

    def save(self, path: str | Path) -> str | Path:
        Path(path).expanduser().write_bytes(self.to_blob())
        return path

    async def asave(self, path: str | Path) -> str | Path:
        Path(path).expanduser().write_bytes(await self.ato_blob())
        return path


async def paint(
    prompt: str,
    *,
    model: str | None = None,
    provider: str | None = None,
    assume_model_exists: bool = False,
    size: str = "1024x1024",
    context: Context | None = None,
    with_: Any = None,
    mask: Any = None,
    params: dict[str, Any] | None = None,
) -> Image:
    from . import config as _config

    cfg = context.config if context else _config()
    model = model or cfg.default_image_model
    model_info, provider_instance = _models.resolve(
        model, provider=provider, assume_exists=assume_model_exists, config=cfg
    )
    return await provider_instance.paint(
        prompt, model=model_info.id, size=size, with_=with_, mask=mask, params=params or {}
    )
