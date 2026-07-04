"""fal.ai protocol — image and video generation.

fal serves models behind two shapes:

* **Synchronous run** (``https://fal.run/{model}``) — used for image models
  (FLUX.2, HunyuanImage, Qwen-Image, ...), which return quickly.
* **Queue** (``https://queue.fal.run/{model}``) — submit returns a
  ``status_url``/``response_url``; we poll until ``COMPLETED`` then fetch the
  result. Used for video models (LTX, Wan, HunyuanVideo, ...), which take
  longer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..errors import Error
from ..image import Image
from ..protocol import Protocol
from ..video import Video

QUEUE_BASE = "https://queue.fal.run"
_TERMINAL_OK = "COMPLETED"
_TERMINAL_BAD = {"FAILED", "ERROR", "CANCELLED"}


class Fal(Protocol):
    # poll cadence for the queue API
    poll_interval: float = 1.0

    # --- image (synchronous run) ----------------------------------------------
    async def paint(
        self,
        prompt: str,
        *,
        model: str,
        size: str,
        with_: Any = None,
        mask: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Image:
        payload: dict[str, Any] = {"prompt": prompt, **_size_payload(size), **(params or {})}
        if with_ is not None:
            payload.setdefault("image_url", with_)
        response = await self.connection.post(f"/{model.lstrip('/')}", payload)
        return self._parse_image(response.body, model)

    def _parse_image(self, data: dict[str, Any], model: str) -> Image:
        images = data.get("images") or []
        first = images[0] if images else None
        if not first:
            raise Error(None, "fal image response contained no images")
        return Image(
            url=first.get("url"),
            data=first.get("b64_json"),
            mime_type=first.get("content_type") or "image/png",
            revised_prompt=data.get("prompt"),
            model_id=model,
            usage={k: v for k, v in data.items() if k != "images"},
        )

    # --- video (queue + poll) --------------------------------------------------
    async def animate(
        self, prompt: str, *, model: str, params: dict[str, Any] | None = None
    ) -> Video:
        submit = await self.connection.post(
            f"{self._queue_base()}/{model.lstrip('/')}", {"prompt": prompt, **(params or {})}
        )
        status_url = submit.body.get("status_url")
        response_url = submit.body.get("response_url")
        if not status_url or not response_url:
            raise Error(submit, "fal queue submission did not return poll URLs")

        await self._await_completion(status_url)
        result = await self.connection.get(response_url)
        return self._parse_video(result.body, model)

    def _queue_base(self) -> str:
        """Queue endpoint, honoring gateway overrides like the sync base does.

        ``fal_queue_base`` wins; otherwise a custom ``fal_api_base`` (proxy /
        self-hosted gateway) is reused for queue traffic; else the public queue.
        """
        return self.config.fal_queue_base or self.config.fal_api_base or QUEUE_BASE

    async def _await_completion(self, status_url: str) -> None:
        deadline = max(1, int(self.config.request_timeout / max(self.poll_interval, 0.01)))
        for _ in range(deadline):
            status_resp = await self.connection.get(status_url)
            status = status_resp.body.get("status")
            if status == _TERMINAL_OK:
                return
            if status in _TERMINAL_BAD:
                raise Error(status_resp, f"fal job {status.lower()}")
            await asyncio.sleep(self.poll_interval)
        raise Error(None, "fal job timed out before completing")

    def _parse_video(self, data: dict[str, Any], model: str) -> Video:
        video = data.get("video") or (data.get("videos") or [{}])[0]
        if not video or not video.get("url"):
            raise Error(None, "fal video response contained no video")
        return Video(
            url=video.get("url"),
            mime_type=video.get("content_type") or "video/mp4",
            model_id=model,
            raw=data,
        )


def _size_payload(size: str | None) -> dict[str, Any]:
    """Map an OpenAI-style ``"WxH"`` size to fal's ``image_size`` field.

    A non-dimension string (e.g. ``"landscape_16_9"``) is passed through as a
    fal size preset.
    """
    if not size:
        return {}
    if "x" in size:
        width, _, height = size.partition("x")
        if width.isdigit() and height.isdigit():
            return {"image_size": {"width": int(width), "height": int(height)}}
    return {"image_size": size}
