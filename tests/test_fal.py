from __future__ import annotations

import pytest
from aioresponses import CallbackResult

import pyllm

from .conftest import sent_json, sent_requests


@pytest.fixture(autouse=True)
def _fal_key():
    pyllm.configure(lambda c: setattr(c, "fal_api_key", "fal-test"))


@pytest.mark.asyncio
async def test_fal_image_paint(mock_http):
    mock_http.post(
        "https://fal.run/fal-ai/flux/dev",
        payload={
            "images": [{"url": "https://cdn.fal/img.png", "content_type": "image/png"}],
            "prompt": "a red panda",
        },
    )
    image = await pyllm.paint(
        "a red panda", provider="fal", model="fal-ai/flux/dev", size="1024x768"
    )
    assert sent_requests(mock_http)
    sent = sent_json(mock_http)
    assert '"image_size"' in sent and '"width": 1024' in sent
    assert image.url == "https://cdn.fal/img.png"
    assert image.mime_type == "image/png"


@pytest.mark.asyncio
async def test_fal_video_animate_queue_poll(mock_http):
    mock_http.post(
        "https://queue.fal.run/fal-ai/ltx-video-13b-distilled",
        payload={
            "request_id": "r1",
            "status_url": "https://queue.fal.run/r1/status",
            "response_url": "https://queue.fal.run/r1",
        },
    )
    status = {"n": 0}

    def status_responder(url, **kwargs):
        status["n"] += 1
        body = {"status": "IN_PROGRESS"} if status["n"] == 1 else {"status": "COMPLETED"}
        return CallbackResult(payload=body)

    mock_http.get("https://queue.fal.run/r1/status", callback=status_responder, repeat=True)
    mock_http.get(
        "https://queue.fal.run/r1",
        payload={"video": {"url": "https://cdn.fal/out.mp4", "content_type": "video/mp4"}},
    )

    # speed up polling
    from pyllm.protocols.fal import Fal

    Fal.poll_interval = 0.0

    video = await pyllm.animate("a timelapse sunrise", model="fal-ai/ltx-video-13b-distilled")
    assert video.url == "https://cdn.fal/out.mp4"
    assert video.mime_type == "video/mp4"
    assert status["n"] == 2  # polled until COMPLETED
