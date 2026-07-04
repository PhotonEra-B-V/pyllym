from __future__ import annotations

import httpx
import pytest
import respx

import pyllm


@pytest.fixture(autouse=True)
def _fal_key():
    pyllm.configure(lambda c: setattr(c, "fal_api_key", "fal-test"))


@pytest.mark.asyncio
@respx.mock
async def test_fal_image_paint():
    route = respx.post("https://fal.run/fal-ai/flux/dev").mock(
        return_value=httpx.Response(
            200,
            json={
                "images": [{"url": "https://cdn.fal/img.png", "content_type": "image/png"}],
                "prompt": "a red panda",
            },
        )
    )
    image = await pyllm.paint(
        "a red panda", provider="fal", model="fal-ai/flux/dev", size="1024x768"
    )
    assert route.called
    sent = route.calls.last.request
    assert b'"image_size"' in sent.content and b'"width":1024' in sent.content
    assert image.url == "https://cdn.fal/img.png"
    assert image.mime_type == "image/png"


@pytest.mark.asyncio
@respx.mock
async def test_fal_video_animate_queue_poll():
    respx.post("https://queue.fal.run/fal-ai/ltx-video-13b-distilled").mock(
        return_value=httpx.Response(
            200,
            json={
                "request_id": "r1",
                "status_url": "https://queue.fal.run/r1/status",
                "response_url": "https://queue.fal.run/r1",
            },
        )
    )
    status = {"n": 0}

    def status_responder(request):
        status["n"] += 1
        body = {"status": "IN_PROGRESS"} if status["n"] == 1 else {"status": "COMPLETED"}
        return httpx.Response(200, json=body)

    respx.get("https://queue.fal.run/r1/status").mock(side_effect=status_responder)
    respx.get("https://queue.fal.run/r1").mock(
        return_value=httpx.Response(
            200, json={"video": {"url": "https://cdn.fal/out.mp4", "content_type": "video/mp4"}}
        )
    )

    # speed up polling
    from pyllm.protocols.fal import Fal

    Fal.poll_interval = 0.0

    video = await pyllm.animate("a timelapse sunrise", model="fal-ai/ltx-video-13b-distilled")
    assert video.url == "https://cdn.fal/out.mp4"
    assert video.mime_type == "video/mp4"
    assert status["n"] == 2  # polled until COMPLETED
