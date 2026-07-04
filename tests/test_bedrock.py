from __future__ import annotations

import httpx
import pytest
import respx

import pyllm

from . import factories as f


def _bedrock_model_id() -> str:
    ids = [m.id for m in pyllm.models.all() if m.provider == "bedrock"]
    assert ids, "expected bedrock models in the registry"
    return ids[0]


@pytest.mark.asyncio
@respx.mock
async def test_bedrock_converse_roundtrip():
    route = respx.post(
        url__regex=r"https://bedrock-runtime\..*\.amazonaws\.com/model/.*/converse$"
    ).mock(return_value=httpx.Response(200, json=f.bedrock_converse("Bedrock says hi")))
    chat = pyllm.create_chat(model=_bedrock_model_id(), provider="bedrock")
    msg = await chat.ask("hi")
    assert route.called
    assert msg.content == "Bedrock says hi"
    assert msg.input_tokens == 9
    assert msg.output_tokens == 7
    # request was SigV4-signed
    auth = route.calls.last.request.headers.get("Authorization", "")
    assert auth.startswith("AWS4-HMAC-SHA256")


def test_bedrock_render_payload_shape():
    chat = pyllm.create_chat(model=_bedrock_model_id(), provider="bedrock")
    chat.add_user_message("hello")
    payload = chat.render()
    assert payload["messages"][0]["role"] == "user"
    assert isinstance(payload["messages"][0]["content"], list)
