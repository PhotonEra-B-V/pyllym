from __future__ import annotations

import re

import pytest

import pyllym

from . import factories as f
from .conftest import sent_requests


def _bedrock_model_id() -> str:
    ids = [m.id for m in pyllym.models.all() if m.provider == "bedrock"]
    assert ids, "expected bedrock models in the registry"
    return ids[0]


@pytest.mark.asyncio
async def test_bedrock_converse_roundtrip(mock_http):
    mock_http.post(
        re.compile(r"https://bedrock-runtime\..*\.amazonaws\.com/model/.*/converse$"),
        payload=f.bedrock_converse("Bedrock says hi"),
    )
    chat = pyllym.create_chat(model=_bedrock_model_id(), provider="bedrock")
    msg = await chat.ask("hi")
    requests = sent_requests(mock_http)
    assert requests
    assert msg.content == "Bedrock says hi"
    assert msg.input_tokens == 9
    assert msg.output_tokens == 7
    # request was SigV4-signed
    auth = (requests[-1].kwargs.get("headers") or {}).get("Authorization", "")
    assert auth.startswith("AWS4-HMAC-SHA256")


def test_bedrock_render_payload_shape():
    chat = pyllym.create_chat(model=_bedrock_model_id(), provider="bedrock")
    chat.add_user_message("hello")
    payload = chat.render()
    assert payload["messages"][0]["role"] == "user"
    assert isinstance(payload["messages"][0]["content"], list)
