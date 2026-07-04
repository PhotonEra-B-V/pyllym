from __future__ import annotations

import httpx
import pytest
import respx

import pyllm

from . import factories as f
from .seed_fixtures import load_json


@pytest.mark.asyncio
@respx.mock
async def test_embedding_single_vector():
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json=f.openai_embedding([[0.1, 0.2, 0.3]], tokens=4))
    )
    emb = await pyllm.embed("hello", model="text-embedding-3-small")
    assert emb.vectors == [0.1, 0.2, 0.3]  # unwrapped for a single input
    assert emb.input_tokens == 4


@pytest.mark.asyncio
@respx.mock
async def test_embedding_batch_vectors():
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(200, json=f.openai_embedding([[0.1], [0.2]]))
    )
    emb = await pyllm.embed(["a", "b"], model="text-embedding-3-small")
    assert emb.vectors == [[0.1], [0.2]]  # list input stays nested


@pytest.mark.asyncio
@respx.mock
async def test_moderation_flagged():
    respx.post("https://api.openai.com/v1/moderations").mock(
        return_value=httpx.Response(200, json=f.openai_moderation(flagged=True))
    )
    result = await pyllm.moderate("something", model="omni-moderation-latest")
    assert result.is_flagged()
    assert "hate" in result.flagged_categories()


@pytest.mark.asyncio
@respx.mock
async def test_image_generation():
    respx.post("https://api.openai.com/v1/images/generations").mock(
        return_value=httpx.Response(200, json=f.openai_image(url="https://cdn/cat.png"))
    )
    image = await pyllm.paint("a cat", model="gpt-image-1.5")
    assert image.url == "https://cdn/cat.png"
    assert image.revised_prompt == "a cat"


@pytest.mark.asyncio
@respx.mock
async def test_chat_from_seeded_fixture_file():
    # Proves the on-disk seeded fixture drives the full parse stack.
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=load_json("openai_chat"))
    )
    msg = await pyllm.create_chat(model="gpt-4o").ask("hi")
    assert msg.content == "Hello!"
    assert msg.input_tokens == 10
