from __future__ import annotations

import pytest

import pyllm

from . import factories as f
from .seed_fixtures import load_json


@pytest.mark.asyncio
async def test_embedding_single_vector(mock_http):
    mock_http.post(
        "https://api.openai.com/v1/embeddings",
        payload=f.openai_embedding([[0.1, 0.2, 0.3]], tokens=4),
    )
    emb = await pyllm.embed("hello", model="text-embedding-3-small")
    assert emb.vectors == [0.1, 0.2, 0.3]  # unwrapped for a single input
    assert emb.input_tokens == 4


@pytest.mark.asyncio
async def test_embedding_batch_vectors(mock_http):
    mock_http.post(
        "https://api.openai.com/v1/embeddings", payload=f.openai_embedding([[0.1], [0.2]])
    )
    emb = await pyllm.embed(["a", "b"], model="text-embedding-3-small")
    assert emb.vectors == [[0.1], [0.2]]  # list input stays nested


@pytest.mark.asyncio
async def test_moderation_flagged(mock_http):
    mock_http.post(
        "https://api.openai.com/v1/moderations", payload=f.openai_moderation(flagged=True)
    )
    result = await pyllm.moderate("something", model="omni-moderation-latest")
    assert result.is_flagged()
    assert "hate" in result.flagged_categories()


@pytest.mark.asyncio
async def test_image_generation(mock_http):
    mock_http.post(
        "https://api.openai.com/v1/images/generations",
        payload=f.openai_image(url="https://cdn/cat.png"),
    )
    image = await pyllm.paint("a cat", model="gpt-image-1.5")
    assert image.url == "https://cdn/cat.png"
    assert image.revised_prompt == "a cat"


@pytest.mark.asyncio
async def test_chat_from_seeded_fixture_file(mock_http):
    # Proves the on-disk seeded fixture drives the full parse stack.
    mock_http.post("https://api.openai.com/v1/chat/completions", payload=load_json("openai_chat"))
    msg = await pyllm.create_chat(model="gpt-4o").ask("hi")
    assert msg.content == "Hello!"
    assert msg.input_tokens == 10
