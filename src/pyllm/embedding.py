"""Embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import models as _models

if TYPE_CHECKING:
    from .context import Context


@dataclass(slots=True)
class Embedding:
    vectors: Any
    model: str
    input_tokens: int = 0


async def embed(
    text: Any,
    *,
    model: str | None = None,
    provider: str | None = None,
    assume_model_exists: bool = False,
    context: Context | None = None,
    dimensions: int | None = None,
) -> Embedding:
    from . import config as _config

    cfg = context.config if context else _config()
    model = model or cfg.default_embedding_model
    model_info, provider_instance = _models.resolve(
        model, provider=provider, assume_exists=assume_model_exists, config=cfg
    )
    return await provider_instance.embed(text, model=model_info.id, dimensions=dimensions)
