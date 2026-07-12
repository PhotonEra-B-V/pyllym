"""Content moderation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import models as _models

if TYPE_CHECKING:
    from .context import Context


class Moderation:
    def __init__(self, *, id: str, model: str, results: list[dict[str, Any]]) -> None:
        self.id = id
        self.model = model
        self.results = results

    @property
    def content(self) -> list[dict[str, Any]]:
        return self.results

    def is_flagged(self) -> bool:
        return any(result.get("flagged") for result in self.results)

    def flagged_categories(self) -> list[str]:
        out: list[str] = []
        for result in self.results:
            categories = result.get("categories") or {}
            out.extend(name for name, flagged in categories.items() if flagged)
        return list(dict.fromkeys(out))

    def category_scores(self) -> dict[str, Any]:
        return (self.results[0].get("category_scores") if self.results else {}) or {}

    def categories(self) -> dict[str, Any]:
        return (self.results[0].get("categories") if self.results else {}) or {}


async def moderate(
    input: Any,
    *,
    model: str | None = None,
    provider: str | None = None,
    assume_model_exists: bool = False,
    context: Context | None = None,
) -> Moderation:
    from . import config as _config

    cfg = context.config if context else _config()
    model = model or cfg.default_moderation_model
    model_info, provider_instance = _models.resolve(
        model, provider=provider, assume_exists=assume_model_exists, config=cfg
    )
    return await provider_instance.moderate(input, model=model_info.id)
