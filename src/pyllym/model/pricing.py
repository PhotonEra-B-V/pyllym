"""Pricing categories for a model."""

from __future__ import annotations

from typing import Any

from .pricing_category import PricingCategory

CATEGORIES = ("text_tokens", "images", "audio_tokens", "embeddings")


def _empty_pricing(data: dict[str, Any] | None) -> bool:
    if not data:
        return True
    for tier in ("standard", "batch"):
        tier_data = data.get(tier)
        if not tier_data:
            continue
        for value in tier_data.values():
            if value and value != 0.0:
                return False
    return True


class Pricing:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        self._data: dict[str, PricingCategory] = {}
        for category in CATEGORIES:
            value = data.get(category)
            if value and not _empty_pricing(value):
                self._data[category] = PricingCategory(value)

    def _category(self, name: str) -> PricingCategory:
        return self._data.get(name) or PricingCategory()

    @property
    def text_tokens(self) -> PricingCategory:
        return self._category("text_tokens")

    @property
    def images(self) -> PricingCategory:
        return self._category("images")

    @property
    def audio_tokens(self) -> PricingCategory:
        return self._category("audio_tokens")

    @property
    def embeddings(self) -> PricingCategory:
        return self._category("embeddings")

    def to_dict(self) -> dict[str, Any]:
        return {name: cat.to_dict() for name, cat in self._data.items()}
