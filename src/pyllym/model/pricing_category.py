"""Standard/batch pricing tiers."""

from __future__ import annotations

from typing import Any

from .pricing_tier import PricingTier


def _empty_tier(tier_data: dict[str, Any] | None) -> bool:
    if not tier_data:
        return True
    return all(v is None or v == 0.0 for v in tier_data.values())


class PricingCategory:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        self.standard = (
            None if _empty_tier(data.get("standard")) else PricingTier(data.get("standard") or {})
        )
        self.batch = (
            None if _empty_tier(data.get("batch")) else PricingTier(data.get("batch") or {})
        )

    @property
    def input(self) -> float | None:
        return self.standard.input_per_million if self.standard else None

    @property
    def output(self) -> float | None:
        return self.standard.output_per_million if self.standard else None

    @property
    def cache_read_input(self) -> float | None:
        return self.standard.cache_read_input_per_million if self.standard else None

    @property
    def cache_write_input(self) -> float | None:
        return self.standard.cache_write_input_per_million if self.standard else None

    @property
    def reasoning_output(self) -> float | None:
        return self.standard.reasoning_output_per_million if self.standard else None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.standard:
            result["standard"] = self.standard.to_dict()
        if self.batch:
            result["batch"] = self.batch.to_dict()
        return result
