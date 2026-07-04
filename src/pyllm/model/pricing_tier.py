"""Non-zero pricing values for one tier."""

from __future__ import annotations

from typing import Any

ATTRIBUTES = (
    "input_per_million",
    "output_per_million",
    "cache_read_input_per_million",
    "cache_write_input_per_million",
    "reasoning_output_per_million",
)


class PricingTier:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._values: dict[str, float] = {}
        for key, value in (data or {}).items():
            if value and value != 0.0:
                self._values[str(key)] = value

    @property
    def input_per_million(self) -> float | None:
        return self._values.get("input_per_million")

    @property
    def output_per_million(self) -> float | None:
        return self._values.get("output_per_million")

    @property
    def cache_read_input_per_million(self) -> float | None:
        return self._values.get("cache_read_input_per_million")

    @property
    def cache_write_input_per_million(self) -> float | None:
        return self._values.get("cache_write_input_per_million")

    @property
    def reasoning_output_per_million(self) -> float | None:
        return self._values.get("reasoning_output_per_million")

    def to_dict(self) -> dict[str, float]:
        return dict(self._values)
