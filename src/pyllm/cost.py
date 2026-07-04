"""Cost of token usage for a model response."""

from __future__ import annotations

from typing import Any

from .model.pricing_category import PricingCategory

COMPONENTS = ("input", "output", "cache_read", "cache_write", "thinking")
PER_MILLION = 1_000_000.0


class Cost:
    @staticmethod
    def aggregate(costs: list[Cost | None]) -> Aggregate:
        return Aggregate.build(costs)

    def __init__(
        self,
        *,
        tokens: Any = None,
        model: Any = None,
        category: str = "text_tokens",
        input_details: dict[str, Any] | None = None,
    ) -> None:
        self.tokens = tokens
        self.model = self._normalize_model(model)
        self.category = str(category)
        self._input_details = input_details

    @property
    def input(self) -> float | None:
        return self._amount_for("input")

    @property
    def output(self) -> float | None:
        return self._amount_for("output")

    @property
    def cache_read(self) -> float | None:
        return self._amount_for("cache_read")

    @property
    def cache_write(self) -> float | None:
        return self._amount_for("cache_write")

    @property
    def thinking(self) -> float | None:
        return self._amount_for("thinking")

    @property
    def total(self) -> float | None:
        if not self.has_tokens():
            return None
        if any(self.is_missing(c) for c in COMPONENTS):
            return None
        costs = [v for c in COMPONENTS if (v := getattr(self, c)) is not None]
        return sum(costs) if costs else None

    def to_dict(self) -> dict[str, float]:
        data = {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "thinking": self.thinking,
            "total": self.total,
        }
        return {k: v for k, v in data.items() if v is not None}

    def has_tokens(self) -> bool:
        return any(self._tokens_for(c) is not None for c in COMPONENTS)

    def is_missing(self, component: str) -> bool:
        if component == "input" and self._detailed_image_input():
            return self._image_input_missing()
        if component == "thinking" and not self._thinking_priced_separately():
            return False
        tokens = self._tokens_for(component)
        return bool(tokens and int(tokens) > 0 and self._price_for(component) is None)

    # --- internals -------------------------------------------------------------
    def _amount_for(self, component: str) -> float | None:
        if component == "input" and self._detailed_image_input():
            return self._image_input_amount()
        token_count = self._tokens_for(component)
        if token_count is None:
            return None
        token_count = int(token_count)
        if token_count == 0:
            return 0.0
        price = self._price_for(component)
        if not price:
            return None
        return token_count * price / PER_MILLION

    def _tokens_for(self, component: str) -> int | None:
        if not self.tokens:
            return None
        if component == "input":
            return self.tokens.input
        if component == "output":
            return self.tokens.output
        if component == "cache_read":
            return self.tokens.cache_read
        if component == "cache_write":
            return self.tokens.cache_write
        if component == "thinking":
            return self.tokens.thinking if self._thinking_priced_separately() else None
        return None

    def _price_for(self, component: str) -> float | None:
        tp = self._text_pricing()
        if component == "input":
            return tp.input
        if component == "output":
            return self._output_pricing().output
        if component == "cache_read":
            return tp.cache_read_input
        if component == "cache_write":
            return tp.cache_write_input
        if component == "thinking":
            return tp.reasoning_output
        return None

    def _text_pricing(self) -> PricingCategory:
        if self.model and getattr(self.model, "pricing", None):
            return self.model.pricing.text_tokens
        return PricingCategory()

    def _image_pricing(self) -> PricingCategory:
        if self.model and getattr(self.model, "pricing", None):
            return self.model.pricing.images
        return PricingCategory()

    def _output_pricing(self) -> PricingCategory:
        img = self._image_pricing()
        return img if (self._is_image_cost() and img.output) else self._text_pricing()

    def _is_image_cost(self) -> bool:
        return self.category in ("image", "images")

    def _detailed_image_input(self) -> bool:
        return (
            self._is_image_cost()
            and isinstance(self._input_details, dict)
            and any(t is not None for _, t, _ in self._image_input_parts())
        )

    def _image_input_amount(self) -> float | None:
        if self._image_input_missing():
            return None
        total = 0.0
        for _, token_count, price in self._image_input_parts():
            if token_count is None or int(token_count) == 0:
                continue
            total += int(token_count) * price / PER_MILLION
        return total

    def _image_input_missing(self) -> bool:
        return any(
            token_count and int(token_count) > 0 and price is None
            for _, token_count, price in self._image_input_parts()
        )

    def _image_input_parts(self) -> list[tuple[str, Any, float | None]]:
        tp = self._text_pricing()
        ip = self._image_pricing()
        return [
            ("text", self._input_detail("text_tokens"), tp.input),
            ("image", self._input_detail("image_tokens"), ip.input or tp.input),
        ]

    def _input_detail(self, key: str) -> Any:
        details = self._input_details or {}
        return details.get(key)

    def _thinking_priced_separately(self) -> bool:
        reasoning_price = self._text_pricing().reasoning_output
        if not reasoning_price:
            return False
        output_price = self._text_pricing().output
        return output_price is None or reasoning_price != output_price

    @staticmethod
    def _normalize_model(model: Any) -> Any:
        from .errors import ModelNotFoundError

        if isinstance(model, str):
            try:
                from . import models as _models

                return _models.find(model)
            except ModelNotFoundError:
                return None
        if hasattr(model, "to_llm"):
            return model.to_llm()
        if hasattr(model, "pricing"):
            return model
        return None


class Aggregate:
    """Sum of several per-message costs with amounts precomputed at build time."""

    @staticmethod
    def build(costs: list[Cost | None]) -> Aggregate:
        kept = [c for c in costs if c is not None and c.has_tokens()]
        missing = [c for c in COMPONENTS if any(cost.is_missing(c) for cost in kept)]
        amounts: dict[str, float | None] = {}
        for component in COMPONENTS:
            values = [v for cost in kept if (v := getattr(cost, component)) is not None]
            amounts[component] = None if (component in missing or not values) else sum(values)
        return Aggregate(amounts=amounts, missing=missing, tokens=bool(kept))

    def __init__(
        self,
        *,
        amounts: dict[str, float | None],
        missing: list[str],
        tokens: bool,
    ) -> None:
        self._amounts = amounts
        self._missing = missing
        self._tokens = tokens

    @property
    def input(self) -> float | None:
        return self._amounts["input"]

    @property
    def output(self) -> float | None:
        return self._amounts["output"]

    @property
    def cache_read(self) -> float | None:
        return self._amounts["cache_read"]

    @property
    def cache_write(self) -> float | None:
        return self._amounts["cache_write"]

    @property
    def thinking(self) -> float | None:
        return self._amounts["thinking"]

    @property
    def total(self) -> float | None:
        if not self._tokens or self._missing:
            return None
        costs = [v for c in COMPONENTS if (v := self._amounts[c]) is not None]
        return sum(costs) if costs else None

    def has_tokens(self) -> bool:
        return self._tokens

    def is_missing(self, component: str) -> bool:
        return component in self._missing

    def to_dict(self) -> dict[str, float]:
        data = {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "thinking": self.thinking,
            "total": self.total,
        }
        return {k: v for k, v in data.items() if v is not None}
