"""Token usage value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Tokens:
    input: int | None = None
    output: int | None = None
    cached: int | None = None
    cache_creation: int | None = None
    thinking: int | None = None

    @classmethod
    def build(
        cls,
        *,
        input: int | None = None,
        output: int | None = None,
        cached: int | None = None,
        cache_creation: int | None = None,
        thinking: int | None = None,
    ) -> Tokens | None:
        if all(v is None for v in (input, output, cached, cache_creation, thinking)):
            return None
        return cls(
            input=input,
            output=output,
            cached=cached,
            cache_creation=cache_creation,
            thinking=thinking,
        )

    @property
    def cache_read(self) -> int | None:
        return self.cached

    @property
    def cache_write(self) -> int | None:
        return self.cache_creation

    def to_dict(self) -> dict[str, Any]:
        data = {
            "input_tokens": self.input,
            "output_tokens": self.output,
            "cached_tokens": self.cached,
            "cache_creation_tokens": self.cache_creation,
            "thinking_tokens": self.thinking,
        }
        return {k: v for k, v in data.items() if v is not None}
