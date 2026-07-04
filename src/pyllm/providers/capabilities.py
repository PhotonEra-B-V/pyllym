"""Base provider capabilities.

Per-provider capability heuristics used mainly when ``list_models`` returns
bare model ids. Since the packaged ``models.json`` is the primary source of
truth, the base implementation returns conservative defaults; providers may
override.
"""

from __future__ import annotations

from typing import Any


class Capabilities:
    @classmethod
    def context_window_for(cls, model_id: str) -> int | None:
        return None

    @classmethod
    def max_tokens_for(cls, model_id: str) -> int | None:
        return None

    @classmethod
    def critical_capabilities_for(cls, model_id: str) -> list[str]:
        return ["streaming"]

    @classmethod
    def pricing_for(cls, model_id: str) -> dict[str, Any]:
        return {}

    @classmethod
    def model_type(cls, model_id: str) -> str:
        return "chat"

    @classmethod
    def model_family(cls, model_id: str) -> str | None:
        return None
