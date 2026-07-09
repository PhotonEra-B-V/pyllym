"""A function call from a model to a Tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    id: str | None = None
    name: str | None = None
    arguments: Any = field(default_factory=dict)
    thought_signature: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "thought_signature": self.thought_signature,
        }
        return {k: v for k, v in data.items() if v is not None}
