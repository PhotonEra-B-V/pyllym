"""Citation value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Citation:
    """Links a span of generated text to its supporting source, normalized
    across providers."""

    url: str | None = None
    title: str | None = None
    cited_text: str | None = None
    text: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    source_index: int | None = None
    start_page: int | None = None
    end_page: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Citation:
        fields = set(cls.__slots__)
        return cls(**{k: v for k, v in data.items() if k in fields})

    # Uniform value-object constructor name (see CLAUDE.md conventions).
    build = from_dict

    def to_dict(self) -> dict[str, Any]:
        data = {
            "url": self.url,
            "title": self.title,
            "cited_text": self.cited_text,
            "text": self.text,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "source_index": self.source_index,
            "start_page": self.start_page,
            "end_page": self.end_page,
        }
        return {k: v for k, v in data.items() if v is not None}
