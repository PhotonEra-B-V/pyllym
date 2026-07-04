"""Input/output modalities of a model."""

from __future__ import annotations

from typing import Any

from .. import utils


class Modalities:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        self.input = [str(x) for x in utils.to_safe_array(data.get("input"))]
        self.output = [str(x) for x in utils.to_safe_array(data.get("output"))]

    def to_dict(self) -> dict[str, list[str]]:
        return {"input": self.input, "output": self.output}
