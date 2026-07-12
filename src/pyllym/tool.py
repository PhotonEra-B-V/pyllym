"""Base class for tools (function calling).

Introspects the ``execute`` method's keyword signature to validate and
infer parameters. Python mirrors this with :mod:`inspect`: subclasses implement
``execute`` (sync or async) with typed keyword parameters, or declare
parameters explicitly via the ``param`` classmethod / ``params_schema``.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, ClassVar

from . import utils

logger = logging.getLogger("pyllym")

_TYPE_MAP = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    list: "array",
    dict: "object",
}


class Parameter:
    def __init__(
        self,
        name: str,
        *,
        type: str = "string",
        desc: str | None = None,
        description: str | None = None,
        required: bool = True,
    ) -> None:
        self.name = name
        self.type = type
        self.description = desc or description
        self.required = required

    def to_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": _map_type(self.type)}
        if self.description:
            schema["description"] = self.description
        if schema["type"] == "array":
            schema["items"] = {"type": "string"}
        return schema


def _map_type(type: Any) -> str:
    if isinstance(type, str):
        key = type.lower()
        return {
            "integer": "integer",
            "int": "integer",
            "number": "number",
            "float": "number",
            "double": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }.get(key, "string")
    return _TYPE_MAP.get(type, "string")


class Tool:
    """Subclass and implement :meth:`execute`.

    Optionally set class attributes ``description`` and ``params_schema`` (a raw
    JSON-schema dict), or declare :class:`Parameter` entries via :meth:`param`.
    """

    description: ClassVar[str | None] = None
    _parameters: ClassVar[dict[str, Parameter]]
    _params_schema: ClassVar[dict[str, Any] | None] = None
    provider_params: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own parameter registry.
        cls._parameters = dict(getattr(cls, "_parameters", {}))

    # --- class DSL -------------------------------------------------------------
    @classmethod
    def param(cls, name: str, **options: Any) -> None:
        cls._parameters[name] = Parameter(name, **options)

    @classmethod
    def parameters(cls) -> dict[str, Parameter]:
        return getattr(cls, "_parameters", {})

    # --- instance API ----------------------------------------------------------
    @property
    def name(self) -> str:
        raw = type(self).__name__
        ascii_name = raw.encode("ascii", "ignore").decode("ascii")
        import re

        ascii_name = re.sub(r"[^a-zA-Z0-9_-]", "-", ascii_name)
        underscored = utils.underscore(ascii_name)
        return underscored.removesuffix("_tool")

    @property
    def params_schema(self) -> dict[str, Any] | None:
        if type(self)._params_schema is not None:
            return type(self)._params_schema
        params = self.parameters()
        if params:
            return _schema_from_parameters(params)
        return _schema_from_parameters(self._inferred_parameters(), allow_empty=True)

    async def call(self, args: Any) -> Any:
        normalized = self._normalize_args(args)
        error = self._validate_keyword_arguments(normalized)
        if error:
            return {"error": f"Invalid tool arguments: {error}"}
        logger.debug("Tool %s called with: %r", self.name, normalized)
        result = self.execute(**normalized)
        if inspect.isawaitable(result):
            result = await result
        logger.debug("Tool %s returned: %r", self.name, result)
        return result

    def execute(self, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement execute()")

    # --- internals -------------------------------------------------------------
    @staticmethod
    def _normalize_args(args: Any) -> dict[str, Any]:
        if args is None:
            return {}
        if isinstance(args, dict):
            return {str(k): v for k, v in args.items()}
        return {}

    def _execute_signature(self) -> tuple[list[str], list[str], bool, bool]:
        sig = inspect.signature(self.execute)
        required: list[str] = []
        optional: list[str] = []
        accepts_extra = False
        accepts_positional = False
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_extra = True
            elif (
                p.kind == inspect.Parameter.VAR_POSITIONAL
                or p.kind == inspect.Parameter.POSITIONAL_ONLY
            ):
                accepts_positional = True
            elif p.kind in (
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if p.default is inspect.Parameter.empty:
                    required.append(p.name)
                else:
                    optional.append(p.name)
        return required, optional, accepts_extra, accepts_positional

    def _validate_keyword_arguments(self, arguments: dict[str, Any]) -> str | None:
        required, optional, accepts_extra, accepts_positional = self._execute_signature()
        if not required and not optional and accepts_positional:
            return None
        argument_keys = set(arguments)
        missing = next((k for k in required if k not in argument_keys), None)
        if missing:
            return f"missing keyword: {missing}"
        if accepts_extra:
            return None
        allowed = set(required) | set(optional)
        unknown = next((k for k in argument_keys if k not in allowed), None)
        if unknown:
            return f"unknown keyword: {unknown}"
        return None

    def _inferred_parameters(self) -> dict[str, Parameter]:
        required, optional, _, _ = self._execute_signature()
        return {name: Parameter(name, required=name in required) for name in (*required, *optional)}


def _schema_from_parameters(
    parameters: dict[str, Parameter], *, allow_empty: bool = False
) -> dict[str, Any] | None:
    if not parameters and not allow_empty:
        return None
    properties = {name: param.to_schema() for name, param in parameters.items()}
    required = [name for name, param in parameters.items() if param.required]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
        "strict": True,
    }
