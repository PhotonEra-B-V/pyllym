"""Config-driven tools: expose allowlisted library callables to the model.

A general chat model is bad at arithmetic and worse at statistics — it will
happily hallucinate a standard deviation. The fix pyllym uses everywhere is
*tools*: hand the model real functions and run the model -> tool -> model loop
so the numbers come from ``numpy``/``statistics``, not the model's guesses.

Writing a :class:`~pyllym.tool.Tool` subclass per function is boilerplate when
the function already exists in a library. This module reads a **TOML file** that
lists callables by dotted path and turns each into a ready-to-use ``Tool``::

    # analysis_tools.toml
    [[tools]]
    path = "numpy.mean"
    description = "Arithmetic mean of a list of numbers."
    [tools.params.a]
    type = "array"
    items = "number"
    description = "The numbers to average."

    [[tools]]
    path = "statistics.median"

Then::

    import pyllym

    tools = pyllym.load_toolset("analysis_tools.toml")
    chat = pyllym.create_chat(model="llama3.1").with_tools(*tools)
    answer = await chat.ask("What's the mean of [1, 2, 3, 4]? Use the tool.")

Security model — **explicit allowlist only**. Only the exact dotted paths named
in the config are imported and callable; there is no wildcard/module expansion
and no model-authored code path. The listed callables run **in-process**, so a
toolset is exactly as trusted as the code it names: treat the ``.toml`` like a
list of imports you are choosing to run. This is defense against the *model*
reaching arbitrary functions, not a sandbox against a hostile *config* author —
review a toolset file the same way you would review an ``import`` statement.
Heavy libraries (numpy, matplotlib, torch, ...) are imported lazily, only when a
toolset actually names them, so the core install stays dependency-light. A path
whose top-level package is not installed (e.g. ``torch.mean`` without torch)
fails with an actionable "not installed — pip install" message, so an optional
library like torch is *usable if the user has it* and a clear no-op otherwise —
pyllym never depends on it.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import tomllib
import types
import typing
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .tool import Tool

__all__ = ["MissingToolPackageError", "load_toolset", "tool_from_path"]

logger = logging.getLogger("pyllym")


class MissingToolPackageError(ConfigurationError):
    """A toolset path names a top-level package that isn't installed.

    A subclass of :class:`~pyllym.errors.ConfigurationError` so callers can
    catch it specifically — :func:`load_toolset` skips such entries by default
    (an opt-in library like torch is only used if the user has it), while
    :func:`tool_from_path` still raises since it names exactly one tool.
    """

    def __init__(self, package: str, dotted: str) -> None:
        self.package = package
        self.dotted = dotted
        super().__init__(
            f"toolset path {dotted!r}: package {package!r} is not installed. "
            f"Install it to use this tool (e.g. `pip install {package}`)."
        )


# Python annotation -> JSON-schema "type". Mirrors tool._TYPE_MAP but works off
# real annotations (including PEP 604 unions and builtin generics) so a
# ``list[float]`` parameter advertises as an array-of-number, not a string.
_SCALAR_JSON_TYPE: dict[Any, str] = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    type(None): "null",
}


def load_toolset(path: str | Path, *, skip_missing: bool = True) -> list[Tool]:
    """Load a TOML toolset file and return one :class:`Tool` per ``[[tools]]``.

    See the module docstring for the file format and security model. Raises
    :class:`~pyllym.errors.ConfigurationError` for a malformed file, a missing
    ``path``, or a dotted path that does not import to a callable.

    ``skip_missing`` (default ``True``): entries whose top-level package is not
    installed are skipped with a warning rather than failing the whole file, so
    a toolset can mix always-present and opt-in libraries (e.g. stdlib +
    ``torch``). Set ``False`` to require every named package be installed.
    """
    file = Path(path)
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"toolset file not found: {file}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in toolset {file}: {exc}") from exc

    entries = data.get("tools")
    if not isinstance(entries, list):
        raise ConfigurationError(f"toolset {file} must define an array of [[tools]] tables")

    tools: list[Tool] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"toolset {file}: [[tools]] entry {index} is not a table")
        dotted = entry.get("path")
        if not isinstance(dotted, str) or not dotted:
            raise ConfigurationError(f"toolset {file}: [[tools]] entry {index} is missing 'path'")
        try:
            tool = tool_from_path(
                dotted,
                name=entry.get("name"),
                description=entry.get("description"),
                params=entry.get("params"),
            )
        except MissingToolPackageError as exc:
            if not skip_missing:
                raise
            logger.warning("toolset %s: skipping %r — %s", file, dotted, exc)
            continue
        if tool.name in seen:
            raise ConfigurationError(
                f"toolset {file}: duplicate tool name {tool.name!r} "
                f"(from {dotted!r}); set an explicit 'name'"
            )
        seen.add(tool.name)
        tools.append(tool)
    return tools


def tool_from_path(
    dotted: str,
    *,
    name: str | None = None,
    description: str | None = None,
    params: dict[str, Any] | None = None,
) -> Tool:
    """Import an allowlisted dotted-path callable and wrap it as a ``Tool``.

    ``params`` optionally overrides per-parameter JSON schema (``type``,
    ``items``, ``description``) — needed when a callable has no useful
    annotations, or to describe a parameter for the model.
    """
    func = _import_callable(dotted)
    schema = _schema_for(func, params or {})
    fallback_name = _default_name(dotted)
    doc = description or _first_doc_line(func) or f"Call {dotted}."

    class _ConfiguredTool(Tool):
        _params_schema = schema

        @property
        def name(self) -> str:
            return name or fallback_name

        @property
        def description(self) -> str:  # type: ignore[override]
            return doc

        async def execute(self, **kwargs: Any) -> Any:
            result = func(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return _jsonable(result)

    return _ConfiguredTool()


def _import_callable(dotted: str) -> Any:
    """Resolve ``pkg.mod.attr`` to a callable, importing only ``dotted`` itself.

    Tries progressively shorter module prefixes so ``numpy.mean`` (attribute on
    a module) and ``pkg.mod.Class.staticmethod`` both resolve.
    """
    if "." not in dotted:
        raise ConfigurationError(f"toolset path {dotted!r} must be a dotted 'module.attr' path")
    parts = dotted.split(".")
    root = parts[0]

    # If the top-level package itself is not installed, say so with an install
    # hint rather than a generic "could not import" — this is the torch/numpy/…
    # opt-in case: the toolset only works if the user has that library.
    try:
        importlib.import_module(root)
    except ModuleNotFoundError as exc:
        if exc.name == root:
            raise MissingToolPackageError(root, dotted) from exc
        # A submodule of an installed package is missing; fall through to the
        # per-prefix loop below, which reports it against the dotted path.

    for split in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:split])
        try:
            obj: Any = importlib.import_module(module_path)
        except ImportError:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            raise ConfigurationError(f"toolset path {dotted!r}: {exc}") from exc
        if not callable(obj):
            raise ConfigurationError(f"toolset path {dotted!r} did not resolve to a callable")
        return obj
    raise ConfigurationError(f"toolset path {dotted!r}: could not import module for it")


def _schema_for(func: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-schema params object from the callable's signature.

    Parameter types come from annotations; ``overrides[name]`` (from the config)
    wins over the inferred schema. A callable with an uninspectable signature
    (many C builtins) yields an open schema unless the config supplies params.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    sig: inspect.Signature | None = None
    hints: dict[str, Any] = {}
    try:
        sig = inspect.signature(func)
        hints = _safe_hints(func)
    except (ValueError, TypeError):
        sig = None

    if sig is not None:
        for pname, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            properties[pname] = _param_schema(hints.get(pname), overrides.get(pname))
            if param.default is inspect.Parameter.empty:
                required.append(pname)

    # Config may describe params the signature didn't expose (e.g. C builtins).
    for pname, spec in overrides.items():
        if pname not in properties:
            properties[pname] = _param_schema(None, spec)
            if isinstance(spec, dict) and spec.get("required", True):
                required.append(pname)

    if not properties:
        # Uninspectable and undescribed: accept an open object so the call still
        # works; the model has only the description to go on.
        return {"type": "object", "properties": {}, "additionalProperties": True}

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _param_schema(annotation: Any, override: Any) -> dict[str, Any]:
    schema = _schema_from_annotation(annotation)
    if isinstance(override, dict):
        if "type" in override:
            schema = {"type": override["type"]}
            if override["type"] == "array":
                items = override.get("items", "string")
                schema["items"] = {"type": items}
        if "items" in override and schema.get("type") == "array":
            schema["items"] = {"type": override["items"]}
        if override.get("description"):
            schema["description"] = override["description"]
    return schema


def _schema_from_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is None or annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = typing.get_origin(annotation)
    if origin in (list, set, tuple, frozenset):
        (item,) = (typing.get_args(annotation) or (str,))[:1] or (str,)
        return {"type": "array", "items": _schema_from_annotation(item)}
    if origin in (dict,):
        return {"type": "object"}
    if origin in (typing.Union, types.UnionType):
        # Pick the first non-None member for the advertised type.
        members = [a for a in typing.get_args(annotation) if a is not type(None)]
        return _schema_from_annotation(members[0]) if members else {"type": "string"}
    return {"type": _SCALAR_JSON_TYPE.get(annotation, "string")}


def _safe_hints(func: Any) -> dict[str, Any]:
    try:
        return typing.get_type_hints(func)
    except Exception:
        # Unresolvable forward refs / exotic annotations: fall back to raw.
        return dict(getattr(func, "__annotations__", {}))


def _first_doc_line(func: Any) -> str | None:
    doc = inspect.getdoc(func)
    if not doc:
        return None
    return doc.strip().splitlines()[0].strip() or None


def _default_name(dotted: str) -> str:
    # e.g. "numpy.mean" -> "numpy_mean", "statistics.median" -> "statistics_median"
    tail = dotted.split(".")[-2:]
    return "_".join(tail).replace("-", "_")


def _jsonable(value: Any) -> Any:
    """Coerce common non-JSON returns (numpy scalars/arrays) to plain Python.

    Kept dependency-free: we duck-type via ``tolist``/``item`` rather than
    importing numpy, so this works for any array-like without adding a dep.
    """
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item") and type(value).__module__ not in ("builtins",):
        try:
            return value.item()
        except Exception:
            pass
    return value
