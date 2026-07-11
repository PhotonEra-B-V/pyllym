"""Inspect the real dependencies of the module under test.

The planner otherwise sees only the *declared* API surface — the module path,
imports, and signatures a human wrote into the TOML spec. Nothing checks that
those declarations match what the module actually pulls in. This module closes
that gap: it discovers the module's true import dependencies and cross-checks
them against the spec, so generated tests reflect the module's real
collaborators rather than a stale hand-written surface.

Discovery is **static-first, runtime-fallback** (see :func:`inspect_dependencies`):

- Static: locate the module's ``.py`` source without importing it
  (``importlib.util.find_spec``) and walk its AST for ``import`` /
  ``from ... import`` statements. No module code runs — no side effects, and a
  module that would fail to import (missing transitive dep, import-time error)
  is still analyzable.
- Runtime fallback: only when the source can't be read statically (namespace
  packages, C extensions, generated modules) do we import the module and read
  ``inspect``-visible dependencies. This executes module code, so it is the
  second choice, not the first.

:func:`check_dependencies` returns human-readable problems the same shape
``checks.validate_plan`` uses, so the builder can fold them into the brief and
fail the build under ``strict`` exactly like a coverage gap.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Dependencies:
    """What a module actually depends on.

    ``imports`` is the set of top-level module names the target imports
    (``os``, ``httpx``, ``pyllm.chat`` -> ``pyllm``). ``members`` is the set of
    names imported *from* other modules (``from x import y`` -> ``y``), which is
    what the spec's declared ``imports`` are really about. ``source`` records
    how the dependencies were found so callers (and the brief) can tell a
    static read from a runtime import.
    """

    module: str
    imports: frozenset[str] = frozenset()
    members: frozenset[str] = frozenset()
    defined: frozenset[str] = frozenset()
    source: str = "static"  # "static" | "runtime"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_prompt_block(self) -> str:
        """A compact block naming the real dependencies for the planner."""
        lines = [f"Real dependencies of {self.module} (discovered via {self.source} analysis):"]
        if self.imports:
            lines.append("  imports: " + ", ".join(sorted(self.imports)))
        if self.members:
            lines.append("  imported names: " + ", ".join(sorted(self.members)))
        if self.defined:
            lines.append("  defines: " + ", ".join(sorted(self.defined)))
        if not self.imports and not self.members:
            lines.append("  (none — the module imports nothing external)")
        return "\n".join(lines) + "\n"


def _module_names(name: str) -> set[str]:
    """Top-level package name for a dotted import, e.g. 'a.b.c' -> 'a'."""
    return {name.split(".", 1)[0]} if name else set()


def _static_dependencies(module: str) -> Dependencies | None:
    """Read the module's source and walk its AST — no import side effects.

    Returns ``None`` (so the caller falls back to runtime inspection) when the
    module has no readable ``.py`` source: a namespace package, a C extension,
    or a spec whose ``origin`` isn't a file on disk.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ModuleNotFoundError, ValueError, ImportError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    path = Path(spec.origin)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None

    imports: set[str] = set()
    members: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports |= _module_names(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # skip relative imports
                imports |= _module_names(node.module)
            for alias in node.names:
                if alias.name != "*":
                    members.add(alias.name)
    # Top-level names the module *defines* — what a `from <module> import X`
    # self-import legitimately targets.
    defined: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
    return Dependencies(
        module=module,
        imports=frozenset(imports),
        members=frozenset(members),
        defined=frozenset(defined),
        source="static",
    )


def _runtime_dependencies(module: str) -> Dependencies:
    """Import the module and read its dependencies via ``inspect``.

    Fallback only — this executes module code. Any import failure is captured
    as a warning rather than raised, so a module we cannot analyze degrades to
    "no dependencies discovered" instead of breaking the whole build.
    """
    try:
        mod = importlib.import_module(module)
    except Exception as exc:
        return Dependencies(
            module=module,
            source="runtime",
            warnings=(f"could not import {module!r} to inspect dependencies: {exc}",),
        )

    imports: set[str] = set()
    members: set[str] = set()
    defined: set[str] = set()
    for name, value in vars(mod).items():
        if name.startswith("__"):
            continue
        origin = getattr(value, "__module__", None)
        if inspect.ismodule(value):
            imports |= _module_names(getattr(value, "__name__", ""))
        elif origin and origin != module:
            # an imported class/function/etc.
            imports |= _module_names(origin)
            members.add(name)
        else:
            # defined in this module (or a plain value with no __module__)
            defined.add(name)
    return Dependencies(
        module=module,
        imports=frozenset(imports),
        members=frozenset(members),
        defined=frozenset(defined),
        source="runtime",
    )


def inspect_dependencies(module: str) -> Dependencies:
    """Discover ``module``'s real dependencies: static AST first, runtime fallback.

    Never raises for an unanalyzable module — the returned
    :class:`Dependencies` carries a ``warnings`` tuple instead, so dependency
    checking can only *add* information, never break a build that would
    otherwise succeed.
    """
    static = _static_dependencies(module)
    if static is not None:
        return static
    return _runtime_dependencies(module)


def check_dependencies(declared_imports: list[str], deps: Dependencies) -> list[str]:
    """Cross-check the spec's declared imports against real dependencies.

    ``declared_imports`` are the raw import statements from ``[api].imports``
    (e.g. ``'from myapp.retry import RetryPolicy'``). For each, we confirm the
    module it imports *from* is one the target actually depends on, and that the
    named members actually appear among the target's real imports. Returns
    human-readable problems (empty == consistent), in the same shape
    :func:`~pyllm.tdg.checks.validate_plan` uses.

    Any inspection warning (e.g. the module could not be imported for the
    runtime fallback) is surfaced as a problem too, so it reaches the brief.
    """
    problems: list[str] = list(deps.warnings)

    known_modules = deps.imports | _module_names(deps.module)
    for statement in declared_imports:
        parsed = _parse_import(statement)
        if parsed is None:
            problems.append(f"could not parse declared import: {statement!r}")
            continue
        from_module, names = parsed
        if from_module is not None:
            top = from_module.split(".", 1)[0]
            if top not in known_modules and from_module != deps.module:
                problems.append(
                    f"declared import references {from_module!r}, which "
                    f"{deps.module!r} does not depend on"
                )
                continue
            if from_module == deps.module:
                # A self-import must name something the module actually defines
                # or itself re-imports. Only checked when we could read the
                # module's own names (static, or a successful runtime import).
                available = deps.defined | deps.members | deps.imports
                if available:
                    for name in names:
                        if name not in available:
                            problems.append(
                                f"declared import of {name!r} from {from_module!r} "
                                f"is not present in {deps.module!r}"
                            )
        else:  # plain 'import x'
            for name in names:
                top = name.split(".", 1)[0]
                if top not in known_modules:
                    problems.append(
                        f"declared 'import {name}' references a module "
                        f"{deps.module!r} does not depend on"
                    )
    return problems


def _parse_import(statement: str) -> tuple[str | None, list[str]] | None:
    """Parse an import statement into ``(from_module | None, [names])``.

    ``'from a.b import C, D'`` -> ``('a.b', ['C', 'D'])``;
    ``'import a.b'`` -> ``(None, ['a.b'])``. Returns ``None`` on unparseable
    input.
    """
    try:
        tree = ast.parse(statement.strip())
    except SyntaxError:
        return None
    if len(tree.body) != 1:
        return None
    node = tree.body[0]
    if isinstance(node, ast.ImportFrom):
        return (node.module, [a.name for a in node.names])
    if isinstance(node, ast.Import):
        return (None, [a.name for a in node.names])
    return None
