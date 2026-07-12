"""AST safety gate for the executable strings in a :class:`TestPlan`.

The TDG pipeline renders LLM-authored (request mode) or hand-authored (plan
mode) strings — a case's ``when``/``then``/``given`` and fixture bodies —
verbatim into a pytest module that someone then runs. Every such string is
therefore executable Python. This module screens those strings *before*
rendering so a hallucinated ``import os`` or a prompt-injected
``__import__('os').system(...)`` fails the build loudly instead of landing on
disk as runnable code.

**This is defense-in-depth, not a sandbox.** It is a denylist over an AST
walk: it reliably catches accidents and low-effort payloads, and it is far
harder to fool than a substring scan, but a determined attacker can construct
Python that evades any static denylist. The real trust boundary is
*procedural*: never execute a freshly generated suite in an environment with
secrets or network access before a human has reviewed the diff, and run
generation/execution of untrusted specs in a sandbox. See the module docs.

The gate also parses each string, so syntactically invalid Python (a common
failure mode of the request-mode model) is caught here rather than at pytest
import time.
"""

from __future__ import annotations

import ast

from .schema import TestPlan

# Root names whose mere use is disallowed in a spec's executable strings.
# These are the common vectors for filesystem/process/network/reflection
# escapes; the API under test should never legitimately need them in a test's
# act-or-assert expression.
_BANNED_NAMES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "importlib",
        "ctypes",
        "builtins",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "compile",
    }
)

# Callables that are dangerous regardless of how they are named/aliased.
_BANNED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "open"})


class _Screen(ast.NodeVisitor):
    """Walk one expression/statement tree collecting policy violations."""

    def __init__(self, where: str) -> None:
        self.where = where
        self.problems: list[str] = []

    def _flag(self, msg: str) -> None:
        self.problems.append(f"{self.where}: {msg}")

    def visit_Import(self, node: ast.Import) -> None:
        names = ", ".join(alias.name for alias in node.names)
        self._flag(f"import statement is not allowed ({names})")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._flag(f"import statement is not allowed (from {node.module or '?'})")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Dunder attribute access is the classic reflection escape
        # (``().__class__.__bases__[0].__subclasses__()`` ...).
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self._flag(f"dunder attribute access is not allowed (.{node.attr})")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BANNED_NAMES:
            self._flag(f"disallowed name {node.id!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
            self._flag(f"call to disallowed builtin {func.id!r}")
        elif isinstance(func, ast.Attribute) and func.attr in _BANNED_CALLS:
            self._flag(f"call to disallowed function {func.attr!r}")
        self.generic_visit(node)


def _scan_source(source: str, where: str, *, mode: str = "eval") -> list[str]:
    """Parse and screen one executable string. ``mode`` is ``ast.parse`` mode."""
    source = source.strip()
    if not source or source == "None":
        return []
    try:
        tree = ast.parse(source, mode=mode)
    except SyntaxError as exc:
        return [f"{where}: not valid Python ({exc.msg})"]
    screen = _Screen(where)
    screen.visit(tree)
    return screen.problems


def scan_plan(plan: TestPlan) -> list[str]:
    """Return human-readable safety problems; an empty list means the plan is clean.

    Screens every executable string that :mod:`~pyllym.tdg.renderer` would emit
    verbatim — each case's ``given`` statements, its ``when`` act expression,
    each ``then`` assertion, and every fixture body line — rejecting imports,
    dangerous builtins/names, and dunder reflection. Fixture bodies and
    ``given`` entries are parsed as statements; ``when`` and ``then`` as
    expressions (matching how the renderer uses them).

    Not a sandbox: see the module docstring. This turns silent code execution
    into a loud build failure and validates the strings are real Python.
    """
    problems: list[str] = []

    for fixture in plan.fixtures:
        for i, line in enumerate(fixture.body):
            problems += _scan_source(line, f"fixture {fixture.name!r} body[{i}]", mode="exec")

    for case in plan.cases:
        tag = case.test_name
        # 'given' entries render as consecutive body lines, so the canonical
        # error-path form spans several entries (``with pytest.raises(...):``
        # then an indented call). Screen them as one block, not line-by-line.
        if case.given:
            problems += _scan_source("\n".join(case.given), f"case {tag!r} given", mode="exec")
        problems += _scan_source(case.when, f"case {tag!r} when", mode="eval")
        for i, expr in enumerate(case.then):
            problems += _scan_source(expr, f"case {tag!r} then[{i}]", mode="eval")

    return problems
