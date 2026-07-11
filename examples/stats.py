"""Let a *local* model do numeric/statistical analysis via tools.

A general chat model is bad at arithmetic and worse at statistics — it will
happily hallucinate a standard deviation. The fix is the same one pyllm uses
everywhere: hand the model *tools* and run the model -> tool -> model loop, so
the numbers come from numpy and the stdlib ``statistics`` module, not the
model's guesses.

Two flavours, both shown here:

1. **Curated stat tools** (``Mean``, ``StdDev``, ``Correlation``,
   ``LinearRegression``) — explicit, deterministic, no model-authored code runs.
   This is the safe default and what you should reach for first.
2. An opt-in **code tool** (``RunCode``, behind ``--allow-code``) — the model
   writes a small Python snippet and we ``exec`` it with the scientific stack
   pre-imported (numpy, scipy, pandas, sympy, scikit-learn, matplotlib as
   ``plt``, seaborn as ``sns`` — whichever are installed). The snippet may also
   ``import`` those libraries; imports are allowlisted, so ``import os`` and the
   like are refused. ``print()`` output is captured and plots are saved to
   ``examples/plots/`` (their path returned). Maximally flexible, but it runs
   model-generated code, so it's gated and import-restricted. This is NOT a
   sandbox: only enable it against a model and data you trust.

Install the sci stack for the code tool with the opt-in extra::

    pip install -e ".[sci]"

    python examples/stats.py
    python examples/stats.py --allow-code
    python examples/stats.py --model llama3.1 "Is study time correlated with grades?"
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

import _bootstrap as boot
import numpy as np

from pyllm import Tool

# Where the code tool writes any plots the model produces.
_PLOT_DIR = Path(__file__).resolve().parent / "plots"

# Optional scientific stack, exposed to the code tool as whatever is installed.
# numpy is required by this example (the curated tools use it); the rest are
# opt-in via `pip install -e ".[sci]"` and simply won't be offered if absent.
# Each entry: injected name -> (module import path, human label for the model).
_SCI_LIBS: dict[str, tuple[str, str]] = {
    "np": ("numpy", "numpy"),
    "scipy": ("scipy", "scipy"),
    "pd": ("pandas", "pandas"),
    "sympy": ("sympy", "sympy"),
    "sklearn": ("sklearn", "scikit-learn"),
    "plt": ("matplotlib.pyplot", "matplotlib (as plt)"),
    "sns": ("seaborn", "seaborn"),
}

# Submodules to eagerly attach so `scipy.stats` / `sklearn.linear_model` resolve
# regardless of the installed version's lazy-import behaviour. Best-effort.
_SCI_SUBMODULES = [
    "scipy.stats",
    "scipy.optimize",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.cluster",
]


def _sci_namespace() -> tuple[dict[str, Any], list[str]]:
    """Import whatever of the sci stack is installed. Returns (namespace, labels).

    matplotlib is forced onto the non-interactive ``Agg`` backend so plotting
    works headless; the code tool captures the current figure to a PNG. Common
    scipy/sklearn submodules are pre-imported so the model can reach them by
    dotted path (e.g. ``scipy.stats``) without an import statement.
    """
    import importlib

    ns: dict[str, Any] = {}
    labels: list[str] = []
    for alias, (path, label) in _SCI_LIBS.items():
        try:
            if path == "matplotlib.pyplot":
                import matplotlib

                matplotlib.use("Agg")  # headless: never try to open a window
            ns[alias] = importlib.import_module(path)
            labels.append(label)
        except ImportError:
            continue
    for path in _SCI_SUBMODULES:
        try:  # attaches the submodule onto its already-imported parent
            importlib.import_module(path)
        except ImportError:
            continue
    return ns, labels


# pyllm infers tool parameters from ``execute``'s signature by *name* only, so
# annotations like ``list[float]`` would still advertise as "string" to the
# model. We set ``_params_schema`` explicitly (via the helpers below) so the
# model sends real JSON arrays; ``_nums`` casts them to floats for numpy.

# A tiny dataset the questions below refer to. Swap in your own numbers.
STUDY_HOURS = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 6.0, 7.5, 8.0, 9.0]
EXAM_SCORES = [52.0, 55.0, 61.0, 60.0, 72.0, 78.0, 80.0, 88.0, 91.0, 95.0]

DEFAULT_QUESTION = (
    "Here is study time in hours per student "
    f"{STUDY_HOURS} and their exam scores {EXAM_SCORES}. "
    "What's the mean and standard deviation of the scores, how strongly are "
    "study time and score correlated, and what score does a linear fit predict "
    "for 5.5 hours of study? Use the tools for every number."
)


# --- Curated, deterministic stat tools (the safe path) ----------------------


def _num_array(desc: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "number"}, "description": desc}


def _schema(required: dict[str, Any], optional: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build an explicit params schema. Needed because pyllm infers types by
    name only, so ``list[float]`` params would otherwise advertise as strings."""
    properties = {**required, **(optional or {})}
    return {"type": "object", "properties": properties, "required": list(required)}


class Mean(Tool):
    description = "Arithmetic mean of a list of numbers."
    _params_schema = _schema({"values": _num_array("The numbers to average.")})

    async def execute(self, *, values: list[float]) -> float:
        return float(np.mean(_nums(values)))


class StdDev(Tool):
    description = "Standard deviation of a list of numbers (population by default)."
    _params_schema = _schema(
        {"values": _num_array("The numbers.")},
        {"sample": {"type": "boolean", "description": "Use the sample stddev (ddof=1)."}},
    )

    async def execute(self, *, values: list[float], sample: bool = False) -> float:
        # ddof=1 gives the sample standard deviation; 0 the population one.
        return float(np.std(_nums(values), ddof=1 if sample else 0))


class Correlation(Tool):
    description = "Pearson correlation coefficient between two equal-length number lists."
    _params_schema = _schema({"x": _num_array("First series."), "y": _num_array("Second series.")})

    async def execute(self, *, x: list[float], y: list[float]) -> float:
        xa, ya = _nums(x), _nums(y)
        if len(xa) != len(ya):
            return float("nan")
        return float(np.corrcoef(xa, ya)[0, 1])


class LinearRegression(Tool):
    description = (
        "Fit y = slope*x + intercept by least squares and optionally predict y "
        "at a given x. Returns slope, intercept and (if predict_x is given) prediction."
    )
    _params_schema = _schema(
        {"x": _num_array("Independent variable."), "y": _num_array("Dependent variable.")},
        {"predict_x": {"type": "number", "description": "If given, predict y at this x."}},
    )

    async def execute(
        self, *, x: list[float], y: list[float], predict_x: float | None = None
    ) -> dict[str, float]:
        slope, intercept = np.polyfit(_nums(x), _nums(y), 1)
        out = {"slope": float(slope), "intercept": float(intercept)}
        if predict_x is not None:
            out["prediction"] = float(slope * predict_x + intercept)
        return out


CURATED: tuple[type[Tool], ...] = (Mean, StdDev, Correlation, LinearRegression)


# --- Opt-in code tool (flexible, gated — NOT a sandbox) ---------------------


_SAFE_BUILTINS = {
    fn.__name__: fn
    for fn in (
        len,
        range,
        min,
        max,
        sum,
        abs,
        round,
        float,
        int,
        bool,
        str,
        list,
        tuple,
        dict,
        set,
        enumerate,
        zip,
        sorted,
        map,
        filter,
        print,
    )
}

# Top-level packages a snippet is allowed to ``import``. The sci stack plus a
# few pure-computation stdlib modules. Everything else (os, sys, subprocess,
# pathlib, socket, ...) raises ImportError — that's the safety boundary.
_IMPORT_ALLOWLIST = {
    "numpy",
    "scipy",
    "pandas",
    "sympy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "statistics",
    "math",
    "cmath",
    "random",
    "itertools",
    "functools",
    "collections",
}


def _guarded_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
    """A restricted ``__import__``: allows the sci stack + safe stdlib only.

    Models habitually write ``import numpy as np`` etc., so rather than fight
    that, we let imports through but only for allowlisted top-level packages.
    ``import os`` / ``from pathlib import Path`` still raise ImportError.
    """
    root = name.split(".")[0]
    if root not in _IMPORT_ALLOWLIST:
        raise ImportError(f"import of {name!r} is not allowed in this tool")
    import importlib

    module = importlib.import_module(name)
    # Mirror real __import__: for `import a.b` return the top package `a`;
    # for `from a.b import c` (fromlist set) return the submodule `a.b`.
    return module if fromlist else importlib.import_module(root)


class RunCode(Tool):
    """Execute a model-written Python snippet with the sci stack pre-imported.

    The available libraries are discovered at construction time from whatever is
    installed, and named in the tool description so the model knows what it can
    use. Imports are restricted to an allowlist (the sci stack + safe stdlib), so
    the snippet can't reach os, sys, subprocess, or the filesystem.
    """

    _base_description = (
        "Run a short Python snippet for analysis the other tools don't cover. "
        "The scientific stack is pre-imported (numpy as `np`, pandas as `pd`, "
        "matplotlib.pyplot as `plt`, seaborn as `sns`, plus `scipy`, `sympy`, "
        "`sklearn`); the stdlib `statistics` module is available as `stats`. You "
        "may also write normal imports of these libraries. Assign your final "
        "answer to a variable named `result`. To make a plot, use `plt`/`sns` "
        "and leave the figure current — it is saved to a PNG and its path "
        "returned. `print()` output is captured and returned too."
    )

    def __init__(self) -> None:
        super().__init__()
        self._sci, self._labels = _sci_namespace()

    @property
    def description(self) -> str:  # type: ignore[override]
        libs = ", ".join(self._labels) if self._labels else "none installed"
        return f"{self._base_description}\nInstalled libraries: {libs}."

    async def execute(self, *, code: str) -> Any:
        import contextlib
        import io
        import statistics

        builtins = {**_SAFE_BUILTINS, "__import__": _guarded_import}
        namespace: dict[str, Any] = {
            **self._sci,
            "stats": statistics,
            "__builtins__": builtins,
        }
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, namespace)  # gated behind --allow-code; trusted use only
        except Exception as exc:  # surface the error back to the model to retry
            return {"error": f"{type(exc).__name__}: {exc}"}

        # Assemble output from any of: an explicit `result`, captured stdout, a plot.
        out: dict[str, Any] = {}
        if "result" in namespace:
            out["result"] = _jsonable(namespace["result"])
        printed = buffer.getvalue().strip()
        if printed:
            out["stdout"] = printed
        plot_path = self._save_plot()
        if plot_path:
            out["plot"] = plot_path
        if not out:
            return {"error": "snippet produced no `result`, output, or plot"}
        return out

    def _save_plot(self) -> str | None:
        """If matplotlib drew anything, save the current figure to a PNG."""
        plt = self._sci.get("plt")
        if plt is None or not plt.get_fignums():
            return None
        path = str(_PLOT_DIR / f"plot_{len(list(_PLOT_DIR.glob('plot_*.png')))}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close("all")
        return path


def _nums(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays into plain JSON-friendly Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question", nargs="?", default=DEFAULT_QUESTION, help="What to ask the model."
    )
    parser.add_argument(
        "--model", default=boot.MODEL, help="Local model id (default: %(default)s)."
    )
    parser.add_argument(
        "--allow-code",
        action="store_true",
        help="Also expose the RunCode tool (runs model-authored code against the sci stack).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    boot.setup()

    tools: list[type[Tool]] = list(CURATED)
    if args.allow_code:
        _PLOT_DIR.mkdir(exist_ok=True)
        tools.append(RunCode)

    chat = boot.chat(model=args.model).with_tools(*tools)
    answer = await chat.ask(args.question)

    print(answer.content)
    tool_calls = [m.role for m in chat.messages if m.role == "tool"]
    print(f"\n({len(tool_calls)} tool call(s); code tool {'on' if args.allow_code else 'off'})")


if __name__ == "__main__":
    asyncio.run(main())
