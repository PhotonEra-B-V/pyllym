"""CLI: ``python -m pyllym.bdd features/ --out tests/generated --model gpt-5.4``."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from ..connection import aclose
from .builder import BuildResult, build


async def _build_and_close(*args: Any, **kwargs: Any) -> list[BuildResult]:
    # Close the per-loop shared HTTP pools before asyncio.run tears the loop down.
    try:
        return await build(*args, **kwargs)
    finally:
        await aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pyllym.bdd",
        description="Turn a .toml spec into a red pytest suite plus build briefs.",
    )
    parser.add_argument(
        "source",
        help="a .toml spec file or a directory containing them",
    )
    parser.add_argument("--out", default="tests/generated", help="output directory")
    parser.add_argument("--model", default=None, help="model id for the planner")
    parser.add_argument("--provider", default=None, help="provider slug, if ambiguous")
    parser.add_argument(
        "--api-hint", default=None, help="pin the target module / existing signatures"
    )
    parser.add_argument(
        "--toml-mode",
        choices=("auto", "plan", "request"),
        default="auto",
        help="TOML build mode: 'plan' uses hand-written [[cases]] (no LLM), "
        "'request' has the planner LLM fill the plan; 'auto' (default) detects "
        "from the presence of [[cases]]",
    )
    args = parser.parse_args(argv)

    chat_kwargs = {}
    if args.model:
        chat_kwargs["model"] = args.model
    if args.provider:
        chat_kwargs["provider"] = args.provider

    results = asyncio.run(
        _build_and_close(
            args.source,
            args.out,
            api_hint=args.api_hint,
            toml_mode=args.toml_mode,
            **chat_kwargs,
        )
    )
    for result in results:
        print(f"{result.plan.feature}: {len(result.plan.cases)} tests")
        for path in (result.test_path, result.plan_path, result.brief_path):
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
