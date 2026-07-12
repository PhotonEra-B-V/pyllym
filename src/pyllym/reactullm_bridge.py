"""pyllym as the request-mode **planner backend** for reactuLLM-sdd.

reactuLLM-sdd (TypeScript) compiles a TOML spec into a red React Testing Library
suite. In *request mode* it needs a model to fill a typed ``TestPlan`` via
structured output. Rather than couple the two repos over HTTP, they agree on a
**shared JSON contract** — ``reactullm-pyllum.contract.json`` — and pyllym reads
that file and fulfils its side. Neither repo imports the other; the contract is
the only interface.

The whole connection is gated by ONE environment variable,
``REACTULLM_PYLLUM_CONTRACT`` (an absolute path to the contract):

* **set**   → the bridge is active; load the contract from that path.
* **unset** → the bridge is OFF. :func:`is_enabled` is ``False`` and nothing
  errors (reactuLLM falls back to its own client / plan mode on its side).

``REACTULLM_PYLLUM_MODEL`` selects the model pyllym uses to fill the plan
(consulted only when the contract path is set). Provider API keys follow
pyllym's existing config precedence (env vars / :func:`pyllym.configure`).

The contract is authoritative. Its ``test_plan_schema`` is a JSON Schema
(draft 2019-09) with **camelCase** keys (``testName``, ``isAsync``,
``edgeCase``, ``requiredIds``); pyllym feeds it straight to
:meth:`Chat.with_schema` so the returned JSON keys match exactly — reactuLLM
consumes the object directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .chat import Chat

CONTRACT_PATH_ENV = "REACTULLM_PYLLUM_CONTRACT"
MODEL_ENV = "REACTULLM_PYLLUM_MODEL"

#: Contract ``version`` values this bridge knows how to fulfil. The schema is
#: expected to drift; a silent drift produces plans reactuLLM rejects, so an
#: unsupported version fails loudly (:class:`UnsupportedContractVersion`).
SUPPORTED_CONTRACT_VERSIONS: frozenset[int] = frozenset({1})

#: Model used to fill the plan when ``REACTULLM_PYLLUM_MODEL`` is unset.
DEFAULT_MODEL = "gpt-4o"


class UnsupportedContractVersion(RuntimeError):
    """Raised when a contract's ``version`` is outside the supported set."""

    def __init__(self, version: Any) -> None:
        super().__init__(
            f"reactuLLM contract version {version!r} is unsupported; "
            f"this bridge supports {sorted(SUPPORTED_CONTRACT_VERSIONS)}"
        )
        self.version = version


class BridgeDisabled(RuntimeError):
    """Raised when the bridge is used but ``REACTULLM_PYLLUM_CONTRACT`` is unset."""

    def __init__(self) -> None:
        super().__init__(
            f"reactuLLM bridge is disabled: set {CONTRACT_PATH_ENV} to the "
            "absolute path of reactullm-pyllum.contract.json to enable it"
        )


# --- switch + contract loading -------------------------------------------------


def contract_path(explicit: str | None = None) -> str | None:
    """The resolved contract path, or ``None`` when the bridge is off.

    An explicit argument wins; otherwise the ``REACTULLM_PYLLUM_CONTRACT`` env
    var is consulted. An empty / whitespace-only value counts as unset.
    """
    raw = explicit if explicit is not None else os.environ.get(CONTRACT_PATH_ENV)
    if raw is None or not raw.strip():
        return None
    return raw


def is_enabled(explicit: str | None = None) -> bool:
    """Whether the bridge is switched on (the contract path resolves)."""
    return contract_path(explicit) is not None


def default_model() -> str:
    """The model id pyllym uses to fill the plan (``REACTULLM_PYLLUM_MODEL``)."""
    raw = os.environ.get(MODEL_ENV)
    return raw if raw and raw.strip() else DEFAULT_MODEL


def load_contract(path: str | None = None) -> dict[str, Any]:
    """Load and version-check the contract JSON.

    Raises :class:`BridgeDisabled` when no path resolves (env unset and no
    argument), and :class:`UnsupportedContractVersion` when the contract's
    ``version`` is outside :data:`SUPPORTED_CONTRACT_VERSIONS`.
    """
    resolved = contract_path(path)
    if resolved is None:
        raise BridgeDisabled()
    data = json.loads(Path(resolved).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"reactuLLM contract at {resolved!r} is not a JSON object")
    version = data.get("version")
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        raise UnsupportedContractVersion(version)
    return data


def test_plan_target(contract: dict[str, Any]) -> dict[str, Any]:
    """The structured-output target: the contract's ``test_plan_schema`` object.

    Returned as-is (a JSON Schema dict). :meth:`Chat.with_schema` accepts a
    plain JSON Schema directly, and its camelCase keys / ``default``s /
    ``required`` are carried through untouched so the returned JSON matches the
    schema exactly.
    """
    schema = contract.get("test_plan_schema")
    if not isinstance(schema, dict):
        raise ValueError("reactuLLM contract is missing a 'test_plan_schema' object")
    return schema


# --- the one structured-output turn -------------------------------------------


async def plan_from_spec(
    spec_prompt: str,
    *,
    model: str | None = None,
    contract_path: str | None = None,
    chat: Chat | None = None,
) -> dict[str, Any]:
    """Fill reactuLLM's ``TestPlan`` for one spec prompt via structured output.

    ``spec_prompt`` is the text reactuLLM produces from a spec (``spec.toPrompt()``),
    passed through verbatim. ``model`` defaults to ``REACTULLM_PYLLUM_MODEL``;
    ``contract_path`` defaults to ``REACTULLM_PYLLUM_CONTRACT``. Returns the parsed
    plan dict — the same object reactuLLM's ``ChatClient.ask`` would return as
    ``content``, with camelCase keys honoring ``test_plan_schema``.

    ``chat`` is injectable for tests. If the model cannot honor the schema
    (non-object content), a clear :class:`ValueError` is raised rather than
    returning malformed JSON — reactuLLM treats non-object content as a failure.
    """
    contract = load_contract(contract_path)
    target = test_plan_target(contract)
    instructions = contract.get("planner_instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("reactuLLM contract is missing 'planner_instructions'")

    chosen_model = model or default_model()
    chat = (chat or Chat(model=chosen_model)).with_instructions(instructions).with_schema(target)
    message = await chat.ask(spec_prompt)
    if not isinstance(message.content, dict):
        raise ValueError(
            "reactuLLM planner model returned non-object content; "
            "the model could not honor test_plan_schema"
        )
    return message.content


# --- CLI ----------------------------------------------------------------------


async def _plan_and_close(spec_prompt: str, *, model: str | None) -> dict[str, Any]:
    from .connection import aclose

    try:
        return await plan_from_spec(spec_prompt, model=model)
    finally:
        # Close the per-loop shared HTTP pools before asyncio.run tears the loop down.
        await aclose()


def main(argv: list[str] | None = None) -> int:
    """``python -m pyllym.reactullm_bridge spec.prompt.txt --out plan.json``.

    Reads a spec prompt from a file (or stdin with ``-``), fills the ``TestPlan``
    via the shared contract, and writes it as JSON to stdout or ``--out``. Honors
    ``REACTULLM_PYLLUM_CONTRACT`` / ``REACTULLM_PYLLUM_MODEL``; ``--model``
    overrides. Exits non-zero (without producing output) when the bridge is
    disabled.
    """
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m pyllym.reactullm_bridge",
        description="Fill reactuLLM-sdd's request-mode TestPlan via the shared contract.",
    )
    parser.add_argument(
        "prompt",
        help="a file holding reactuLLM's spec prompt, or '-' to read stdin",
    )
    parser.add_argument("--out", default=None, help="write the plan JSON here (default: stdout)")
    parser.add_argument(
        "--model", default=None, help=f"model id (default: ${MODEL_ENV} or {DEFAULT_MODEL})"
    )
    args = parser.parse_args(argv)

    if not is_enabled():
        print(
            f"reactuLLM bridge is disabled: set {CONTRACT_PATH_ENV} to the "
            "absolute path of reactullm-pyllum.contract.json to enable it",
            file=sys.stderr,
        )
        return 2

    spec_prompt = sys.stdin.read() if args.prompt == "-" else Path(args.prompt).read_text("utf-8")

    plan = asyncio.run(_plan_and_close(spec_prompt, model=args.model))
    rendered = json.dumps(plan, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
