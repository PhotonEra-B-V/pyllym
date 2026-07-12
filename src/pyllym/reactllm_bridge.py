"""pyllum as the request-mode **planner backend** for reactllm-sdd.

reactllm-sdd (TypeScript) compiles a TOML spec into a red React Testing Library
suite. In *request mode* it needs a model to fill a typed ``TestPlan`` via
structured output. Rather than couple the two repos over HTTP, they agree on a
**shared JSON contract** — ``reactllm-pyllum.contract.json`` — and pyllum reads
that file and fulfils its side. Neither repo imports the other; the contract is
the only interface.

The whole connection is gated by ONE environment variable,
``REACTLLM_PYLLUM_CONTRACT`` (an absolute path to the contract):

* **set**   → the bridge is active; load the contract from that path.
* **unset** → the bridge is OFF. :func:`is_enabled` is ``False`` and nothing
  errors (reactllm falls back to its own client / plan mode on its side).

``REACTLLM_PYLLUM_MODEL`` selects the model pyllum uses to fill the plan
(consulted only when the contract path is set). Provider API keys follow
pyllum's existing config precedence (env vars / :func:`pyllym.configure`).

The contract is authoritative. Its ``test_plan_schema`` is a JSON Schema
(draft 2019-09) with **camelCase** keys (``testName``, ``isAsync``,
``edgeCase``, ``requiredIds``); pyllum feeds it straight to
:meth:`Chat.with_schema` so the returned JSON keys match exactly — reactllm
consumes the object directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .chat import Chat

CONTRACT_PATH_ENV = "REACTLLM_PYLLUM_CONTRACT"
MODEL_ENV = "REACTLLM_PYLLUM_MODEL"

#: Contract ``version`` values this bridge knows how to fulfil. The schema is
#: expected to drift; a silent drift produces plans reactllm rejects, so an
#: unsupported version fails loudly (:class:`UnsupportedContractVersion`).
SUPPORTED_CONTRACT_VERSIONS: frozenset[int] = frozenset({1})

#: Model used to fill the plan when ``REACTLLM_PYLLUM_MODEL`` is unset.
DEFAULT_MODEL = "gpt-4o"


class UnsupportedContractVersion(RuntimeError):
    """Raised when a contract's ``version`` is outside the supported set."""

    def __init__(self, version: Any) -> None:
        super().__init__(
            f"reactllm contract version {version!r} is unsupported; "
            f"this bridge supports {sorted(SUPPORTED_CONTRACT_VERSIONS)}"
        )
        self.version = version


class BridgeDisabled(RuntimeError):
    """Raised when the bridge is used but ``REACTLLM_PYLLUM_CONTRACT`` is unset."""

    def __init__(self) -> None:
        super().__init__(
            f"reactllm bridge is disabled: set {CONTRACT_PATH_ENV} to the "
            "absolute path of reactllm-pyllum.contract.json to enable it"
        )


# --- switch + contract loading -------------------------------------------------


def contract_path(explicit: str | None = None) -> str | None:
    """The resolved contract path, or ``None`` when the bridge is off.

    An explicit argument wins; otherwise the ``REACTLLM_PYLLUM_CONTRACT`` env
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
    """The model id pyllum uses to fill the plan (``REACTLLM_PYLLUM_MODEL``)."""
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
        raise ValueError(f"reactllm contract at {resolved!r} is not a JSON object")
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
        raise ValueError("reactllm contract is missing a 'test_plan_schema' object")
    return schema


# --- the one structured-output turn -------------------------------------------


async def plan_from_spec(
    spec_prompt: str,
    *,
    model: str | None = None,
    contract_path: str | None = None,  # noqa: A002 — mirrors public kwarg name
    chat: Chat | None = None,
) -> dict[str, Any]:
    """Fill reactllm's ``TestPlan`` for one spec prompt via structured output.

    ``spec_prompt`` is the text reactllm produces from a spec (``spec.toPrompt()``),
    passed through verbatim. ``model`` defaults to ``REACTLLM_PYLLUM_MODEL``;
    ``contract_path`` defaults to ``REACTLLM_PYLLUM_CONTRACT``. Returns the parsed
    plan dict — the same object reactllm's ``ChatClient.ask`` would return as
    ``content``, with camelCase keys honoring ``test_plan_schema``.

    ``chat`` is injectable for tests. If the model cannot honor the schema
    (non-object content), a clear :class:`ValueError` is raised rather than
    returning malformed JSON — reactllm treats non-object content as a failure.
    """
    contract = load_contract(contract_path)
    target = test_plan_target(contract)
    instructions = contract.get("planner_instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("reactllm contract is missing 'planner_instructions'")

    chosen_model = model or default_model()
    chat = (
        (chat or Chat(model=chosen_model))
        .with_instructions(instructions)
        .with_schema(target)
    )
    message = await chat.ask(spec_prompt)
    if not isinstance(message.content, dict):
        raise ValueError(
            "reactllm planner model returned non-object content; "
            "the model could not honor test_plan_schema"
        )
    return message.content
