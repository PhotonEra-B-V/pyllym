"""pyllym as the **runtime LLM backend** for a shipped reactuLLM-sdd frontend.

This is the runtime twin of :mod:`pyllym.reactullm_bridge`. Where the bridge
fills a build-time ``TestPlan``, this module serves the *runtime* envelope: the
request/response contract a shipped React frontend and this backend both speak
when the app's **end users** invoke LLM functionality. It governs live calls,
not test generation.

The seam, in one paragraph
--------------------------
The frontend builds an :class:`LLMRequest` and POSTs it to whatever endpoint it
already uses. You validate it against the contract's ``request_schema``, **route
on ``task``** (a stable id both sides agree on — not free-form prose), set the
system prompt to ``instructions`` (or the task's registered default when
empty), and run **one** pyllym turn. If the request carried a non-null
``schema``, the turn is constrained to it (structured output) and the object is
returned in ``response.data``; otherwise the free-form answer is returned in
``response.text``. On failure, ``ok=false`` with a structured ``error``. The
``task`` is always echoed back.

This mirrors pyllym's ``chat.with_instructions(...).with_schema(...).ask(input)``.

Gating
------
Like the other reactuLLM contracts, the whole endpoint is gated by ONE
environment variable, ``REACTULLM_PYLLUM_RUNTIME_CONTRACT`` (an absolute path to
``reactullm-pyllum.runtime.json``):

* **set**   → the runtime handler is active; the contract is loaded from there.
* **unset** → the handler is OFF. :func:`is_enabled` is ``False`` and nothing
  errors (the frontend falls back on its side).

``REACTULLM_PYLLUM_MODEL`` selects the default model. Provider API keys follow
pyllym's existing config precedence; **credentials stay server-side** and are
never echoed.

Framework-agnostic
------------------
This module is pure pyllym: pydantic envelope models plus an async
:func:`handle` coroutine. It imports no web framework. Wire it into FastAPI (or
anything) with a two-line adapter::

    from pyllym.reactullm_runtime import handle, LLMRequest, LLMResponse

    @router.post("/api/v1/llm", response_model=LLMResponse, response_model_by_alias=True)
    async def run_llm(req: LLMRequest) -> LLMResponse:
        return await handle(req)

Security posture — ``task`` is the authorization key. The browser must not be
able to pick the model or inject a system prompt the server did not register;
``variables`` is untrusted browser input, allow-listed per task and never a
carrier for secrets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .chat import Chat
from .errors import Error, OverloadedError, RateLimitError

CONTRACT_PATH_ENV = "REACTULLM_PYLLUM_RUNTIME_CONTRACT"
MODEL_ENV = "REACTULLM_PYLLUM_MODEL"

#: Runtime-contract ``version`` values this handler knows how to serve. A silent
#: schema drift would produce envelopes the frontend rejects, so an unsupported
#: version fails loudly (:class:`UnsupportedContractVersion`).
SUPPORTED_CONTRACT_VERSIONS: frozenset[int] = frozenset({1})

#: Model used when neither the task nor ``REACTULLM_PYLLUM_MODEL`` picks one.
DEFAULT_MODEL = "gpt-4o"

#: The closed set of machine-routable failure classes (the contract's enum).
ERROR_CODES: frozenset[str] = frozenset(
    {
        "schema_violation",
        "unknown_task",
        "rate_limited",
        "provider_error",
        "invalid_request",
        "internal",
    }
)


class UnsupportedContractVersion(RuntimeError):
    """Raised when the runtime contract's ``version`` is outside the supported set."""

    def __init__(self, version: Any) -> None:
        super().__init__(
            f"reactuLLM runtime contract version {version!r} is unsupported; "
            f"this handler supports {sorted(SUPPORTED_CONTRACT_VERSIONS)}"
        )
        self.version = version


class RuntimeDisabled(RuntimeError):
    """Raised when the handler is used but ``REACTULLM_PYLLUM_RUNTIME_CONTRACT`` is unset."""

    def __init__(self) -> None:
        super().__init__(
            f"reactuLLM runtime handler is disabled: set {CONTRACT_PATH_ENV} to "
            "the absolute path of reactullm-pyllum.runtime.json to enable it"
        )


# --- switch + contract loading -------------------------------------------------


def contract_path(explicit: str | None = None) -> str | None:
    """The resolved runtime-contract path, or ``None`` when the handler is off.

    An explicit argument wins; otherwise ``REACTULLM_PYLLUM_RUNTIME_CONTRACT`` is
    consulted. An empty / whitespace-only value counts as unset.
    """
    raw = explicit if explicit is not None else os.environ.get(CONTRACT_PATH_ENV)
    if raw is None or not raw.strip():
        return None
    return raw


def is_enabled(explicit: str | None = None) -> bool:
    """Whether the runtime handler is switched on (the contract path resolves)."""
    return contract_path(explicit) is not None


def default_model() -> str:
    """The model id used when a task does not pin one (``REACTULLM_PYLLUM_MODEL``)."""
    raw = os.environ.get(MODEL_ENV)
    return raw if raw and raw.strip() else DEFAULT_MODEL


def load_contract(path: str | None = None) -> dict[str, Any]:
    """Load and version-check the runtime contract JSON.

    Raises :class:`RuntimeDisabled` when no path resolves (env unset and no
    argument), and :class:`UnsupportedContractVersion` when the contract's
    ``version`` is outside :data:`SUPPORTED_CONTRACT_VERSIONS`.
    """
    resolved = contract_path(path)
    if resolved is None:
        raise RuntimeDisabled()
    data = json.loads(Path(resolved).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"reactuLLM runtime contract at {resolved!r} is not a JSON object")
    version = data.get("version")
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        raise UnsupportedContractVersion(version)
    return data


def request_schema(contract: dict[str, Any]) -> dict[str, Any]:
    """The contract's ``request_schema`` object (JSON Schema for the envelope in)."""
    schema = contract.get("request_schema")
    if not isinstance(schema, dict):
        raise ValueError("reactuLLM runtime contract is missing a 'request_schema' object")
    return schema


def response_schema(contract: dict[str, Any]) -> dict[str, Any]:
    """The contract's ``response_schema`` object (JSON Schema for the envelope out)."""
    schema = contract.get("response_schema")
    if not isinstance(schema, dict):
        raise ValueError("reactuLLM runtime contract is missing a 'response_schema' object")
    return schema


# --- the wire envelope (camelCase on the wire) --------------------------------


class LLMError(BaseModel):
    """The ``error`` sub-object; present iff ``ok=false``."""

    code: str
    message: str


class LLMRequest(BaseModel):
    """The inbound runtime envelope. Wire keys are camelCase.

    The wire ``schema`` key collides with pydantic's ``BaseModel.schema`` /
    Python's convention, so it is aliased to ``schema_`` on the Python side;
    ``populate_by_name=True`` lets callers construct with either name.
    """

    model_config = ConfigDict(populate_by_name=True)

    task: str
    input: str
    instructions: str = ""
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    variables: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """The outbound runtime envelope. Serialize ``by_alias=True`` on the wire.

    Invariant: on ``ok=true`` exactly one of ``data``/``text`` is non-null,
    chosen by whether the request carried a schema; on ``ok=false`` both are
    null and ``error`` is set.
    """

    model_config = ConfigDict(populate_by_name=True)

    task: str
    ok: bool
    data: dict[str, Any] | None = None
    text: str | None = None
    error: LLMError | None = None


# --- the task registry (task -> handler config) -------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """Server-side policy for one ``task``.

    ``task`` is the authorization key: the browser routes on it but never picks
    the model or an arbitrary system prompt. ``default_prompt`` is used when the
    request's ``instructions`` are empty; ``allowed_variables`` is the allow-list
    of ``variables`` keys honored for this task (everything else is dropped);
    ``model`` optionally pins a model; ``max_input`` optionally caps input size.
    """

    default_prompt: str
    allowed_variables: frozenset[str] = frozenset()
    model: str | None = None
    max_input: int | None = None


@dataclass
class TaskRegistry:
    """A ``task -> TaskConfig`` map. Only registered tasks are servable."""

    tasks: dict[str, TaskConfig] = field(default_factory=dict)

    def register(self, task: str, config: TaskConfig) -> TaskRegistry:
        self.tasks[task] = config
        return self

    def get(self, task: str) -> TaskConfig | None:
        return self.tasks.get(task)


# --- request validation (dependency-free, structural) -------------------------


class _ValidationError(Exception):
    """Internal: a request failed structural validation against request_schema."""


def _validate_against_schema(value: Any, schema: dict[str, Any], where: str = "$") -> None:
    """Minimal structural validation against a JSON Schema subset.

    ``jsonschema`` is not a pyllym dependency; this covers what the runtime
    request_schema uses: object/array/string/boolean/null (incl. union type
    lists), ``required``, and ``additionalProperties: false``. Raises
    :class:`_ValidationError` on the first mismatch.
    """
    types = schema.get("type")
    allowed = types if isinstance(types, list) else [types] if types is not None else []

    def matches(t: str) -> bool:
        if t == "object":
            return isinstance(value, dict)
        if t == "array":
            return isinstance(value, list)
        if t == "string":
            return isinstance(value, str)
        if t == "boolean":
            return isinstance(value, bool)
        if t == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if t == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if t == "null":
            return value is None
        return True

    if allowed and not any(matches(t) for t in allowed):
        raise _ValidationError(f"{where}: expected type {allowed}, got {type(value).__name__}")

    if "object" in allowed and isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise _ValidationError(f"{where}: missing required key {key!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise _ValidationError(f"{where}: unexpected keys {sorted(extra)}")
        for key, subschema in props.items():
            if key in value:
                _validate_against_schema(value[key], subschema, f"{where}.{key}")
    elif "array" in allowed and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                _validate_against_schema(item, item_schema, f"{where}[{i}]")


# --- error mapping ------------------------------------------------------------


def _error_code_for(exc: Exception) -> str:
    """Map a raised exception to a contract ``error.code``.

    Throttling → ``rate_limited``; any other provider/network fault (including
    context-length and server errors) → ``provider_error``; anything unhandled →
    ``internal``. Callers map schema/routing failures directly at their site.
    """
    if isinstance(exc, (RateLimitError, OverloadedError)):
        return "rate_limited"
    if isinstance(exc, Error):  # any wrapped provider/network fault
        return "provider_error"
    return "internal"


def _fail(task: str, code: str, message: str) -> LLMResponse:
    """Build an ``ok=false`` envelope, never leaking keys or stack traces."""
    return LLMResponse(task=task, ok=False, error=LLMError(code=code, message=message[:200]))


# --- the one runtime turn -----------------------------------------------------


async def handle(
    request: LLMRequest,
    *,
    registry: TaskRegistry,
    contract_path: str | None = None,
    chat: Chat | None = None,
) -> LLMResponse:
    """Serve one runtime envelope: validate → route → one turn → reply.

    ``registry`` maps each ``task`` to its server-side policy (default prompt,
    allowed ``variables`` keys, optional model / max input). ``contract_path``
    defaults to ``REACTULLM_PYLLUM_RUNTIME_CONTRACT`` and is used to validate the
    incoming request against ``request_schema``. ``chat`` is injectable for
    tests (its instructions/schema are set here).

    The ``task`` is echoed on every response. On ``ok=true`` exactly one of
    ``data``/``text`` is non-null, chosen by whether the request carried a
    schema. Failures map to the contract's ``error.code`` enum and never leak
    secrets, keys, or stack traces.
    """
    task = request.task

    # 1. Validate the incoming request against the contract's request_schema.
    try:
        contract = load_contract(contract_path)
        _validate_against_schema(request.model_dump(by_alias=True), request_schema(contract))
    except (_ValidationError, ValueError) as exc:
        return _fail(task, "invalid_request", f"invalid request: {exc}")
    except (RuntimeDisabled, UnsupportedContractVersion) as exc:
        return _fail(task, "internal", str(exc))

    # 2. Route on task — the authorization key. Unknown -> unknown_task.
    cfg = registry.get(task)
    if cfg is None:
        return _fail(task, "unknown_task", f"no handler for task {task!r}")

    # 3. Per-task policy: cap input, pick model, allow-list variables.
    if cfg.max_input is not None and len(request.input) > cfg.max_input:
        return _fail(task, "invalid_request", "input exceeds the maximum size for this task")
    system = request.instructions or cfg.default_prompt
    model = cfg.model or default_model()
    variables = {k: v for k, v in request.variables.items() if k in cfg.allowed_variables}

    try:
        user_input = request.input.format(**variables) if variables else request.input
    except (KeyError, IndexError, ValueError) as exc:
        return _fail(task, "invalid_request", f"variable interpolation failed: {exc}")

    # 4. The single turn.
    turn = (chat or Chat(model=model)).with_instructions(system)
    if request.schema_ is not None:
        turn = turn.with_schema(request.schema_)  # JSON Schema straight through

    try:
        message = await turn.ask(user_input)
    except Exception as exc:  # narrowed to a contract code, never re-raised
        return _fail(task, _error_code_for(exc), str(exc))

    content = message.content

    # 5. Shape the payload by whether a schema was requested.
    if request.schema_ is not None:
        if not isinstance(content, dict):
            # Model returned prose where structured output was required.
            return _fail(task, "schema_violation", "model did not honor the requested schema")
        return LLMResponse(task=task, ok=True, data=content)

    if isinstance(content, dict):
        # No schema requested but the model still produced structured content;
        # a free-form task must reply with text, so serialize it.
        content = json.dumps(content)
    return LLMResponse(task=task, ok=True, text=str(content))
