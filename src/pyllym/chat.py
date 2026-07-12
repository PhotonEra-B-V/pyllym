"""A conversation with an AI model.

Async-first: :meth:`ask` awaits a full completion (running the agentic tool
loop), :meth:`stream` is an async generator of chunks. Tool ``execute`` methods
may be sync or async; concurrent tool execution uses :func:`asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from . import models as _models
from . import utils
from .content import Content, RawContent
from .cost import Cost
from .errors import Error, InvalidToolChoiceError
from .message import Message
from .thinking import ThinkingConfig
from .tool import Tool

if TYPE_CHECKING:
    from .configuration import Configuration
    from .context import Context

OnChunk = Callable[[Any], Any]


class Chat:
    def __init__(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        assume_model_exists: bool = False,
        context: Context | None = None,
    ) -> None:
        if assume_model_exists and not provider:
            raise ValueError("Provider must be specified if assume_model_exists is true")
        self._context = context
        from . import config as _config

        self._config: Configuration = context.config if context else _config()
        model_id = model or self._config.default_model
        self.with_model(model_id, provider=provider, assume_exists=assume_model_exists)
        self._temperature: float | None = None
        self.messages: list[Message] = []
        self.tools: dict[str, Tool] = {}
        self.tool_prefs: dict[str, Any] = {"choice": None, "calls": None}
        self._concurrency = self._normalize_tool_concurrency(self._config.tool_concurrency)
        self.params: dict[str, Any] = {}
        self.headers: dict[str, str] = {}
        self.schema: dict[str, Any] | None = None
        self._thinking: ThinkingConfig | None = None
        self._citations = False
        self._protocol: str | None = None
        self._callbacks: dict[str, list[Callable[..., Any]]] = {
            "before_message": [],
            "after_message": [],
            "before_tool_call": [],
            "after_tool_result": [],
        }

    # --- asking ----------------------------------------------------------------
    async def ask(
        self, message: Any = None, *, with_: Any = None, on_chunk: OnChunk | None = None
    ) -> Message:
        self.add_user_message(message, with_=with_)
        return await self.run_until_done(on_chunk=on_chunk)

    say = ask

    def add_user_message(self, message: Any = None, *, with_: Any = None) -> Chat:
        """Stage a user message without sending it."""
        self.add_message(role="user", content=self._build_content(message, with_))
        return self

    async def send(self, on_chunk: OnChunk | None = None) -> Message:
        """Call the model once and append its response. The model's move."""
        result = await self._provider_completion(on_chunk)
        # Streaming already fired before_message via _wrap_streaming_callback.
        if on_chunk is None:
            await self._run_callbacks("before_message")
        return await self._finalize_completion(result)

    async def execute_tools(self) -> Chat:
        """Run the pending tool calls and append their results. Our move."""
        last = self.messages[-1] if self.messages else None
        if last and last.is_tool_call():
            await self._execute_pending_tool_calls(last)
        return self

    async def step(self, on_chunk: OnChunk | None = None) -> Message | None:
        if self.is_complete():
            return None
        last = self.messages[-1] if self.messages else None
        if last and last.is_tool_call():
            await self.execute_tools()
            return None
        return await self.send(on_chunk)

    async def run_until_done(self, on_chunk: OnChunk | None = None) -> Message:
        """Run the agentic loop to completion: step until nothing is left."""
        if not self.messages:
            raise Error("Nothing to send: add a user message first (add_user_message or ask)")
        while not self.is_complete():
            await self.step(on_chunk)
        return self.messages[-1]

    def is_complete(self) -> bool:
        last = self.messages[-1] if self.messages else None
        if last is None:
            return True
        if last.role in ("user", "tool"):
            return False
        return not last.is_tool_call()

    async def stream(self, message: Any = None, *, with_: Any = None) -> AsyncIterator[Any]:
        """Drive a completion, yielding chunks as they arrive."""
        if message is not None or with_ is not None:
            self.add_user_message(message, with_=with_)
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        async def on_chunk(chunk: Any) -> None:
            await queue.put(chunk)

        async def run() -> None:
            try:
                await self.run_until_done(on_chunk=on_chunk)
            finally:
                await queue.put(sentinel)

        task = asyncio.ensure_future(run())
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            # If the consumer stopped early (break / aclose), cancel the
            # producer instead of silently running the completion to the end.
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # --- builders --------------------------------------------------------------
    def with_instructions(self, instructions: str, *, append: bool = False) -> Chat:
        if append:
            self._append_system_instruction(instructions)
        else:
            self._replace_system_instruction(instructions)
        return self

    def with_tool(
        self,
        tool: Tool | type[Tool] | None,
        *,
        choice: Any = None,
        calls: Any = None,
        concurrency: Any = ...,
    ) -> Chat:
        if tool is not None:
            instance = tool() if isinstance(tool, type) else tool
            self.tools[instance.name] = instance
        self._update_tool_options(choice=choice, calls=calls)
        if concurrency is not ...:
            self._concurrency = self._normalize_tool_concurrency(concurrency)
        return self

    def with_tools(
        self,
        *tools: Tool | type[Tool],
        replace: bool = False,
        choice: Any = None,
        calls: Any = None,
    ) -> Chat:
        if replace:
            self.tools.clear()
        for tool in tools:
            if tool is not None:
                self.with_tool(tool)
        self._update_tool_options(choice=choice, calls=calls)
        return self

    def with_model(
        self, model_id: str, *, provider: str | None = None, assume_exists: bool = False
    ) -> Chat:
        self.model, self.provider = _models.resolve(
            model_id, provider=provider, assume_exists=assume_exists, config=self._config
        )
        self._connection = self.provider.connection
        return self

    def with_temperature(self, temperature: float) -> Chat:
        self._temperature = temperature
        return self

    def with_thinking(self, *, effort: str | None = None, budget: int | None = None) -> Chat:
        if effort is None and budget is None:
            raise ValueError("with_thinking requires effort or budget")
        self._thinking = ThinkingConfig(effort=effort, budget=budget)
        return self

    def with_citations(self, enabled: bool = True) -> Chat:
        self._citations = enabled
        return self

    def with_context(self, context: Context) -> Chat:
        self._context = context
        self._config = context.config
        self.with_model(self.model.id, provider=self.provider.slug, assume_exists=True)
        return self

    def with_params(self, **params: Any) -> Chat:
        self.params = params
        return self

    def with_protocol(self, protocol: str) -> Chat:
        self._protocol = protocol
        return self

    def with_headers(self, **headers: str) -> Chat:
        self.headers = headers
        return self

    def with_schema(self, schema: Any) -> Chat:
        # Resolve a JSON Schema from: a pydantic model class/instance, an object
        # exposing to_json_schema(), or a plain schema dict. Detect the pydantic
        # case *before* instantiating, so passing a model class works.
        if hasattr(schema, "model_json_schema"):
            raw = schema.model_json_schema()
        elif isinstance(schema, type):
            instance = schema()
            raw = instance.to_json_schema() if hasattr(instance, "to_json_schema") else instance
        elif hasattr(schema, "to_json_schema"):
            raw = schema.to_json_schema()
        else:
            raw = schema
        self.schema = self._normalize_schema_payload(raw)
        return self

    # --- callbacks -------------------------------------------------------------
    def before_message(self, callback: Callable[..., Any]) -> Chat:
        self._callbacks["before_message"].append(callback)
        return self

    def after_message(self, callback: Callable[..., Any]) -> Chat:
        self._callbacks["after_message"].append(callback)
        return self

    def before_tool_call(self, callback: Callable[..., Any]) -> Chat:
        self._callbacks["before_tool_call"].append(callback)
        return self

    def after_tool_result(self, callback: Callable[..., Any]) -> Chat:
        self._callbacks["after_tool_result"].append(callback)
        return self

    # --- misc ------------------------------------------------------------------
    def __iter__(self):
        return iter(self.messages)

    @property
    def cost(self) -> Any:
        return Cost.aggregate([m.cost(model=m.model_info or self.model) for m in self.messages])

    def add_message(self, message_or_attributes: Any = None, **attrs: Any) -> Message:
        if attrs:
            message_or_attributes = attrs
        message = self._coerce_message(message_or_attributes)
        if self.provider:
            message = self.provider.preprocess_message(
                message, model=self.model, protocol=self._protocol
            )
        self.messages.append(message)
        return message

    async def add_completion(self, response: Message) -> Message:
        """Receive a completion produced out-of-band, running the same callbacks."""
        await self._run_callbacks("before_message")
        return await self._finalize_completion(response)

    async def _finalize_completion(self, response: Message) -> Message:
        self._normalize_schema_response(response)
        self.add_message(response)
        await self._run_callbacks("after_message", response)
        return response

    def render(self) -> dict[str, Any]:
        return self.provider.render(
            self.messages,
            tools=self.tools,
            tool_prefs=self.tool_prefs,
            temperature=self._temperature,
            model=self.model,
            params=self.params,
            schema=self.schema,
            thinking=self._thinking,
            citations=self._citations,
            protocol=self._protocol,
        )

    # --- internals -------------------------------------------------------------
    def _coerce_message(self, value: Any) -> Message:
        if hasattr(value, "to_llm"):
            value = value.to_llm()
        return value if isinstance(value, Message) else Message(value)

    async def _provider_completion(self, on_chunk: OnChunk | None) -> Message:
        wrapped = await self._wrap_streaming_callback(on_chunk)
        return await self.provider.complete(
            self.messages,
            tools=self.tools,
            tool_prefs=self.tool_prefs,
            temperature=self._temperature,
            model=self.model,
            params=self.params,
            headers=self.headers,
            schema=self.schema,
            thinking=self._thinking,
            citations=self._citations,
            protocol=self._protocol,
            on_chunk=wrapped,
        )

    async def _wrap_streaming_callback(self, on_chunk: OnChunk | None) -> OnChunk | None:
        if on_chunk is None:
            return None
        await self._run_callbacks("before_message")
        return on_chunk

    def _normalize_schema_response(self, response: Message) -> None:
        if not (self.schema and isinstance(response.content, str) and not response.is_tool_call()):
            return
        try:
            response.content = json.loads(response.content)
        except json.JSONDecodeError:
            pass

    async def _run_callbacks(self, name: str, *args: Any) -> None:
        for callback in self._callbacks[name]:
            result = callback(*args)
            if inspect.isawaitable(result):
                await result

    async def _execute_pending_tool_calls(self, response: Message) -> None:
        tool_calls = response.tool_calls or {}
        if self._concurrency:
            results = await asyncio.gather(
                *(self._execute_tool_with_callbacks(tc) for tc in tool_calls.values())
            )
            for tc, result in zip(tool_calls.values(), results, strict=True):
                await self._run_callbacks("before_message")
                await self._add_tool_result_message(tc, result)
        else:
            for tc in tool_calls.values():
                await self._run_callbacks("before_message")
                result = await self._execute_tool_with_callbacks(tc)
                await self._add_tool_result_message(tc, result)
        if self._forced_tool_choice():
            self.tool_prefs["choice"] = None

    async def _execute_tool_with_callbacks(self, tool_call: Any) -> Any:
        await self._run_callbacks("before_tool_call", tool_call)
        result = await self._execute_tool(tool_call)
        await self._run_callbacks("after_tool_result", result)
        return result

    async def _add_tool_result_message(self, tool_call: Any, result: Any) -> Message:
        content = result if self._content_like(result) else str(result)
        message = self.add_message(role="tool", content=content, tool_call_id=tool_call.id)
        await self._run_callbacks("after_message", message)
        return message

    async def _execute_tool(self, tool_call: Any) -> Any:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            available = json.dumps(list(self.tools))
            return {
                "error": f"Model tried to call unavailable tool `{tool_call.name}`. "
                f"Available tools: {available}."
            }
        return await tool.call(tool_call.arguments)

    def _update_tool_options(self, *, choice: Any, calls: Any) -> None:
        if choice is not None:
            normalized = self._normalize_tool_choice(choice)
            valid = {"auto", "none", "required", *self.tools}
            if normalized not in valid:
                raise InvalidToolChoiceError(
                    f"Invalid tool choice: {choice}. Valid choices are: {', '.join(valid)}"
                )
            self.tool_prefs["choice"] = normalized
        if calls is not None:
            self.tool_prefs["calls"] = self._normalize_calls(calls)

    @staticmethod
    def _normalize_calls(calls: Any) -> str:
        # bool is a subclass of int, so check it first: calls=True must not
        # silently match the `1` alias for "one".
        if not isinstance(calls, bool):
            if calls == "many":
                return "many"
            if calls == "one" or calls == 1:
                return "one"
        raise ValueError(f"Invalid calls value: {calls!r}. Valid values are: 'many', 'one', or 1")

    def _normalize_tool_choice(self, choice: Any) -> str:
        if isinstance(choice, str):
            return choice
        if isinstance(choice, type):
            matched = next((n for n, t in self.tools.items() if isinstance(t, choice)), None)
            if matched:
                return matched
            return utils.underscore(choice.__name__).removesuffix("_tool")
        if hasattr(choice, "name"):
            return choice.name
        return str(choice)

    TOOL_CONCURRENCY_MODES = ("asyncio",)

    @classmethod
    def _normalize_tool_concurrency(cls, concurrency: Any) -> Any:
        if concurrency is None or concurrency is False:
            return None
        if concurrency is True:
            return "asyncio"
        normalized = str(concurrency)
        if normalized not in cls.TOOL_CONCURRENCY_MODES:
            raise ValueError(
                f"Unknown tool concurrency: {concurrency!r}. "
                f"Available modes: {', '.join(cls.TOOL_CONCURRENCY_MODES)}"
            )
        return normalized

    def _forced_tool_choice(self) -> bool:
        return bool(self.tool_prefs["choice"]) and self.tool_prefs["choice"] not in ("auto", "none")

    def _build_content(self, message: Any, attachments: Any) -> Any:
        if self._content_like(message):
            return message
        return Content(message, attachments)

    @staticmethod
    def _content_like(obj: Any) -> bool:
        return isinstance(obj, (Content, RawContent))

    def _append_system_instruction(self, instructions: str) -> None:
        system = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]
        system.append(Message(role="system", content=instructions))
        self.messages = system + non_system

    def _replace_system_instruction(self, instructions: str) -> None:
        system = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]
        if not system:
            system = [Message(role="system", content=instructions)]
        else:
            system[0].content = instructions
            system = [system[0]]
        self.messages = system + non_system

    # --- schema normalization --------------------------------------------------
    def _normalize_schema_payload(self, raw_schema: Any) -> dict[str, Any] | None:
        if raw_schema is None:
            return None
        if not isinstance(raw_schema, dict):
            return raw_schema
        schema = utils.deep_dup(raw_schema)
        schema_def = utils.deep_dup(schema.get("schema") or schema)
        if "strict" in schema:
            strict = schema["strict"]
        elif isinstance(schema_def, dict) and "strict" in schema_def:
            strict = schema_def.pop("strict")
        else:
            strict = None
        payload = {
            "name": self._sanitize_schema_name(schema.get("name") or "response"),
            "schema": schema_def,
            "strict": True if strict is None else strict,
            "description": schema.get("description"),
        }
        return {k: v for k, v in payload.items() if v is not None}

    @staticmethod
    def _sanitize_schema_name(name: Any) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name))
        return sanitized or "response"
