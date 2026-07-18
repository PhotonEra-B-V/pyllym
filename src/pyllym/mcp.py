"""MCP (Model Context Protocol) client support: remote tools as ``Tool`` objects.

Where :mod:`pyllym.toolset` turns *local library callables* into tools, this
module turns tools served by an **external MCP server** into the same
:class:`~pyllym.tool.Tool` objects, so they drop straight into
``chat.with_tools(...)``::

    import pyllym

    server = pyllym.MCPServer.stdio(
        "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"
    )
    async with server as fs:
        chat = pyllym.create_chat(model="claude-sonnet-5").with_tools(*fs.tools)
        answer = await chat.ask("List the files in /tmp and summarize them.")

    async with pyllym.MCPServer.http("https://example.com/mcp", headers={...}) as remote:
        chat = pyllym.create_chat(model="gpt-5").with_tools(*remote.tools)

Requires the official MCP Python SDK, installed via the ``mcp`` extra
(``pip install "pyllym[mcp]"``). The SDK is imported lazily — importing
:mod:`pyllym` (or even this module) never pulls it in.

Design notes
------------
- :class:`MCPServer` owns the session lifecycle (transport + initialize) as an
  async context manager. Tools are snapshotted on ``__aenter__``; call
  :meth:`MCPServer.refresh` to re-list after a ``tools/list_changed``.
- The adapter core, :func:`tools_from_session`, works against anything with
  ``list_tools()`` / ``call_tool(name, arguments)`` coroutine methods, so the
  Tool-wrapping logic is independent of (and testable without) the SDK.
- Tool results: ``structuredContent`` wins when present; otherwise text blocks
  are joined; non-text blocks are represented as small dicts. A result with
  ``isError`` set comes back as ``{"error": ...}`` — the same convention
  :meth:`Tool.call` uses for invalid arguments — so the model can recover.

Security model — an MCP server is an *arbitrary external program or endpoint*:
a stdio server is code you execute, an HTTP server is a service you trust with
your tool arguments (which may contain conversation data). Connect only to
servers you would be comfortable running by hand; pyllym adds no sandbox.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from .errors import ConfigurationError
from .tool import Tool

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ["MCPNotInstalledError", "MCPServer", "MCPTool", "tools_from_session"]

logger = logging.getLogger("pyllym")


class MCPNotInstalledError(ConfigurationError):
    """The optional MCP SDK is not installed.

    A subclass of :class:`~pyllym.errors.ConfigurationError`, mirroring
    :class:`~pyllym.toolset.MissingToolPackageError` — MCP support is opt-in.
    """

    def __init__(self) -> None:
        super().__init__(
            "MCP support requires the official MCP SDK. "
            'Install it with the extra: pip install "pyllym[mcp]".'
        )


class MCPSession(Protocol):
    """The slice of an MCP client session the adapter needs."""

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any: ...


class MCPTool(Tool):
    """A remote MCP tool adapted to the :class:`~pyllym.tool.Tool` interface.

    Name, description, and JSON schema come from the server's ``tools/list``
    entry; ``execute`` forwards the arguments over the live session.
    """

    def __init__(
        self,
        session: MCPSession,
        *,
        name: str,
        description: str | None,
        schema: dict[str, Any] | None,
        prefix: str | None = None,
    ) -> None:
        self._session = session
        self._remote_name = name
        self._name = f"{prefix}{name}" if prefix else name
        self._description = description or f"MCP tool {name}."
        self._schema = schema

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:  # type: ignore[override]
        return self._description

    @property
    def params_schema(self) -> dict[str, Any] | None:
        return self._schema

    async def execute(self, **kwargs: Any) -> Any:
        result = await self._session.call_tool(self._remote_name, kwargs)
        return _convert_result(result)


async def tools_from_session(
    session: MCPSession,
    *,
    prefix: str | None = None,
    only: Iterable[str] | None = None,
) -> list[Tool]:
    """List a session's tools and wrap each as an :class:`MCPTool`.

    ``prefix`` namespaces the advertised tool names (useful when combining
    several servers whose tool names could collide); the remote call still uses
    the server's original name. ``only`` restricts to the named remote tools.
    """
    wanted = set(only) if only is not None else None
    tools: list[Tool] = []
    cursor: str | None = None
    while True:
        listing = await (session.list_tools(cursor) if cursor else session.list_tools())
        for entry in getattr(listing, "tools", []):
            name = entry.name
            if wanted is not None and name not in wanted:
                continue
            tools.append(
                MCPTool(
                    session,
                    name=name,
                    description=getattr(entry, "description", None),
                    schema=getattr(entry, "inputSchema", None),
                    prefix=prefix,
                )
            )
        cursor = getattr(listing, "nextCursor", None)
        if not cursor:
            break
    if wanted is not None:
        found = {t._remote_name for t in tools if isinstance(t, MCPTool)}
        missing = wanted - found
        if missing:
            logger.warning("MCP server did not offer requested tools: %s", sorted(missing))
    return tools


class MCPServer:
    """A connection to one MCP server, usable as an async context manager.

    Construct via :meth:`stdio` (spawn a local server process) or :meth:`http`
    (streamable-HTTP endpoint), enter the context, and read :attr:`tools`.
    """

    def __init__(
        self,
        *,
        transport: str,
        command: str | None = None,
        args: tuple[str, ...] = (),
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        url: str | None = None,
        headers: Mapping[str, str] | None = None,
        prefix: str | None = None,
        only: Iterable[str] | None = None,
    ) -> None:
        self._transport = transport
        self._command = command
        self._args = args
        self._env = dict(env) if env else None
        self._cwd = cwd
        self._url = url
        self._headers = dict(headers) if headers else None
        self._prefix = prefix
        self._only = tuple(only) if only is not None else None
        self._stack: Any = None
        self._session: MCPSession | None = None
        self.tools: list[Tool] = []

    # --- constructors ----------------------------------------------------------
    @classmethod
    def stdio(
        cls,
        command: str,
        *args: str,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        prefix: str | None = None,
        only: Iterable[str] | None = None,
    ) -> MCPServer:
        """A server spawned as a subprocess, speaking MCP over stdio."""
        return cls(
            transport="stdio",
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            prefix=prefix,
            only=only,
        )

    @classmethod
    def http(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        prefix: str | None = None,
        only: Iterable[str] | None = None,
    ) -> MCPServer:
        """A remote server speaking MCP over streamable HTTP."""
        return cls(transport="http", url=url, headers=headers, prefix=prefix, only=only)

    # --- lifecycle -------------------------------------------------------------
    async def __aenter__(self) -> MCPServer:
        from contextlib import AsyncExitStack

        try:
            from mcp import ClientSession
        except ModuleNotFoundError as exc:
            raise MCPNotInstalledError() from exc

        self._stack = AsyncExitStack()
        try:
            if self._transport == "stdio":
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                assert self._command is not None
                params = StdioServerParameters(
                    command=self._command,
                    args=list(self._args),
                    env=self._env,
                    cwd=self._cwd,
                )
                read, write = await self._stack.enter_async_context(stdio_client(params))
            else:
                from mcp.client.streamable_http import streamablehttp_client

                assert self._url is not None
                read, write, _ = await self._stack.enter_async_context(
                    streamablehttp_client(self._url, headers=self._headers)
                )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            raise
        self._session = session
        await self.refresh()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        stack, self._stack, self._session, self.tools = self._stack, None, None, []
        if stack is not None:
            await stack.aclose()

    async def refresh(self) -> list[Tool]:
        """Re-list the server's tools (e.g. after a ``tools/list_changed``)."""
        if self._session is None:
            raise ConfigurationError("MCPServer is not connected; use `async with` to connect")
        self.tools = await tools_from_session(self._session, prefix=self._prefix, only=self._only)
        return self.tools


def _convert_result(result: Any) -> Any:
    """Flatten an MCP ``CallToolResult`` into plain JSON-able Python."""
    structured = getattr(result, "structuredContent", None)
    parts: list[Any] = []
    for block in getattr(result, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(block.text)
        elif kind == "image":
            parts.append(
                {
                    "type": "image",
                    "mime_type": getattr(block, "mimeType", None),
                    "data": getattr(block, "data", None),
                }
            )
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            parts.append(
                {
                    "type": "resource",
                    "uri": str(getattr(resource, "uri", "")),
                    "text": getattr(resource, "text", None),
                }
            )
        else:
            parts.append(str(block))
    text = "\n".join(p for p in parts if isinstance(p, str))
    payload: Any
    if structured is not None:
        payload = structured
    elif len(parts) == 1:
        payload = parts[0]
    elif all(isinstance(p, str) for p in parts):
        payload = text
    else:
        payload = parts
    if getattr(result, "isError", False):
        return {"error": payload if payload not in (None, "", []) else "MCP tool call failed"}
    return payload
