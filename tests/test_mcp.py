"""Tests for pyllym.mcp — the adapter core, with a fake MCP session.

The official ``mcp`` SDK is an optional extra; these tests exercise everything
above the transport (tool listing, wrapping, calling, result flattening)
against a stand-in session, so they run without the SDK installed.
"""

from __future__ import annotations

import sys
from contextlib import AsyncExitStack, asynccontextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import pyllym
from pyllym.errors import ConfigurationError
from pyllym.mcp import (
    MCPServer,
    MCPTool,
    _convert_result,
    _enter_http_transport,
    tools_from_session,
)


def _tool_entry(name: str, description: str | None = None, schema: dict | None = None) -> Any:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema
        or {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )


def _text_result(text: str, *, is_error: bool = False) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        structuredContent=None,
        isError=is_error,
    )


class FakeSession:
    """Duck-typed stand-in for ``mcp.ClientSession``."""

    def __init__(self, tools: list[Any], results: dict[str, Any] | None = None) -> None:
        self._tools = tools
        self._results = results or {}
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def list_tools(self, cursor: str | None = None) -> Any:
        return SimpleNamespace(tools=self._tools, nextCursor=None)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        self.calls.append((name, arguments))
        return self._results.get(name, _text_result("ok"))


# --- tools_from_session ------------------------------------------------------


async def test_wraps_listed_tools_with_metadata():
    session = FakeSession([_tool_entry("read_file", "Read a file.")])
    tools = await tools_from_session(session)
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, MCPTool)
    assert tool.name == "read_file"
    assert tool.description == "Read a file."
    assert tool.params_schema is not None
    assert "path" in tool.params_schema["properties"]


async def test_prefix_namespaces_name_but_calls_remote_name():
    session = FakeSession([_tool_entry("search")])
    (tool,) = await tools_from_session(session, prefix="docs_")
    assert tool.name == "docs_search"
    await tool.call({"path": "x"})
    assert session.calls == [("search", {"path": "x"})]


async def test_only_filters_tools():
    session = FakeSession([_tool_entry("a"), _tool_entry("b"), _tool_entry("c")])
    tools = await tools_from_session(session, only={"a", "c"})
    assert sorted(t.name for t in tools) == ["a", "c"]


async def test_pagination_follows_cursor():
    class PagedSession(FakeSession):
        async def list_tools(self, cursor: str | None = None) -> Any:
            if cursor is None:
                return SimpleNamespace(tools=[_tool_entry("first")], nextCursor="page2")
            return SimpleNamespace(tools=[_tool_entry("second")], nextCursor=None)

    tools = await tools_from_session(PagedSession([]))
    assert [t.name for t in tools] == ["first", "second"]


async def test_missing_description_gets_fallback():
    session = FakeSession([_tool_entry("bare")])
    (tool,) = await tools_from_session(session)
    assert "bare" in tool.description


# --- MCPTool.call ------------------------------------------------------------


async def test_call_forwards_arguments_and_returns_text():
    session = FakeSession([_tool_entry("echo")], {"echo": _text_result("hello")})
    (tool,) = await tools_from_session(session)
    assert await tool.call({"path": "p"}) == "hello"
    assert session.calls == [("echo", {"path": "p"})]


async def test_error_result_becomes_error_dict():
    session = FakeSession([_tool_entry("boom")], {"boom": _text_result("kaput", is_error=True)})
    (tool,) = await tools_from_session(session)
    assert await tool.call({"path": "p"}) == {"error": "kaput"}


async def test_structured_content_wins_over_text():
    result = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="{}")],
        structuredContent={"count": 3},
        isError=False,
    )
    session = FakeSession([_tool_entry("stats")], {"stats": result})
    (tool,) = await tools_from_session(session)
    assert await tool.call({"path": "p"}) == {"count": 3}


# --- _convert_result ---------------------------------------------------------


def test_multiple_text_blocks_join():
    result = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="one"),
            SimpleNamespace(type="text", text="two"),
        ],
        structuredContent=None,
        isError=False,
    )
    assert _convert_result(result) == "one\ntwo"


def test_image_block_becomes_dict():
    result = SimpleNamespace(
        content=[SimpleNamespace(type="image", mimeType="image/png", data="b64==")],
        structuredContent=None,
        isError=False,
    )
    assert _convert_result(result) == {"type": "image", "mime_type": "image/png", "data": "b64=="}


def test_empty_error_result_gets_placeholder_message():
    result = SimpleNamespace(content=[], structuredContent=None, isError=True)
    assert _convert_result(result) == {"error": "MCP tool call failed"}


# --- MCPServer / façade ------------------------------------------------------


async def test_refresh_before_connect_raises():
    server = MCPServer.stdio("some-server")
    with pytest.raises(ConfigurationError):
        await server.refresh()


async def test_aenter_without_sdk_raises_actionable_error():
    try:
        import mcp  # noqa: F401

        pytest.skip("mcp SDK installed; not-installed path untestable")
    except ModuleNotFoundError:
        pass
    with pytest.raises(pyllym.MCPNotInstalledError, match=r"pyllym\[mcp\]"):
        async with MCPServer.stdio("some-server"):
            pass


def test_facade_exports():
    assert pyllym.MCPServer is MCPServer
    assert pyllym.MCPTool is MCPTool
    assert pyllym.tools_from_session is tools_from_session
    assert issubclass(pyllym.MCPNotInstalledError, ConfigurationError)


async def test_mcp_tools_plug_into_chat_registry():
    session = FakeSession([_tool_entry("lookup")])
    (tool,) = await tools_from_session(session)
    chat = pyllym.create_chat(model="gpt-4o", assume_model_exists=True, provider="openai")
    chat.with_tools(tool)
    assert "lookup" in chat.tools


# --- _enter_http_transport (SDK 1.x vs 2.x shapes) ---------------------------


def _install_fake_streamable_http(monkeypatch: pytest.MonkeyPatch, **attrs: Any) -> None:
    """Register a stand-in ``mcp.client.streamable_http`` module exposing ``attrs``."""
    root = ModuleType("mcp")
    client = ModuleType("mcp.client")
    transport = ModuleType("mcp.client.streamable_http")
    for name, value in attrs.items():
        setattr(transport, name, value)
    root.client = client  # type: ignore[attr-defined]
    client.streamable_http = transport  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", root)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", transport)


async def test_http_transport_early_v1_uses_headers_kwarg_and_drops_session_id(monkeypatch):
    seen: dict[str, Any] = {}
    closed: list[str] = []

    @asynccontextmanager
    async def streamablehttp_client(url: str, *, headers: dict[str, str] | None = None):
        seen.update(url=url, headers=headers)
        try:
            yield "read", "write", lambda: "session-id"
        finally:
            closed.append("transport")

    _install_fake_streamable_http(monkeypatch, streamablehttp_client=streamablehttp_client)
    async with AsyncExitStack() as stack:
        streams = await _enter_http_transport(stack, "https://x/mcp", {"Authorization": "b t"})
    assert streams == ("read", "write")
    assert seen == {"url": "https://x/mcp", "headers": {"Authorization": "b t"}}
    assert closed == ["transport"]


async def test_http_transport_uses_v2_httpx_client_for_headers(monkeypatch):
    seen: dict[str, Any] = {}
    closed: list[str] = []

    class FakeHttpClient:
        def __init__(self, headers: dict[str, str] | None) -> None:
            self.headers = headers

        async def __aenter__(self) -> FakeHttpClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            closed.append("http_client")

    def create_mcp_http_client(headers: dict[str, str] | None = None) -> FakeHttpClient:
        return FakeHttpClient(headers)

    @asynccontextmanager
    async def streamable_http_client(url: str, *, http_client: Any = None):
        seen.update(url=url, http_client=http_client)
        try:
            yield ("read", "write")  # TransportStreams: a 2-tuple
        finally:
            closed.append("transport")

    _install_fake_streamable_http(
        monkeypatch,
        streamable_http_client=streamable_http_client,
        create_mcp_http_client=create_mcp_http_client,
    )
    async with AsyncExitStack() as stack:
        streams = await _enter_http_transport(stack, "https://x/mcp", {"Authorization": "b t"})
    assert streams == ("read", "write")
    assert seen["url"] == "https://x/mcp"
    assert seen["http_client"].headers == {"Authorization": "b t"}
    # The transport is torn down before the client we own, and the client is closed at all.
    assert closed == ["transport", "http_client"]


async def test_http_transport_prefers_new_entry_point_and_tolerates_v1_triple(monkeypatch):
    """Late 1.x exposes both names; the new one still yields a 3-tuple there."""
    seen: dict[str, Any] = {}

    class FakeHttpClient:
        def __init__(self, headers: dict[str, str] | None) -> None:
            self.headers = headers

        async def __aenter__(self) -> FakeHttpClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            pass

    @asynccontextmanager
    async def streamablehttp_client(url: str, *, headers: dict[str, str] | None = None):
        raise AssertionError("the original entry point must not be used when the new one exists")
        yield  # pragma: no cover

    @asynccontextmanager
    async def streamable_http_client(url: str, *, http_client: Any = None):
        seen.update(http_client=http_client)
        yield ("read", "write", lambda: "session-id")

    _install_fake_streamable_http(
        monkeypatch,
        streamablehttp_client=streamablehttp_client,
        streamable_http_client=streamable_http_client,
        create_mcp_http_client=lambda headers=None: FakeHttpClient(headers),
    )
    async with AsyncExitStack() as stack:
        assert await _enter_http_transport(stack, "https://x/mcp", {"A": "b"}) == ("read", "write")
    assert seen["http_client"].headers == {"A": "b"}


async def test_http_transport_v2_without_headers_lets_sdk_build_client(monkeypatch):
    seen: dict[str, Any] = {}

    def create_mcp_http_client(headers: dict[str, str] | None = None) -> Any:
        raise AssertionError("no client should be built when there are no headers")

    @asynccontextmanager
    async def streamable_http_client(url: str, *, http_client: Any = None):
        seen.update(http_client=http_client)
        yield ("read", "write")

    _install_fake_streamable_http(
        monkeypatch,
        streamable_http_client=streamable_http_client,
        create_mcp_http_client=create_mcp_http_client,
    )
    async with AsyncExitStack() as stack:
        assert await _enter_http_transport(stack, "https://x/mcp", None) == ("read", "write")
    assert seen == {"http_client": None}
