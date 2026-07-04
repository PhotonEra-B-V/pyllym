from __future__ import annotations

import pytest

import pyllm
from pyllm import Message, Tool, utils
from pyllm.content import Content
from pyllm.errors import InvalidRoleError


def test_underscore():
    assert utils.underscore("HTTPProxyTool") == "http_proxy_tool"
    assert utils.underscore("OpenAI") == "open_ai"


def test_deep_merge():
    assert utils.deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_message_roles_and_validation():
    assert Message(role="user", content="hi").content == "hi"
    with pytest.raises(InvalidRoleError):
        Message(role="wizard", content="x")


def test_content_format_collapses_to_text():
    assert Content("just text").format() == "just text"


def test_models_registry_lookup():
    m = pyllm.models.find("gpt-4o")
    assert m.provider == "openai"
    assert pyllm.models.find("claude-sonnet-4-6").provider == "anthropic"
    assert len(pyllm.models.chat_models()) > 100


def test_model_not_found():
    from pyllm.errors import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        pyllm.models.find("nonexistent-model-xyz")


def test_tool_schema_inference():
    class AddTool(Tool):
        description = "Add numbers"

        def execute(self, *, a: int, b: int = 0):
            return a + b

    tool = AddTool()
    assert tool.name == "add"
    schema = tool.params_schema
    assert set(schema["properties"]) == {"a", "b"}
    assert schema["required"] == ["a"]


@pytest.mark.asyncio
async def test_tool_call_validation_and_execute():
    class EchoTool(Tool):
        def execute(self, *, text: str):
            return text.upper()

    tool = EchoTool()
    assert await tool.call({"text": "hi"}) == "HI"
    result = await tool.call({"wrong": "x"})
    assert "error" in result
