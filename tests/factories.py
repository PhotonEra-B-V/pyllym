"""Canonical provider-response builders — the single source of truth for test
fixtures. :mod:`tests.seed_fixtures` serializes these to disk; tests may also
call them directly.
"""

from __future__ import annotations

import json
from typing import Any


# --- OpenAI Chat Completions ---------------------------------------------------
def openai_tool_call(
    name: str, args: dict[str, Any], *, id: str = "call_1"
) -> list[dict[str, Any]]:
    return [
        {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
    ]


def openai_chat(
    content: str | None = "Hello!",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "gpt-4o",
    usage: tuple[int, int] = (10, 5),
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {
        "id": "cmpl-test",
        "model": model,
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": sum(usage),
        },
    }


def openai_sse(*text_chunks: str, model: str = "gpt-4o", usage: tuple[int, int] = (1, 3)) -> bytes:
    lines = []
    for chunk in text_chunks:
        payload = {"model": model, "choices": [{"delta": {"content": chunk}}]}
        lines.append(f"data: {json.dumps(payload)}\n\n")
    final = {
        "model": model,
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }
    lines.append(f"data: {json.dumps(final)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def openai_embedding(vectors: list[list[float]], *, tokens: int = 3) -> dict[str, Any]:
    return {
        "data": [{"embedding": v, "index": i} for i, v in enumerate(vectors)],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": tokens},
    }


def openai_moderation(*, flagged: bool = False) -> dict[str, Any]:
    categories = {"hate": flagged, "violence": False}
    return {
        "id": "modr-test",
        "model": "omni-moderation-latest",
        "results": [
            {
                "flagged": flagged,
                "categories": categories,
                "category_scores": {"hate": 0.9 if flagged else 0.01},
            }
        ],
    }


def openai_image(
    *, url: str | None = "https://cdn/img.png", b64: str | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {"revised_prompt": "a cat"}
    if url:
        entry["url"] = url
    if b64:
        entry["b64_json"] = b64
    return {"created": 0, "data": [entry], "usage": {"total_tokens": 0}}


# --- Anthropic Messages --------------------------------------------------------
def anthropic_message(
    text: str = "Hi from Claude",
    *,
    tool_use: dict[str, Any] | None = None,
    model: str = "claude-sonnet-4-6",
    usage: tuple[int, int] = (12, 7),
    stop_reason: str | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if tool_use:
        content.append({"type": "tool_use", **tool_use})
    return {
        "id": "msg-test",
        "model": model,
        "stop_reason": stop_reason or ("tool_use" if tool_use else "end_turn"),
        "content": content,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


# --- Gemini generateContent ----------------------------------------------------
def gemini_response(
    text: str | None = "Hi from Gemini",
    *,
    function_call: dict[str, Any] | None = None,
    model: str = "gemini-2.5-flash",
    tokens: tuple[int, int] = (8, 4),
) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    if text is not None:
        parts.append({"text": text})
    if function_call:
        parts.append({"functionCall": function_call})
    return {
        "modelVersion": model,
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": tokens[0], "candidatesTokenCount": tokens[1]},
    }


def gemini_sse(*text_chunks: str, model: str = "gemini-2.5-flash") -> bytes:
    lines = []
    for chunk in text_chunks:
        payload = {"candidates": [{"content": {"parts": [{"text": chunk}]}}], "modelVersion": model}
        lines.append(f"data: {json.dumps(payload)}\n\n")
    final = {
        "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": len(text_chunks)},
        "modelVersion": model,
    }
    lines.append(f"data: {json.dumps(final)}\n\n")
    return "".join(lines).encode()


# --- Bedrock Converse ----------------------------------------------------------
def bedrock_converse(
    text: str = "Hi from Bedrock",
    *,
    tool_use: dict[str, Any] | None = None,
    tokens: tuple[int, int] = (9, 7),
    stop_reason: str | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"text": text}]
    if tool_use:
        content.append({"toolUse": tool_use})
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop_reason or ("tool_use" if tool_use else "end_turn"),
        "usage": {"inputTokens": tokens[0], "outputTokens": tokens[1], "totalTokens": sum(tokens)},
    }
