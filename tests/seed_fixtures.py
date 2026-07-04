"""Fixture seeder.

Serializes the canonical response builders in :mod:`tests.factories` to
``tests/fixtures/data/`` so fixtures can be inspected, diffed, and reused across
tests. Deterministic — safe to re-run.

    python -m tests.seed_fixtures        # (re)generate the fixture files
"""

from __future__ import annotations

import json
from pathlib import Path

from . import factories as f

DATA_DIR = Path(__file__).parent / "fixtures" / "data"

# name -> (kind, payload). kind "json" writes .json, "sse" writes .sse bytes.
JSON_FIXTURES = {
    "openai_chat": f.openai_chat(),
    "openai_chat_tool": f.openai_chat(tool_calls=f.openai_tool_call("weather", {"city": "Paris"})),
    "openai_embedding": f.openai_embedding([[0.1, 0.2, 0.3]]),
    "openai_moderation_flagged": f.openai_moderation(flagged=True),
    "openai_image": f.openai_image(),
    "anthropic_message": f.anthropic_message(),
    "anthropic_tool": f.anthropic_message(
        "let me check", tool_use={"id": "tu1", "name": "weather", "input": {"city": "Paris"}}
    ),
    "gemini_response": f.gemini_response(),
    "gemini_tool": f.gemini_response(
        None, function_call={"name": "weather", "args": {"city": "Paris"}}
    ),
    "bedrock_converse": f.bedrock_converse(),
}

SSE_FIXTURES = {
    "openai_stream": f.openai_sse("Hel", "lo!"),
    "gemini_stream": f.gemini_sse("Hel", "lo!"),
}


def seed_all(dest: Path = DATA_DIR) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in JSON_FIXTURES.items():
        path = dest / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        written.append(path)
    for name, blob in SSE_FIXTURES.items():
        path = dest / f"{name}.sse"
        path.write_bytes(blob)
        written.append(path)
    return written


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / f"{name}.json").read_text())


def load_sse(name: str) -> bytes:
    return (DATA_DIR / f"{name}.sse").read_bytes()


if __name__ == "__main__":
    paths = seed_all()
    print(f"seeded {len(paths)} fixtures into {DATA_DIR}")
    for p in paths:
        print(" ", p.name)
