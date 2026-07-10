"""Streaming chat against local Gemma 4 — tokens print as they arrive.

python examples/stream.py "Write a haiku about the terminal"
"""

from __future__ import annotations

import asyncio
import sys

import _bootstrap as boot


async def main() -> None:
    boot.setup()
    prompt = " ".join(sys.argv[1:]) or "Write a haiku about async Python."
    async for chunk in boot.chat().stream(prompt):
        print(chunk.content or "", end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
