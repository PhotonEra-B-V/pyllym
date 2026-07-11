"""Minimal one-shot chat against local Gemma 4.

python examples/chat.py "Explain async/await in one sentence"
"""

from __future__ import annotations

import asyncio
import sys

import _bootstrap as boot


async def main() -> None:
    boot.setup()
    prompt = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    msg = await boot.chat().ask(prompt)
    print(msg.content)
    print(f"\n[tokens: in={msg.input_tokens} out={msg.output_tokens}]")


if __name__ == "__main__":
    asyncio.run(main())
