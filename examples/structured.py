"""Structured (typed) output against local Gemma 4 via a Pydantic schema.

python examples/structured.py
"""

from __future__ import annotations

import asyncio

import _bootstrap as boot
from pydantic import BaseModel


class Recipe(BaseModel):
    title: str
    ingredients: list[str]
    minutes: int


async def main() -> None:
    boot.setup()
    chat = boot.chat().with_schema(Recipe)
    msg = await chat.ask("Give me a quick pancake recipe.")
    if isinstance(msg.content, dict):
        recipe = Recipe.model_validate(msg.content)
        print(f"{recipe.title} ({recipe.minutes} min)")
        for item in recipe.ingredients:
            print(f"  - {item}")
    else:
        # Some local models don't guarantee JSON; show what came back.
        print("Model returned unstructured text:\n")
        print(msg.content)


if __name__ == "__main__":
    asyncio.run(main())
