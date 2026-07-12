# examples

Small, runnable scripts for poking at `pyllym` from the terminal against a
**local Gemma 4 model served by Ollama**.

## Prerequisites

- Ollama running with Gemma 4 pulled:
  ```bash
  ollama pull gemma4          # or gemma4:31b
  ollama serve                # usually already running
  ```
- `pyllym` importable. From the repo root, either install it editable:
  ```bash
  pip install -e ".[dev]"
  ```
  or just run the scripts from the repo root (they add `src/` to the path).

## Run

```bash
python examples/chat.py "Explain async/await in one sentence"
python examples/stream.py "Write a haiku about the terminal"
python examples/structured.py
python examples/tools.py
python examples/stats.py
pip install -e ".[sci]"                           # sci stack for the code tool
python examples/stats.py --allow-code             # also expose the code tool
python examples/moderation.py
python examples/moderation.py --no-model          # wordlist only, no LLM
python examples/moderation.py --model gemma4:latest --language Finnish
```

`stats.py` lets the local model do real numeric/statistical work by handing it
tools backed by **numpy** and the stdlib `statistics` module — so the numbers
come from the library, not the model's guesses. It exposes curated,
deterministic tools (`Mean`, `StdDev`, `Correlation`, `LinearRegression`) by
default, plus an opt-in `RunCode` code tool (`--allow-code`) that `exec`s a
model-written snippet with the **scientific stack** pre-imported — numpy (`np`),
pandas (`pd`), matplotlib (`plt`), seaborn (`sns`), plus `scipy`, `sympy` and
`scikit-learn`. The model can `import` those libraries too (imports are
allowlisted, so `import os` is refused), `print()` output is captured, and any
plot is saved to `examples/plots/`. The code tool is **not a sandbox** — only
enable it for a model and data you trust. Install the stack with the opt-in
extra: `pip install -e ".[sci]"` (the curated tools only need numpy).

`moderation.py` moderates each quote with **two local signals**: a
deterministic **wordlist** (the authority — fast, consistent, catches profanity
and slurs the model refuses to flag) plus the **local model** as a secondary
signal for nuance the wordlist misses. A quote is flagged if either fires;
flagged categories are labeled with their source. Both the model (`--model`)
and the reason language (`--language`) are selectable, or drop the LLM entirely
with `--no-model`.

Override the model or endpoint with env vars:

```bash
GEMMA_MODEL=gemma4:31b python examples/chat.py "hi"
OLLAMA_BASE=http://localhost:11434/v1 python examples/chat.py "hi"
```

`_bootstrap.py` centralizes the Ollama config so each script stays tiny.
