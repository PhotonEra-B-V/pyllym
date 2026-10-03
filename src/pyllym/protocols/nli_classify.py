"""Decisions from an open NLI classifier served by vLLM (e.g. OpenJev).

Open decision models such as OpenJev are natural-language-inference
classifiers: given a (premise, hypothesis) pair they return probabilities for
entailment / contradiction / neutral. vLLM serves them on its ``/classify``
endpoint (``vllm serve <model> --runner pooling --convert classify``), which
lives at the server root rather than under ``/v1``.

``decide()`` maps each typed question onto zero-shot NLI: the state is the
premise and every option becomes a hypothesis, all sent in one batch.

* ``choice`` / ``score`` — each option's entailment probability, normalized
  across the options, is its probability; ``confidence`` is the top one.
* ``yes_no`` — the hypothesis is the ``true`` criterion (else the
  instructions); P(yes) is entailment / (entailment + contradiction).

Options (keyword arguments to ``decide``):

* ``nli_labels`` — the classifier's output order; defaults to OpenJev's
  ``("contradiction", "entailment", "neutral")``. MNLI checkpoints such as
  ``bart-large-mnli`` use ``("contradiction", "neutral", "entailment")``.
* ``hypothesis_template`` — formats each option; ``{option}`` and
  ``{instructions}`` are available. Defaults to ``"This is about {option}."``.
* ``pair_template`` — how premise and hypothesis are joined into one input;
  defaults to OpenJev's ``"Premise: {premise}\\nHypothesis: {hypothesis}"``.

Everything else on this protocol is plain Chat Completions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..decision import Answer, Decision, Question
from ..tokens import Tokens
from .chat_completions import ChatCompletions

DEFAULT_LABELS = ("contradiction", "entailment", "neutral")
DEFAULT_HYPOTHESIS_TEMPLATE = "This is about {option}."
DEFAULT_PAIR_TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"
# vLLM's per-request cap on classify inputs.
MAX_INPUTS = 256


class NLIClassify(ChatCompletions):
    def classify_url(self) -> str:
        base = self.provider.api_base.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/classify"

    async def decide(
        self, state: Any, questions: dict[str, Question], *, model: str, **options: Any
    ) -> Decision:
        labels = list(options.get("nli_labels") or DEFAULT_LABELS)
        entailment = labels.index("entailment")
        contradiction = labels.index("contradiction")
        hypothesis_template = options.get("hypothesis_template") or DEFAULT_HYPOTHESIS_TEMPLATE
        pair_template = options.get("pair_template") or DEFAULT_PAIR_TEMPLATE

        premise = state if isinstance(state, str) else json.dumps(state, default=str)
        hypotheses = {
            name: _hypotheses(question, hypothesis_template) for name, question in questions.items()
        }
        inputs = [
            pair_template.format(premise=premise, hypothesis=h)
            for name in hypotheses
            for h in hypotheses[name]
        ]

        rows: list[list[float]] = []
        input_tokens = 0
        for start in range(0, len(inputs), MAX_INPUTS):
            response = await self.connection.post(
                self.classify_url(), {"model": model, "input": inputs[start : start + MAX_INPUTS]}
            )
            body = response.body or {}
            data = sorted(body.get("data") or [], key=lambda row: row.get("index", 0))
            rows.extend(row["probs"] for row in data)
            input_tokens += (body.get("usage") or {}).get("prompt_tokens") or 0

        answers: dict[str, Answer] = {}
        cursor = 0
        for name, question in questions.items():
            chunk = rows[cursor : cursor + len(hypotheses[name])]
            cursor += len(hypotheses[name])
            if question.type == "yes_no":
                answers[name] = _yes_no_answer(chunk[0], entailment, contradiction)
            else:
                answers[name] = _ranked_answer(question, [p[entailment] for p in chunk])
        return Decision(
            answers=answers, model=model, tokens=Tokens.build(input=input_tokens or None)
        )


def _hypotheses(question: Question, template: str) -> list[str]:
    instructions = question.instructions or ""
    criteria = question.criteria or ()
    if question.type == "yes_no":
        true = criteria.get("true") if isinstance(criteria, Mapping) else None
        return [true or instructions]
    texts = list(criteria.values()) if isinstance(criteria, Mapping) else list(criteria)
    return [template.format(option=text, instructions=instructions) for text in texts]


def _normalize(scores: list[float]) -> list[float]:
    total = sum(scores)
    if total <= 0:
        return [1 / len(scores)] * len(scores)
    return [s / total for s in scores]


def _ranked_answer(question: Question, entailments: list[float]) -> Answer:
    probs = _normalize(entailments)
    best = max(range(len(probs)), key=probs.__getitem__)
    if question.type == "choice":
        labels = question.options
        return Answer(
            type="choice",
            choice=labels[best],
            probabilities=dict(zip(labels, probs, strict=True)),
            confidence=probs[best],
        )
    keys = [str(i) for i in range(len(probs))]
    return Answer(
        type="score",
        score=sum(i * p for i, p in enumerate(probs)),
        probabilities=dict(zip(keys, probs, strict=True)),
        legend=dict(zip(keys, question.options, strict=True)),
        confidence=probs[best],
    )


def _yes_no_answer(probs: list[float], entailment: int, contradiction: int) -> Answer:
    yes, no = probs[entailment], probs[contradiction]
    return Answer(type="yes_no", probability=yes / (yes + no) if yes + no > 0 else 0.5)
