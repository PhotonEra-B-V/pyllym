"""TypeSafe's ``systemone`` decision protocol (Jev).

Jev never writes text: it serves a single ``POST /v1/systemone`` endpoint
that takes a ``state`` plus named, typed ``questions`` and answers each with
a calibrated distribution. There is no chat, tools or streaming.

Wire shape::

    {"model": "jev-latest", "state": "...",
     "questions": {"team": {"type": "choice", "instructions": "...",
                            "criteria": {"billing": "...", "technical": "..."}}}}
    -> {"model": "jev-1.13.0",
        "answers": {"team": {"type": "choice", "choice": "billing",
                             "probabilities": {...}, "confidence": 0.82}},
        "usage": {"input_tokens": 312, "output_tokens": 48}}

Score questions send ``criteria`` as an ordered list (2-10 levels) and answer
with ``score``, ``legend`` and ``probabilities``; the yes/no type is spelled
``noul`` on the wire and answers with ``noul``, the probability of yes.
"""

from __future__ import annotations

from typing import Any

from .. import utils
from ..decision import Answer, Decision, Question
from ..errors import Error
from ..protocol import Protocol
from ..tokens import Tokens

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10

_WIRE_TYPES = {"choice": "choice", "score": "score", "yes_no": "noul"}


class SystemOne(Protocol):
    def decision_url(self) -> str:
        return "systemone"

    def render_payload(self, messages: Any, **kwargs: Any) -> dict[str, Any]:
        raise Error(None, "Jev is a decision model and cannot chat; use pyllym.decide()")

    def render_decision_payload(
        self, state: Any, questions: dict[str, Question], *, model: str, **options: Any
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "state": state,
            "questions": {name: self._format_question(q) for name, q in questions.items()},
        }
        return utils.deep_merge(payload, options.get("params") or {})

    def parse_decision_response(
        self, response: Any, *, model: str, questions: dict[str, Question]
    ) -> Decision:
        body = response.body or {}
        usage = body.get("usage") or {}
        return Decision(
            answers={
                name: self._parse_answer(data) for name, data in (body.get("answers") or {}).items()
            },
            model=body.get("model") or model,
            tokens=Tokens.build(input=usage.get("input_tokens"), output=usage.get("output_tokens")),
        )

    @staticmethod
    def _format_question(question: Question) -> dict[str, Any]:
        count = len(question.options)
        if question.type == "choice" and count > MAX_CHOICE_OPTIONS:
            raise ValueError(f"Jev choice questions take at most {MAX_CHOICE_OPTIONS} options")
        if question.type == "score" and count > MAX_SCORE_LEVELS:
            raise ValueError(f"Jev score questions take at most {MAX_SCORE_LEVELS} levels")
        return {**question.to_dict(), "type": _WIRE_TYPES[question.type]}

    @staticmethod
    def _parse_answer(data: dict[str, Any]) -> Answer:
        if data.get("type") in ("noul", "null"):
            probability = data.get("noul", data.get("null"))
            return Answer.build({"type": "yes_no", "probability": probability})
        return Answer.build(data)
