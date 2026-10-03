"""Decisions: typed, calibrated answers instead of text.

Decision models ("System One" models such as TypeSafe's Jev) take a piece of
context — the *state* — plus named, typed questions, and return a probability
distribution per question rather than generated text::

    decision = await pyllym.decide(
        "Customer says they were charged twice for one order.",
        {
            "team": Question.choice({"billing": "Charges, refunds", "technical": "Bugs"}),
            "severity": Question.score(["None", "Limited", "Material"]),
            "needs_human": Question.yes_no("Does this need a human reviewer?"),
        },
    )
    decision["team"].choice        # "billing"
    decision["severity"].level     # "Limited"
    decision["needs_human"].probability

The question/answer shapes here are provider-neutral; each protocol maps them
onto its wire format (TypeSafe's ``systemone`` endpoint, an NLI classifier
served by vLLM, ...). OpenAI's Decisions API (``/v1/decisions``) is still
invite-only with an unpublished schema; it slots in as one more protocol once
documented.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from . import models as _models
from .tokens import Tokens

if TYPE_CHECKING:
    from .context import Context

QuestionType = Literal["choice", "score", "yes_no"]

_QUESTION_TYPES: tuple[str, ...] = ("choice", "score", "yes_no")
# Wire spellings of the yes/no type seen in the wild ("noul" at TypeSafe).
_YES_NO_ALIASES = ("yes_no", "noul", "null")


@dataclass(frozen=True, slots=True)
class Question:
    """One typed question.

    ``criteria`` depends on ``type``:

    * ``choice`` — ``{label: description}``, at least two options.
    * ``score`` — ordered levels, lowest first, at least two.
    * ``yes_no`` — optional ``{"true": ..., "false": ...}`` descriptions.
    """

    type: QuestionType
    instructions: str | None = None
    criteria: Mapping[str, str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.type not in _QUESTION_TYPES:
            raise ValueError(f"Unknown question type: {self.type!r}")
        if self.type in ("choice", "score") and len(self.criteria or ()) < 2:
            raise ValueError(f"A {self.type} question needs at least two options")

    @classmethod
    def choice(
        cls, options: Mapping[str, str] | Sequence[str], *, instructions: str | None = None
    ) -> Question:
        """Pick one label. A plain list of labels uses each label as its description."""
        criteria = (
            dict(options) if isinstance(options, Mapping) else {label: label for label in options}
        )
        return cls("choice", instructions, criteria)

    @classmethod
    def score(cls, levels: Sequence[str], *, instructions: str | None = None) -> Question:
        """Place the state on an ordered scale (``levels`` lowest first)."""
        return cls("score", instructions, tuple(levels))

    @classmethod
    def yes_no(
        cls, instructions: str, *, true: str | None = None, false: str | None = None
    ) -> Question:
        """The probability that the answer is yes."""
        criteria = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
        return cls("yes_no", instructions, criteria or None)

    @property
    def options(self) -> list[str]:
        """Choice labels or score levels, in order (empty for yes/no)."""
        if self.type == "yes_no" or not self.criteria:
            return []
        return list(self.criteria)

    @classmethod
    def build(cls, data: Mapping[str, Any] | Question) -> Question:
        """Coerce a :meth:`to_dict`-shaped (or wire-shaped) mapping into a Question."""
        if isinstance(data, Question):
            return data
        type_ = data.get("type")
        criteria = data.get("criteria")
        instructions = data.get("instructions")
        if type_ in _YES_NO_ALIASES:
            return cls("yes_no", instructions, dict(criteria) if criteria else None)
        if type_ == "score":
            return cls.score(criteria or (), instructions=instructions)
        if type_ == "choice":
            return cls.choice(criteria or {}, instructions=instructions)
        raise ValueError(f"Unknown question type: {type_!r}")

    def to_dict(self) -> dict[str, Any]:
        criteria: Any = self.criteria
        if isinstance(criteria, tuple):
            criteria = list(criteria)
        elif criteria is not None:
            criteria = dict(criteria)
        data = {"type": self.type, "instructions": self.instructions, "criteria": criteria}
        return {k: v for k, v in data.items() if v is not None}


@dataclass(frozen=True, slots=True)
class Answer:
    """The model's answer to one :class:`Question`.

    * ``choice`` answers carry ``choice``, ``probabilities`` per label and ``confidence``.
    * ``score`` answers carry ``score`` (probability-weighted position on the
      0-based level index), ``probabilities`` per level, ``legend`` (level index
      -> level text) and ``confidence``.
    * ``yes_no`` answers carry ``probability`` (of yes).
    """

    type: QuestionType
    choice: str | None = None
    score: float | None = None
    probability: float | None = None
    probabilities: Mapping[str, float] = field(default_factory=dict)
    confidence: float | None = None
    legend: Mapping[str, str] = field(default_factory=dict)

    @property
    def value(self) -> str | float | None:
        """The headline answer: the chosen label, the score, or P(yes)."""
        if self.type == "choice":
            return self.choice
        if self.type == "score":
            return self.score
        return self.probability

    @property
    def level(self) -> str | None:
        """For a score answer, the text of the most likely level."""
        if self.type != "score" or not self.probabilities:
            return None
        best = max(self.probabilities, key=lambda k: self.probabilities[k])
        return self.legend.get(best, best)

    @property
    def yes(self) -> bool | None:
        """For a yes/no answer, whether yes is more likely than not."""
        if self.probability is None:
            return None
        return self.probability >= 0.5

    @classmethod
    def build(cls, data: Mapping[str, Any]) -> Answer:
        type_ = "yes_no" if data.get("type") in _YES_NO_ALIASES else data.get("type")
        return cls(
            type=type_,  # type: ignore[arg-type]
            choice=data.get("choice"),
            score=data.get("score"),
            probability=data.get("probability"),
            probabilities=dict(data.get("probabilities") or {}),
            confidence=data.get("confidence"),
            legend={str(k): v for k, v in (data.get("legend") or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "type": self.type,
            "choice": self.choice,
            "score": self.score,
            "probability": self.probability,
            "probabilities": dict(self.probabilities) or None,
            "confidence": self.confidence,
            "legend": dict(self.legend) or None,
        }
        return {k: v for k, v in data.items() if v is not None}


@dataclass(slots=True)
class Decision:
    """Answers keyed by question name."""

    answers: dict[str, Answer]
    model: str
    tokens: Tokens | None = None

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)

    @property
    def input_tokens(self) -> int:
        return (self.tokens.input if self.tokens else None) or 0

    @property
    def output_tokens(self) -> int:
        return (self.tokens.output if self.tokens else None) or 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {name: answer.to_dict() for name, answer in self.answers.items()},
            "tokens": self.tokens.to_dict() if self.tokens else None,
        }


async def decide(
    state: Any,
    questions: Mapping[str, Question | Mapping[str, Any]],
    *,
    model: str | None = None,
    provider: str | None = None,
    assume_model_exists: bool = False,
    context: Context | None = None,
    **options: Any,
) -> Decision:
    from . import config as _config

    if not questions:
        raise ValueError("decide() needs at least one question")
    cfg = context.config if context else _config()
    model = model or cfg.default_decision_model
    model_info, provider_instance = _models.resolve(
        model, provider=provider, assume_exists=assume_model_exists, config=cfg
    )
    built = {name: Question.build(q) for name, q in questions.items()}
    return await provider_instance.decide(state, built, model=model_info.id, **options)
