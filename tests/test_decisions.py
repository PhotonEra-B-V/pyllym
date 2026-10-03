from __future__ import annotations

import json

import pytest

import pyllym
from pyllym import Answer, Decision, Question

from .conftest import sent_requests

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
VLLM_BASE = "http://localhost:8000/v1"
CLASSIFY_URL = "http://localhost:8000/classify"

QUESTIONS = {
    "team": Question.choice(
        {"billing": "Charges or refunds", "technical": "Product errors"},
        instructions="Which team should investigate?",
    ),
    "severity": Question.score(["No impact", "Limited impact", "Material impact"]),
    "needs_human": Question.yes_no("Does this need a human?", true="An account change"),
}


def jev_response() -> dict:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "team": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.85, "technical": 0.15},
                "confidence": 0.82,
            },
            "severity": {
                "type": "score",
                "score": 1.2,
                "legend": {"0": "No impact", "1": "Limited impact", "2": "Material impact"},
                "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3},
                "confidence": 0.7,
            },
            "needs_human": {"type": "noul", "noul": 0.91},
        },
        "usage": {"input_tokens": 312, "output_tokens": 48},
    }


@pytest.fixture
def vllm(monkeypatch):
    monkeypatch.setattr(pyllym.config(), "vllm_api_base", VLLM_BASE)


# --- TypeSafe (Jev) -------------------------------------------------------------


@pytest.mark.asyncio
async def test_jev_decide_round_trip(mock_http):
    mock_http.post(TYPESAFE_URL, payload=jev_response())

    decision = await pyllym.decide("Charged twice for one order.", QUESTIONS)

    request = sent_requests(mock_http)[-1]
    assert request.kwargs["headers"]["Authorization"] == "Bearer ts-test"
    assert request.kwargs["json"] == {
        "model": "jev-latest",
        "state": "Charged twice for one order.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should investigate?",
                "criteria": {"billing": "Charges or refunds", "technical": "Product errors"},
            },
            "severity": {
                "type": "score",
                "criteria": ["No impact", "Limited impact", "Material impact"],
            },
            "needs_human": {
                "type": "noul",
                "instructions": "Does this need a human?",
                "criteria": {"true": "An account change"},
            },
        },
    }
    assert isinstance(decision, Decision)
    assert decision.model == "jev-1.13.0"
    assert (decision.input_tokens, decision.output_tokens) == (312, 48)
    assert decision["team"].choice == "billing"
    assert decision["team"].value == "billing"
    assert decision["team"].confidence == 0.82
    assert decision["severity"].score == 1.2
    assert decision["severity"].level == "Limited impact"
    assert decision["needs_human"].type == "yes_no"
    assert decision["needs_human"].probability == 0.91
    assert decision["needs_human"].yes is True
    assert list(decision) == ["team", "severity", "needs_human"]


@pytest.mark.asyncio
async def test_jev_accepts_dict_questions_and_null_spelling(mock_http):
    mock_http.post(
        TYPESAFE_URL,
        payload={"model": "jev-latest", "answers": {"ok": {"type": "null", "null": 0.2}}},
    )
    decision = await pyllym.decide(
        "state", {"ok": {"type": "noul", "instructions": "Is it fine?"}}, model="jev-preview"
    )
    sent = sent_requests(mock_http)[-1].kwargs["json"]
    assert sent["model"] == "jev-preview"
    assert sent["questions"]["ok"]["type"] == "noul"
    assert decision["ok"].probability == 0.2
    assert decision["ok"].yes is False
    assert decision.tokens is None


@pytest.mark.asyncio
async def test_jev_unregistered_version_and_custom_base(mock_http, monkeypatch):
    monkeypatch.setattr(pyllym.config(), "typesafe_api_base", "https://gw.example/typesafe/v1")
    mock_http.post("https://gw.example/typesafe/v1/systemone", payload=jev_response())
    decision = await pyllym.decide("s", QUESTIONS, model="jev-2.0.0", provider="typesafe")
    assert decision["team"].choice == "billing"


@pytest.mark.asyncio
async def test_jev_rejects_too_many_options():
    too_many = Question.choice([f"o{i}" for i in range(256)])
    with pytest.raises(ValueError, match="255"):
        await pyllym.decide("s", {"q": too_many})
    with pytest.raises(ValueError, match="10 levels"):
        await pyllym.decide("s", {"q": Question.score([str(i) for i in range(11)])})


@pytest.mark.asyncio
async def test_jev_cannot_chat():
    chat = pyllym.create_chat(model="jev-latest")
    with pytest.raises(pyllym.Error, match="decide"):
        await chat.ask("hi")


@pytest.mark.asyncio
async def test_providers_without_decisions_say_so():
    with pytest.raises(NotImplementedError, match="does not support decisions"):
        await pyllym.decide("s", QUESTIONS, model="gpt-5.4")


@pytest.mark.asyncio
async def test_decide_needs_questions():
    with pytest.raises(ValueError, match="at least one question"):
        await pyllym.decide("s", {})


# --- vLLM NLI classifier (OpenJev) ----------------------------------------------


def classify_response(rows: list[list[float]], tokens: int = 10) -> dict:
    return {
        "object": "list",
        "model": "openjev-4b",
        "data": [
            {"index": i, "label": "x", "probs": probs, "num_classes": 3}
            for i, probs in enumerate(rows)
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "completion_tokens": 0},
    }


@pytest.mark.asyncio
async def test_openjev_on_vllm(mock_http, vllm):
    # Rows are [contradiction, entailment, neutral], one per hypothesis in
    # question order: 2 team options, 3 severity levels, 1 yes/no.
    mock_http.post(
        CLASSIFY_URL,
        payload=classify_response(
            [
                [0.1, 0.6, 0.3],
                [0.7, 0.2, 0.1],
                [0.8, 0.1, 0.1],
                [0.2, 0.3, 0.5],
                [0.4, 0.4, 0.2],
                [0.25, 0.75, 0.0],
            ],
            tokens=120,
        ),
    )

    decision = await pyllym.decide(
        {"ticket": "Charged twice."}, QUESTIONS, model="openjev-4b", provider="vllm"
    )

    sent = sent_requests(mock_http)[-1].kwargs["json"]
    assert sent["model"] == "openjev-4b"
    assert sent["input"][0] == (
        'Premise: {"ticket": "Charged twice."}\nHypothesis: This is about Charges or refunds.'
    )
    assert sent["input"][5].endswith("Hypothesis: An account change")
    assert len(sent["input"]) == 6

    team = decision["team"]
    assert team.choice == "billing"
    assert team.probabilities == pytest.approx({"billing": 0.75, "technical": 0.25})
    assert team.confidence == pytest.approx(0.75)

    severity = decision["severity"]
    assert severity.probabilities == pytest.approx({"0": 0.125, "1": 0.375, "2": 0.5})
    assert severity.score == pytest.approx(0.375 + 2 * 0.5)
    assert severity.level == "Material impact"

    assert decision["needs_human"].probability == pytest.approx(0.75)
    assert decision.input_tokens == 120
    assert decision.model == "openjev-4b"


@pytest.mark.asyncio
async def test_nli_label_order_and_templates_are_configurable(mock_http, vllm):
    # bart-large-mnli order: [contradiction, neutral, entailment]
    mock_http.post(CLASSIFY_URL, payload=classify_response([[0.1, 0.1, 0.8], [0.6, 0.3, 0.1]]))
    decision = await pyllym.decide(
        "I want my money back",
        {"intent": Question.choice(["refund", "upgrade"], instructions="Intent?")},
        model="facebook/bart-large-mnli",
        provider="vllm",
        nli_labels=("contradiction", "neutral", "entailment"),
        hypothesis_template="{instructions} {option}",
        pair_template="{premise} || {hypothesis}",
    )
    sent = sent_requests(mock_http)[-1].kwargs["json"]
    assert sent["input"] == [
        "I want my money back || Intent? refund",
        "I want my money back || Intent? upgrade",
    ]
    assert decision["intent"].choice == "refund"
    assert decision["intent"].probabilities["refund"] == pytest.approx(0.8 / 0.9)


@pytest.mark.asyncio
async def test_nli_batches_over_vllm_input_cap(mock_http, vllm):
    labels = [f"o{i}" for i in range(300)]
    mock_http.post(CLASSIFY_URL, payload=classify_response([[0.5, 0.5, 0.0]] * 256, tokens=5))
    mock_http.post(
        CLASSIFY_URL,
        payload=classify_response([[0.5, 0.5, 0.0]] * 43 + [[0.0, 1.0, 0.0]], tokens=5),
    )
    decision = await pyllym.decide(
        "s", {"q": Question.choice(labels)}, model="openjev-4b", provider="vllm"
    )
    requests = sent_requests(mock_http)
    assert [len(r.kwargs["json"]["input"]) for r in requests] == [256, 44]
    assert decision["q"].choice == "o299"
    assert decision.input_tokens == 10


# --- value objects ----------------------------------------------------------------


def test_question_validation():
    with pytest.raises(ValueError, match="at least two"):
        Question.choice(["only"])
    with pytest.raises(ValueError, match="at least two"):
        Question.score(["one"])
    with pytest.raises(ValueError, match="Unknown question type"):
        Question.build({"type": "essay"})


def test_question_round_trip():
    for question in QUESTIONS.values():
        assert Question.build(question.to_dict()) == question
    assert Question.choice(["a", "b"]).criteria == {"a": "a", "b": "b"}
    assert Question.score(["lo", "hi"]).options == ["lo", "hi"]
    assert Question.yes_no("ok?").to_dict() == {"type": "yes_no", "instructions": "ok?"}


def test_answer_and_decision_round_trip():
    answer = Answer(type="score", score=1.5, probabilities={"0": 0.5, "1": 0.5}, legend={"0": "lo"})
    assert Answer.build(answer.to_dict()) == answer
    decision = Decision(answers={"s": answer}, model="jev-latest", tokens=pyllym.Tokens(input=3))
    assert json.loads(json.dumps(decision.to_dict())) == {
        "model": "jev-latest",
        "answers": {"s": answer.to_dict()},
        "tokens": {"input_tokens": 3},
    }
    assert decision.output_tokens == 0


def test_jev_models_are_registered():
    model = pyllym.models.find("jev-latest")
    assert (model.provider, model.type) == ("typesafe", "decision")
    assert {m.id for m in pyllym.models.decision_models()} >= {"jev-latest", "jev-1.13.0"}
    assert pyllym.config().default_decision_model == "jev-latest"
