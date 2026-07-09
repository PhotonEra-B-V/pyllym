"""TOML front-end for the BDD builder — one spec file, two modes.

A TOML spec is the single authoring surface for ``pyllm.bdd``: the API
surface, coverable rules and edge cases, optional ``sequenceDiagram`` blocks
(parsed by the existing :mod:`pyllm.bdd.sequence`), and — in *plan mode* —
hand-written test cases.

Two modes, auto-detected from the presence of ``[[cases]]`` unless
``[meta].mode`` says otherwise:

- **plan mode** — the spec carries ``[[cases]]``; :meth:`TomlSpec.to_test_plan`
  builds a :class:`~pyllm.bdd.schema.TestPlan` deterministically, with no LLM
  in the loop. The plan is rendered by ``render_tests`` and accepted by
  ``validate_plan`` when the cases cover every rule/edge/message id.
- **request mode** — the spec is loose (rules and edge cases, no cases);
  :meth:`TomlSpec.to_prompt` produces a prompt block for the planner LLM,
  which fills the ``TestPlan`` schema. The watertight guarantee still lives in
  ``validate_plan`` + template rendering; the model never writes pytest text.

Value objects are frozen dataclasses with ``to_dict()``, a ``slug`` property,
and globally-unique ``M``-id numbering across every ``[[sequences]]`` block via
the ``start`` arg to ``sequence.parse``.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .schema import ApiSurface, FixtureDef, TestCase, TestPlan
from .sequence import SequenceDiagram
from .sequence import parse as parse_sequence

Mode = Literal["plan", "request"]


@dataclass(frozen=True)
class Meta:
    feature_name: str
    description: str = ""
    mode: Mode | None = None
    async_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "description": self.description,
            "mode": self.mode,
            "async_required": self.async_required,
        }


@dataclass(frozen=True)
class Api:
    module: str
    imports: tuple[str, ...] = ()
    signatures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "imports": list(self.imports),
            "signatures": list(self.signatures),
        }

    def to_surface(self) -> ApiSurface:
        return ApiSurface(
            module=self.module,
            imports=list(self.imports),
            signatures=list(self.signatures),
        )


@dataclass(frozen=True)
class Coverable:
    """A rule or edge case carrying a coverable id (``rule_1``, ``edge_1``)."""

    id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass(frozen=True)
class Case:
    """A hand-written test case (plan mode); mirrors ``schema.TestCase``."""

    test_name: str
    scenario: str
    given: tuple[str, ...] = ()
    when: str = "None"
    then: tuple[str, ...] = ()
    fixtures: tuple[str, ...] = ()
    covers: tuple[str, ...] = ()
    is_async: bool = False
    edge_case: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "scenario": self.scenario,
            "given": list(self.given),
            "when": self.when,
            "then": list(self.then),
            "fixtures": list(self.fixtures),
            "covers": list(self.covers),
            "is_async": self.is_async,
            "edge_case": self.edge_case,
            "notes": self.notes,
        }

    def to_test_case(self) -> TestCase:
        return TestCase(
            scenario=self.scenario,
            test_name=self.test_name,
            given=list(self.given),
            when=self.when,
            then=list(self.then),
            fixtures=list(self.fixtures),
            covers=list(self.covers),
            is_async=self.is_async,
            edge_case=self.edge_case,
            notes=self.notes,
        )


@dataclass(frozen=True)
class Fixture:
    """A hand-written fixture (plan mode); mirrors ``schema.FixtureDef``."""

    name: str
    body: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "body": list(self.body)}

    def to_fixture_def(self) -> FixtureDef:
        return FixtureDef(name=self.name, body=list(self.body))


@dataclass(frozen=True)
class TomlSpec:
    meta: Meta
    api: Api
    rules: tuple[Coverable, ...] = ()
    edge_cases: tuple[Coverable, ...] = ()
    sequences: tuple[SequenceDiagram, ...] = ()
    cases: tuple[Case, ...] = ()
    fixtures: tuple[Fixture, ...] = ()
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "api": self.api.to_dict(),
            "rules": [r.to_dict() for r in self.rules],
            "edge_cases": [e.to_dict() for e in self.edge_cases],
            "sequences": [d.to_dict() for d in self.sequences],
            "cases": [c.to_dict() for c in self.cases],
            "fixtures": [f.to_dict() for f in self.fixtures],
            "source_path": self.source_path,
        }

    # -- identity ---------------------------------------------------------

    @property
    def slug(self) -> str:
        """Filesystem/identifier-safe name: ``Retry policy`` -> ``retry_policy``."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.meta.feature_name.lower()).strip("_")
        return slug or "feature"

    @property
    def mode(self) -> Mode:
        """Explicit ``[meta].mode`` wins; else 'plan' iff any ``[[cases]]``."""
        if self.meta.mode is not None:
            return self.meta.mode
        return "plan" if self.cases else "request"

    # -- coverage sets (feed straight into checks.validate_plan) ----------

    def scenario_names(self) -> set[str]:
        """Scenario names every plan must cover (from ``[[cases]]``)."""
        return {c.scenario for c in self.cases}

    def message_ids(self) -> set[str]:
        """Sequence-diagram message ids (``M1``, ...) across all diagrams."""
        return {m.id for d in self.sequences for m in d.messages}

    def required_ids(self) -> set[str]:
        """Rule + edge-case ids, folded into one set (the contract's field)."""
        return {r.id for r in self.rules} | {e.id for e in self.edge_cases}

    # -- plan mode --------------------------------------------------------

    def to_test_plan(self) -> TestPlan:
        """Deterministically build a :class:`TestPlan` from ``[[cases]]``.

        The resulting plan renders via ``render_tests`` and is accepted by
        ``validate_plan`` when the hand-written cases cover every rule, edge
        case, and sequence-diagram message id.
        """
        return TestPlan(
            feature=self.meta.feature_name,
            api=self.api.to_surface(),
            fixtures=[f.to_fixture_def() for f in self.fixtures],
            cases=[c.to_test_case() for c in self.cases],
            required_ids=sorted(self.required_ids()),
        )

    # -- request mode -----------------------------------------------------

    def to_prompt(self) -> str:
        """A readable prompt block for the planner LLM (request mode).

        Lists the API surface, every rule and edge case with its id, and any
        sequence diagrams, instructing that every id be claimed via 'covers'.
        """
        lines: list[str] = [f"Feature: {self.meta.feature_name}"]
        if self.meta.description:
            lines.extend(f"  {line}" for line in self.meta.description.splitlines())

        lines.append("")
        lines.append("API surface (exercise only this):")
        lines.append(f"  module: {self.api.module}")
        for imp in self.api.imports:
            lines.append(f"  import: {imp}")
        for sig in self.api.signatures:
            lines.append(f"  signature: {sig}")

        if self.rules:
            lines.append("")
            lines.append("Rules (cover every id via 'covers'):")
            lines.extend(f"  {r.id}: {r.text}" for r in self.rules)

        if self.edge_cases:
            lines.append("")
            lines.append("Edge cases (cover every id via 'covers'):")
            lines.extend(f"  {e.id}: {e.text}" for e in self.edge_cases)

        if self.sequences:
            lines.append("")
            lines.append("Sequence-diagram messages (cover every id via 'covers'):")
            for diagram in self.sequences:
                lines.append(diagram.to_annotated())

        if self.meta.async_required:
            lines.append("")
            lines.append("This feature is async: set is_async=true and use 'await' in the act.")

        lines.append("")
        lines.append(
            "Every rule id, edge-case id, and sequence message id above MUST be "
            "claimed by at least one test case via its 'covers' list."
        )
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(v) for v in value)


def _coverables(entries: Any) -> tuple[Coverable, ...]:
    out: list[Coverable] = []
    for entry in entries or ():
        out.append(Coverable(id=str(entry["id"]), text=str(entry.get("text", ""))))
    return tuple(out)


def _cases(entries: Any) -> tuple[Case, ...]:
    out: list[Case] = []
    for entry in entries or ():
        out.append(
            Case(
                test_name=str(entry["test_name"]),
                scenario=str(entry.get("scenario", "")),
                given=_str_tuple(entry.get("given")),
                when=str(entry.get("when", "None")),
                then=_str_tuple(entry.get("then")),
                fixtures=_str_tuple(entry.get("fixtures")),
                covers=_str_tuple(entry.get("covers")),
                is_async=bool(entry.get("is_async", False)),
                edge_case=bool(entry.get("edge_case", False)),
                notes=str(entry.get("notes", "")),
            )
        )
    return tuple(out)


def _fixtures(entries: Any) -> tuple[Fixture, ...]:
    out: list[Fixture] = []
    for entry in entries or ():
        out.append(Fixture(name=str(entry["name"]), body=_str_tuple(entry.get("body"))))
    return tuple(out)


def _sequences(entries: Any) -> tuple[SequenceDiagram, ...]:
    """Parse each ``[[sequences]]`` mermaid string, keeping ids globally unique.

    Each diagram starts one past the running message count via the ``start``
    arg to ``sequence.parse``, so ``M``-ids stay unique across all blocks.
    """
    diagrams: list[SequenceDiagram] = []
    for entry in entries or ():
        mermaid = str(entry["mermaid"])
        next_id = 1 + sum(len(d.messages) for d in diagrams)
        diagrams.append(parse_sequence(mermaid, start=next_id))
    return tuple(diagrams)


def parse(text: str, *, source_path: str | None = None) -> TomlSpec:
    """Parse a TOML spec into a :class:`TomlSpec`.

    Raises ``ValueError`` when ``[meta].feature_name`` is missing.
    """
    data = tomllib.loads(text)

    meta_raw = data.get("meta") or {}
    feature_name = meta_raw.get("feature_name")
    if not feature_name:
        raise ValueError("TOML spec missing required [meta].feature_name")
    mode_raw = meta_raw.get("mode")
    if mode_raw is not None and mode_raw not in ("plan", "request"):
        raise ValueError(f"[meta].mode must be 'plan' or 'request', got {mode_raw!r}")
    meta = Meta(
        feature_name=str(feature_name),
        description=str(meta_raw.get("description", "")),
        mode=mode_raw,
        async_required=bool(meta_raw.get("async_required", False)),
    )

    api_raw = data.get("api") or {}
    module = api_raw.get("module")
    if not module:
        raise ValueError("TOML spec missing required [api].module")
    api = Api(
        module=str(module),
        imports=_str_tuple(api_raw.get("imports")),
        signatures=_str_tuple(api_raw.get("signatures")),
    )

    return TomlSpec(
        meta=meta,
        api=api,
        rules=_coverables(data.get("rules")),
        edge_cases=_coverables(data.get("edge_cases")),
        sequences=_sequences(data.get("sequences")),
        cases=_cases(data.get("cases")),
        fixtures=_fixtures(data.get("fixtures")),
        source_path=source_path,
    )


def parse_file(path: str | Path) -> TomlSpec:
    path = Path(path)
    return parse(path.read_text(encoding="utf-8"), source_path=str(path))
