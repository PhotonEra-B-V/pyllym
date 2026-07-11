"""Minimal Mermaid ``sequenceDiagram`` parser producing frozen value objects.

Sequence diagrams complement a TOML spec's rules and cases: rules say *what*
must be true, diagrams say *who talks to whom, in what order*. Each message gets a
stable id (``M1``, ``M2``, ...) so test cases can claim coverage of it and
the builder can verify — mechanically — that no interaction was dropped.

Covers the common subset: participants/actors (with aliases), call arrows
(``->>``, ``->``), reply arrows (``-->>``, ``-->``), async arrows (``-)``,
``--)``), ``+``/``-`` activation markers, and ``alt``/``else``/``opt``/
``loop``/``par``/``critical`` blocks (recorded as context on each message).
Notes and styling directives are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ARROW = re.compile(
    r"^(?P<sender>[^-<>:]+?)\s*"
    r"(?P<arrow>-->>|->>|--\)|-\)|-->|->)\s*"
    r"(?P<activation>[+-]?)\s*"
    r"(?P<receiver>[^:]+?)\s*:\s*"
    r"(?P<text>.*)$"
)
_KINDS = {
    "->>": "call",
    "->": "call",
    "-->>": "reply",
    "-->": "reply",
    "-)": "async",
    "--)": "async",
}
_BLOCK_OPENERS = ("alt", "opt", "loop", "par", "critical", "rect", "break")


@dataclass(frozen=True)
class SequenceMessage:
    id: str
    sender: str
    receiver: str
    text: str
    kind: str  # call | reply | async
    context: tuple[str, ...] = ()  # enclosing alt/opt/loop labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "text": self.text,
            "kind": self.kind,
            "context": list(self.context),
        }


@dataclass(frozen=True)
class SequenceDiagram:
    participants: tuple[str, ...]
    messages: tuple[SequenceMessage, ...]
    source: str  # normalized mermaid source, for verbatim re-emission

    def to_dict(self) -> dict[str, Any]:
        return {
            "participants": list(self.participants),
            "messages": [m.to_dict() for m in self.messages],
            "source": self.source,
        }

    def to_annotated(self) -> str:
        """One line per message with its id — the planner prompt format."""
        lines = []
        for m in self.messages:
            where = f" [{' / '.join(m.context)}]" if m.context else ""
            lines.append(f"{m.id}: {m.sender} -[{m.kind}]-> {m.receiver}: {m.text}{where}")
        return "\n".join(lines)


def parse(text: str, *, start: int = 1) -> SequenceDiagram:
    """Parse a mermaid ``sequenceDiagram`` block.

    ``start`` sets the first message number, so multiple diagrams attached to
    one feature keep globally unique ids.
    """
    participants: list[str] = []
    messages: list[SequenceMessage] = []
    context: list[str] = []
    source_lines: list[str] = []
    seen_header = False
    n = start

    def note_participant(name: str) -> None:
        if name not in participants:
            participants.append(name)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        source_lines.append(line)
        if line == "sequenceDiagram":
            seen_header = True
            continue
        word, _, rest = line.partition(" ")
        if word in ("participant", "actor"):
            # "participant A as Alias" declares A; messages use the bare name.
            note_participant(rest.split(" as ")[0].strip())
            continue
        if word in _BLOCK_OPENERS:
            context.append(f"{word} {rest.strip()}".strip())
            continue
        if word == "else":
            if context:
                context[-1] = f"else {rest.strip()}".strip()
            continue
        if word == "end":
            if context:
                context.pop()
            continue
        match = _ARROW.match(line)
        if match is None:
            continue  # notes, activate/deactivate lines, styling
        sender = match["sender"].strip()
        receiver = match["receiver"].strip()
        note_participant(sender)
        note_participant(receiver)
        messages.append(
            SequenceMessage(
                id=f"M{n}",
                sender=sender,
                receiver=receiver,
                text=match["text"].strip(),
                kind=_KINDS[match["arrow"]],
                context=tuple(context),
            )
        )
        n += 1

    if not seen_header:
        raise ValueError("no 'sequenceDiagram' header found in mermaid source")
    return SequenceDiagram(
        participants=tuple(participants),
        messages=tuple(messages),
        source="\n".join(source_lines),
    )
