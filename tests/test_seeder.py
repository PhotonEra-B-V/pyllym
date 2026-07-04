from __future__ import annotations

from . import factories as f
from . import seed_fixtures


def test_seed_all_is_deterministic(tmp_path):
    first = {p.name: p.read_bytes() for p in seed_fixtures.seed_all(tmp_path)}
    second = {p.name: p.read_bytes() for p in seed_fixtures.seed_all(tmp_path)}
    assert first == second
    assert len(first) == len(seed_fixtures.JSON_FIXTURES) + len(seed_fixtures.SSE_FIXTURES)


def test_seeded_json_matches_factory():
    on_disk = seed_fixtures.load_json("openai_chat")
    assert on_disk == f.openai_chat()


def test_seeded_sse_matches_factory():
    assert seed_fixtures.load_sse("openai_stream") == f.openai_sse("Hel", "lo!")
