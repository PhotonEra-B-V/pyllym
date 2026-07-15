from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import pyllym
from pyllym.errors import ConfigurationError
from pyllym.toolset import tool_from_path


def _write(tmp_path: Path, body: str) -> Path:
    file = tmp_path / "toolset.toml"
    file.write_text(textwrap.dedent(body), encoding="utf-8")
    return file


# --- tool_from_path: schema inference & execution ---------------------------


@pytest.mark.asyncio
async def test_tool_from_stdlib_callable_infers_and_runs():
    tool = tool_from_path("statistics.mean")
    assert tool.name == "statistics_mean"
    # statistics.mean(data) -> the single param is inferred from the signature.
    schema = tool.params_schema
    assert schema is not None
    assert "data" in schema["properties"]
    assert await tool.call({"data": [1, 2, 3, 4]}) == 2.5


@pytest.mark.asyncio
async def test_annotation_maps_list_param_to_array():
    def total(values: list[float]) -> float:
        return sum(values)

    # Register the local callable via its module path.
    globals()["_total_for_test"] = total
    tool = tool_from_path(f"{__name__}._total_for_test")
    prop = tool.params_schema["properties"]["values"]
    assert prop["type"] == "array"
    assert prop["items"] == {"type": "number"}
    assert await tool.call({"values": [1.5, 2.5]}) == 4.0


def test_name_and_description_overrides():
    tool = tool_from_path("statistics.median", name="mid", description="the middle value")
    assert tool.name == "mid"
    assert tool.description == "the middle value"


def test_param_override_wins_over_inference():
    tool = tool_from_path(
        "statistics.mean",
        params={"data": {"type": "array", "items": "number", "description": "numbers"}},
    )
    prop = tool.params_schema["properties"]["data"]
    assert prop == {"type": "array", "items": {"type": "number"}, "description": "numbers"}


# --- load_toolset: file parsing & guarantees --------------------------------


def test_load_toolset_reads_entries(tmp_path: Path):
    file = _write(
        tmp_path,
        """
        [[tools]]
        path = "statistics.mean"
        description = "average"

        [[tools]]
        path = "statistics.median"
        name = "mid"
        """,
    )
    tools = pyllym.load_toolset(file)
    assert [t.name for t in tools] == ["statistics_mean", "mid"]


def test_missing_file_raises_configuration_error(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="not found"):
        pyllym.load_toolset(tmp_path / "nope.toml")


def test_entry_without_path_raises(tmp_path: Path):
    file = _write(tmp_path, '[[tools]]\ndescription = "no path"\n')
    with pytest.raises(ConfigurationError, match="missing 'path'"):
        pyllym.load_toolset(file)


def test_duplicate_names_raise(tmp_path: Path):
    file = _write(
        tmp_path,
        """
        [[tools]]
        path = "statistics.mean"

        [[tools]]
        path = "statistics.mean"
        """,
    )
    with pytest.raises(ConfigurationError, match="duplicate tool name"):
        pyllym.load_toolset(file)


# --- allowlist / safety boundary --------------------------------------------


def test_unimportable_path_raises():
    with pytest.raises(ConfigurationError):
        tool_from_path("nonexistent_pkg_xyz.func")


def test_missing_optional_package_gives_install_hint():
    # torch is not a pyllym dependency; a toolset naming it must fail with an
    # actionable "not installed / pip install" message, not a generic error.
    from pyllym.toolset import MissingToolPackageError

    with pytest.raises(MissingToolPackageError, match="not installed") as exc:
        tool_from_path("torch.mean")
    assert exc.value.package == "torch"
    # Specific type so callers can catch it, still a ConfigurationError.
    assert isinstance(exc.value, ConfigurationError)


def test_load_toolset_skips_missing_package_by_default(tmp_path: Path):
    file = _write(
        tmp_path,
        """
        [[tools]]
        path = "statistics.mean"

        [[tools]]
        path = "torch.mean"
        """,
    )
    # torch absent -> that entry is skipped, the stdlib one still loads.
    tools = pyllym.load_toolset(file)
    assert [t.name for t in tools] == ["statistics_mean"]


def test_load_toolset_strict_reraises_missing_package(tmp_path: Path):
    from pyllym.toolset import MissingToolPackageError

    file = _write(tmp_path, '[[tools]]\npath = "torch.mean"\n')
    with pytest.raises(MissingToolPackageError):
        pyllym.load_toolset(file, skip_missing=False)


@pytest.mark.asyncio
async def test_installed_optional_package_is_usable(monkeypatch):
    # Simulate an installed optional lib (e.g. torch) by registering a stub
    # module; the toolset must then import and call it with no special-casing.
    import sys
    import types as _types

    stub = _types.ModuleType("fake_torch_lib")
    stub.tensor_mean = lambda values: sum(values) / len(values)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_torch_lib", stub)

    tool = tool_from_path("fake_torch_lib.tensor_mean")
    assert await tool.call({"values": [2, 4, 6]}) == 4.0


def test_non_callable_path_raises():
    # statistics.pi does not exist; use a real non-callable attribute instead.
    with pytest.raises(ConfigurationError, match="callable"):
        tool_from_path("statistics.__doc__")


def test_bare_name_without_dot_raises():
    with pytest.raises(ConfigurationError, match="dotted"):
        tool_from_path("len")
