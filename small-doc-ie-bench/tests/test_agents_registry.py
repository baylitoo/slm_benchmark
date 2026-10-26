"""Where the agent registry lives (#562).

The dependency map (#528) listed `agents.json` as one of the `serving_home()`
readers, and #562 asked whether agents should become catalog-backed or be
accepted as per-node config. Neither: an agent is not node state at all.
Nothing outside the `agents` package reads the registry and the only caller is
the api router, so a serving node has no use for the file.

It sat under `DOCIE_SERVING_HOME` because that was the one shared volume, not
because the data is node-scoped. These tests pin that the location is now
named directly, and that the old default is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.agents.registry import default_agents_path


def test_the_agents_path_defaults_to_the_serving_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged for an existing deployment: the historical location."""
    monkeypatch.delenv("DOCIE_AGENTS_PATH", raising=False)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))

    assert default_agents_path() == tmp_path / "serving" / "agents.json"


def test_the_agents_path_can_be_set_independently_of_the_serving_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fleet must be able to place central config without the serving home --
    a per-node concept -- deciding where fleet-wide agent definitions live."""
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.setenv("DOCIE_AGENTS_PATH", str(tmp_path / "central" / "agents.json"))

    assert default_agents_path() == tmp_path / "central" / "agents.json"


def test_an_empty_override_falls_back_to_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset-but-present env var is how compose passes "no value"; it must
    not resolve the registry to the current directory."""
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.setenv("DOCIE_AGENTS_PATH", "   ")

    assert default_agents_path() == tmp_path / "serving" / "agents.json"
