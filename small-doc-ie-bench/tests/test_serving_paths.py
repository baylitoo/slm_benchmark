from pathlib import Path

import pytest

from docie_bench.agents.registry import default_agents_path
from docie_bench.serving.paths import serving_home
from docie_bench.serving.recency import recency_dir


def test_the_environment_names_the_serving_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    assert serving_home() == tmp_path


def test_without_the_environment_the_home_is_under_the_user_share_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCIE_SERVING_HOME", raising=False)
    assert serving_home() == Path.home() / ".local" / "share" / "docie-bench" / "serving"


def test_the_environment_is_read_on_every_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "first"))
    assert serving_home() == tmp_path / "first"
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "second"))
    assert serving_home() == tmp_path / "second"


def test_every_consumer_resolves_under_the_same_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    assert default_agents_path() == tmp_path / "agents.json"
    assert recency_dir() == tmp_path / "recency"
