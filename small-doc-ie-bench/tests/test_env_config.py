"""env_config.int_env: shared by docs_search and sql_agent, factored out once
duplicated identically across both."""

from __future__ import annotations

import pytest

from docie_bench.mcp_servers.env_config import int_env

ENV_VAR = "DOCIE_TEST_INT_ENV"


def test_int_env_falls_back_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert int_env(ENV_VAR, 42) == 42


def test_int_env_falls_back_when_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "not-an-int")
    assert int_env(ENV_VAR, 42) == 42


def test_int_env_uses_the_override_when_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "7")
    assert int_env(ENV_VAR, 42) == 7
