"""Dates MCP server: current_date() reproducibility override (#351)."""

from __future__ import annotations

import datetime as dt

import pytest

from docie_bench.mcp_servers import dates


def test_current_date_reads_wall_clock_when_env_is_unset(monkeypatch) -> None:
    monkeypatch.delenv(dates.FIXED_TODAY_ENV, raising=False)
    monkeypatch.setattr(dates.dt, "date", _FixedDate)

    assert dates.current_date() == "2020-01-15"


def test_current_date_returns_the_fixed_date_when_env_is_set(monkeypatch) -> None:
    monkeypatch.setenv(dates.FIXED_TODAY_ENV, "2026-08-31")

    assert dates.current_date() == "2026-08-31"


def test_current_date_raises_on_an_invalid_fixed_date(monkeypatch) -> None:
    monkeypatch.setenv(dates.FIXED_TODAY_ENV, "not-a-date")

    with pytest.raises(ValueError, match=dates.FIXED_TODAY_ENV):
        dates.current_date()


class _FixedDate(dt.date):
    @classmethod
    def today(cls) -> _FixedDate:
        return cls(2020, 1, 15)
