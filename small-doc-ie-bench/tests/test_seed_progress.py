"""Pollable seed-progress sidecar: write / read / clear / path containment."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from docie_bench.inngest.serving_api import seed_progress
from docie_bench.serving import seed_progress as sp


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    return tmp_path


def test_write_then_read_roundtrips(home: Path) -> None:
    payload = {"stage": "download", "percent": 42.5, "received_bytes": 10, "total_bytes": 24}
    sp.write_progress("seed:abc123", payload)
    assert sp.read_progress("seed:abc123") == payload


def test_read_absent_is_none(home: Path) -> None:
    assert sp.read_progress("seed:never-written") is None


def test_clear_removes_sidecar(home: Path) -> None:
    sp.write_progress("seed:x", {"percent": 99})
    sp.clear_progress("seed:x")
    assert sp.read_progress("seed:x") is None
    # Idempotent — clearing an absent one is a no-op.
    sp.clear_progress("seed:x")


def test_channel_is_slugified_not_traversed(home: Path) -> None:
    # A channel with path separators must not escape the progress dir.
    sp.write_progress("seed:../../etc/passwd", {"percent": 1})
    # It still round-trips under a slugified, contained name...
    assert sp.read_progress("seed:../../etc/passwd") == {"percent": 1}
    # ...and every sidecar lives inside <home>/seed-progress.
    written = list((home / "seed-progress").glob("*.json"))
    assert written
    for path in written:
        assert (home / "seed-progress").resolve() in path.resolve().parents


def test_endpoint_returns_progress(home: Path) -> None:
    sp.write_progress("seed:live", {"percent": 33.0, "stage": "download"})
    result = asyncio.run(seed_progress(channel="seed:live"))
    assert result["channel"] == "seed:live"
    assert result["progress"]["percent"] == 33.0


def test_endpoint_null_when_absent(home: Path) -> None:
    assert asyncio.run(seed_progress(channel="seed:none"))["progress"] is None
