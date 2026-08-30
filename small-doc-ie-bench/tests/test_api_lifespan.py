from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
from docie_bench.storage import db


def test_lifespan_shutdown_disposes_the_engine(tmp_path: Path) -> None:
    """The app's lifespan must actually call dispose_engine() on shutdown --
    proving the shutdown path is real, not wired-but-dead code (#322)."""
    db.init_engine(f"sqlite:///{tmp_path / 'lifespan.db'}")
    assert db.get_session_factory() is not None
    try:
        with TestClient(api.app) as client:
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert db.get_session_factory() is not None
        # Exiting the TestClient context triggers the ASGI lifespan shutdown
        # event, which must call dispose_engine() and tear down the factory.
        assert db.get_session_factory() is None
    finally:
        db.dispose_engine()


def test_lifespan_shutdown_calls_dispose_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spy directly on storage.db.dispose_engine to confirm the lifespan
    context manager invokes it after yield, not merely that its effects
    happen to line up."""
    db.init_engine(f"sqlite:///{tmp_path / 'lifespan_spy.db'}")
    calls = []
    original = db.dispose_engine

    def spy() -> None:
        calls.append(True)
        original()

    monkeypatch.setattr(api, "dispose_engine", spy)
    try:
        with TestClient(api.app) as client:
            assert calls == []
            client.get("/healthz")
        assert calls == [True]
    finally:
        db.dispose_engine()
