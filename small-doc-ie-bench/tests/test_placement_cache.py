"""`store:` resolution survives an unreachable catalog (#532, phase 1 of #527).

The node side already honours #527's hard requirement -- the reconciler repairs
and saves `deployments.json` with no database and just skips the publish, so
models keep answering. The central resolver did not: it read the catalog on
every request, so with Postgres down a healthy, loaded, reachable model became
unroutable. These tests pin the four acceptance criteria on #532.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.model_store import StoreEntry
from docie_bench.serving.placement_resolver import (
    PlacementNotFoundError,
    resolve_store_profile,
)

MODEL = "lfm2.5-350m"


@pytest.fixture
def sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


@pytest.fixture(autouse=True)
def _empty_cache() -> Iterator[None]:
    from docie_bench.serving import placement_resolver

    placement_resolver._RESOLUTION_CACHE.clear()
    placement_resolver._ROUND_ROBIN_COUNTERS.clear()
    try:
        yield
    finally:
        placement_resolver._RESOLUTION_CACHE.clear()
        placement_resolver._ROUND_ROBIN_COUNTERS.clear()


def _seed(name: str = MODEL, *, port: int = 8091) -> None:
    ModelCatalog().upsert(
        StoreEntry(
            name=name,
            family="openai_chat",
            model_path=Path(f"/models/{name}/model.gguf"),
        )
    )
    ModelCatalog().record_placement(
        name,
        model_name=name,
        engine="llama-server",
        endpoint=f"http://worker:{port}/v1",
        state="ready",
    )


def test_a_resolved_model_still_routes_once_the_catalog_is_gone(
    sqlite_catalog: None,
) -> None:
    """Acceptance: resolve, stop Postgres, resolve again -- still routes."""
    _seed()
    before = resolve_store_profile(MODEL)

    db.dispose_engine()  # Postgres goes away

    after = resolve_store_profile(MODEL)

    assert after.base_url == before.base_url
    assert after.model == before.model


def test_a_model_never_resolved_fails_by_naming_the_outage(sqlite_catalog: None) -> None:
    """Acceptance: an un-cached model fails honestly, and the message
    distinguishes "not in the catalog" from "catalog unreachable"."""
    _seed()
    resolve_store_profile(MODEL)  # warms only this model

    db.dispose_engine()

    with pytest.raises(PlacementNotFoundError) as caught:
        resolve_store_profile("never-seen")

    message = str(caught.value)
    assert "unreachable" in message
    assert "not been resolved recently" in message


def test_an_expired_entry_does_not_route(sqlite_catalog: None) -> None:
    """Acceptance: cached entries expire; an expired entry does not route."""
    from docie_bench.serving import placement_resolver

    _seed()
    resolve_store_profile(MODEL)
    db.dispose_engine()

    # Age the entry past the TTL rather than waiting for it.
    stamped, entry, placements = placement_resolver._RESOLUTION_CACHE[MODEL]
    ttl = placement_resolver.get_settings().serving_placement_cache_ttl_seconds
    placement_resolver._RESOLUTION_CACHE[MODEL] = (stamped - ttl - 1, entry, placements)

    with pytest.raises(PlacementNotFoundError, match="unreachable"):
        resolve_store_profile(MODEL)


def test_the_absence_of_a_model_is_never_cached(sqlite_catalog: None) -> None:
    """Acceptance: a model unknown during an outage must not be remembered as
    unknown. Deploying it later must resolve on the first try."""
    from docie_bench.serving import placement_resolver

    with pytest.raises(PlacementNotFoundError):
        resolve_store_profile(MODEL)  # not seeded yet

    assert MODEL not in placement_resolver._RESOLUTION_CACHE

    _seed()
    assert resolve_store_profile(MODEL).base_url == "http://worker:8091/v1"


def test_a_successful_read_always_wins_over_the_cache(sqlite_catalog: None) -> None:
    """The cache is advisory. A redeploy to a new port must take effect at
    once, not after the TTL."""
    _seed(port=8091)
    assert resolve_store_profile(MODEL).base_url == "http://worker:8091/v1"

    ModelCatalog().record_placement(
        MODEL,
        model_name=MODEL,
        engine="llama-server",
        endpoint="http://worker:8099/v1",
        state="ready",
    )

    assert resolve_store_profile(MODEL).base_url == "http://worker:8099/v1"


def test_disabling_the_cache_restores_the_previous_behaviour(
    sqlite_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TTL 0 is the documented escape hatch: refuse rather than serve a
    last-known-good placement."""
    from docie_bench.settings import get_settings

    _seed()
    resolve_store_profile(MODEL)

    monkeypatch.setenv("DOCIE_SERVING_PLACEMENT_CACHE_TTL", "0")
    get_settings.cache_clear()
    db.dispose_engine()
    try:
        with pytest.raises(PlacementNotFoundError):
            resolve_store_profile(MODEL)
    finally:
        get_settings.cache_clear()
