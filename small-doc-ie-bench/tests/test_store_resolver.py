"""``store:<name>`` resolution: a deployed store model becomes an extraction target.

The resolver turns a catalog entry + live placement into a ModelProfile, with
the load-bearing style precedence: probed negotiated_style > purpose-built
family style (nuextract3/nuextract_v1 templates are delivered out-of-band and
must never be swapped for an engine default) > per-engine generic default.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.model_store import StoreEntry
from docie_bench.serving.placement_resolver import (
    ENGINE_DEFAULT_STYLE,
    PlacementNotFoundError,
    PlacementNotReadyError,
    endpoint_is_loopback,
    resolve_store_profile,
)

_ENDPOINT = "http://127.0.0.1:8088/v1"


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


def _seed(name: str, family: str) -> None:
    ModelCatalog().upsert(
        StoreEntry(name=name, family=family, model_path=Path(f"/models/{name}/model.gguf"))
    )


def _place(
    name: str,
    *,
    engine: str = "llama-server",
    state: str = "ready",
    negotiated_style: str | None = None,
    endpoint: str = _ENDPOINT,
) -> None:
    ModelCatalog().record_placement(
        name,
        model_name=name,
        engine=engine,
        endpoint=endpoint,
        state=state,
        negotiated_style=negotiated_style,
    )


def test_resolve_store_llama_server_default_style(_sqlite_catalog: None) -> None:
    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", engine="llama-server")

    profile = resolve_store_profile("qwen2.5-1.5b")
    assert profile.name == "store:qwen2.5-1.5b"
    assert profile.model == "qwen2.5-1.5b"  # deployment name == llama-server --alias
    assert profile.base_url == _ENDPOINT
    assert profile.api_key == "local-not-used"
    assert profile.response_format_style == "openai_json_schema"
    assert ENGINE_DEFAULT_STYLE["llama-server"] == "openai_json_schema"


def test_resolve_store_records_activity(_sqlite_catalog: None) -> None:
    """Every live resolution bumps model_activity — the signal a future
    autoscale-up decision would read (see catalog.ModelActivity)."""
    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", engine="llama-server")

    assert ModelCatalog().get_activity("qwen2.5-1.5b") is None

    resolve_store_profile("qwen2.5-1.5b")
    activity = ModelCatalog().get_activity("qwen2.5-1.5b")
    assert activity is not None
    assert activity["window_count"] == 1

    resolve_store_profile("qwen2.5-1.5b")
    assert ModelCatalog().get_activity("qwen2.5-1.5b")["window_count"] == 2


def test_resolve_store_survives_activity_write_failure(
    _sqlite_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Activity tracking is best-effort — a DB hiccup on the bump must never
    break a resolution that otherwise succeeded cleanly."""
    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", engine="llama-server")

    def boom(self: ModelCatalog, model_name: str) -> None:
        raise RuntimeError("db hiccup")

    monkeypatch.setattr(ModelCatalog, "record_activity", boom)

    profile = resolve_store_profile("qwen2.5-1.5b")
    assert profile.name == "store:qwen2.5-1.5b"


def test_resolve_store_ollama_default_style(_sqlite_catalog: None) -> None:
    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", engine="ollama")

    profile = resolve_store_profile("qwen2.5-1.5b")
    # Ollama's json_schema path returns empty content on several models.
    assert profile.response_format_style == "json_object"


def test_resolve_store_family_contract_wins(_sqlite_catalog: None) -> None:
    """A purpose-built family must keep its declared style: swapping in the
    engine default (openai_json_schema) would silently disable NuExtract3's
    out-of-band template and vision extraction."""
    _seed("nuextract3", family="nuextract3")
    _place("nuextract3", engine="llama-server")

    profile = resolve_store_profile("nuextract3")
    assert profile.response_format_style == "nuextract3"
    assert profile.vision is True
    assert profile.prompt_profile == "nuextract3"
    assert profile.temperature == 0.2  # family default_temperature
    assert profile.stop_sequences == ()


def test_resolve_store_nuextract_v1_family_style_and_stops(_sqlite_catalog: None) -> None:
    _seed("nuextract-v1", family="nuextract_v1")
    _place("nuextract-v1", engine="ollama")

    profile = resolve_store_profile("nuextract-v1")
    # In-prompt template family: style "none" is binding, not ollama's json_object.
    assert profile.response_format_style == "none"
    assert profile.prompt_profile == "nuextract_v1"
    assert profile.stop_sequences == ("<|end-output|>",)


def test_resolve_store_negotiated_style_overrides(_sqlite_catalog: None) -> None:
    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", engine="llama-server", negotiated_style="json_object")

    profile = resolve_store_profile("qwen2.5-1.5b")
    assert profile.response_format_style == "json_object"


def test_resolve_store_no_placement_raises_actionable(_sqlite_catalog: None) -> None:
    _seed("qwen2.5-1.5b", family="openai_chat")

    with pytest.raises(PlacementNotFoundError) as excinfo:
        resolve_store_profile("qwen2.5-1.5b")
    # The message must name the deploy step, not just say "not found".
    assert "Deploy it first" in str(excinfo.value)
    assert "docie up qwen2.5-1.5b" in str(excinfo.value)


def test_resolve_store_unknown_model_raises_seed_hint(_sqlite_catalog: None) -> None:
    with pytest.raises(PlacementNotFoundError) as excinfo:
        resolve_store_profile("never-seeded")
    assert "seed it first" in str(excinfo.value)


def test_resolve_store_not_ready_raises(_sqlite_catalog: None) -> None:
    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", state="starting")

    with pytest.raises(PlacementNotReadyError) as excinfo:
        resolve_store_profile("qwen2.5-1.5b")
    assert "starting" in str(excinfo.value)


async def test_api_resolve_profile_store_maps_error_to_http(_sqlite_catalog: None) -> None:
    from docie_bench.api import resolve_profile

    with pytest.raises(HTTPException) as missing:
        await resolve_profile("store:never-seeded")
    assert missing.value.status_code == 404

    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", state="starting")
    # A placement row exists (not live yet) -- trigger_deployment_load tries
    # to fire a reload, but this test's fixture has no deployments.json
    # record / reachable Inngest client for it to fire against, so it
    # degrades to None and the original 409 stands (see the next test file
    # for the case where the trigger genuinely fires).
    with pytest.raises(HTTPException) as not_ready:
        await resolve_profile("store:qwen2.5-1.5b")
    assert not_ready.value.status_code == 409


async def test_api_resolve_profile_triggers_load_and_returns_202(
    _sqlite_catalog: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The core of the "available models, load on first request" behavior: a
    # store model that isn't live yet gets a load/deploy fired for it and an
    # honest 202 "starting" response, instead of a bare 404/409. This test
    # stubs trigger_deployment_load itself (its own real logic -- catalog
    # lookup, never-deployed vs evicted branching, Inngest event shape -- is
    # exercised by test_serving_control_plane.py); here we're proving
    # resolve_profile actually calls it and turns a hit into a 202.
    from docie_bench import api

    async def fake_trigger(name: str) -> tuple[str, float] | None:
        assert name == "qwen2.5-1.5b"
        return name, 42.0

    monkeypatch.setattr(api, "trigger_deployment_load", fake_trigger)

    _seed("qwen2.5-1.5b", family="openai_chat")  # catalog entry, no placement at all
    with pytest.raises(HTTPException) as excinfo:
        await api.resolve_profile("store:qwen2.5-1.5b")
    assert excinfo.value.status_code == 202
    detail = excinfo.value.detail
    assert detail["status"] == "loading"
    assert detail["deployment"] == "qwen2.5-1.5b"
    assert detail["eta_seconds"] == 42.0
    assert "42s" in detail["message"]


async def test_api_resolve_profile_unknown_catalog_name_still_404s(
    _sqlite_catalog: None,
) -> None:
    # trigger_deployment_load itself (not mocked here) must decline to fire
    # anything for a name that was never seeded -- there's genuinely nothing
    # to load, so the original 404 has to survive.
    from docie_bench.api import resolve_profile

    with pytest.raises(HTTPException) as excinfo:
        await resolve_profile("store:genuinely-never-seeded")
    assert excinfo.value.status_code == 404


# ── PR-C: load-balancing across scaled replicas ─────────────────────────────


def _place_replica(
    record_name: str,
    model_name: str,
    *,
    endpoint: str,
    state: str = "ready",
    engine: str = "llama-server",
) -> None:
    """A placement row whose RECORD name differs from its MODEL name — i.e. a
    scaled replica (control_plane names replicas <base>-2/-3 but keeps
    model_name=<base>)."""
    ModelCatalog().record_placement(
        record_name,
        model_name=model_name,
        engine=engine,
        endpoint=endpoint,
        state=state,
    )


def test_resolve_load_balances_across_ready_replicas(_sqlite_catalog: None) -> None:
    _seed("qwen", family="openai_chat")
    _place_replica("qwen", "qwen", endpoint="http://worker:8091/v1")
    _place_replica("qwen-2", "qwen", endpoint="http://worker:8092/v1")
    _place_replica("qwen-3", "qwen", endpoint="http://worker:8093/v1")

    # A deterministic chooser (sorted by name) reaches every replica across the
    # three indices — proving all live replicas are candidates.
    seen = set()
    for idx in (0, 1, 2):
        profile = resolve_store_profile(
            "qwen",
            chooser=lambda xs, _i=idx: sorted(xs, key=lambda p: p["name"])[_i],
        )
        # Every replica answers to the same routable model id (the base alias).
        assert profile.model == "qwen"
        seen.add(profile.base_url)
    assert seen == {
        "http://worker:8091/v1",
        "http://worker:8092/v1",
        "http://worker:8093/v1",
    }


def test_resolve_only_balances_over_ready_replicas(_sqlite_catalog: None) -> None:
    _seed("qwen", family="openai_chat")
    _place_replica("qwen", "qwen", endpoint="http://worker:8091/v1", state="starting")
    _place_replica("qwen-2", "qwen", endpoint="http://worker:8092/v1", state="ready")

    def chooser(candidates):  # the chooser must only ever see the ready replica
        assert [p["name"] for p in candidates] == ["qwen-2"]
        return candidates[0]

    profile = resolve_store_profile("qwen", chooser=chooser)
    assert profile.base_url == "http://worker:8092/v1"


def test_resolve_single_replica_is_unchanged(_sqlite_catalog: None) -> None:
    _seed("qwen", family="openai_chat")
    _place_replica("qwen", "qwen", endpoint="http://worker:8091/v1")

    # One live replica: the chooser is never consulted (would raise if it were).
    def boom(_candidates):
        raise AssertionError("chooser must not run for a single replica")

    profile = resolve_store_profile("qwen", chooser=boom)
    assert profile.base_url == "http://worker:8091/v1"


def test_resolve_all_replicas_not_ready_raises(_sqlite_catalog: None) -> None:
    _seed("qwen", family="openai_chat")
    _place_replica("qwen", "qwen", endpoint="http://worker:8091/v1", state="starting")
    _place_replica("qwen-2", "qwen", endpoint="http://worker:8092/v1", state="loading")

    with pytest.raises(PlacementNotReadyError):
        resolve_store_profile("qwen")


@pytest.mark.parametrize(
    ("url", "loopback"),
    [
        ("http://127.0.0.1:8088/v1", True),
        ("http://localhost:8088/v1", True),
        ("http://[::1]:8088/v1", True),
        ("http://127.1.2.3:8088/v1", True),
        ("http://0.0.0.0:8088/v1", True),  # unspecified: unreachable as recorded
        ("http://worker:8088/v1", False),
        ("http://192.168.1.20:8088/v1", False),
        ("https://models.example/v1", False),
    ],
)
def test_endpoint_is_loopback(url: str, loopback: bool) -> None:
    assert endpoint_is_loopback(url) is loopback


async def test_api_resolve_profile_rejects_worker_loopback_endpoint(_sqlite_catalog: None) -> None:
    """The placement endpoint is recorded from the WORKER's point of view; in
    the documented api/worker compose topology the API can never reach the
    worker's 127.0.0.1, so resolving it 'successfully' would burn
    timeout_seconds x retries per request. Fail fast with 501 (worker-only)."""
    from docie_bench.api import resolve_profile

    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b")  # loopback _ENDPOINT

    with pytest.raises(HTTPException) as excinfo:
        await resolve_profile("store:qwen2.5-1.5b")
    assert excinfo.value.status_code == 501
    assert "127.0.0.1:8088" in excinfo.value.detail
    # The message must point at the working alternatives.
    assert "/v1/studio/extract" in excinfo.value.detail


async def test_api_resolve_profile_accepts_advertised_endpoint(_sqlite_catalog: None) -> None:
    """A non-loopback (advertised) endpoint passes the guard untouched."""
    from docie_bench.api import resolve_profile

    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", endpoint="http://worker:8088/v1")

    profile = await resolve_profile("store:qwen2.5-1.5b")
    assert profile.base_url == "http://worker:8088/v1"


def test_inngest_resolve_profile_allows_worker_local_endpoint(_sqlite_catalog: None) -> None:
    """The worker resolves its own deployments, so loopback IS reachable there
    — the guard is API-only."""
    from docie_bench.inngest.functions import _resolve_profile

    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b")

    assert _resolve_profile(model_profile="store:qwen2.5-1.5b").base_url == _ENDPOINT


def test_inngest_resolve_profile_store_raises(_sqlite_catalog: None) -> None:
    """DELIBERATE divergence from the module's never-raises convention: an
    explicit store: ref must never silently fall back to the env default —
    that fallback IS the deploy/extraction disconnect being fixed."""
    from docie_bench.inngest.functions import _resolve_profile

    with pytest.raises(PlacementNotFoundError):
        _resolve_profile(model_profile="store:never-seeded")


def test_inngest_resolve_profile_unknown_plain_name_refuses(_sqlite_catalog: None) -> None:
    """Post-#63 the shared resolver REFUSES unknown explicit selectors (the old
    silent env-default fallback was the closed backdoor), for plain names and
    store: refs alike."""
    from docie_bench.inngest.functions import _resolve_profile
    from docie_bench.serving.profile_resolver import ProfileResolutionError

    with pytest.raises(ProfileResolutionError):
        _resolve_profile(model_profile="no-such-profile")
