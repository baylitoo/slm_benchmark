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
) -> None:
    ModelCatalog().record_placement(
        name,
        model_name=name,
        engine=engine,
        endpoint=_ENDPOINT,
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


def test_api_resolve_profile_store_maps_error_to_http(_sqlite_catalog: None) -> None:
    from docie_bench.api import resolve_profile

    with pytest.raises(HTTPException) as missing:
        resolve_profile("store:never-seeded")
    assert missing.value.status_code == 404

    _seed("qwen2.5-1.5b", family="openai_chat")
    _place("qwen2.5-1.5b", state="starting")
    with pytest.raises(HTTPException) as not_ready:
        resolve_profile("store:qwen2.5-1.5b")
    assert not_ready.value.status_code == 409


def test_inngest_resolve_profile_store_raises(_sqlite_catalog: None) -> None:
    """DELIBERATE divergence from the module's never-raises convention: an
    explicit store: ref must never silently fall back to the env default —
    that fallback IS the deploy/extraction disconnect being fixed."""
    from docie_bench.inngest.functions import _resolve_profile

    with pytest.raises(PlacementNotFoundError):
        _resolve_profile("store:never-seeded")


def test_inngest_resolve_profile_plain_names_still_default(_sqlite_catalog: None) -> None:
    from docie_bench.inngest.functions import _resolve_profile

    profile = _resolve_profile("no-such-profile")  # unchanged: falls back, no raise
    assert profile.name == "no-such-profile"
