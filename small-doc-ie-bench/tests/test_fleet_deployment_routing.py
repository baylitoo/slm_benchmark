"""A deployment name resolves on any node, not just this one (#562).

`deployment=` and the bare-name route synthesize their profile from the
supervisor's `deployments.json`, which on the api is one node's file. So a
deployment running perfectly well on another machine resolved to nothing and
the request was REFUSED -- loudly, unlike the listing, but still wrong.

The local file keeps winning wherever it has an answer: it is fresher (saved
every reconcile cycle with no database involved) and carries launch detail a
placement row does not.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)

REMOTE = "nuextract3"
MISSING_CONFIG = "configs/__no_such_models_file__.yaml"


@pytest.fixture(autouse=True)
def _empty_resolution_cache() -> Iterator[None]:
    """#532's last-known-good cache is process-global and would otherwise let
    one test's successful resolution answer another test's outage."""
    from docie_bench.serving import placement_resolver

    placement_resolver._RESOLUTION_CACHE.clear()
    placement_resolver._ROUND_ROBIN_COUNTERS.clear()
    try:
        yield
    finally:
        placement_resolver._RESOLUTION_CACHE.clear()
        placement_resolver._ROUND_ROBIN_COUNTERS.clear()


@pytest.fixture
def sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


def _place(
    name: str,
    *,
    node_id: str,
    state: str = "ready",
    endpoint: str | None = None,
    model_name: str | None = None,
) -> None:
    ModelCatalog().record_placement(
        name,
        model_name=model_name if model_name is not None else name,
        engine="llama-server",
        endpoint=endpoint if endpoint is not None else f"http://{node_id}:8088/v1",
        state=state,
        node_id=node_id,
    )


def _resolve(**kwargs: object) -> object:
    # No local deployments at all: `deployments=[]` is the injection seam, and
    # stands for "this node's file knows nothing about it".
    return resolve_extraction_profile(
        deployments=[], models_config_path=MISSING_CONFIG, **kwargs
    )


def test_an_explicit_deployment_on_another_node_resolves(sqlite_catalog: None) -> None:
    """Acceptance: a worker resolving a deployment name finds one owned by any
    node, not only its own."""
    _place(REMOTE, node_id="node-b")

    profile = _resolve(deployment=REMOTE)

    assert profile.base_url == "http://node-b:8088/v1"
    assert profile.name == REMOTE
    assert profile.model == REMOTE


def test_a_bare_name_on_another_node_resolves(sqlite_catalog: None) -> None:
    """The `model_profile` route addresses a live deployment by name too."""
    _place(REMOTE, node_id="node-b")

    profile = _resolve(model_profile=REMOTE)

    assert profile.base_url == "http://node-b:8088/v1"


def test_the_served_id_is_the_launch_alias_not_the_record_name(
    sqlite_catalog: None,
) -> None:
    """A replica is addressed by the alias the runtime answers to, which is the
    store model -- sending the record name would 400 upstream."""
    _place(f"{REMOTE}-2", node_id="node-b", model_name=REMOTE)

    profile = _resolve(deployment=f"{REMOTE}-2")

    assert profile.name == f"{REMOTE}-2"  # the honest label
    assert profile.model == REMOTE  # what goes upstream


@pytest.mark.parametrize(
    ("state", "endpoint"),
    [("loading", "http://node-b:8088/v1"), ("ready", ""), ("failed", "")],
)
def test_a_placement_that_is_not_servable_is_not_a_fallback(
    sqlite_catalog: None, state: str, endpoint: str
) -> None:
    """The route's contract is unchanged: it must resolve to a LIVE deployment
    or refuse. A row that is not ready, or carries no endpoint, is not one."""
    _place(REMOTE, node_id="node-b", state=state, endpoint=endpoint)

    with pytest.raises(ProfileResolutionError, match="not a live"):
        _resolve(deployment=REMOTE)


def test_an_unknown_deployment_still_refuses(sqlite_catalog: None) -> None:
    with pytest.raises(ProfileResolutionError, match="not a live"):
        _resolve(deployment="never-deployed")


def test_resolution_is_unchanged_without_a_catalog() -> None:
    """No DATABASE_URL and nothing cached: no remote entries, and the refusal
    is the old one -- routing must not acquire a database dependency.

    "Nothing cached" is load-bearing: #532 deliberately keeps resolving a model
    it resolved before the catalog went away, so a previously-seen model would
    resolve here and that would be correct.
    """
    db.dispose_engine()

    with pytest.raises(ProfileResolutionError, match="not a live"):
        _resolve(deployment=REMOTE)


# --- the model name still routes after minting (#530) --------------------


def test_the_model_name_resolves_to_a_live_replica(sqlite_catalog: None) -> None:
    """Record names are minted since #530, so the model's own name is no longer
    one of them. A caller naming the MODEL must still reach a live replica --
    that is what keeps `deployment=<model>` working.
    """
    from docie_bench.serving.model_store import StoreEntry

    ModelCatalog().upsert(
        StoreEntry(
            name=REMOTE,
            family="openai_chat",
            model_path=Path(f"/models/{REMOTE}/model.gguf"),
        )
    )
    # A minted record name, with the model as its launch alias.
    _place(f"{REMOTE}-a1b2c3", node_id="node-b", model_name=REMOTE)

    profile = _resolve(deployment=REMOTE)

    assert profile.base_url == "http://node-b:8088/v1"
    assert profile.model == REMOTE


def test_a_bare_model_name_resolves_too(sqlite_catalog: None) -> None:
    from docie_bench.serving.model_store import StoreEntry

    ModelCatalog().upsert(
        StoreEntry(
            name=REMOTE,
            family="openai_chat",
            model_path=Path(f"/models/{REMOTE}/model.gguf"),
        )
    )
    _place(f"{REMOTE}-a1b2c3", node_id="node-b", model_name=REMOTE)

    assert _resolve(model_profile=REMOTE).base_url == "http://node-b:8088/v1"


def test_a_model_with_no_live_replica_still_refuses(sqlite_catalog: None) -> None:
    """The fallback must not soften the refusal: a model in the catalog with
    nothing serving is still an error, not a silent env fallback."""
    from docie_bench.serving.model_store import StoreEntry

    ModelCatalog().upsert(
        StoreEntry(
            name=REMOTE,
            family="openai_chat",
            model_path=Path(f"/models/{REMOTE}/model.gguf"),
        )
    )
    _place(f"{REMOTE}-a1b2c3", node_id="node-b", model_name=REMOTE, state="loading")

    with pytest.raises(ProfileResolutionError):
        _resolve(deployment=REMOTE)
