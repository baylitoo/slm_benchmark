"""Unloading a deployment that another node owns (#536).

The last of #536's mutations is "remote deploy, load, unload, pin, repair
against a named node". Unload is the one that works with the machinery that
exists, and the reason is worth stating because it is also why the other three
do not: a stop is the single intent ``plan_convergence`` already derives
correctly for a deployment that is present on its node (``to_stop``). Load
needs a plan state for "present but not running", delete needs an explicitly
declared removal, and pin/repair are not expressible in a desired spec at all
-- see #590.

Lifecycle EVENTS cannot be aimed: they are delivered to whichever node Inngest
picks. Desired state is the one channel that is node-scoped, which is why a
remote action has to go through it and why the response carries no channel.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.model_store import StoreEntry
from docie_bench.settings import get_settings

MODEL = "lfm2.5-350m"
RECORD = "lfm2.5-350m-ab12"
OWNER = "node-b"
GIB = 1024**3


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A catalog that knows a deployment THIS process has no local record for."""
    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", "node-a")
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    catalog = ModelCatalog()
    catalog.upsert(
        StoreEntry(
            name=MODEL,
            family="openai_chat",
            model_path=Path(f"/models/{MODEL}/model.gguf"),
        ),
        size_bytes=GIB // 2,
    )
    catalog.publish_node_snapshot(
        node_id=OWNER,
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
        store_models=[MODEL],
    )
    catalog.record_placement(
        RECORD,
        model_name=MODEL,
        engine="llama-server",
        endpoint=f"http://{OWNER}:8088/v1",
        state="ready",
        node_id=OWNER,
    )
    try:
        yield
    finally:
        db.dispose_engine()
        get_settings.cache_clear()


def _unload(name: str = RECORD) -> dict[str, Any]:
    from docie_bench.inngest.serving_api import unload_deployment

    return asyncio.run(unload_deployment(name, tenant=None))  # type: ignore[arg-type]


def _desired_rows(node_id: str = OWNER) -> list[dict[str, Any]]:
    rows = ModelCatalog().desired_snapshot(node_id)["deployments"]
    return list(rows) if isinstance(rows, list) else []


def test_a_remote_unload_is_recorded_for_the_owning_node(remote: None) -> None:
    result = _unload()

    rows = _desired_rows()
    assert len(rows) == 1
    assert rows[0]["name"] == RECORD
    assert rows[0]["desired_state"] == "stopped"
    assert OWNER in str(result["detail"])


def test_a_remote_unload_returns_no_channel_to_wait_on(remote: None) -> None:
    """The owning node converges on its own cycle, so there is no progress
    stream. A caller that waited on one would wait forever."""
    result = _unload()

    assert result["channel"] is None
    assert result["event_ids"] == []


def test_the_recorded_spec_names_the_model_so_a_later_start_can_use_it(
    remote: None,
) -> None:
    """A stop does not read the spec, but a later start does -- so the row has
    to carry something a reconciler could act on."""
    _unload()

    assert _desired_rows()[0]["spec"]["model"] == MODEL


def test_an_existing_spec_is_preserved_rather_than_replaced(remote: None) -> None:
    """The deployment was targeted at this node with a chosen context length.
    Overwriting the spec with a thin one would silently drop that the next time
    it started."""
    ModelCatalog().set_desired_deployment(
        OWNER,
        RECORD,
        desired_state="running",
        spec={"model": MODEL, "context_length": 32768, "port": 8099},
    )

    _unload()

    spec = _desired_rows()[0]["spec"]
    assert spec["context_length"] == 32768
    assert spec["port"] == 8099
    assert _desired_rows()[0]["desired_state"] == "stopped"


def test_a_spec_without_a_model_gains_one_instead_of_being_left_unstartable(
    remote: None,
) -> None:
    """Preserving the row blindly is not enough. A spec that carries the
    operator's context length but no model would leave a later start with
    nothing to launch, so the model is merged in rather than the dict
    replaced."""
    ModelCatalog().set_desired_deployment(
        OWNER, RECORD, desired_state="running", spec={"context_length": 32768}
    )

    _unload()

    spec = _desired_rows()[0]["spec"]
    assert spec["model"] == MODEL
    assert spec["context_length"] == 32768


def test_unloading_twice_changes_nothing_the_second_time(remote: None) -> None:
    """The local unload is documented as idempotent; the remote one says the
    same rather than bumping a generation for no change."""
    _unload()
    generation = _desired_rows()[0]["generation"]

    second = _unload()

    assert _desired_rows()[0]["generation"] == generation
    assert "already recorded as stopped" in str(second["detail"])


def test_a_silent_owner_is_reported_rather_than_refused(remote: None) -> None:
    """The intent is durable and applies when the node returns, so recording it
    is right -- but saying nothing would let an operator read "unload
    requested" as "unloaded"."""
    with db.session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, OWNER)
        assert row is not None
        row.updated_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)

    result = _unload()

    assert "stopped reporting" in str(result["detail"])
    assert _desired_rows()[0]["desired_state"] == "stopped"


def test_an_unknown_deployment_is_still_a_404(remote: None) -> None:
    """The remote route must not turn a typo into a desired row nothing will
    ever read."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _unload("not-a-deployment")

    assert excinfo.value.status_code == 404
    assert _desired_rows() == []


def test_the_other_verbs_still_refuse_and_say_which_one_works(
    remote: None,
) -> None:
    """Load, delete, pin and repair need plan states that do not exist (#590),
    so they refuse rather than writing desired state that would be ignored --
    and the refusal names the verb that does work."""
    from fastapi import HTTPException

    from docie_bench.inngest.serving_api import delete_deployment, load_deployment

    for endpoint in (load_deployment, delete_deployment):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(endpoint(RECORD, tenant=None))  # type: ignore[arg-type]
        assert excinfo.value.status_code == 409
        assert "Unload works remotely" in str(excinfo.value.detail)

    assert _desired_rows() == []
