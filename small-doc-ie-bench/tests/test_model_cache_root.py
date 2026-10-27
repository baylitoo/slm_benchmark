"""The model cache has its own root, separate from durable node state (#595).

`serving_home` holds what a node cannot lose: `node_id` is its identity,
`deployments.json` is what it is running, the applied-state file is how far it
has converged. Weights are the opposite — once a shared repository can supply
them, the local copy is a cache that should be clearable and rebuildable.

They were the same volume, so "clear the cache" and "destroy this node's
identity" were the same command. #595 lists this split as a prerequisite for
the remote backend rather than a follow-up, which is why it ships on its own.

The property that matters most here is the boring one: the default must be
byte-identical to the old path, or an upgraded node cannot find weights it
already has.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from docie_bench.serving.paths import (
    MODEL_STORE_DIR_ENV_VAR,
    model_store_root,
    serving_home,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.delenv(MODEL_STORE_DIR_ENV_VAR, raising=False)


def test_the_default_is_exactly_where_blobs_already_live() -> None:
    """The upgrade property. A node that already holds weights must find them
    without anything being moved, so the default cannot be *equivalent* to the
    old path -- it has to be the same path."""
    assert model_store_root() == serving_home() / "models"


def test_an_override_moves_only_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: the cache can live on a large cheap disk while
    identity and desired state stay on the durable one."""
    state_root = serving_home()
    monkeypatch.setenv(MODEL_STORE_DIR_ENV_VAR, "/mnt/bigdisk/models")

    assert model_store_root() == Path("/mnt/bigdisk/models")
    assert serving_home() == state_root


def test_clearing_the_cache_cannot_touch_node_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stated as the invariant an operator relies on: with the override set,
    the cache root contains none of the files a node cannot lose.

    Without the split this test is impossible to write -- the cache root IS
    the state root, so deleting it takes `node_id` with it.
    """
    from docie_bench.serving.applied_state import applied_state_path
    from docie_bench.serving.node_identity import node_id_path

    cache = tmp_path / "cache"
    monkeypatch.setenv(MODEL_STORE_DIR_ENV_VAR, str(cache))

    for durable in (node_id_path(), applied_state_path(), serving_home() / "deployments.json"):
        assert cache not in durable.parents, f"{durable} would be destroyed with the cache"


def test_without_the_override_they_still_overlap() -> None:
    """The default is honest about what it is: on a single-machine deployment
    the cache still lives under the state root, and clearing it there is still
    dangerous. The override is what makes the separation real -- this records
    that the default has NOT silently changed behaviour.
    """
    assert serving_home() in model_store_root().parents


def test_an_empty_override_is_ignored_rather_than_obeyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset variable in a compose file arrives as "". Treating that as a
    path would put the store at the process's working directory."""
    monkeypatch.setenv(MODEL_STORE_DIR_ENV_VAR, "   ")

    assert model_store_root() == serving_home() / "models"


def test_the_override_is_read_per_call_not_cached() -> None:
    """Matching `serving_home`'s own contract: tests and the CLI set this per
    process after import."""
    first = model_store_root()
    os.environ[MODEL_STORE_DIR_ENV_VAR] = "/mnt/elsewhere"  # not a temp path
    try:
        assert model_store_root() != first
    finally:
        del os.environ[MODEL_STORE_DIR_ENV_VAR]


def test_every_store_construction_goes_through_the_helper() -> None:
    """Eight call sites computed `serving_home() / "models"` by hand. One of
    them drifting would split a node's blobs across two directories, which is
    the same failure `serving_home` itself was introduced to prevent.
    """
    import inspect

    from docie_bench.inngest import functions, serving_api
    from docie_bench.serving import cli, control_plane, profile_resolver, resources

    for module in (functions, serving_api, cli, control_plane, profile_resolver, resources):
        source = inspect.getsource(module)
        assert 'serving_home() / "models"' not in source, module.__name__
