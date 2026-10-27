"""The model cache has its own root, separate from durable node state (#595).

`serving_home` holds what a node cannot lose: `node_id` is its identity,
`deployments.json` is what it is running, the applied-state file is how far it
has converged. Weights are the opposite — a cache, rebuildable by seeding or
pulling again.

**What the default actually costs.** The store is a CHILD of the state root
(`<serving_home>/models`), so removing the store directory does not remove
`node_id` beside it. The cost is shared PLACEMENT: one volume holds both, so
the weights cannot live on a different disk, their growth consumes the volume
the durable state is on, and destroying or resizing that volume takes both.

So the separation is a property of a CONFIGURED cache root, not of the
default. The default deliberately keeps the old shared-volume arrangement, and
the tests below say which assertions hold in which case.

The property that matters most is the boring one: the default must be
byte-identical to the old path, or an upgraded node cannot find weights it
already has.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from docie_bench.serving.paths import (
    MODEL_STORE_DIR_ENV_VAR,
    InvalidModelStoreRootError,
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


def test_a_configured_cache_root_holds_no_durable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The invariant, and it holds only WITH a configured root.

    With one set, deleting the cache root cannot take anything a node needs to
    come back. Under the default this assertion is simply false -- the cache
    root sits inside the state root -- which is what the next test records.
    """
    from docie_bench.serving.applied_state import applied_state_path
    from docie_bench.serving.node_identity import node_id_path

    cache = tmp_path / "cache"
    monkeypatch.setenv(MODEL_STORE_DIR_ENV_VAR, str(cache))

    for durable in (node_id_path(), applied_state_path(), serving_home() / "deployments.json"):
        assert cache not in durable.parents, f"{durable} would be destroyed with the cache"


def test_the_default_still_nests_the_cache_inside_the_state_root() -> None:
    """Recorded explicitly so the compatibility default is not mistaken for
    the separation.

    Under the default the weights are a child of the state directory. That is
    not dangerous in itself -- removing `<serving_home>/models` leaves
    `node_id` alone -- but the two share a volume, so only a configured root
    gives independent placement. This asserts the default has NOT silently
    changed.
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


@pytest.mark.parametrize("relative", ["", "..", "../.."])
def test_a_root_that_swallows_the_state_root_is_refused(
    monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    """A configured cache root is a promise that its contents are disposable.
    One that is, or contains, the state root turns "reclaim the cache" into
    "delete this node's identity and its deployment records" -- the hazard the
    split exists to remove.

    `..` is included because the check resolves before comparing; a lexical
    match alone would miss `<serving_home>/..`.
    """
    target = serving_home() if not relative else serving_home().joinpath(relative)
    monkeypatch.setenv(MODEL_STORE_DIR_ENV_VAR, str(target))

    with pytest.raises(InvalidModelStoreRootError, match="disposable|contains"):
        model_store_root()


def test_the_guard_does_not_apply_to_the_compatible_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default IS inside the state root, and every existing node already
    has that layout -- refusing it would stop them all from starting. The
    check is about a configuration an operator wrote expecting disposability.
    """
    monkeypatch.delenv(MODEL_STORE_DIR_ENV_VAR, raising=False)

    assert model_store_root() == serving_home() / "models"


def test_an_explicit_root_inside_the_state_root_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside is fine; the refusal is about ENCOMPASSING. Writing the default
    path out explicitly must behave exactly like leaving it unset."""
    monkeypatch.setenv(MODEL_STORE_DIR_ENV_VAR, str(serving_home() / "models"))

    assert model_store_root() == serving_home() / "models"


def test_an_unrelated_root_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_STORE_DIR_ENV_VAR, "/mnt/bigdisk/models")

    assert model_store_root() == Path("/mnt/bigdisk/models")


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
