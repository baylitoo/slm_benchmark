"""The one place that answers "where does serving state live?".

``DOCIE_SERVING_HOME`` names the volume that the api, serving and worker
containers all mount: the deployment registry, the model store, ``agents.json``,
the resource sidecars and the seed-progress files are all under it. Eight call
sites used to repeat this expression, three of them carrying a comment asking
the reader to keep them in sync by hand; a single one of them drifting would
split the platform's state across two directories.

The environment is read on every call, not cached in ``Settings``: tests and the
CLI set ``DOCIE_SERVING_HOME`` per process after import.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Everything a deployment name may contribute to a filename. A name is minted
# as ``<store-name>-<hex>`` and is otherwise operator-supplied, so anything
# outside this set is replaced rather than trusted.
_LOG_NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def serving_home() -> Path:
    """The shared serving-state root."""
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME",
            Path.home() / ".local" / "share" / "docie-bench" / "serving",
        )
    )


#: Environment override for the model blob store.
MODEL_STORE_DIR_ENV_VAR = "DOCIE_MODEL_STORE_DIR"


class InvalidModelStoreRootError(ValueError):
    """A configured cache root would encompass durable node state."""


def model_store_root() -> Path:
    """Where this node keeps model weights on disk.

    ``serving_home`` holds state a node cannot lose: ``node_id`` is its
    identity, ``deployments.json`` is what it is running, and the applied-state
    file is how far it has converged. Weights are the opposite -- once a shared
    repository can supply them, the local copy is a CACHE, rebuildable by
    pulling again.

    **What the default actually costs.** Today the store is a CHILD of the
    state root (``<serving_home>/models``), so removing the store directory
    does not touch ``node_id`` beside it. The cost is that the two share one
    VOLUME: the weights cannot be put on a different disk, their growth
    consumes the volume the durable state lives on, and any volume-level
    operation -- destroying it, resizing it, moving it to another host --
    takes both. #595 lists the split as a prerequisite for the remote backend
    because a cache you cannot place or size independently is not really a
    cache.

    The default is EXACTLY the current location, so an existing node finds its
    blobs where it left them and nothing has to be moved to upgrade. The
    separation is what an explicit override buys; the default deliberately
    keeps the old shared-volume arrangement.

    A configured root is REFUSED when it would encompass the state root -- see
    :func:`_reject_root_covering_state`.
    """
    override = os.environ.get(MODEL_STORE_DIR_ENV_VAR, "").strip()
    if not override:
        return serving_home() / "models"
    root = Path(override)
    _reject_root_covering_state(root)
    return root


def _reject_root_covering_state(root: Path) -> None:
    """Refuse a cache root that is, or contains, ``serving_home``.

    The point of configuring a cache root is that its contents are disposable.
    A root of ``/app/.serving`` (the state root) or ``/app`` (above it) makes
    "clear the cache" mean "delete this node's identity and its deployment
    records" -- turning the safety this split exists to provide into the
    hazard it was meant to remove.

    Deliberately NOT applied to the default. The default IS a child of the
    state root, which is the compatible arrangement every existing node
    already has; refusing it would mean no node could start. The check is
    about a configuration an operator wrote expecting disposability.

    Resolved before comparing so ``/app/.serving/..`` is caught too. Paths that
    do not exist yet still resolve lexically, which is the normal case on a
    fresh node.
    """
    state = serving_home().expanduser().resolve()
    candidate = root.expanduser().resolve()
    if candidate == state or candidate in state.parents:
        raise InvalidModelStoreRootError(
            f"{MODEL_STORE_DIR_ENV_VAR}={str(root)!r} resolves to {candidate}, "
            f"which is or contains the serving state root {state}. The model "
            f"cache is meant to be disposable, and this would put node_id, "
            f"deployments.json and the applied-state file inside it. Point it "
            f"at a directory beside or below them, or leave it unset to keep "
            f"the default {state / 'models'}."
        )


def deployment_log_path(logs_dir: Path, name: str) -> Path:
    """Where ``name``'s runtime stdout lives, guaranteed inside ``logs_dir``.

    The writer opened ``logs_dir / f"{name}.log"`` directly, and a deployment
    name is operator-supplied through the deploy request. A name carrying
    parent segments therefore made the supervisor ``mkdir -p`` an arbitrary
    directory and append the runtime's output into it (``runtime.start`` opens
    the file ``"ab"``), which is the whole reason this function exists.

    **Mapped, not refused.** A node restarting with an existing deployment
    whose name predates this would otherwise lose its log, or worse, refuse to
    start it. Replacing the unsafe characters keeps every name servable and
    every log findable.

    Reader and writer MUST both come through here. They used to build the path
    independently -- the reader with a containment check, the writer without --
    so any name they disagreed about was a log that existed and could not be
    read.
    """
    safe = _LOG_NAME_SAFE.sub("_", name).strip(".") or "deployment"
    return logs_dir / f"{safe}.log"
