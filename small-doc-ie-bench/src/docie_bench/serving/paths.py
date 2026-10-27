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
