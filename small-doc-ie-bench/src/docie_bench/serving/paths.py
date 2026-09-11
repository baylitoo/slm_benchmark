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
from pathlib import Path


def serving_home() -> Path:
    """The shared serving-state root."""
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME",
            Path.home() / ".local" / "share" / "docie-bench" / "serving",
        )
    )
