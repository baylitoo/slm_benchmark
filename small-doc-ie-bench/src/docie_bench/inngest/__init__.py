"""Inngest integration for DocIE Studio.

This package turns the framework's core operations (document extraction,
benchmark runs, model deployment) into durable, event-driven Inngest
functions served by a long-lived Connect worker. It is *additive*: the
existing hand-rolled orchestrator (``docie_bench.orchestrator``) is left
untouched. Fire an event (``inngest_client.send``) and a worker picks it up;
results stream back over an Inngest realtime channel.

Note: ``import inngest`` inside this package resolves to the third-party
``inngest`` SDK (absolute imports), not this subpackage.
"""

from __future__ import annotations

# NOTE: do not re-export the `functions` list here — binding the name `functions`
# on this package would shadow the `docie_bench.inngest.functions` submodule
# (`import docie_bench.inngest.functions` would resolve to the list). Import the
# client from here; import the function registry straight from the submodule:
#   from docie_bench.inngest.functions import functions
from docie_bench.inngest.client import inngest_client

__all__ = ["inngest_client"]
