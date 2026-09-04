"""Resolve a ``store:<name>`` model reference to a ready-to-use ModelProfile.

This is the single seam that connects a *deployment* (a ``ModelPlacement`` row
written by the deploy job) to an *extraction* (a ``ModelProfile`` the LLM
client consumes). Both ``docie_bench.api.resolve_profile`` and
``docie_bench.inngest.functions._resolve_profile`` delegate ``store:`` refs
here, so deploying a store model is all it takes to extract with it — no
models.yaml edit, no env change.

Style precedence (the load-bearing rule):

1. ``placement.negotiated_style`` — the probed known-good style, once the
   probe-at-deploy pass fills it in.
2. The family's declared ``response_format_style`` when the family delivers
   its template out-of-band (``nuextract3``'s ``chat_template_kwargs``,
   ``nuextract_v1``'s in-prompt format). Substituting an engine default here
   would silently break the family contract — e.g. emitting
   ``openai_json_schema`` for a nuextract3 placement disables vision
   extraction.
3. ``ENGINE_DEFAULT_STYLE[engine]`` for generic OpenAI-chat families.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import urllib.parse
from collections.abc import Callable, Sequence
from typing import Any

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.model_store import FAMILIES, FamilyContract, TemplateDelivery

logger = logging.getLogger(__name__)

STORE_PROFILE_PREFIX = "store:"

# Per-engine default response-format style for GENERIC families only (see the
# module docstring for the full precedence rule).
ENGINE_DEFAULT_STYLE: dict[str, str] = {
    # Strongest style llama.cpp supports; the client's negotiation ladder
    # downgrades from there if the grammar fails to compile.
    "llama-server": "openai_json_schema",
    # Ollama's json_schema path returns empty content on several models, so
    # json_object is the safe universal default (see llm.model_catalog).
    "ollama": "json_object",
    # The transformers shim free-generates (it ignores response_format), so it
    # cannot enforce a grammar — rely on the prompt for JSON, never advertise a
    # constraint the server drops. (Last-resort path; best-effort extraction.)
    "transformers": "none",
}

_FALLBACK_STYLE = "openai_json_schema"


class PlacementError(RuntimeError):
    """A ``store:<name>`` reference could not be resolved to a live deployment."""


class PlacementNotFoundError(PlacementError):
    """No catalog entry or no live placement for the referenced store model."""


class PlacementNotReadyError(PlacementError):
    """A placement exists but its deployment is not serving yet."""


def _resolve_style(
    placement: dict[str, object], contract: FamilyContract | None, engine: str
) -> str:
    negotiated = placement.get("negotiated_style")
    if negotiated:
        return str(negotiated)
    if contract is not None and contract.template_delivery != TemplateDelivery.OPENAI_JSON_SCHEMA:
        # Purpose-built family: its template is delivered out-of-band, so its
        # declared style is binding — never swap in an engine default.
        return contract.response_format_style
    return ENGINE_DEFAULT_STYLE.get(engine, _FALLBACK_STYLE)


def endpoint_is_loopback(url: str) -> bool:
    """True when ``url``'s host is only reachable from its own machine.

    The deploy runtime records its endpoint from the WORKER's point of view, so
    a loopback (127.0.0.0/8, ``localhost``, ``::1``) or unspecified (0.0.0.0)
    host is worker-local: in the split api/worker compose topology no other
    container can ever reach it. Callers that run in a different process/host
    than the deploy runtime should reject such endpoints up front instead of
    burning the client's timeout on doomed connect retries.
    """
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _placement_is_live(placement: dict[str, Any]) -> bool:
    # PR-1 contract: a non-live row stores endpoint "" (the column is NOT NULL),
    # and EVERY reader must treat "" as "no live endpoint" — never route into it,
    # whatever the state column momentarily says.
    return (
        str(placement.get("state") or "") == "ready"
        and bool(str(placement.get("endpoint") or "").strip())
    )


# Round-robin state for replica routing: one counter per store model, guarded
# by a lock because resolution runs on concurrent request threads. See
# round_robin_choice for the consistency model.
_ROUND_ROBIN_LOCK = threading.Lock()
_ROUND_ROBIN_COUNTERS: dict[str, int] = {}


def round_robin_choice(
    model_name: str, candidates: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Pick the next live replica of ``model_name`` in rotation.

    Candidates are ordered by record name (``base`` < ``base-2`` < …) so the
    rotation sequence is deterministic, then indexed by a per-model counter.

    CONSISTENCY MODEL (stated honestly): the counter is process-local and
    in-memory. Each api/worker process rotates independently, a restart resets
    the counter to zero, and nothing is coordinated across processes — the
    guarantee is per-process rotation (successive calls in one process never
    hit the same replica twice while several are live), which composes into
    approximately uniform distribution across processes. When the live set
    changes size between calls (a replica died or scaled in/out), the modulo
    re-maps the counter and one endpoint may be skipped or repeated once —
    acceptable for load balancing, but this function alone has no session
    affinity (successive calls for the same conversation can land on
    different replicas). See :func:`session_affinity_choice`, which layers
    per-conversation pinning on top of this same rotation.
    """
    ordered = sorted(candidates, key=lambda p: str(p.get("name") or ""))
    with _ROUND_ROBIN_LOCK:
        index = _ROUND_ROBIN_COUNTERS.get(model_name, 0)
        _ROUND_ROBIN_COUNTERS[model_name] = index + 1
    return ordered[index % len(ordered)]


# Session-affinity state: (model_name, session_id) -> pinned replica record
# name. Guarded by the same lock discipline as the round-robin counters
# above (a separate lock, since the two are never taken together). See
# session_affinity_choice for the consistency model.
_SESSION_AFFINITY_LOCK = threading.Lock()
_SESSION_AFFINITY: dict[str, str] = {}


def session_affinity_choice(
    session_id: str, model_name: str, candidates: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Pick a replica of ``model_name`` for ``session_id``, pinning later
    turns of the same conversation to the same replica.

    Each top-level ``/v1/chat/completions`` call is independently
    round-robined by default, which can bounce a multi-turn conversation
    across replicas turn to turn and defeat llama-server's prefix-KV cache.
    Reusing the ``session_id`` already threaded through the Playground for
    docs-search continuity gives this a free, request-visible affinity key.

    CONSISTENCY MODEL (stated as honestly as round_robin_choice's own): the
    pin map is process-local and in-memory. Each api/worker process pins
    independently, a restart drops every pin (those sessions just
    round-robin again from a cold start, same as a brand-new session), and
    nothing is coordinated across processes — acceptable because
    round_robin_choice already makes the same trade-off for load balancing.
    The map also only grows (one entry per distinct ``session_id`` seen,
    never evicted) until a process restart. Note also that
    ``resolve_store_profile``'s single-live-replica shortcut bypasses this
    function entirely, so a pin recorded while two replicas were live is not
    refreshed during a window where only one survives — if the pinned
    replica later revives, the stale pin still points at it, which is
    correct routing either way.

    Behavior:
    - No existing pin for ``session_id``: falls back to
      :func:`round_robin_choice` and records the pick as the new pin.
    - An existing pin whose replica is still in ``candidates`` (still live):
      returns that replica — deterministic reuse across turns.
    - An existing pin whose replica is no longer live: falls back to
      :func:`round_robin_choice` for THIS request and re-pins to the new
      pick, so subsequent turns in the same session follow the replacement.
    """
    key = f"{model_name}:{session_id}"
    with _SESSION_AFFINITY_LOCK:
        pinned_name = _SESSION_AFFINITY.get(key)
    if pinned_name is not None:
        for candidate in candidates:
            if str(candidate.get("name") or "") == pinned_name:
                return candidate
    choice = round_robin_choice(model_name, candidates)
    with _SESSION_AFFINITY_LOCK:
        _SESSION_AFFINITY[key] = str(choice.get("name") or "")
    return choice


def _record_activity_best_effort(name: str, catalog: ModelCatalog) -> None:
    """Bump ``model_activity`` for ``name`` — best-effort, mirroring
    ``profile_resolver._catalog_family``'s swallow pattern for a similar
    advisory lookup. An activity-tracking hiccup (a DB error, or a rare
    concurrent-insert race on the model's very first request) must never
    break resolution for a request that otherwise resolved cleanly.
    """
    try:
        catalog.record_activity(name)
    except Exception:  # noqa: BLE001 - see docstring
        logger.debug("model-activity bump failed for %r", name, exc_info=True)


def resolve_store_profile(
    name: str,
    *,
    catalog: ModelCatalog | None = None,
    chooser: Callable[[Sequence[dict[str, Any]]], dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> ModelProfile:
    """Build the ModelProfile that extracts against a live placement of ``name``.

    LOAD BALANCING (PR-C): a scaled model has several placement rows (one per
    replica, all sharing ``model_name``); this picks ONE live replica per call.
    The default pick is :func:`round_robin_choice` — a process-local rotation
    over the live replicas (see its docstring for the consistency model);
    tests inject a deterministic ``chooser`` instead. Dead replicas are
    skipped naturally: only rows with ``state="ready"`` and a non-empty
    endpoint are candidates, and the serving reconciler republishes every
    row's observed state each cycle, so a crashed replica leaves the rotation
    within one reconcile interval. A single-instance model has exactly one
    live row, so the pick is a no-op and behaviour is unchanged.

    SESSION AFFINITY: when ``session_id`` is given (and no explicit
    ``chooser`` is injected), the pick goes through
    :func:`session_affinity_choice` instead of a bare round-robin, so later
    turns of the same conversation land on the same replica and keep
    llama-server's prefix-KV cache warm. A ``session_id`` of ``None`` (the
    default) round-robins exactly as before — existing callers are
    unaffected.

    Raises :class:`PlacementNotFoundError` when the model is not in the catalog
    or has no placement at all, and :class:`PlacementNotReadyError` when
    deployments exist but none is serving yet. May raise
    ``CatalogUnavailableError`` when DATABASE_URL is not configured.
    """
    catalog = catalog if catalog is not None else ModelCatalog()
    entry = catalog.get(name)
    if entry is None:
        raise PlacementNotFoundError(
            f"store model {name!r} is not in the catalog; seed it first"
        )
    placements = catalog.list_placements_for_model(name)
    if not placements:
        raise PlacementNotFoundError(
            f"No live placement for store model {name!r}. Deploy it first "
            f"(POST /v1/serving/deploy or `docie up {name}`), then retry."
        )
    live = [p for p in placements if _placement_is_live(p)]
    if not live:
        # No replica is servable — give an honest reason from the freshest row
        # (list_placements_for_model returns freshest-first).
        freshest = placements[0]
        state = str(freshest.get("state") or "")
        if state != "ready":
            raise PlacementNotReadyError(
                f"store model {name!r} placement is {state!r}, not ready — "
                f"wait for the deploy to finish or redeploy."
            )
        raise PlacementNotReadyError(
            f"store model {name!r} placement advertises no live endpoint — "
            f"wait for the deploy/reload to finish or redeploy."
        )
    if len(live) == 1:
        placement = live[0]
    elif chooser is not None:
        placement = chooser(live)
    elif session_id:
        placement = session_affinity_choice(session_id, name, live)
    else:
        placement = round_robin_choice(name, live)
    _record_activity_best_effort(name, catalog)
    contract = FAMILIES.get(str(entry.get("family") or ""))
    engine = str(placement.get("engine") or "")
    return ModelProfile(
        name=f"{STORE_PROFILE_PREFIX}{name}",
        # The model id sent upstream is the llama-server --alias, which is the
        # BASE store name (model_name) — NOT the record name. They are equal for
        # a single deploy, but a scaled replica's record is `<base>-2` while its
        # alias stays `<base>`, so routing to it with the record name would ask
        # for a model the server does not answer to.
        model=str(placement.get("model_name") or placement["name"]),
        base_url=str(placement.get("endpoint") or "").rstrip("/"),
        api_key="local-not-used",
        response_format_style=_resolve_style(placement, contract, engine),
        prompt_profile=contract.prompt_profile if contract else "strict_extraction_v1",
        vision=bool(contract.vision) if contract else False,
        stop_sequences=tuple(contract.stop_sequences) if contract else (),
        temperature=contract.default_temperature if contract else 0.0,
        # The family's generation tuning, not ModelProfile's bare defaults
        # (900/180s) -- profile_resolver.py's family-synthesis path already
        # carries these (guarding NuExtract3's 4096/600s requirement); this
        # sibling store: routing path never did, so a store-deployed
        # NuExtract3 silently ran capped at 900 tokens / timing out at 180s.
        max_tokens=contract.default_max_tokens if contract else 900,
        timeout_seconds=contract.default_timeout_seconds if contract else 180.0,
    )


__all__ = [
    "ENGINE_DEFAULT_STYLE",
    "STORE_PROFILE_PREFIX",
    "PlacementError",
    "PlacementNotFoundError",
    "PlacementNotReadyError",
    "endpoint_is_loopback",
    "resolve_store_profile",
    "round_robin_choice",
    "session_affinity_choice",
]
