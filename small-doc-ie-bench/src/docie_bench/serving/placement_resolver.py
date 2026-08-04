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
import random
import urllib.parse
from collections.abc import Callable, Sequence
from typing import Any

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.model_store import FAMILIES, FamilyContract, TemplateDelivery

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


def resolve_store_profile(
    name: str,
    *,
    catalog: ModelCatalog | None = None,
    chooser: Callable[[Sequence[dict[str, Any]]], dict[str, Any]] = random.choice,
) -> ModelProfile:
    """Build the ModelProfile that extracts against a live placement of ``name``.

    LOAD BALANCING (PR-C): a scaled model has several placement rows (one per
    replica, all sharing ``model_name``); this picks ONE live replica per call.
    ``chooser`` defaults to ``random.choice`` — stateless, so it distributes
    evenly across replicas even from many concurrent workers with no shared
    counter; tests inject a deterministic chooser. A single-instance model has
    exactly one live row, so the pick is a no-op and behaviour is unchanged.

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
    placement = live[0] if len(live) == 1 else chooser(live)
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
    )


__all__ = [
    "ENGINE_DEFAULT_STYLE",
    "STORE_PROFILE_PREFIX",
    "PlacementError",
    "PlacementNotFoundError",
    "PlacementNotReadyError",
    "endpoint_is_loopback",
    "resolve_store_profile",
]
