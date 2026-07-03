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


def resolve_store_profile(name: str, *, catalog: ModelCatalog | None = None) -> ModelProfile:
    """Build the ModelProfile that extracts against the live placement of ``name``.

    Raises :class:`PlacementNotFoundError` when the model is not in the catalog
    or has no live placement, and :class:`PlacementNotReadyError` when the
    deployment exists but is not serving yet. May raise
    ``CatalogUnavailableError`` when DATABASE_URL is not configured.
    """
    catalog = catalog if catalog is not None else ModelCatalog()
    entry = catalog.get(name)
    if entry is None:
        raise PlacementNotFoundError(
            f"store model {name!r} is not in the catalog; seed it first"
        )
    placement = catalog.get_placement_for_model(name)
    if placement is None:
        raise PlacementNotFoundError(
            f"No live placement for store model {name!r}. Deploy it first "
            f"(POST /v1/serving/deploy or `docie up {name}`), then retry."
        )
    state = str(placement.get("state") or "")
    if state != "ready":
        raise PlacementNotReadyError(
            f"store model {name!r} placement is {state!r}, not ready — "
            f"wait for the deploy to finish or redeploy."
        )
    contract = FAMILIES.get(str(entry.get("family") or ""))
    engine = str(placement.get("engine") or "")
    return ModelProfile(
        name=f"{STORE_PROFILE_PREFIX}{name}",
        # The deployment name is the llama-server --alias / Ollama model name.
        model=str(placement["name"]),
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
    "resolve_store_profile",
]
