"""Resolve an extraction request to exactly one ``ModelProfile``.

This is the single routing seam shared by the sync API (``docie_bench.api``) and
the Inngest worker (``docie_bench.inngest.functions``). It reuses the gateway's
routing *primitive* (:func:`docie_bench.serving.gateway.resolve_profile` —
name-first, then unique upstream id, ambiguity-safe) but applies it over an
*enriched* table: the static ``configs/models.yaml`` profiles PLUS one synthesized
passthrough profile per **live** deployment (from the supervisor's
``deployments.json``). Routing to a live deployment is therefore the gateway
mechanism applied to the live-deployment registry, invoked in-process —
``OpenAICompatibleClient`` then forwards to that deployment's own runtime endpoint.

Precedence (see the PR design's ``extract_contract``):

1. ``deployment`` — must resolve to a LIVE deployment (``state == "ready"`` AND an
   ``endpoint``); otherwise REFUSE (never fall back to env).
2. ``model_profile`` — a ``models.yaml`` profile name/upstream id OR a
   live-deployment name; otherwise REFUSE.
3. neither — the honest default ``settings.default_model_profile`` loaded FROM
   ``models.yaml``. The env-derived profile is reached only as a labeled last
   resort (``name="env_fallback"``) when ``models.yaml`` is entirely absent.

Per-runtime served-id rule: ``llama-server`` answers to ``--alias`` and vLLM to
``--served-model-name=alias``, so the synthesized ``profile.model`` is
``spec.launch.alias`` for ``llamacpp``/``vllm`` and ``spec.launch.model`` for
``ollama``/``remote``. This is the #1 landmine: the "Auto"/store deploy path sets
``model=<GGUF filesystem path>``, so sending that as the model id would 400 the
upstream — the alias is the addressable served id.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from docie_bench.llm.model_profiles import ModelProfile, load_model_profiles
from docie_bench.serving.gateway import GatewayRoutingError
from docie_bench.serving.gateway import resolve_profile as gateway_resolve_profile
from docie_bench.serving.model_store import FAMILIES
from docie_bench.serving.runtime import LifecycleState, RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentRecord, PersistentSupervisor
from docie_bench.settings import Settings, get_settings

logger = logging.getLogger("docie_bench.serving.profile_resolver")

DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")

# Runtimes whose served id is the launch *alias* (they are started with
# ``--alias`` / ``--served-model-name``); everything else answers to the model id.
_ALIAS_SERVED_RUNTIMES = frozenset({RuntimeKind.LLAMACPP, RuntimeKind.VLLM})


class ProfileResolutionError(ValueError):
    """A deployment / model_profile selector could not be resolved to a profile."""


@dataclass(frozen=True)
class _Traits:
    """The family traits inherited when synthesizing a profile from its family.

    Carries the template-sensitive traits (style / prompt / vision / stop) AND the
    generation params (temperature / max_tokens / timeout) so a family-synthesized
    profile runs with the family's intended tuning rather than bare ModelProfile
    defaults — e.g. NuExtract3 needs 4096 tokens, not the 900 default.
    """

    response_format_style: str
    prompt_profile: str
    vision: bool
    stop_sequences: tuple[str, ...] = ()
    temperature: float = 0.0
    max_tokens: int = 900
    timeout_seconds: float = 180.0


def _serving_home() -> Path:
    # Must match ``control_plane.from_defaults`` so the resolver reads the SAME
    # deployments.json the worker writes (the shared ``serving-state`` volume).
    return Path(
        os.environ.get(
            "DOCIE_SERVING_HOME",
            Path.home() / ".local" / "share" / "docie-bench" / "serving",
        )
    )


def _default_live_deployments() -> list[DeploymentRecord]:
    """Read deployments FRESH from disk (uncached), returning typed records.

    A ``PersistentSupervisor`` loads ``deployments.json`` in ``__init__`` and never
    reloads, so a new one per call is an always-fresh, side-effect-free read
    (no ``_save``, no mkdir) — the same freshness rationale as
    ``serving_api._control_plane`` (which rebuilds ``ControlPlane.from_defaults``
    per request). Building the supervisor directly avoids the async
    ``ControlPlane`` facade (whose ``list_deployments`` returns JSON dicts, not the
    typed ``DeploymentRecord`` this module needs) and the planner/psutil setup.
    """
    supervisor = PersistentSupervisor(_serving_home() / "deployments.json")
    return list(supervisor.list())


def _is_live(record: DeploymentRecord) -> bool:
    """A deployment is selectable iff it is READY and advertises an endpoint."""
    return record.state == LifecycleState.READY and bool(record.endpoint)


def _served_id(launch: RuntimeLaunchSpec) -> str:
    """The id the upstream actually answers to (alias for llamacpp/vllm)."""
    if launch.runtime in _ALIAS_SERVED_RUNTIMES:
        return launch.alias
    return launch.model


def _match_yaml_profile(
    served_id: str, yaml_profiles: dict[str, ModelProfile]
) -> ModelProfile | None:
    """A models.yaml profile whose upstream ``model`` equals the served id.

    Deterministic (sorted by name) when several match (e.g. ``nux`` / ``nux_think``
    share ``nuextract3``) — only the template traits are inherited, the base_url is
    overridden with the deployment's own endpoint regardless.
    """
    matches = sorted(
        (profile for profile in yaml_profiles.values() if profile.model == served_id),
        key=lambda profile: profile.name,
    )
    return matches[0] if matches else None


def _store_family(name: str) -> str | None:
    """The deployment's family from the on-disk model-store index (no DB).

    ``StoreEntry.family`` is recorded in ``<serving_home>/models/index.json`` at
    seed/deploy time, so family/vision/generation inheritance needs no
    DATABASE_URL. Read the index file directly (not via ``ModelStore``, whose
    ``__init__`` mkdirs) to keep resolution side-effect-free. Best-effort: a
    missing/corrupt index or an absent entry yields ``None`` (fall back to the
    catalog).
    """
    index_path = _serving_home() / "models" / "index.json"
    try:
        if not index_path.is_file():
            return None
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - a disk hiccup must not break routing
        logger.warning("model-store index read failed for %r", name, exc_info=True)
        return None
    entry = data.get(name) if isinstance(data, dict) else None
    if not isinstance(entry, dict):
        return None
    family = entry.get("family")
    return str(family) if family else None


def _catalog_family(name: str) -> str | None:
    """The deployment's family from the Postgres catalog (best-effort; needs DB).

    Only consulted when the on-disk store entry is absent. No DATABASE_URL (unit
    tests) or a transient DB error must never break routing — it yields ``None``.
    """
    try:
        from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

        try:
            view = ModelCatalog().get(name)
        except CatalogUnavailableError:
            return None
    except Exception:  # pragma: no cover - a DB hiccup must not break routing
        logger.warning("model-catalog family lookup failed for %r", name, exc_info=True)
        return None
    if not view:
        return None
    family = view.get("family")
    return str(family) if family else None


def _family_traits(name: str) -> _Traits | None:
    """Traits from the deployment's store-entry family (on-disk store, then catalog).

    Family is read from the on-disk ``StoreEntry.family`` FIRST (no DATABASE_URL),
    falling back to the Postgres catalog only when the store entry is absent. The
    matched ``FamilyContract`` is the single source of truth for the family's
    template traits AND its generation defaults (temperature / max_tokens /
    timeout), so a family-synthesized profile carries the family's intended tuning
    instead of bare ModelProfile defaults. Best-effort: no family found -> ``None``
    (the caller then uses conservative defaults).
    """
    family_name = _store_family(name) or _catalog_family(name)
    if family_name is None:
        return None
    family = FAMILIES.get(family_name)
    if family is None:
        return None
    return _Traits(
        response_format_style=family.response_format_style,
        prompt_profile=family.prompt_profile,
        vision=family.vision,
        stop_sequences=family.stop_sequences,
        temperature=family.default_temperature,
        max_tokens=family.default_max_tokens,
        timeout_seconds=family.default_timeout_seconds,
    )


def _synthesize_profile(
    record: DeploymentRecord, yaml_profiles: dict[str, ModelProfile]
) -> ModelProfile:
    """Build a passthrough ModelProfile that routes to one live deployment.

    ``name`` is the deployment's own name (the honest label surfaced in the
    ExtractionResponse), ``base_url`` its live endpoint, ``model`` the per-runtime
    served id. Template traits (style / prompt / vision, and — when a whole profile
    matches — its tuning too) are inherited from a matching models.yaml profile,
    else the store-entry family, else conservative defaults.
    """
    launch = record.spec.launch
    served_id = _served_id(launch)
    base_url = (record.endpoint or "").rstrip("/")

    match = _match_yaml_profile(served_id, yaml_profiles)
    if match is not None:
        # Inherit the full matching profile (temperature/max_tokens/stop/style/…),
        # repointed at this deployment. Force passthrough — a matched profile could
        # in principle be an ocr/pipeline kind, which must not leak here.
        return replace(
            match,
            name=record.spec.name,
            model=served_id,
            base_url=base_url,
            kind="passthrough",
        )

    traits = _family_traits(record.spec.name)
    if traits is None:
        traits = _Traits(
            response_format_style="openai_json_schema",
            prompt_profile="strict_extraction_v1",
            vision=False,
        )
    return ModelProfile(
        name=record.spec.name,
        model=served_id,
        base_url=base_url,
        api_key="local-not-used",
        response_format_style=traits.response_format_style,
        prompt_profile=traits.prompt_profile,
        vision=traits.vision,
        stop_sequences=traits.stop_sequences,
        temperature=traits.temperature,
        max_tokens=traits.max_tokens,
        timeout_seconds=traits.timeout_seconds,
    )


def _env_fallback_profile(settings: Settings) -> ModelProfile:
    """Last-resort profile from OPENAI_COMPAT_* env — reached only when models.yaml
    is entirely absent. Labeled ``env_fallback`` (never a real profile name) so the
    ExtractionResponse never mislabels an env-synthesized route as a config profile.
    """
    return ModelProfile(
        name="env_fallback",
        model=settings.openai_compat_model,
        base_url=settings.openai_compat_base_url.rstrip("/"),
        api_key=settings.openai_compat_api_key.get_secret_value(),
        response_format_style=settings.openai_compat_response_format_style,
        timeout_seconds=settings.openai_compat_timeout_seconds,
    )


def build_profile_table(
    yaml_profiles: dict[str, ModelProfile],
    live: dict[str, DeploymentRecord],
) -> dict[str, ModelProfile]:
    """Merge the models.yaml table with one synthesized profile per live deployment.

    Live deployments are added last, so a deployment whose name equals a yaml
    profile name routes to the (running) deployment.
    """
    table: dict[str, ModelProfile] = dict(yaml_profiles)
    for name, record in live.items():
        table[name] = _synthesize_profile(record, yaml_profiles)
    return table


def resolve_extraction_profile(
    *,
    deployment: str | None = None,
    model_profile: str | None = None,
    models_config_path: str | Path = DEFAULT_MODELS_CONFIG,
    deployments: Sequence[DeploymentRecord] | None = None,
    settings: Settings | None = None,
) -> ModelProfile:
    """Resolve one extraction request to a single ModelProfile (see module docstring).

    ``deployments`` is an injection seam for tests (a stub returning fabricated
    ``DeploymentRecord``s); production reads them fresh from disk. Raises
    :class:`ProfileResolutionError` on an unknown/not-ready explicit selector or a
    missing default — callers translate that to their surface (HTTP 400 / channel
    error). Never silently falls back to env for an explicit selector.
    """
    settings = settings or get_settings()
    path = Path(models_config_path)
    yaml_profiles = load_model_profiles(path) if path.exists() else {}
    records = list(deployments) if deployments is not None else _default_live_deployments()
    live = {record.spec.name: record for record in records if _is_live(record)}

    # (1) explicit deployment selector -> a live deployment or refuse.
    if deployment:
        record = live.get(deployment)
        if record is None:
            raise ProfileResolutionError(
                f"deployment {deployment!r} is not a live (ready) deployment"
            )
        return _synthesize_profile(record, yaml_profiles)

    table = build_profile_table(yaml_profiles, live)

    # (2) explicit model_profile -> a config profile / upstream id / deployment name.
    if model_profile:
        try:
            return gateway_resolve_profile(model_profile, table)
        except GatewayRoutingError as exc:
            raise ProfileResolutionError(exc.message) from exc

    # (3) default -> studio_default from models.yaml (honest label); env only as a
    # labeled last resort when models.yaml is entirely absent.
    default_name = settings.default_model_profile
    if default_name in table:
        return table[default_name]
    if not yaml_profiles:
        return _env_fallback_profile(settings)
    raise ProfileResolutionError(
        f"default model profile {default_name!r} is not defined in {path}"
    )


__all__ = [
    "ProfileResolutionError",
    "resolve_extraction_profile",
    "build_profile_table",
    "DEFAULT_MODELS_CONFIG",
]
