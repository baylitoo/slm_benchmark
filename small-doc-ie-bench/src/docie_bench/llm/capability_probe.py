"""Runtime capability probe for the response-format negotiation path.

A profile declares one ``response_format_style``, but whether a given endpoint
actually *honours* it is a property of the live runtime (small Ollama models
return empty content for ``json_schema``). This module records, per endpoint,
which styles were confirmed to work — via ``GET /models`` plus one cheap canary
extraction — so the negotiation ladder can be pruned to the strongest confirmed
style instead of burning a downgrade round-trip on every real document.

The cache is module-level and keyed by ``(base_url, model)`` because the
extraction path builds a fresh client per document; a per-client cache would
re-probe (and re-burn tokens) on every call. Entries carry a TTL and a profile
fingerprint so a config change or an expiry forces a fresh probe rather than
re-introducing the empty-content bug from a stale result.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.response_format import is_generic_style, style_ladder

if TYPE_CHECKING:
    from docie_bench.llm.openai_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

DEFAULT_PROBE_TTL_SECONDS = 900.0

# Minimal schema/prompt used by the canary. Trivial on purpose: the point is to
# learn whether the runtime returns *any* valid JSON object for a style, not to
# extract anything.
_CANARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "string"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_CANARY_SYSTEM = "You are a JSON API. Reply with a single JSON object and nothing else."
_CANARY_USER = 'Return exactly this JSON object: {"ok": "yes"}'


@dataclass(frozen=True)
class CapabilityProbe:
    """Observed response-format behaviour of one endpoint at ``probed_at``."""

    base_url: str
    model: str
    declared_style: str
    effective_style: str | None
    confirmed_styles: tuple[str, ...]
    rejected_styles: tuple[str, ...]
    advertised_styles: tuple[str, ...] | None
    vision: bool | None
    source: str  # "probe" | "skipped" | "error"
    fingerprint: str
    probed_at: float = field(default=0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "declared_style": self.declared_style,
            "effective_style": self.effective_style,
            "confirmed_styles": list(self.confirmed_styles),
            "rejected_styles": list(self.rejected_styles),
            "advertised_styles": (
                list(self.advertised_styles) if self.advertised_styles is not None else None
            ),
            "vision": self.vision,
            "source": self.source,
        }


_PROBE_CACHE: dict[tuple[str, str], CapabilityProbe] = {}


def reset_probe_cache() -> None:
    """Clear the shared probe cache. Intended for tests and reconfiguration."""
    _PROBE_CACHE.clear()


def profile_probe_fingerprint(profile: ModelProfile) -> str:
    """Fingerprint the profile fields that change what a probe would observe.

    Changing any of these invalidates a cached probe so a stale result cannot
    re-introduce the empty-content bug after a profile edit.
    """
    payload = "|".join(
        str(part)
        for part in (
            profile.base_url,
            profile.model,
            profile.response_format_style,
            profile.prompt_profile,
            profile.vision,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_cached_probe(
    profile: ModelProfile,
    *,
    ttl_seconds: float = DEFAULT_PROBE_TTL_SECONDS,
    now: float | None = None,
) -> CapabilityProbe | None:
    """Return a live cached probe for ``profile`` or ``None`` if it must be re-run.

    A cached entry is discarded when its fingerprint no longer matches the
    profile (config changed) or when it is older than ``ttl_seconds`` (expired).
    """
    key = (profile.base_url, profile.model)
    cached = _PROBE_CACHE.get(key)
    if cached is None:
        return None
    if cached.fingerprint != profile_probe_fingerprint(profile):
        return None
    current = time.monotonic() if now is None else now
    if current - cached.probed_at > ttl_seconds:
        return None
    return cached


def store_probe(probe: CapabilityProbe) -> None:
    _PROBE_CACHE[(probe.base_url, probe.model)] = probe


def cached_probe_for_endpoint(base_url: str, model: str) -> CapabilityProbe | None:
    """Best-effort lookup used by the negotiation path (no TTL/fingerprint check).

    The runtime downgrade ladder is unconditional, so a slightly stale hint here
    only affects which rung it *starts* on, never correctness.
    """
    return _PROBE_CACHE.get((base_url, model))


async def probe_endpoint(
    client: OpenAICompatibleClient,
    *,
    ttl_seconds: float = DEFAULT_PROBE_TTL_SECONDS,
    force: bool = False,
    now: float | None = None,
) -> CapabilityProbe:
    """Probe (or return a cached) capability record for ``client``'s endpoint.

    Best-effort: transport failures degrade to ``source="error"`` rather than
    raising, so a probe can never turn a runnable benchmark into a hard failure.
    """
    profile = client.profile
    if not force:
        cached = get_cached_probe(profile, ttl_seconds=ttl_seconds, now=now)
        if cached is not None:
            return cached

    fingerprint = profile_probe_fingerprint(profile)
    probed_at = time.monotonic() if now is None else now
    advertised, vision = await _advertised_metadata(client)

    declared = profile.response_format_style
    # Only the generic JSON family exhibits the empty-content defect and can be
    # safely canaried with a trivial schema; purpose-built styles are recorded
    # as declared-and-skipped so the manifest still documents the decision.
    if not is_generic_style(declared):
        probe = CapabilityProbe(
            base_url=profile.base_url,
            model=profile.model,
            declared_style=declared,
            effective_style=declared,
            confirmed_styles=(),
            rejected_styles=(),
            advertised_styles=advertised,
            vision=vision,
            source="skipped",
            fingerprint=fingerprint,
            probed_at=probed_at,
        )
        store_probe(probe)
        return probe

    confirmed: list[str] = []
    rejected: list[str] = []
    effective: str | None = None
    source = "probe"
    for style in style_ladder(declared):
        if style == "none":
            # The terminal repair rung always "works" (empty -> parse failure is
            # handled by the caller), so there is no point spending a canary on it.
            break
        try:
            honored = await client.probe_style(style, schema=_CANARY_SCHEMA)
        except Exception:  # noqa: BLE001 - probe is best-effort
            logger.warning(
                "capability probe canary failed",
                extra={
                    "docie_model_profile": profile.name,
                    "docie_model": profile.model,
                    "docie_response_format_style": style,
                },
                exc_info=True,
            )
            source = "error"
            break
        if honored:
            confirmed.append(style)
            effective = style
            break
        rejected.append(style)

    if effective is None and source == "probe":
        # Every canaried style returned empty/invalid; fall back to the terminal
        # repair rung so real calls still have somewhere to land.
        effective = "none"

    probe = CapabilityProbe(
        base_url=profile.base_url,
        model=profile.model,
        declared_style=declared,
        effective_style=effective,
        confirmed_styles=tuple(confirmed),
        rejected_styles=tuple(rejected),
        advertised_styles=advertised,
        vision=vision,
        source=source,
        fingerprint=fingerprint,
        probed_at=probed_at,
    )
    store_probe(probe)
    logger.info(
        "capability probe complete",
        extra={
            "docie_model_profile": profile.name,
            "docie_model": profile.model,
            "docie_declared_style": declared,
            "docie_effective_style": effective,
            "docie_rejected_styles": rejected,
            "docie_probe_source": source,
        },
    )
    return probe


async def _advertised_metadata(
    client: OpenAICompatibleClient,
) -> tuple[tuple[str, ...] | None, bool | None]:
    """Best-effort ``GET /models`` capability read; never raises."""
    try:
        capabilities = await client.discover_capabilities(force=True)
    except Exception:  # noqa: BLE001 - advertised metadata is optional
        return None, None
    styles = capabilities.response_format_styles
    advertised = tuple(sorted(styles)) if styles is not None else None
    return advertised, capabilities.vision


async def run_capability_probes(
    profiles: list[ModelProfile],
    *,
    client_factory: Callable[[ModelProfile], OpenAICompatibleClient] | None = None,
    ttl_seconds: float = DEFAULT_PROBE_TTL_SECONDS,
) -> dict[str, dict[str, Any]]:
    """Probe each passthrough profile once and return a manifest-ready mapping.

    Solution kinds (ocr/pipeline) are served by local adapters, not an
    OpenAI-compatible endpoint, so they are not probed.
    """
    from docie_bench.llm.openai_client import OpenAICompatibleClient

    factory = client_factory or OpenAICompatibleClient
    results: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        if profile.kind != "passthrough":
            continue
        client = factory(profile)
        try:
            probe = await probe_endpoint(client, ttl_seconds=ttl_seconds)
        finally:
            await client.aclose()
        results[profile.name] = probe.as_dict()
    return results
