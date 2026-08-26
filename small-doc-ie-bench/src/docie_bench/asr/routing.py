"""Resolve public transcription selectors to live managed ASR deployments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from docie_bench.serving.profile_resolver import _default_live_deployments, _is_live
from docie_bench.serving.runtime import RuntimeKind
from docie_bench.serving.supervisor import DeploymentRecord


class ASRRoutingError(ValueError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ASRRoute:
    deployment: str
    base_url: str
    model: str


def resolve_asr_route(
    selector: str,
    *,
    deployments: Sequence[DeploymentRecord] | None = None,
) -> ASRRoute:
    """Resolve by deployment name first, then by unique served alias.

    Only ready ASR runtime records participate. This is also the model
    allowlist: arbitrary Hub ids can never become download instructions.
    """

    records = list(deployments) if deployments is not None else _default_live_deployments()
    asr_records = [r for r in records if r.spec.launch.runtime == RuntimeKind.ASR]
    live = [r for r in asr_records if _is_live(r)]
    exact = [r for r in live if r.spec.name == selector]
    matches = exact or [r for r in live if r.spec.launch.alias == selector]
    if not matches:
        known = [
            r
            for r in asr_records
            if r.spec.name == selector or r.spec.launch.alias == selector
        ]
        if known:
            raise ASRRoutingError(
                f"ASR deployment {selector!r} is not ready", status_code=503
            )
        raise ASRRoutingError(
            f"Unknown ASR deployment or alias {selector!r}", status_code=404
        )
    if len(matches) > 1:
        names = ", ".join(sorted(r.spec.name for r in matches))
        raise ASRRoutingError(
            f"ASR alias {selector!r} is ambiguous across live deployments: {names}",
            status_code=409,
        )
    record = matches[0]
    return ASRRoute(
        deployment=record.spec.name,
        base_url=str(record.endpoint).rstrip("/"),
        model=record.spec.launch.alias,
    )
