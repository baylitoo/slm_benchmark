"""Bridge declarative routing policies to the benchmark's model profiles.

The routing engine in :mod:`docie_bench.extract.routing` is profile-agnostic: a
stage is just a named ``ExtractionStage``. The benchmark binds those stages to
configured model profiles by *name convention* — a policy stage named
``ollama_qwen3_4b`` runs the ``ollama_qwen3_4b`` profile from ``models.yaml``.
Adding a model to a cascade is therefore just adding the profile and referencing
its name as a stage; no extra mapping config is required.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from docie_bench.extract.routing import (
    ExtractionRouter,
    ExtractionServiceStage,
    RoutingPolicy,
)
from docie_bench.extract.service import ExtractionService
from docie_bench.llm.model_profiles import ModelProfile


def load_routing_policy(path: str | Path) -> RoutingPolicy:
    """Load and validate a declarative routing policy from YAML or JSON."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Routing policy {path} must be a mapping, got {type(raw).__name__}")
    return RoutingPolicy.model_validate(raw)


def resolve_routing_profiles(
    policy: RoutingPolicy, profiles: Mapping[str, ModelProfile]
) -> list[ModelProfile]:
    """Return the distinct profiles a policy references, in first-seen stage order.

    Raises ``ValueError`` if any stage name does not match a configured profile so
    the failure is reported up front rather than mid-run.
    """
    unknown = sorted({stage.name for stage in policy.stages if stage.name not in profiles})
    if unknown:
        raise ValueError(
            f"Routing policy references unknown model profiles {unknown}; "
            "each stage name must match a profile in the models config."
        )
    ordered: list[ModelProfile] = []
    seen: set[str] = set()
    for stage in policy.stages:
        if stage.name not in seen:
            seen.add(stage.name)
            ordered.append(profiles[stage.name])
    return ordered


def build_extraction_router(
    policy: RoutingPolicy, profiles: Mapping[str, ModelProfile]
) -> ExtractionRouter:
    """Build a router whose stages map 1:1 to model profiles by name."""
    resolve_routing_profiles(policy, profiles)
    stages = [
        ExtractionServiceStage(stage.name, ExtractionService(profiles[stage.name]))
        for stage in policy.stages
    ]
    return ExtractionRouter(stages=stages, policy=policy)
