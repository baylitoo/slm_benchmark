"""Unit tests for the shared extraction profile resolver.

Pure unit tests: the live-deployment source is INJECTED (fabricated
``DeploymentRecord``s) so nothing here touches the network, a runtime, or a DB —
mirroring ``tests/test_gateway.py``'s profiles-dict injection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.serving.profile_resolver import (
    ProfileResolutionError,
    resolve_extraction_profile,
)
from docie_bench.serving.runtime import LifecycleState, RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentRecord, DeploymentSpec

_MODELS_YAML = """\
profiles:
  studio_default:
    model: qwen3:4b
    base_url: http://localhost:11434/v1
    api_key: local-not-used
    response_format_style: openai_json_schema
    prompt_profile: strict_extraction_v1

  local_llamacpp:
    model: local-model
    base_url: http://llm-llamacpp:8000/v1
    api_key: local-not-used

  nuextract3:
    model: nuextract3
    base_url: http://localhost:8088/v1
    api_key: local-not-used
    response_format_style: nuextract3
    prompt_profile: nuextract3
    vision: true
"""


@pytest.fixture
def models_config(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(_MODELS_YAML, encoding="utf-8")
    return path


def _record(
    *,
    name: str,
    runtime: RuntimeKind,
    model: str,
    alias: str,
    state: LifecycleState = LifecycleState.READY,
    endpoint: str | None = "http://127.0.0.1:8088/v1",
) -> DeploymentRecord:
    return DeploymentRecord(
        spec=DeploymentSpec(
            name=name,
            launch=RuntimeLaunchSpec(runtime=runtime, model=model, alias=alias),
        ),
        state=state,
        endpoint=endpoint,
    )


# ── (a) ready llamacpp store-deploy: served id is the ALIAS, not the GGUF path ──


def test_llamacpp_store_deploy_uses_alias_not_gguf_path(models_config: Path) -> None:
    # serve_store_model sets alias=entry.name, model=<GGUF path>, runtime LLAMACPP.
    record = _record(
        name="inv",
        runtime=RuntimeKind.LLAMACPP,
        model="/app/.serving/models/inv/model.gguf",
        alias="inv",
        endpoint="http://127.0.0.1:8088/v1",
    )
    profile = resolve_extraction_profile(
        deployment="inv", models_config_path=models_config, deployments=[record]
    )
    assert profile.name == "inv"  # honest label
    assert profile.model == "inv"  # the served alias, NOT the filesystem path
    assert profile.base_url == "http://127.0.0.1:8088/v1"  # the live endpoint
    assert profile.kind == "passthrough"


# ── (b) vllm -> alias; ollama/remote -> model ──────────────────────────────────


def test_vllm_served_id_is_alias(models_config: Path) -> None:
    record = _record(
        name="vllm-dep", runtime=RuntimeKind.VLLM, model="org/big-model", alias="served-alias"
    )
    profile = resolve_extraction_profile(
        deployment="vllm-dep", models_config_path=models_config, deployments=[record]
    )
    assert profile.model == "served-alias"


def test_ollama_served_id_is_model(models_config: Path) -> None:
    record = _record(
        name="olla", runtime=RuntimeKind.OLLAMA, model="qwen2.5:1.5b", alias="olla"
    )
    profile = resolve_extraction_profile(
        deployment="olla", models_config_path=models_config, deployments=[record]
    )
    assert profile.model == "qwen2.5:1.5b"


def test_remote_served_id_is_model(models_config: Path) -> None:
    record = _record(
        name="rem",
        runtime=RuntimeKind.REMOTE,
        model="gpt-ish",
        alias="rem",
        endpoint="http://remote:9000/v1",
    )
    profile = resolve_extraction_profile(
        deployment="rem", models_config_path=models_config, deployments=[record]
    )
    assert profile.model == "gpt-ish"
    assert profile.base_url == "http://remote:9000/v1"


# ── (c) non-ready / endpoint-less deployments are excluded and REFUSED ──────────


@pytest.mark.parametrize(
    ("state", "endpoint"),
    [
        (LifecycleState.STOPPED, "http://127.0.0.1:8088/v1"),
        (LifecycleState.FAILED, "http://127.0.0.1:8088/v1"),
        (LifecycleState.STARTING, "http://127.0.0.1:8088/v1"),
        (LifecycleState.DEGRADED, "http://127.0.0.1:8088/v1"),
        (LifecycleState.READY, None),  # ready but no endpoint yet
    ],
)
def test_non_live_deployment_selector_refuses(
    models_config: Path, state: LifecycleState, endpoint: str | None
) -> None:
    record = _record(
        name="pending",
        runtime=RuntimeKind.LLAMACPP,
        model="/p/model.gguf",
        alias="pending",
        state=state,
        endpoint=endpoint,
    )
    with pytest.raises(ProfileResolutionError):
        resolve_extraction_profile(
            deployment="pending", models_config_path=models_config, deployments=[record]
        )


def test_unknown_deployment_selector_refuses(models_config: Path) -> None:
    with pytest.raises(ProfileResolutionError):
        resolve_extraction_profile(
            deployment="ghost", models_config_path=models_config, deployments=[]
        )


# ── (d) None -> studio_default loaded from models.yaml (no mislabel) ────────────


def test_default_resolves_studio_default_from_yaml(models_config: Path) -> None:
    profile = resolve_extraction_profile(models_config_path=models_config, deployments=[])
    assert profile.name == "studio_default"  # honest label, not "env_fallback"
    assert profile.model == "qwen3:4b"
    assert profile.base_url == "http://localhost:11434/v1"


# ── (e) model_profile still resolves a models.yaml profile ─────────────────────


def test_model_profile_resolves_yaml_profile(models_config: Path) -> None:
    profile = resolve_extraction_profile(
        model_profile="local_llamacpp", models_config_path=models_config, deployments=[]
    )
    assert profile.name == "local_llamacpp"
    assert profile.base_url == "http://llm-llamacpp:8000/v1"


def test_model_profile_can_name_a_live_deployment(models_config: Path) -> None:
    record = _record(
        name="my-dep", runtime=RuntimeKind.LLAMACPP, model="/p/m.gguf", alias="my-dep"
    )
    profile = resolve_extraction_profile(
        model_profile="my-dep", models_config_path=models_config, deployments=[record]
    )
    assert profile.name == "my-dep"
    assert profile.base_url == "http://127.0.0.1:8088/v1"


def test_unknown_model_profile_refuses(models_config: Path) -> None:
    with pytest.raises(ProfileResolutionError):
        resolve_extraction_profile(
            model_profile="nope", models_config_path=models_config, deployments=[]
        )


# ── (f) env fallback ONLY when models.yaml is absent, labeled "env_fallback" ────


def test_env_fallback_only_when_yaml_absent(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"
    profile = resolve_extraction_profile(models_config_path=missing, deployments=[])
    assert profile.name == "env_fallback"  # honest label, never a config profile name


def test_default_missing_from_present_yaml_refuses(tmp_path: Path) -> None:
    # models.yaml exists but lacks studio_default -> misconfig, refuse (never env).
    path = tmp_path / "models.yaml"
    path.write_text(
        "profiles:\n  other:\n    model: m\n    base_url: http://x/v1\n    api_key: k\n",
        encoding="utf-8",
    )
    with pytest.raises(ProfileResolutionError):
        resolve_extraction_profile(models_config_path=path, deployments=[])


# ── (g) vision / template traits inherited ─────────────────────────────────────


def test_vision_inherited_from_matching_yaml_profile(models_config: Path) -> None:
    # A deployment whose served id (alias) == the nuextract3 profile's upstream id
    # inherits that profile's vision + template style.
    record = _record(
        name="nux-live", runtime=RuntimeKind.LLAMACPP, model="/p/nux.gguf", alias="nuextract3"
    )
    profile = resolve_extraction_profile(
        deployment="nux-live", models_config_path=models_config, deployments=[record]
    )
    assert profile.name == "nux-live"  # honest deployment label
    assert profile.model == "nuextract3"
    assert profile.vision is True
    assert profile.response_format_style == "nuextract3"
    assert profile.prompt_profile == "nuextract3"


def test_no_match_falls_back_to_conservative_defaults(models_config: Path) -> None:
    # No yaml match, no DB catalog (unavailable in unit tests) -> safe defaults.
    record = _record(
        name="mystery", runtime=RuntimeKind.LLAMACPP, model="/p/x.gguf", alias="mystery"
    )
    profile = resolve_extraction_profile(
        deployment="mystery", models_config_path=models_config, deployments=[record]
    )
    assert profile.vision is False
    assert profile.response_format_style == "openai_json_schema"
    assert profile.prompt_profile == "strict_extraction_v1"


# ── precedence: deployment wins over model_profile ─────────────────────────────


def test_deployment_selector_wins_over_model_profile(models_config: Path) -> None:
    record = _record(
        name="win", runtime=RuntimeKind.LLAMACPP, model="/p/w.gguf", alias="win"
    )
    profile = resolve_extraction_profile(
        deployment="win",
        model_profile="local_llamacpp",
        models_config_path=models_config,
        deployments=[record],
    )
    assert profile.name == "win"
    assert profile.base_url == "http://127.0.0.1:8088/v1"


# ── wrappers: functions._resolve_profile (worker) & api.resolve_profile ─────────
#
# The wrappers call the DEFAULT deployment source; monkeypatch it so these stay
# pure unit tests against a fabricated registry. Both use the repo's real
# configs/models.yaml (present at CWD), which defines studio_default + a live
# nuextract3 profile the injected deployment matches.


@pytest.fixture
def inject_deployments(monkeypatch: pytest.MonkeyPatch):
    from docie_bench.serving import profile_resolver

    def _install(records: list[DeploymentRecord]) -> None:
        monkeypatch.setattr(
            profile_resolver, "_default_live_deployments", lambda: list(records)
        )

    return _install


def test_functions_resolve_deployment_routes_and_unknown_raises(inject_deployments) -> None:
    from docie_bench.inngest import functions

    record = _record(
        name="live-dep", runtime=RuntimeKind.LLAMACPP, model="/p/m.gguf", alias="live-dep"
    )
    inject_deployments([record])

    routed = functions._resolve_profile(deployment="live-dep")
    assert routed.name == "live-dep"
    assert routed.base_url == "http://127.0.0.1:8088/v1"

    # Precedence: deployment wins over a model_profile.
    both = functions._resolve_profile(deployment="live-dep", model_profile="studio_default")
    assert both.name == "live-dep"

    # Unknown/not-ready explicit selector now RAISES (surfaced on the error topic).
    with pytest.raises(ProfileResolutionError):
        functions._resolve_profile(deployment="not-ready")


def test_functions_resolve_default_is_studio_default(inject_deployments) -> None:
    from docie_bench.inngest import functions

    inject_deployments([])
    profile = functions._resolve_profile()
    assert profile.name == "studio_default"  # honest label, not env_fallback


def test_api_resolve_profile_400s_on_unknown(inject_deployments) -> None:
    from fastapi import HTTPException

    from docie_bench import api

    inject_deployments([])
    with pytest.raises(HTTPException) as exc:
        api.resolve_profile("does-not-exist")
    assert exc.value.status_code == 400


def test_api_resolve_profile_none_is_studio_default(inject_deployments) -> None:
    from docie_bench import api

    inject_deployments([])
    profile = api.resolve_profile(None)
    assert profile.name == "studio_default"
