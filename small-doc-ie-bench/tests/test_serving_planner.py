from __future__ import annotations

from docie_bench.serving.planner import (
    HostResources,
    PlanningRequest,
    ResourcePlanner,
    RuntimeName,
)
from docie_bench.serving.registry import (
    ArtifactManifest,
    BenchmarkRecord,
    ModelManifest,
    RuntimeCompatibilityRecord,
)


def _artifact(name: str = "model.gguf", size_gb: int = 4) -> ArtifactManifest:
    return ArtifactManifest(
        name=name,
        digest="a" * 64,
        size_bytes=size_gb * 1024**3,
    )


def _model(**overrides: object) -> ModelManifest:
    values: dict[str, object] = {
        "model_id": "local/model",
        "source": "local",
        "revision": "pinned-sha",
        "artifacts": (_artifact(),),
        "quantization": "Q4_K_M",
        "required_memory_gb": 5,
        "required_disk_gb": 4,
        "context_length": 8192,
        "supported_tasks": ("generation", "structured_output"),
    }
    values.update(overrides)
    return ModelManifest(**values)


def _cpu(**overrides: object) -> HostResources:
    values: dict[str, object] = {
        "cpu_cores": 16,
        "memory_gb": 16,
        "disk_gb": 100,
        "available_runtimes": frozenset(
            {RuntimeName.VLLM, RuntimeName.LLAMACPP, RuntimeName.OLLAMA}
        ),
    }
    values.update(overrides)
    return HostResources(**values)


def test_cpu_gguf_recommends_llamacpp_and_explains_vllm_incompatibility() -> None:
    recommendation = ResourcePlanner().recommend(
        PlanningRequest(
            model=_model(),
            resources=_cpu(),
            required_features=frozenset({"structured_output"}),
        )
    )

    assert recommendation.runtime == RuntimeName.LLAMACPP
    vllm = next(plan for plan in recommendation.candidates if plan.runtime == RuntimeName.VLLM)
    assert not vllm.compatible
    assert "quantization" in " ".join(vllm.reasons)
    assert "Recommend llamacpp" in recommendation.explanation


def test_gpu_safetensors_recommends_vllm() -> None:
    model = _model(
        artifacts=(_artifact("model.safetensors", 8),),
        quantization=None,
        required_memory_gb=10,
    )
    resources = _cpu(gpu_count=1, gpu_memory_gb=24)

    recommendation = ResourcePlanner().plan(PlanningRequest(model=model, resources=resources))

    assert recommendation.runtime == RuntimeName.VLLM
    assert recommendation.selected is not None
    assert recommendation.selected.configuration == {"device": "gpu"}


def test_planner_rejects_resource_feature_context_and_explicit_compatibility_failures() -> None:
    model = _model(
        required_memory_gb=20,
        runtime_compatibility={
            "ollama": RuntimeCompatibilityRecord(
                compatible=False,
                reason="model import probe failed",
            )
        },
    )
    request = PlanningRequest(
        model=model,
        resources=_cpu(memory_gb=8, disk_gb=2, available_runtimes=frozenset({RuntimeName.OLLAMA})),
        context_length=16_384,
        required_features=frozenset({"batching"}),
    )

    recommendation = ResourcePlanner().recommend(request)

    assert recommendation.selected is None
    assert "No compatible runtime" in recommendation.explanation
    reasons = {plan.runtime: " ".join(plan.reasons) for plan in recommendation.candidates}
    assert "exceeds model limit" in reasons[RuntimeName.VLLM]
    assert "does not support required features" in reasons[RuntimeName.LLAMACPP]
    assert "model import probe failed" in reasons[RuntimeName.OLLAMA]
    assert "no remote endpoint" in reasons[RuntimeName.REMOTE]


def test_remote_endpoint_is_compatible_without_local_resources() -> None:
    request = PlanningRequest(
        model=_model(required_memory_gb=100, required_disk_gb=100),
        resources=_cpu(
            memory_gb=1,
            disk_gb=0,
            available_runtimes=frozenset(),
            remote_endpoints={"local/model": "https://models.example/v1"},
        ),
        preferred_runtime=RuntimeName.REMOTE,
    )

    recommendation = ResourcePlanner().recommend(request)

    assert recommendation.runtime == RuntimeName.REMOTE
    assert recommendation.selected is not None
    assert recommendation.selected.endpoint == "https://models.example/v1"
    assert recommendation.selected.estimated_memory_gb == 0
    assert recommendation.selected.required_disk_gb == 0


def test_benchmark_history_changes_compatible_runtime_ranking() -> None:
    model = _model(
        artifacts=(_artifact("model.safetensors"),),
        quantization=None,
        benchmark_history=(
            BenchmarkRecord(
                runtime="ollama",
                tokens_per_second=200,
                p95_latency_ms=100,
                structured_output_validity=1,
            ),
        ),
    )

    recommendation = ResourcePlanner().recommend(PlanningRequest(model=model, resources=_cpu()))

    assert recommendation.runtime == RuntimeName.OLLAMA


# ── one footprint math: planner == the canonical resources formula ─────────────


def test_planner_memory_is_the_canonical_footprint_formula() -> None:
    """The planner previously carried a second, divergent formula (x1.2
    weights, no mmproj, per-runtime overhead guesses) and could approve a
    deploy the fit gate rejects. It must now price EXACTLY what
    resources.predict_footprint_bytes prices."""
    from docie_bench.serving.resources import predict_footprint_bytes

    gib = 1024**3
    model = _model(
        artifacts=(_artifact(), _artifact(name="mmproj-f16.gguf", size_gb=1)),
        required_memory_gb=5,
        context_length=8192,
    )
    request = PlanningRequest(model=model, resources=_cpu(), concurrency=2)

    plan = ResourcePlanner().evaluate(request, RuntimeName.LLAMACPP)

    expected = predict_footprint_bytes(
        5 * gib, context_length=8192, n_parallel=2, mmproj_bytes=1 * gib
    )
    assert plan.estimated_memory_gb == round(expected / gib, 4)


def test_planner_holds_back_the_safety_margin() -> None:
    """A model that fits in raw RAM but NOT under the configured margin must be
    refused — the Sizing tab and the fit gate hold the margin back, and the
    planner approving what they reject caused real OOMs."""
    from docie_bench.serving.sizing import safety_margin_bytes
    from docie_bench.settings import get_settings

    gib = 1024**3
    planner = ResourcePlanner()
    resources = _cpu(memory_gb=16, gpu_count=0)
    margin_gb = (
        safety_margin_bytes(16 * gib, get_settings().serving_sizing_margin_fraction) / gib
    )
    headroom = 16 - margin_gb

    fits = planner.evaluate(
        PlanningRequest(model=_model(required_memory_gb=2), resources=resources),
        RuntimeName.LLAMACPP,
    )
    over = planner.evaluate(
        PlanningRequest(
            model=_model(required_memory_gb=int(headroom)), resources=resources
        ),
        RuntimeName.LLAMACPP,
    )

    assert fits.compatible is True
    assert over.compatible is False
    assert any("safety margin" in reason for reason in over.reasons)


def test_planner_context_defaults_to_the_deploy_default() -> None:
    """No context anywhere => price at the deploy default (8192), the same
    constant every deploy surface applies — not llama-server's 4096 fallback."""
    from docie_bench.serving.resources import (
        DEFAULT_DEPLOY_CONTEXT_LENGTH,
        predict_footprint_bytes,
    )

    gib = 1024**3
    model = _model(context_length=None)
    plan = ResourcePlanner().evaluate(
        PlanningRequest(model=model, resources=_cpu()), RuntimeName.LLAMACPP
    )
    expected = predict_footprint_bytes(
        5 * gib, context_length=DEFAULT_DEPLOY_CONTEXT_LENGTH
    )
    assert plan.estimated_memory_gb == round(expected / gib, 4)
