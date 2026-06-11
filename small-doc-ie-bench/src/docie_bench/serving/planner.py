from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from docie_bench.serving.registry import BenchmarkRecord, ModelManifest


class RuntimeName(StrEnum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OLLAMA = "ollama"
    REMOTE = "remote"


class HostResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cpu_cores: int = Field(ge=1)
    memory_gb: float = Field(gt=0)
    disk_gb: float = Field(ge=0)
    gpu_count: int = Field(default=0, ge=0)
    gpu_memory_gb: float = Field(default=0, ge=0)
    architecture: str = "x86_64"
    available_runtimes: frozenset[RuntimeName] = Field(
        default_factory=lambda: frozenset(
            {RuntimeName.VLLM, RuntimeName.LLAMACPP, RuntimeName.OLLAMA}
        )
    )
    remote_endpoints: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_gpu(self) -> HostResources:
        if self.gpu_count == 0 and self.gpu_memory_gb:
            raise ValueError("gpu_memory_gb requires at least one GPU")
        return self


class PlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelManifest
    resources: HostResources
    context_length: int | None = Field(default=None, ge=1)
    concurrency: int = Field(default=1, ge=1)
    required_features: frozenset[str] = Field(default_factory=frozenset)
    preferred_runtime: RuntimeName | None = None
    remote_endpoint: str | None = None


class RuntimePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: RuntimeName
    compatible: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    estimated_memory_gb: float = Field(ge=0)
    required_disk_gb: float = Field(ge=0)
    score: float | None = None
    endpoint: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)


class RuntimeRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    selected: RuntimePlan | None
    candidates: tuple[RuntimePlan, ...]
    explanation: str

    @property
    def runtime(self) -> RuntimeName | None:
        return self.selected.runtime if self.selected else None


class ResourcePlanner:
    """Conservative runtime compatibility and resource recommendation planner."""

    _FEATURES: dict[RuntimeName, frozenset[str]] = {
        RuntimeName.VLLM: frozenset(
            {
                "generation",
                "structured_output",
                "vision",
                "embeddings",
                "tool_calls",
                "lora",
                "batching",
            }
        ),
        RuntimeName.LLAMACPP: frozenset(
            {"generation", "structured_output", "vision", "embeddings", "tool_calls", "lora"}
        ),
        RuntimeName.OLLAMA: frozenset(
            {"generation", "structured_output", "vision", "embeddings", "tool_calls"}
        ),
        RuntimeName.REMOTE: frozenset(
            {
                "generation",
                "structured_output",
                "vision",
                "embeddings",
                "tool_calls",
                "lora",
                "batching",
            }
        ),
    }

    def plan(self, request: PlanningRequest) -> RuntimeRecommendation:
        return self.recommend(request)

    def recommend(self, request: PlanningRequest) -> RuntimeRecommendation:
        plans = tuple(self.evaluate(request, runtime) for runtime in RuntimeName)
        compatible = [plan for plan in plans if plan.compatible]
        compatible.sort(key=lambda plan: (-float(plan.score or 0), plan.runtime.value))
        selected = compatible[0] if compatible else None
        if selected is None:
            explanation = (
                f"No compatible runtime for {request.model.model_id}; inspect candidate reasons "
                "before launch."
            )
        else:
            detail = selected.reasons[-1] if selected.reasons else "compatibility checks passed"
            explanation = (
                f"Recommend {selected.runtime.value} for {request.model.model_id}: {detail}; "
                f"ranking score {selected.score:.2f}."
            )
        return RuntimeRecommendation(
            model_id=request.model.model_id,
            selected=selected,
            candidates=plans,
            explanation=explanation,
        )

    def evaluate(self, request: PlanningRequest, runtime: RuntimeName) -> RuntimePlan:
        model = request.model
        resources = request.resources
        reasons: list[str] = []
        failures: list[str] = []
        warnings: list[str] = []
        endpoint: str | None = None
        configuration: dict[str, Any] = {}

        explicit = model.runtime_compatibility.get(runtime.value)
        if explicit is not None:
            reasons.append(explicit.reason)
            if not explicit.compatible:
                return self._incompatible(request, runtime, reasons)

        unsupported = sorted(request.required_features - self._FEATURES[runtime])
        if unsupported:
            failures.append(f"runtime does not support required features: {', '.join(unsupported)}")

        context_length = request.context_length or model.context_length
        if model.context_length and context_length and context_length > model.context_length:
            failures.append(
                f"requested context {context_length} exceeds model limit {model.context_length}"
            )

        memory = self._estimate_memory_gb(request, runtime)
        disk = model.required_disk_gb or self._artifact_disk_gb(model)

        if runtime == RuntimeName.REMOTE:
            endpoint = request.remote_endpoint or resources.remote_endpoints.get(model.model_id)
            endpoint = endpoint or resources.remote_endpoints.get("default")
            if endpoint is None:
                failures.append("no remote endpoint is configured")
            else:
                reasons.append("remote endpoint avoids local model memory and disk requirements")
            memory = 0
            disk = 0
            configuration["endpoint"] = endpoint
        else:
            if runtime not in resources.available_runtimes:
                failures.append(f"{runtime.value} is not installed on the target host")
            if disk > resources.disk_gb:
                failures.append(
                    f"requires {disk:.2f} GB disk but only {resources.disk_gb:.2f} GB is available"
                )
            self._runtime_checks(request, runtime, memory, failures, warnings, configuration)

        reasons.extend(failures)
        if failures:
            return RuntimePlan(
                runtime=runtime,
                compatible=False,
                reasons=tuple(reasons),
                warnings=tuple(warnings),
                estimated_memory_gb=memory,
                required_disk_gb=disk,
                endpoint=endpoint,
                configuration=configuration,
            )

        reasons.append("resource and compatibility checks passed")
        score = self._score(request, runtime)
        if request.preferred_runtime == runtime:
            score += 10
            reasons.append("requested preferred runtime")
        return RuntimePlan(
            runtime=runtime,
            compatible=True,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            estimated_memory_gb=memory,
            required_disk_gb=disk,
            score=round(score, 4),
            endpoint=endpoint,
            configuration=configuration,
        )

    def _runtime_checks(
        self,
        request: PlanningRequest,
        runtime: RuntimeName,
        memory: float,
        failures: list[str],
        warnings: list[str],
        configuration: dict[str, Any],
    ) -> None:
        model = request.model
        resources = request.resources
        quantization = (model.quantization or "").lower()
        has_gguf = any(artifact.name.lower().endswith(".gguf") for artifact in model.artifacts)

        if runtime == RuntimeName.VLLM:
            if has_gguf or quantization.startswith(("q2_", "q3_", "q4_", "q5_", "q6_")):
                failures.append(
                    "vLLM compatibility for this GGUF-style quantization is not verified"
                )
            if resources.gpu_count:
                configuration["device"] = "gpu"
                if memory > resources.gpu_memory_gb:
                    failures.append(
                        f"requires {memory:.2f} GB GPU memory but only "
                        f"{resources.gpu_memory_gb:.2f} GB is available"
                    )
            else:
                configuration["device"] = "cpu"
                if resources.architecture.lower() not in {"x86_64", "amd64"}:
                    failures.append("vLLM CPU support is not verified for this architecture")
                if memory > resources.memory_gb:
                    failures.append(
                        f"requires {memory:.2f} GB memory but only "
                        f"{resources.memory_gb:.2f} GB is available"
                    )
                warnings.append(
                    "vLLM CPU performance and quantization support require benchmarking"
                )
        elif runtime == RuntimeName.LLAMACPP:
            configuration["device"] = "gpu" if resources.gpu_count else "cpu"
            if not has_gguf:
                failures.append("llama.cpp requires a verified GGUF artifact")
            available = resources.memory_gb + resources.gpu_memory_gb
            if memory > available:
                failures.append(
                    f"requires {memory:.2f} GB combined memory but only "
                    f"{available:.2f} GB is available"
                )
        elif runtime == RuntimeName.OLLAMA:
            configuration["device"] = "gpu" if resources.gpu_count else "cpu"
            if memory > resources.memory_gb + resources.gpu_memory_gb:
                failures.append(
                    f"requires {memory:.2f} GB combined memory but only "
                    f"{resources.memory_gb + resources.gpu_memory_gb:.2f} GB is available"
                )
            warnings.append("Ollama runtime compatibility must be confirmed by an import probe")

    def _estimate_memory_gb(self, request: PlanningRequest, runtime: RuntimeName) -> float:
        if runtime == RuntimeName.REMOTE:
            return 0
        model = request.model
        weights = model.required_memory_gb or max(0.1, self._artifact_disk_gb(model) * 1.2)
        context = request.context_length or model.context_length or 4096
        kv_cache = 0.25 * request.concurrency * (context / 4096)
        overhead = 1.0 if runtime == RuntimeName.VLLM else 0.5
        return round(weights + kv_cache + overhead, 4)

    @staticmethod
    def _artifact_disk_gb(model: ModelManifest) -> float:
        return round(sum(artifact.size_bytes for artifact in model.artifacts) / (1024**3), 4)

    def _score(self, request: PlanningRequest, runtime: RuntimeName) -> float:
        resources = request.resources
        base: float = {
            RuntimeName.VLLM: 85 if resources.gpu_count else 62,
            RuntimeName.LLAMACPP: 75 if not resources.gpu_count else 68,
            RuntimeName.OLLAMA: 55,
            RuntimeName.REMOTE: 50,
        }[runtime]
        records = [
            record for record in request.model.benchmark_history if record.runtime == runtime
        ]
        if records:
            base += max(self._benchmark_score(record) for record in records)
        return float(base)

    @staticmethod
    def _benchmark_score(record: BenchmarkRecord) -> float:
        score = min(record.tokens_per_second or 0, 200) / 10
        score += (record.structured_output_validity or 0) * 10
        score -= min(record.p95_latency_ms or 0, 10_000) / 1000
        return score

    def _incompatible(
        self, request: PlanningRequest, runtime: RuntimeName, reasons: list[str]
    ) -> RuntimePlan:
        return RuntimePlan(
            runtime=runtime,
            compatible=False,
            reasons=tuple(reasons),
            estimated_memory_gb=self._estimate_memory_gb(request, runtime),
            required_disk_gb=request.model.required_disk_gb
            or self._artifact_disk_gb(request.model),
        )


RuntimePlanner = ResourcePlanner
ResourceSpec = HostResources


def recommend_runtime(request: PlanningRequest) -> RuntimeRecommendation:
    return ResourcePlanner().recommend(request)


def check_runtime_compatibility(request: PlanningRequest, runtime: RuntimeName) -> RuntimePlan:
    return ResourcePlanner().evaluate(request, runtime)
