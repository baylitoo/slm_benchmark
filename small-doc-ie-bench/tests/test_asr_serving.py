"""Managed ASR runtime, store, routing, and control-plane coverage."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import docie_bench.asr.server as asr_server
from docie_bench.asr.backend import ASRModelLoadError
from docie_bench.asr.models import (
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionSegment,
)
from docie_bench.asr.routing import ASRRoutingError, resolve_asr_route
from docie_bench.asr.server import create_asr_app
from docie_bench.serving.arch_registry import resolve_family
from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor
from docie_bench.serving.hf_hub import HfGgufFile, _artifact_options, _is_snapshot_file
from docie_bench.serving.model_store import FAMILIES, ModelStore, ModelStoreError
from docie_bench.serving.runtime import (
    ASRRuntime,
    HealthResult,
    LifecycleState,
    RuntimeKind,
    RuntimeLaunchSpec,
    RuntimeProcess,
    default_runtime_adapters,
)
from docie_bench.serving.supervisor import (
    DeploymentRecord,
    DeploymentSpec,
    PersistentSupervisor,
)


def _wav(payload: bytes = b"audio") -> bytes:
    size = (36 + len(payload)).to_bytes(4, "little")
    return b"RIFF" + size + b"WAVEfmt " + b"\x00" * 24 + b"data" + payload


class FakeBackend:
    model_name = "snapshot"
    backend_name = "fake"

    def transcribe(
        self, audio_path: Path, options: TranscriptionOptions
    ) -> TranscriptionResult:
        assert audio_path.is_file()
        return TranscriptionResult(
            text="managed speech",
            language=options.language or "en",
            duration=1.0,
            model=self.model_name,
            backend=self.backend_name,
            segments=(TranscriptionSegment(id=0, start=0, end=1, text="managed speech"),),
        )


def _record(
    name: str = "speech-one",
    *,
    alias: str = "whisper-small",
    state: LifecycleState = LifecycleState.READY,
    endpoint: str | None = "http://serving:8093/v1",
) -> DeploymentRecord:
    return DeploymentRecord(
        spec=DeploymentSpec(
            name=name,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.ASR,
                model="/models/whisper/snapshot",
                alias=alias,
                port=8093,
            ),
        ),
        state=state,
        endpoint=endpoint,
    )


def test_internal_runtime_is_model_pinned_and_openai_compatible() -> None:
    with TestClient(
        create_asr_app(model_id="/snapshot", alias="whisper-small", backend=FakeBackend())
    ) as client:
        health = client.get("/healthz")
        assert health.json() == {"status": "ok", "model": "whisper-small", "kind": "asr"}
        unknown = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", _wav(), "audio/wav")},
            data={"model": "download-something"},
        )
        assert unknown.status_code == 404
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", _wav(), "audio/wav")},
            data={"model": "whisper-small", "language": "fr", "response_format": "json"},
        )
    assert response.status_code == 200
    assert response.json() == {"text": "managed speech"}


def test_internal_runtime_refuses_readiness_when_model_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenBackend:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def load(self) -> None:
            raise ASRModelLoadError("broken snapshot")

    monkeypatch.setattr(asr_server, "FasterWhisperBackend", BrokenBackend)
    with (
        pytest.raises(ASRModelLoadError, match="broken snapshot"),
        TestClient(create_asr_app(model_id="/broken", alias="broken")),
    ):
        pass


def test_asr_route_accepts_live_name_or_unique_alias() -> None:
    record = _record()
    assert resolve_asr_route("speech-one", deployments=[record]).model == "whisper-small"
    route = resolve_asr_route("whisper-small", deployments=[record])
    assert route.base_url == "http://serving:8093/v1"


def test_asr_route_refuses_cold_wrong_kind_and_ambiguous_alias() -> None:
    cold = _record(state=LifecycleState.STOPPED, endpoint=None)
    with pytest.raises(ASRRoutingError, match="not ready") as not_ready:
        resolve_asr_route("speech-one", deployments=[cold])
    assert not_ready.value.status_code == 503
    llm = DeploymentRecord(
        spec=DeploymentSpec(
            name="llm",
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP, model="/m.gguf", alias="whisper-small"
            ),
        ),
        state=LifecycleState.READY,
        endpoint="http://serving:8088/v1",
    )
    with pytest.raises(ASRRoutingError, match="Unknown ASR"):
        resolve_asr_route("whisper-small", deployments=[llm])
    with pytest.raises(ASRRoutingError, match="ambiguous") as ambiguous:
        resolve_asr_route(
            "whisper-small",
            deployments=[_record("one"), _record("two")],
        )
    assert ambiguous.value.status_code == 409


def test_asr_runtime_command_capability_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object, **kwargs: object) -> object | None:
        if name == "faster_whisper":
            return object()
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    seen: list[str] = []
    adapter = ASRRuntime(
        which=lambda name: None,
        health_get=lambda url, timeout, headers: (
            seen.append(url) or HealthResult(True, 200)
        ),
    )
    spec = RuntimeLaunchSpec(
        runtime=RuntimeKind.ASR,
        model="C:/models/whisper/snapshot",
        alias="whisper-small",
        port=8093,
        device="cpu",
        dtype="int8",
        cpu_threads=6,
        extra_args=("--beam-size", "3", "--no-vad-filter"),
    )
    command = adapter.build_command(spec)
    assert command[:3] == (sys.executable, "-m", "docie_bench.serving.cli")
    assert "asr-runtime" in command
    assert "--compute-type" in command
    assert command[-2:] == ("3", "--no-vad-filter")
    assert adapter.probe(spec).compatible is True
    assert adapter.health(spec).healthy is True
    assert seen == ["http://127.0.0.1:8093/healthz"]
    assert isinstance(default_runtime_adapters()[RuntimeKind.ASR], ASRRuntime)


def test_asr_runtime_probe_fails_before_spawn_without_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "faster_whisper" else object(),
    )
    spec = RuntimeLaunchSpec(runtime=RuntimeKind.ASR, model="/snapshot", alias="speech")
    capabilities = ASRRuntime(which=lambda _name: None).probe(spec)
    assert capabilities.installed is True
    assert capabilities.compatible is False
    assert "[asr]" in capabilities.reasons[-1]


def test_whisper_family_requires_ctranslate2_artifact() -> None:
    original = resolve_family(
        "whisper",
        has_gguf=False,
        has_safetensors=True,
        has_mmproj=False,
        pipeline_tag="automatic-speech-recognition",
    )
    assert original.verdict == "needs_family"
    converted = resolve_family(
        "whisper",
        has_gguf=False,
        has_safetensors=False,
        has_mmproj=False,
        has_ctranslate2=True,
        pipeline_tag="automatic-speech-recognition",
    )
    assert converted.verdict == "supported"
    assert converted.family == "asr_whisper"
    assert FAMILIES["asr_whisper"].snapshot is True


def test_ctranslate2_snapshot_has_download_and_ram_estimates() -> None:
    assert _is_snapshot_file("model.bin") is True
    assert _is_snapshot_file("pytorch_model.bin") is False
    options, _quant = _artifact_options(
        ggufs=[],
        snapshot_files=[
            HfGgufFile("model.bin", 100_000_000, None, False, False),
            HfGgufFile("config.json", 1_000, None, False, False),
        ],
        include_mmproj=False,
        context_length=4096,
    )
    assert options[0]["download_size_bytes"] == 100_001_000
    assert options[0]["estimated_ram_bytes"] > options[0]["download_size_bytes"]


def test_asr_store_accepts_only_ctranslate2_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"weights")
    (source / "config.json").write_text("{}")
    store = ModelStore(tmp_path / "store")
    entry = store.add_snapshot(
        name="whisper-small", family="asr_whisper", snapshot_dir=source
    )
    assert entry.model_path.joinpath("model.bin").read_bytes() == b"weights"

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "config.json").write_text("{}")
    with pytest.raises(ModelStoreError, match="compatible weights"):
        store.add_snapshot(name="broken", family="asr_whisper", snapshot_dir=invalid)


class CapturingAdapter:
    def __init__(self) -> None:
        self.specs: list[RuntimeLaunchSpec] = []

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> RuntimeProcess:
        del log_path
        self.specs.append(spec)
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", 4455)

    def is_running(self, pid: int | None) -> bool:
        return pid == 4455

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del pid, timeout

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> HealthResult:
        del spec, timeout
        return HealthResult(True, 200)


def test_store_deploy_selects_asr_runtime_and_local_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"weights")
    root = tmp_path / "models"
    ModelStore(root).add_snapshot(
        name="whisper-small", family="asr_whisper", snapshot_dir=source
    )
    adapter = CapturingAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "deployments.json", adapters={RuntimeKind.ASR: adapter}  # type: ignore[dict-item]
    )
    plane = ControlPlane(
        None,
        None,
        _DefaultSupervisor(supervisor, planner=None, model_store_root=root),
        None,
    )  # type: ignore[arg-type]
    record = asyncio.run(plane.up("whisper-small", port=8093))
    assert record["state"] == "ready"
    launch = adapter.specs[-1]
    assert launch.runtime == RuntimeKind.ASR
    assert launch.model.endswith("whisper-small/snapshot")
    assert launch.alias == "whisper-small"
    assert launch.device == "cpu"
    assert launch.dtype == "int8"
