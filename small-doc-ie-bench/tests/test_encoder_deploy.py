"""PR-2: serve_store_model routes analyzer snapshots to the encoder runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.control_plane import ControlPlane, _DefaultSupervisor
from docie_bench.serving.model_store import ModelStore
from docie_bench.serving.runtime import RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import PersistentSupervisor


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


class _CapturingAdapter:
    """A fake runtime adapter that records every launched RuntimeLaunchSpec."""

    def __init__(self) -> None:
        self.next_pid = 200
        self.running: set[int] = set()
        self.specs: list[RuntimeLaunchSpec] = []

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> Any:
        from docie_bench.serving.runtime import RuntimeProcess

        del log_path
        self.specs.append(spec)
        self.next_pid += 1
        self.running.add(self.next_pid)
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", self.next_pid)

    def is_running(self, pid: int | None) -> bool:
        return pid in self.running

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del timeout
        if pid is not None:
            self.running.discard(pid)

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2):
        from docie_bench.serving.runtime import HealthResult

        del spec, timeout
        return HealthResult(True, 200)


def _seed_encoder(root: Path) -> None:
    src = root.parent / "snap"
    src.mkdir(parents=True, exist_ok=True)
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "config.json").write_text("{}")
    ModelStore(root).add_snapshot(
        name="guardrails-pii",
        family="encoder_gliner2",
        snapshot_dir=src,
        source="hf:fastino/GLiNER2-Guardrails-PII-Multi",
    )


def _plane(tmp_path: Path, adapter: _CapturingAdapter) -> Any:
    root = tmp_path / "models"
    _seed_encoder(root)
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.ENCODER: adapter}
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None, model_store_root=root)
    return ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]


def test_encoder_snapshot_deploys_via_encoder_runtime(tmp_path: Path) -> None:
    adapter = _CapturingAdapter()
    plane = _plane(tmp_path, adapter)

    record = asyncio.run(plane.up("guardrails-pii", port=8090))

    assert record["state"] == "ready"
    spec = adapter.specs[-1]
    # Launched on the encoder runtime, with the snapshot DIRECTORY as the model
    # and the family's backend passed explicitly (no name-based guessing).
    assert spec.runtime == RuntimeKind.ENCODER
    assert spec.model.endswith("guardrails-pii/snapshot")
    assert Path(spec.model).is_dir()
    assert list(spec.extra_args) == ["--backend", "gliner2"]


def test_encoder_placement_engine_is_encoder(_sqlite_catalog: None, tmp_path: Path) -> None:
    from docie_bench.serving.catalog import ModelCatalog

    plane = _plane(tmp_path, _CapturingAdapter())
    asyncio.run(plane.up("guardrails-pii", port=8090))

    placement = ModelCatalog().get_placement("guardrails-pii")
    assert placement is not None
    assert placement["engine"] == "encoder"


def test_dense_embedding_snapshot_deploys_with_native_placement(
    _sqlite_catalog: None, tmp_path: Path,
) -> None:
    from docie_bench.serving.catalog import ModelCatalog, available_backends
    from docie_bench.serving.throughput import probe_applicability

    root = tmp_path / "models"
    snapshot = tmp_path / "embedding-snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "config.json").write_text('{"model_type":"embedding_gemma2"}')
    ModelStore(root).add_snapshot(
        name="embeddinggemma-2", family="embeddinggemma2", snapshot_dir=snapshot,
        source="hf:google/embeddinggemma-2",
    )
    adapter = _CapturingAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.DENSE_EMBEDDING: adapter},
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None, model_store_root=root)
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]
    record = asyncio.run(plane.up("embeddinggemma-2", port=8093))
    assert record["state"] == "ready"
    launch = adapter.specs[-1]
    assert launch.runtime == RuntimeKind.DENSE_EMBEDDING
    assert Path(launch.model).is_dir()
    assert launch.model.endswith("embeddinggemma-2/snapshot")
    assert launch.extra_args == ()
    assert available_backends("embeddinggemma2") == ["dense_embedding"]
    placement = ModelCatalog().get_placement("embeddinggemma-2")
    assert placement is not None and placement["engine"] == "dense_embedding"
    probeable, reason = probe_applicability(launch)
    assert not probeable and "/v1/embeddings" in reason


def test_embeddinggemma2_gguf_deploys_with_mean_pooling(tmp_path: Path) -> None:
    root = tmp_path / "models"
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    ModelStore(root).add_gguf(
        name="embeddinggemma-2-gguf", family="embedding", model_gguf=gguf,
        source="hf:ggml-org/embeddinggemma-2-GGUF",
    )
    adapter = _CapturingAdapter()
    supervisor = PersistentSupervisor(
        tmp_path / "state.json", adapters={RuntimeKind.LLAMACPP: adapter},
    )
    wrapper = _DefaultSupervisor(supervisor, planner=None, model_store_root=root)
    plane = ControlPlane(None, None, wrapper, None)  # type: ignore[arg-type]
    record = asyncio.run(plane.up("embeddinggemma-2-gguf", port=8093))
    assert record["state"] == "ready"
    launch = adapter.specs[-1]
    assert launch.runtime == RuntimeKind.LLAMACPP
    assert Path(launch.model).is_file()
    assert launch.extra_args == ("--embedding", "--pooling", "mean")
