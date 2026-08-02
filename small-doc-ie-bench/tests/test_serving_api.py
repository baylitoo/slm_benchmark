"""PR-B: the record-derived /v1/serving/ports admin view.

Pure unit test over the endpoint function: seed ``deployments.json`` in a temp
serving home (the shared on-disk state the api reads), point the settings there,
and assert the shape + that ``recommended_next`` equals the SAME
``PortAllocator.recommend`` the worker uses. No sockets, no worker, no DB.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from docie_bench.inngest.serving_api import deployment_logs, list_store, serving_ports
from docie_bench.serving.control_plane import PortAllocator
from docie_bench.serving.runtime import RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentSpec, PersistentSupervisor
from docie_bench.settings import get_settings


class _FakeAdapter:
    def __init__(self) -> None:
        self.next_pid = 500

    def start(self, spec: RuntimeLaunchSpec, *, log_path: Path | None = None) -> object:
        del log_path
        from docie_bench.serving.runtime import RuntimeProcess

        self.next_pid += 1
        return RuntimeProcess(spec.runtime, f"http://{spec.host}:{spec.port}/v1", self.next_pid)

    def is_running(self, pid: int | None) -> bool:
        return pid is not None

    def shutdown(self, pid: int | None, *, timeout: float = 10) -> None:
        del pid, timeout

    def health(self, spec: RuntimeLaunchSpec, *, timeout: float = 2) -> object:
        del spec, timeout
        from docie_bench.serving.runtime import HealthResult

        return HealthResult(True, 200)


def _seed_deployments(home: Path, ports: dict[str, int]) -> None:
    supervisor = PersistentSupervisor(
        home / "deployments.json", adapters={RuntimeKind.LLAMACPP: _FakeAdapter()}
    )
    for name, port in ports.items():
        supervisor.deploy(
            DeploymentSpec(
                name=name,
                launch=RuntimeLaunchSpec(
                    runtime=RuntimeKind.LLAMACPP,
                    model=f"/models/{name}.gguf",
                    alias=name,
                    port=port,
                ),
            )
        )


@pytest.fixture
def serving_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "serving"
    home.mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_START", "8088")
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_END", "8188")
    get_settings.cache_clear()
    try:
        yield home
    finally:
        get_settings.cache_clear()


def test_ports_endpoint_shape(serving_home: Path) -> None:
    _seed_deployments(serving_home, {"qwen": 8088, "nux": 8089})

    payload = asyncio.run(serving_ports())

    assert payload["range"] == {"start": 8088, "end": 8188}
    assert payload["used"] == [8088, 8089]
    by_port = {d["port"]: d for d in payload["deployments"]}
    assert set(by_port) == {8088, 8089}
    assert by_port[8088]["name"] == "qwen"
    assert 8090 not in payload["used"]

    # recommended_next is the SAME function the worker's allocate() derives from.
    allocator = PortAllocator(range_start=8088, range_end=8188)
    expected = allocator.recommend(bind_host="127.0.0.1", reserved=set(payload["used"]))
    assert payload["recommended_next"] == expected == 8090


def test_ports_recommended_excludes_used(serving_home: Path) -> None:
    _seed_deployments(serving_home, {"a": 8088, "b": 8090})

    payload = asyncio.run(serving_ports())

    assert payload["recommended_next"] not in payload["used"]
    # 8089 is the lowest record-free port (8088 used, 8090 used).
    assert payload["recommended_next"] == 8089
    assert all(port not in payload["used"] for port in payload["free_sample"])


def test_ports_empty_when_no_deployments(serving_home: Path) -> None:
    payload = asyncio.run(serving_ports())

    assert payload["used"] == []
    assert payload["deployments"] == []
    assert payload["recommended_next"] == 8088  # first pick unchanged for a single deploy


def test_deployment_logs_tail(serving_home: Path) -> None:
    _seed_deployments(serving_home, {"qwen": 8088})
    logs_dir = serving_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "qwen.log").write_text("line1\nline2\nline3\n", encoding="utf-8")

    payload = asyncio.run(deployment_logs("qwen", lines=2))

    assert payload["name"] == "qwen"
    assert payload["lines"] == ["line2", "line3"]


def test_deployment_logs_missing_file_is_empty(serving_home: Path) -> None:
    _seed_deployments(serving_home, {"qwen": 8088})
    payload = asyncio.run(deployment_logs("qwen"))
    assert payload["lines"] == []


def test_deployment_logs_rejects_path_traversal(serving_home: Path) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        asyncio.run(deployment_logs("../../etc/passwd"))
    assert exc.value.status_code in (400, 404)


def test_store_reads_on_disk_without_a_catalog(serving_home: Path) -> None:
    """A model seeded WITHOUT DATABASE_URL still shows in Models — the store is
    read from the on-disk index, not only the Postgres catalog (the desync
    this closes). Covers a GGUF and an encoder snapshot (directory entry)."""
    import docie_bench.storage.db as db
    from docie_bench.serving.model_store import ModelStore

    db.dispose_engine()  # no catalog configured
    models_root = serving_home / "models"
    store = ModelStore(models_root)
    gguf = serving_home / "m.gguf"
    gguf.write_bytes(b"GGUF-weights")
    store.add_gguf(name="lfm2.5-350m", family="lfm2", model_gguf=gguf)
    snap = serving_home / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"w")
    (snap / "config.json").write_text("{}")
    store.add_snapshot(name="guardrails-pii", family="encoder_gliner2", snapshot_dir=snap)

    view = asyncio.run(list_store())

    by_name = {e["name"]: e for e in view}
    assert set(by_name) == {"lfm2.5-350m", "guardrails-pii"}
    assert by_name["guardrails-pii"]["analyzer"] is True
    assert by_name["guardrails-pii"]["available_backends"] == ["encoder"]
    assert by_name["guardrails-pii"]["size_bytes"] == len(b"w") + len("{}")
    assert "model_path" not in by_name["guardrails-pii"]
    assert by_name["lfm2.5-350m"]["family"] == "lfm2"
