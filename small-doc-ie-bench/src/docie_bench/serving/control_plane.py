"""Control-plane facade for model-serving operations.

The facade deliberately depends on small protocols. Runtime, registry, supervisor,
and planner implementations can evolve independently while callers use one stable
operations API.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
from collections.abc import Awaitable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar("T")
Result = object | Awaitable[object]


class Registry(Protocol):
    def list_models(self) -> Result: ...

    def get_model(self, name: str) -> Result: ...

    def pull_model(
        self,
        name: str,
        *,
        runtime: str | None,
        revision: str | None,
        trust_remote_code: bool,
    ) -> Result: ...

    def remove_model(self, name: str) -> Result: ...


class RuntimeCatalog(Protocol):
    def list_runtimes(self) -> Result: ...

    def probe_runtime(self, name: str) -> Result: ...


class Supervisor(Protocol):
    def list_deployments(self) -> Result: ...

    def deployment_status(self, name: str) -> Result: ...

    def serve(
        self,
        model: str,
        *,
        name: str | None,
        runtime: str | None,
        replicas: int,
    ) -> Result: ...

    def serve_store_model(
        self,
        name: str,
        *,
        port: int,
        context_length: int,
    ) -> Result: ...

    def start(self, name: str) -> Result: ...

    def stop(self, name: str) -> Result: ...


class Planner(Protocol):
    def plan(self, model: str, *, runtime: str | None, replicas: int) -> Result: ...


@dataclass(frozen=True)
class ControlPlane:
    """Coordinate registry, runtime, deployment, and planning operations."""

    registry: Registry
    runtimes: RuntimeCatalog
    supervisor: Supervisor
    planner: Planner

    @classmethod
    def from_defaults(cls) -> ControlPlane:
        """Build the local control plane from the serving implementation modules."""
        import psutil

        from docie_bench.serving.planner import HostResources, ResourcePlanner, RuntimeName
        from docie_bench.serving.registry import ModelRegistry
        from docie_bench.serving.runtime import default_runtime_adapters
        from docie_bench.serving.supervisor import PersistentSupervisor

        home = Path(
            os.environ.get(
                "DOCIE_SERVING_HOME",
                Path.home() / ".local" / "share" / "docie-bench" / "serving",
            )
        )
        registry = _DefaultRegistry(ModelRegistry(home / "registry"))
        runtimes = _DefaultRuntimes(default_runtime_adapters())
        planner = _DefaultPlanner(
            ResourcePlanner(),
            registry.backend,
            HostResources(
                cpu_cores=psutil.cpu_count(logical=True) or 1,
                memory_gb=round(psutil.virtual_memory().available / (1024**3), 4),
                disk_gb=round(shutil.disk_usage(home.parent).free / (1024**3), 4),
                available_runtimes=frozenset(
                    RuntimeName(name) for name in runtimes.available_names()
                ),
            ),
        )
        supervisor = _DefaultSupervisor(
            PersistentSupervisor(home / "deployments.json"),
            planner,
            model_store_root=home / "models",
        )
        return cls(
            registry=cast(Registry, registry),
            runtimes=cast(RuntimeCatalog, runtimes),
            supervisor=cast(Supervisor, supervisor),
            planner=cast(Planner, planner),
        )

    async def list_models(self) -> object:
        return to_data(await _resolve(self.registry.list_models()))

    async def show_model(self, name: str) -> object:
        return to_data(await _resolve(self.registry.get_model(_required(name, "model"))))

    async def pull_model(
        self,
        name: str,
        *,
        runtime: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> object:
        return to_data(
            await _resolve(
                self.registry.pull_model(
                    _required(name, "model"),
                    runtime=_optional(runtime),
                    revision=_optional(revision),
                    trust_remote_code=trust_remote_code,
                )
            )
        )

    async def remove_model(self, name: str) -> object:
        return to_data(await _resolve(self.registry.remove_model(_required(name, "model"))))

    async def list_runtimes(self) -> object:
        return to_data(await _resolve(self.runtimes.list_runtimes()))

    async def probe_runtime(self, name: str) -> object:
        return to_data(await _resolve(self.runtimes.probe_runtime(_required(name, "runtime"))))

    async def list_deployments(self) -> object:
        return to_data(await _resolve(self.supervisor.list_deployments()))

    async def deployment_status(self, name: str) -> object:
        return to_data(
            await _resolve(self.supervisor.deployment_status(_required(name, "deployment")))
        )

    async def serve(
        self,
        model: str,
        *,
        name: str | None = None,
        runtime: str | None = None,
        replicas: int = 1,
    ) -> object:
        return to_data(
            await _resolve(
                self.supervisor.serve(
                    _required(model, "model"),
                    name=_optional(name),
                    runtime=_optional(runtime),
                    replicas=_replicas(replicas),
                )
            )
        )

    async def up(
        self,
        name: str,
        *,
        port: int = 8088,
        context_length: int = 8192,
    ) -> object:
        # serve_store_model is synchronous and now blocks in await_ready() (a
        # bounded time.sleep poll until the model is serving). Run it in a thread
        # so it does not stall the worker's asyncio loop — the Inngest Connect
        # heartbeats and any concurrent realtime/extraction steps must keep
        # flowing on the scale-1 worker while a large GGUF loads.
        return to_data(
            await asyncio.to_thread(
                self.supervisor.serve_store_model,
                _required(name, "model"),
                port=port,
                context_length=context_length,
            )
        )

    async def start(self, name: str) -> object:
        return to_data(await _resolve(self.supervisor.start(_required(name, "deployment"))))

    async def stop(self, name: str) -> object:
        return to_data(await _resolve(self.supervisor.stop(_required(name, "deployment"))))

    async def plan(
        self,
        model: str,
        *,
        runtime: str | None = None,
        replicas: int = 1,
    ) -> object:
        return to_data(
            await _resolve(
                self.planner.plan(
                    _required(model, "model"),
                    runtime=_optional(runtime),
                    replicas=_replicas(replicas),
                )
            )
        )


async def _resolve(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


def _required(value: str, label: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{label} must not be empty")
    return clean


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    clean = value.strip()
    return clean or None


def _replicas(value: int) -> int:
    if value < 1:
        raise ValueError("replicas must be at least 1")
    return value


class _DefaultRegistry:
    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def list_models(self) -> object:
        return self.backend.list_models()

    def get_model(self, name: str) -> object:
        return self.backend.get(name)

    def pull_model(
        self,
        name: str,
        *,
        runtime: str | None,
        revision: str | None,
        trust_remote_code: bool,
    ) -> object:
        del runtime
        from docie_bench.serving.registry import ModelManifest, TrustPolicy

        manifest_path = Path(name)
        if not manifest_path.is_file():
            raise ValueError(
                "the local registry backend pulls from a manifest JSON path; "
                "configure a provider-backed Registry to pull by model identity"
            )
        manifest = ModelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        updates: dict[str, object] = {
            "trust_policy": (
                TrustPolicy.ALLOW_REMOTE_CODE if trust_remote_code else TrustPolicy.DENY_REMOTE_CODE
            )
        }
        if revision is not None:
            updates["revision"] = revision
        manifest = manifest.model_copy(update=updates)
        sources = {
            artifact.name: artifact.source
            for artifact in manifest.artifacts
            if artifact.source is not None
        }
        if len(sources) != len(manifest.artifacts):
            raise ValueError("every manifest artifact must provide a source for pull")
        return self.backend.pull(manifest, sources)

    def remove_model(self, name: str) -> object:
        return self.backend.remove(name)


class _DefaultRuntimes:
    def __init__(self, adapters: Mapping[Any, Any]) -> None:
        self.adapters = adapters

    def available_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.adapters)

    def list_runtimes(self) -> object:
        return [self._probe(name) for name in sorted(self.adapters, key=str)]

    def probe_runtime(self, name: str) -> object:
        for runtime in self.adapters:
            if str(runtime) == name:
                return self._probe(runtime)
        raise ValueError(f"unknown runtime: {name}")

    def _probe(self, runtime: object) -> object:
        from docie_bench.serving.runtime import RuntimeLaunchSpec

        endpoint = "http://127.0.0.1:1/v1" if str(runtime) == "remote" else None
        spec = RuntimeLaunchSpec(
            runtime=cast(Any, runtime),
            model="__probe__",
            alias="__probe__",
            endpoint=endpoint,
        )
        return self.adapters[runtime].probe(spec)


class _DefaultPlanner:
    def __init__(self, backend: Any, registry: Any, resources: Any) -> None:
        self.backend = backend
        self.registry = registry
        self.resources = resources

    def plan(self, model: str, *, runtime: str | None, replicas: int) -> Any:
        from docie_bench.serving.planner import PlanningRequest, RuntimeName

        return self.backend.plan(
            PlanningRequest(
                model=self.registry.get(model),
                resources=self.resources,
                concurrency=replicas,
                preferred_runtime=RuntimeName(runtime) if runtime else None,
            )
        )


class _DefaultSupervisor:
    def __init__(
        self,
        backend: Any,
        planner: _DefaultPlanner,
        model_store_root: Path | None = None,
    ) -> None:
        self.backend = backend
        self.planner = planner
        self.model_store_root = model_store_root

    def list_deployments(self) -> object:
        return self.backend.list()

    def deployment_status(self, name: str) -> object:
        return self.backend.get(name)

    def serve(
        self,
        model: str,
        *,
        name: str | None,
        runtime: str | None,
        replicas: int,
    ) -> object:
        from docie_bench.serving.runtime import RuntimeKind, RuntimeLaunchSpec
        from docie_bench.serving.supervisor import DeploymentSpec

        if replicas != 1:
            raise ValueError("the local process supervisor supports exactly one replica")
        deployment_name = name or model.rsplit("/", maxsplit=1)[-1]
        if runtime is None:
            recommendation = self.planner.plan(model, runtime=None, replicas=replicas)
            if recommendation.runtime is None:
                raise ValueError(recommendation.explanation)
            runtime = str(recommendation.runtime)
        spec = DeploymentSpec(
            name=deployment_name,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind(runtime),
                model=model,
                alias=deployment_name,
            ),
        )
        return self.backend.deploy(spec)

    def serve_store_model(
        self, name: str, *, port: int = 8088, context_length: int = 8192
    ) -> object:
        from docie_bench.serving.model_store import ModelStore, ModelStoreError
        from docie_bench.serving.runtime import RuntimeKind, RuntimeLaunchSpec
        from docie_bench.serving.supervisor import DeploymentSpec

        if self.model_store_root is None:
            raise ValueError("model store is not configured")
        store = ModelStore(self.model_store_root)  # lazy -> avoids from_defaults mkdir side effect
        try:
            entry = store.entry(name)
        except ModelStoreError as exc:
            raise ModelStoreError(
                f"{exc} Seed it first (ModelStore.seed_from_ollama / add_gguf — "
                f"see serving/README.md), then re-run `docie up {name}`."
            ) from exc
        # Bind vs advertise are independent seams: `host` becomes llama-server's
        # --host (the bind interface), while `endpoint` is the URL recorded on the
        # DeploymentRecord and used by the health check + placement catalog. The
        # advertised host defaults to 127.0.0.1 (host-native CLI); compose sets
        # DOCIE_ADVERTISE_HOST to the worker service name so api/bench containers
        # resolve it (matching the DOCIE_SERVING_HOME os.environ precedent above).
        advertise_host = os.environ.get("DOCIE_ADVERTISE_HOST", "127.0.0.1")
        spec = DeploymentSpec(
            name=entry.name,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP,
                model=entry.model_path.as_posix(),
                alias=entry.name,
                # SECURITY: Bind 0.0.0.0 so api/bench containers on the compose network can reach the runtime; llama-server is auth-less, so this exposes the model to anything sharing the network/host — acceptable only on a trusted private network.
                host="0.0.0.0",  # noqa: S104 - deliberate cross-container bind, see above
                port=port,
                endpoint=f"http://{advertise_host}:{port}/v1",
                context_length=context_length,
                extra_args=store.family_launch_args(name),
            ),
            # A large GGUF can take a while to load; keep the readiness poll from
            # tripping reconcile's degrade-and-kill while the model is still
            # coming up (there is no background reconcile, so this threshold only
            # matters during await_ready below).
            health_failure_threshold=60,
        )
        # Spawn, then wait for the model to finish loading. Without this the sole
        # post-spawn probe sees "Connection refused" and the record freezes at
        # STARTING forever; await_ready re-probes until it is honestly serving.
        self.backend.deploy(spec)
        return self.backend.await_ready(spec.name)

    def start(self, name: str) -> object:
        from docie_bench.serving.supervisor import DesiredState

        record = self.backend.get(name)
        return self.backend.deploy(replace(record.spec, desired_state=DesiredState.RUNNING))

    def stop(self, name: str) -> object:
        result = self.backend.stop(name)
        _clear_placement(name)
        return result

    def remove(self, name: str) -> object:
        result = self.backend.remove(name)
        _clear_placement(name)
        return result


def _clear_placement(name: str) -> None:
    """Drop the catalog placement of a stopped/removed deployment (best-effort).

    Without this, ``store:<model>`` would keep resolving to a dead endpoint
    after ``docie stop``. Best-effort by design: a missing DATABASE_URL or a DB
    hiccup must never block stopping a local process.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        ModelCatalog().clear_placement(name)
    except CatalogUnavailableError:
        pass  # no DATABASE_URL -> nothing was ever recorded; nothing to clear
    except Exception:  # noqa: BLE001 - staleness cleanup must not fail the stop
        logger.warning("could not clear catalog placement for %r", name, exc_info=True)


def to_data(value: object) -> object:
    """Recursively convert common backend values to deterministic JSON-safe data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return to_data(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return to_data(asdict(value))
    if hasattr(value, "model_dump"):
        return to_data(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(key): to_data(item) for key, item in items}
    if isinstance(value, Set):
        return [to_data(item) for item in sorted(value, key=str)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_data(item) for item in value]
    if hasattr(value, "__dict__"):
        public = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        return to_data(public)
    return str(value)
