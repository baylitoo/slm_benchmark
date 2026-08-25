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
import re
import shutil
import socket
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH

if TYPE_CHECKING:
    from docie_bench.serving.runtime import RuntimeLaunchSpec

from docie_bench.serving.placement import (
    clear_placement,
    mark_placement_ready,
    mark_placement_stopped,
    record_placement,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
Result = object | Awaitable[object]


class ResizeUnsupportedError(ValueError):
    """A resize was requested against a runtime that ignores --ctx-size.

    Only llama.cpp (``RuntimeKind.LLAMACPP``) honors a context-length launch
    override; every other runtime (encoder / transformers / multi-vector
    safetensors snapshots, remote endpoints) would silently ignore it, so
    resize refuses up front instead of pretending to apply a no-op.
    """


class ResizeAdmissionError(RuntimeError):
    """A resize was rejected by the RAM admission gate; nothing was touched.

    Raised BEFORE the new-sized shadow instance is spawned — the currently
    running deployment is completely unaffected by a resize that would not
    fit, mirroring the load path's fit-before-evict discipline.
    """


def reachable_launch(
    launch: RuntimeLaunchSpec,
    *,
    bind_host: str,
    advertise_host: str,
) -> RuntimeLaunchSpec:
    """Split the conflated launch ``host`` into a BIND host and an ADVERTISE host.

    The runtime process binds ``bind_host`` (all interfaces inside its container,
    via ``build_command``'s ``--host``) while the DeploymentRecord advertises
    ``http://{advertise_host}:{port}/v1`` — reusing the existing ``spec.endpoint``
    override that ``RuntimeAdapter.endpoint`` returns verbatim (runtime.py:240)
    while ``build_command`` still binds ``spec.host``. This is what makes the
    recorded endpoint cross-container reachable instead of a worker-local
    ``127.0.0.1`` that only resolves to the deployer's own loopback.

    REMOTE keeps its user-supplied endpoint untouched: that URL is the real
    upstream, not a local bind target, so overriding it would break routing.
    """
    from docie_bench.serving.runtime import RuntimeKind

    if launch.runtime == RuntimeKind.REMOTE:
        return launch
    return replace(
        launch,
        host=bind_host,
        endpoint=f"http://{advertise_host}:{launch.port}/v1",
    )


# Advertise hosts that are same-host by construction: a deploy on this node is
# always reachable at these, so the round-robin guard never applies to them.
_LOOPBACK_ADVERTISE_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Bound on the reallocate-on-bind-collision loop in serve_store_model. Keeps a
# genuinely wedged host from spinning through the whole range × restarts; each
# attempt already re-tries the launch up to the spec's max_restarts internally.
_MAX_PORT_ALLOC_ATTEMPTS = 3


def _socket_is_free(host: str, port: int) -> bool:
    """Return True if ``(host, port)`` can be bound right now (worker-local probe).

    Best-effort liveness: tries to ``bind()`` a throwaway TCP socket with
    ``SO_REUSEADDR`` deliberately OFF, so a port already held by a live runtime
    fails the bind and reads as taken. Portable (no ``SO_REUSEPORT``, which is
    Linux-only) so it behaves the same on Windows dev and the Linux worker. This
    only sees the prober's OWN network namespace — the api container cannot use it
    to see the worker's binds — and it is racy by construction (the port can be
    grabbed between this probe and the real ``llama-server`` bind). The launch's
    bind is the authoritative arbiter; this just pre-filters the obvious cases.

    The family is resolved from ``host`` via ``getaddrinfo`` rather than hard-wiring
    ``AF_INET``: an IPv6 bind host (e.g. ``::`` / ``::1``) cannot bind on an IPv4
    socket, so an IPv4-only probe would fail *every* port and wedge allocation into
    a bogus range-exhausted error.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    family, socktype, proto, _canonname, sockaddr = infos[0]
    with socket.socket(family, socktype, proto) as sock:
        try:
            sock.bind(sockaddr)
        except OSError:
            return False
        return True


class PortAllocator:
    """Pick a serving port free of both deployment records and live sockets.

    ``recommend`` is a pure function of ``(range, reserved)`` — the lowest port in
    the window not held by a record — and is shared by the ``/ports`` API hint and
    the worker's ``allocate`` so the two cannot disagree in *logic*. ``allocate``
    additionally socket-probes (worker-local) and returns the first port that is
    both record-free and bindable. They can still differ in *result* because only
    the worker probes sockets; that is expected and documented, never a bug.

    NOT race-free: a probed-free port can be grabbed before ``llama-server`` binds
    it, and a worker cannot see a concurrent host-native ``docie up`` (its
    supervisor is lru-cached and never reloads ``deployments.json``). This is a
    best-effort pre-filter; the real bind is the arbiter and the deploy path
    reallocates on a bind-collision exit.
    """

    def __init__(
        self,
        *,
        range_start: int,
        range_end: int,
        probe: Callable[[str, int], bool] = _socket_is_free,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= range_start <= 65535 or not 1 <= range_end <= 65535:
            raise ValueError("port range bounds must be between 1 and 65535")
        if range_start > range_end:
            raise ValueError("port range start must not exceed end")
        self.range_start = range_start
        self.range_end = range_end
        self._probe = probe
        self._clock = clock

    def _ports(self) -> range:
        return range(self.range_start, self.range_end + 1)

    def _exhausted(self) -> RuntimeError:
        return RuntimeError(
            f"no free port in range {self.range_start}-{self.range_end}; stop a "
            f"deployment or widen DOCIE_SERVING_PORT_RANGE_START/"
            f"DOCIE_SERVING_PORT_RANGE_END"
        )

    def recommend(self, *, bind_host: str, reserved: Set[int]) -> int:
        """Lowest record-free port in the window (pure; no socket probing)."""
        del bind_host  # record-derived only; symmetry with allocate()
        for port in self._ports():
            if port not in reserved:
                return port
        raise self._exhausted()

    def allocate(self, *, bind_host: str, reserved: Set[int]) -> int:
        """First port that is both record-free and bindable on ``bind_host``."""
        for port in self._ports():
            if port in reserved:
                continue
            if self._probe(bind_host, port):
                return port
        raise self._exhausted()


def _resolve_ipv4(host: str) -> tuple[str, ...]:
    """Return the distinct IPv4 addresses ``host`` resolves to (``()`` on failure).

    Fail-open: an unresolvable name yields ``()`` (the caller must treat "can't
    resolve" as "can't prove non-deterministic" and allow the deploy) rather than
    raising, so a transient/absent DNS entry never blocks a legitimate deploy.
    """
    try:
        _name, _aliases, addresses = socket.gethostbyname_ex(host)
    except OSError:
        return ()
    # gethostbyname_ex may repeat addresses; keep first-seen order, deduped.
    return tuple(dict.fromkeys(addresses))


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
        max_tokens: int | None = None,
    ) -> Result: ...

    def serve_store_model(
        self,
        name: str,
        *,
        port: int | None,
        context_length: int,
        max_tokens: int | None = None,
    ) -> Result: ...

    def start(self, name: str) -> Result: ...

    def stop(self, name: str) -> Result: ...

    def remove(self, name: str) -> Result: ...

    def load(self, name: str) -> Result: ...

    def unload(self, name: str) -> Result: ...

    def repair(self, name: str, *, port: int | None) -> Result: ...

    def reconfigure(
        self,
        name: str,
        *,
        context_length: int,
        max_tokens: int | None,
    ) -> Result: ...

    def resize_store_model(self, name: str, *, context_length: int) -> Result: ...

    def pin(self, name: str, *, pinned: bool) -> Result: ...


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
        from docie_bench.serving.resources import read_node_memory
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
                # cgroup-v2-first, same reader as the fit gate and the sizing
                # snapshot: in a container psutil describes the whole VM, and
                # the planner would approve deploys against RAM the cgroup
                # never grants.
                memory_gb=round(read_node_memory().free_bytes / (1024**3), 4),
                disk_gb=round(shutil.disk_usage(home.parent).free / (1024**3), 4),
                # Only the planner-scorable chat runtimes: non-chat kinds (the
                # encoder shim) are deploy-explicit (`runtime="encoder"`) and
                # must never be auto-recommended for a chat model.
                available_runtimes=frozenset(
                    RuntimeName(name)
                    for name in runtimes.available_names()
                    if name in {member.value for member in RuntimeName}
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
        max_tokens: int | None = None,
    ) -> object:
        # Threaded like up(): the runtime-specified deploy path blocks in
        # deploy + await_ready (a bounded time.sleep poll while the model
        # loads) plus a DNS resolve in the advertise guard. Running that on
        # the event loop would stall the Inngest Connect heartbeats for the
        # whole load (design doc fix #4). Argument validation stays eager so
        # bad input raises before a thread is spawned; _resolve keeps the
        # awaitable-backend contract for async test fakes.
        kwargs: dict[str, Any] = {
            "name": _optional(name),
            "runtime": _optional(runtime),
            "replicas": _replicas(replicas),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        result = await asyncio.to_thread(
            self.supervisor.serve,
            _required(model, "model"),
            **kwargs,
        )
        return to_data(await _resolve(result))

    async def up(
        self,
        name: str,
        *,
        port: int | None = None,
        context_length: int = DEFAULT_DEPLOY_CONTEXT_LENGTH,
        max_tokens: int | None = None,
        deployment_name: str | None = None,
    ) -> object:
        # serve_store_model is synchronous and now blocks in await_ready() (a
        # bounded time.sleep poll until the model is serving). Run it in a thread
        # so it does not stall the worker's asyncio loop — the Inngest Connect
        # heartbeats and any concurrent realtime/extraction steps must keep
        # flowing on the scale-1 worker while a large GGUF loads.
        # ``deployment_name`` (scale) names the record while ``name`` stays the
        # store-entry lookup — see serve_store_model.
        kwargs = {
            "port": port,
            "context_length": context_length,
            "deployment_name": deployment_name,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return to_data(
            await asyncio.to_thread(
                self.supervisor.serve_store_model,
                _required(name, "model"),
                **kwargs,
            )
        )

    async def start(self, name: str) -> object:
        # Threaded: start() re-deploys (spawn + immediate health probe), which
        # blocks on process spawn and the probe timeout — same rationale as
        # serve()/stop() below.
        result = await asyncio.to_thread(self.supervisor.start, _required(name, "deployment"))
        return to_data(await _resolve(result))

    async def stop(self, name: str) -> object:
        # Threaded: stop() reconciles to STOPPED, which blocks in the
        # adapter's terminate/kill shutdown timeouts (up to ~10s per process)
        # and the placement UPDATE. Must not stall Connect heartbeats.
        result = await asyncio.to_thread(self.supervisor.stop, _required(name, "deployment"))
        return to_data(await _resolve(result))

    async def remove(self, name: str) -> object:
        """Real teardown (PR-1): kill the process, drop the record, free the
        port, DELETE the placement row — the previously-unreachable
        ``_DefaultSupervisor.remove``, now on the facade so the
        ``serving/delete.requested`` job and the API can actually delete.
        Threaded like ``up()``: the shutdown can block up to the adapter's
        terminate/kill timeouts and must not stall the Connect heartbeats.
        """
        result = await asyncio.to_thread(self.supervisor.remove, _required(name, "deployment"))
        return to_data(await _resolve(result))

    async def load(self, name: str) -> object:
        """Bring a cold/evicted deployment hot (PR-4, §4 cold-start-on-demand).

        Idempotent under the serving-side per-deployment load lock: N
        concurrent loads (scaled workers, api button, step retries) funnel to
        ONE spawn, an already-hot deployment is a no-op. Blocks through a
        size-aware ``await_ready`` — threaded so a multi-minute GGUF load
        never stalls the Connect heartbeats.
        """
        result = await asyncio.to_thread(self.supervisor.load, _required(name, "deployment"))
        return to_data(await _resolve(result))

    async def unload(self, name: str) -> object:
        """Evict a deployment: kill the process, KEEP record + port + row.

        The DISTINCT path from ``stop`` (design fix #3): activation flips to
        ``managed`` so the next request auto-reloads it, and the placement row
        is UPDATEd to ``phase=evicted`` — never cleared.
        """
        result = await asyncio.to_thread(self.supervisor.unload, _required(name, "deployment"))
        return to_data(await _resolve(result))

    async def repair(self, name: str, *, port: int | None = None) -> object:
        """Recover a stuck/failed deployment WITHOUT delete+recreate.

        Rebuilds the SAME launch on a different port and redeploys, resetting
        the restart budget (the changed launch makes ``deploy`` kill the old
        pid and zero the failure counters). ``port=None`` auto-reallocates a
        free port — which side-steps a port still held by an orphan process
        (the EADDRINUSE bind-loop); an explicit ``port`` is honored verbatim.
        Threaded like the other spawn paths so it never stalls the Connect
        heartbeats.
        """
        result = await asyncio.to_thread(
            self.supervisor.repair, _required(name, "deployment"), port=port
        )
        return to_data(await _resolve(result))

    async def reconfigure(
        self,
        name: str,
        *,
        context_length: int,
        max_tokens: int | None,
    ) -> object:
        """Replace editable launch defaults while preserving deployment identity.

        A running deployment is restarted on its existing port. A stopped or
        managed/offloaded deployment keeps its desired state and is only
        updated on disk, so editing configuration never unexpectedly loads it.
        """
        result = await asyncio.to_thread(
            self.supervisor.reconfigure,
            _required(name, "deployment"),
            context_length=context_length,
            max_tokens=max_tokens,
        )
        return to_data(await _resolve(result))

    async def resize_store_model(self, name: str, *, context_length: int) -> object:
        """Change a live store deployment's context window with zero downtime.

        A HOT deployment is resized by drain-and-relaunch (a shadow instance
        at the new context length is spawned, awaited to READY, routed to,
        and only THEN does the old process stop); a stopped/offloaded
        deployment has no process to drain and is edited in place, mirroring
        ``reconfigure``. Threaded: RAM admission reads live node memory and a
        hot resize blocks through the shadow's ``await_ready`` poll — neither
        may stall the Connect heartbeats.
        """
        result = await asyncio.to_thread(
            self.supervisor.resize_store_model,
            _required(name, "deployment"),
            context_length=context_length,
        )
        return to_data(await _resolve(result))

    async def pin(self, name: str, *, pinned: bool = True) -> object:
        """Set/clear the eviction shield on a deployment (PR-4)."""
        result = await asyncio.to_thread(
            lambda: self.supervisor.pin(_required(name, "deployment"), pinned=pinned)
        )
        return to_data(await _resolve(result))

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
        *,
        advertise_host: str | None = None,
        bind_host: str | None = None,
        resolve_host: Callable[[str], Sequence[str]] | None = None,
        port_range: tuple[int, int] | None = None,
        port_probe: Callable[[str, int], bool] | None = None,
    ) -> None:
        self.backend = backend
        self.planner = planner
        self.model_store_root = model_store_root
        # None => resolve lazily from settings at deploy time (so the running
        # container's DOCIE_SERVING_* env wins); tests inject explicit hosts to
        # avoid the get_settings lru_cache dance.
        self._advertise_host = advertise_host
        self._bind_host = bind_host
        # Name-resolution hook for the deterministic-addressing guard; injectable
        # so tests can simulate a scaled (multi-A-record) service without DNS.
        self._resolve_host = resolve_host or _resolve_ipv4
        # Port-allocation window + socket prober. None => read the window from
        # settings at deploy time (container env wins); tests inject a fixed range
        # and a fake prober so they never touch real sockets.
        self._port_range = port_range
        self._port_probe = port_probe
        # PR-4 load coordinator (per-deployment load locks + fit/eviction),
        # built lazily on first load() so read-only constructions (the api's
        # per-request ControlPlane) never touch the footprints sidecars. The
        # lazy init is guarded by _coordinator_lock: two concurrent first
        # loads (cold-start pileup — exactly when the coordinator's locks
        # matter) must never each construct a coordinator with its own lock
        # map, which would void the single-spawn/single-admission guarantee.
        self._load_coordinator: Any = None
        self._coordinator_lock = threading.Lock()

    def _reachability_hosts(self) -> tuple[str, str]:
        """Return ``(bind_host, advertise_host)`` for a deploy on this node."""
        if self._bind_host is not None and self._advertise_host is not None:
            return self._bind_host, self._advertise_host
        from docie_bench.settings import get_settings

        settings = get_settings()
        bind = self._bind_host if self._bind_host is not None else settings.serving_bind_host
        advertise = (
            self._advertise_host
            if self._advertise_host is not None
            else settings.serving_advertise_host
        )
        return bind, advertise

    def _port_allocator(self) -> PortAllocator:
        """Build the allocator; range from settings unless a test injected one."""
        if self._port_range is not None:
            start, end = self._port_range
        else:
            from docie_bench.settings import get_settings

            settings = get_settings()
            start = settings.serving_port_range_start
            end = settings.serving_port_range_end
        probe = self._port_probe if self._port_probe is not None else _socket_is_free
        return PortAllocator(range_start=start, range_end=end, probe=probe)

    def _reserved_ports(self, *, exclude_name: str | None = None) -> set[int]:
        """Ports held by every non-removed deployment record (the record-scan).

        Every record reserves its port regardless of state — a STOPPED record can
        be ``start()``-ed and must not lose its port — with the sole exception of
        the deployment being (re)deployed right now, so a redeploy can reclaim its
        own port instead of drifting to a new one. Only ``remove()`` frees a port.
        """
        reserved: set[int] = set()
        for record in self.backend.list():
            if exclude_name is not None and record.spec.name == exclude_name:
                continue
            reserved.add(record.spec.launch.port)
        return reserved

    def _guard_deterministic_advertise(self, advertise_host: str) -> None:
        """Refuse a deploy whose advertise host does not name a single node.

        A deployed runtime lives on exactly ONE node — the container that ran the
        deploy. If the advertised name round-robins across replicas (e.g. a
        ``docker compose up --scale worker>1`` service, whose embedded DNS returns
        one A-record per replica), the persisted endpoint may later resolve to a
        replica that never ran the deploy, so ``worker`` deploys become
        *intermittently* unreachable instead of reliably working. This turns that
        silent, load-balancer-roulette failure into a clear, actionable error at
        deploy time (finding 2 / PR-1 follow-up).

        Detection assumes Compose's per-replica DNS round-robin (N A-records under
        ``--scale N``); a Swarm VIP would collapse to a single virtual IP and not
        trip this — that model is out of scope for this compose stack.

        Fail-open by design: a same-host loopback advertise (local ``docie up``),
        an unresolvable name, or a single resolved address is always allowed — we
        only refuse when we can positively prove ambiguity (>1 distinct address).
        """
        if advertise_host in _LOOPBACK_ADVERTISE_HOSTS:
            return
        addresses = tuple(self._resolve_host(advertise_host))
        if len(addresses) <= 1:
            return
        raise ValueError(
            f"advertise host {advertise_host!r} resolves to {len(addresses)} "
            f"addresses ({', '.join(addresses)}); the deployed runtime lives on "
            f"exactly one node, so a round-robin service name (e.g. a "
            f"--scale worker>1 compose service) would record an endpoint that "
            f"resolves to a replica which never ran the deploy — an intermittent "
            f"failure. Pin deploys to a single-replica service: keep the deploy "
            f"worker at scale=1, or point DOCIE_SERVING_ADVERTISE_HOST at a "
            f"dedicated single-replica 'serving' service so the advertised name "
            f"always resolves to the one node running the runtime."
        )

    def list_deployments(self) -> object:
        return self.backend.list()

    def deployment_status(self, name: str) -> object:
        return self.backend.get(name)

    def _deploy_with_port_reallocation(
        self,
        *,
        launch_and_await: Callable[[int], Any],
        bind_host: str,
        exclude_name: str,
        port_override: int | None,
    ) -> Any:
        """Deploy through a bounded allocate → deploy → await loop, reallocating the
        port only on a started-then-exited FAILED record (the bind-collision
        signature). Shared by serve() and serve_store_model so the vLLM and
        store-model paths recover from a raced port identically instead of only the
        latter having the loop.

        ``launch_and_await(port)`` builds the spec at ``port``, deploys it, and
        returns the awaited record — the caller owns the runtime-specific spec so
        this stays generic. An explicit ``port_override`` is a command, not a
        suggestion: honored verbatim with no probing and no reallocation, letting
        the real bind arbitrate. A slow-loading model keeps is_running True (never
        FAILED+exited), so it never triggers a realloc; an unfixable failure
        (missing binary) never sets exited_after_start, so the range is not churned.
        """
        from docie_bench.serving.runtime import LifecycleState

        allocator = self._port_allocator()
        # Ports already tried in THIS call: after deploy() replaces the record, the
        # just-failed port is no longer record-reserved, so without this the loop
        # could re-pick it (relying only on the socket probe / external binder).
        # Accumulating them keeps reallocation deterministic and prober-independent.
        tried: set[int] = set()
        record: Any = None
        for _attempt in range(_MAX_PORT_ALLOC_ATTEMPTS):
            if port_override is not None:
                chosen = port_override
            else:
                reserved = self._reserved_ports(exclude_name=exclude_name) | tried
                chosen = allocator.allocate(bind_host=bind_host, reserved=reserved)
            tried.add(chosen)
            record = launch_and_await(chosen)
            if record.state == LifecycleState.READY:
                break
            # Reallocate ONLY on a started-then-exited failure and never for an
            # explicit override; the exit reason (the runtime's real stderr tail) is
            # already on record.last_error via the supervisor.
            exited = getattr(record, "exited_after_start", False)
            if (
                port_override is not None
                or record.state != LifecycleState.FAILED
                or not exited
            ):
                break
            # else: loop -> allocate the next free port and rebuild the spec THROUGH
            # reachable_launch so the advertised endpoint reflects the new port.
        return record

    def serve(
        self,
        model: str,
        *,
        name: str | None,
        runtime: str | None,
        replicas: int,
        max_tokens: int | None = None,
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
        bind_host, advertise_host = self._reachability_hosts()
        runtime_kind = RuntimeKind(runtime)
        # REMOTE advertises the user's real upstream (reachable_launch skips it),
        # so its advertise host plays no role — don't fail-fast on it, don't
        # allocate a local port, and don't await/reallocate (it binds nothing here).
        if runtime_kind == RuntimeKind.REMOTE:
            launch = reachable_launch(
                RuntimeLaunchSpec(
                    runtime=runtime_kind,
                    model=model,
                    alias=deployment_name,
                    max_tokens=max_tokens,
                ),
                bind_host=bind_host,
                advertise_host=advertise_host,
            )
            return self.backend.deploy(DeploymentSpec(name=deployment_name, launch=launch))

        self._guard_deterministic_advertise(advertise_host)

        def _launch_and_await(chosen: int) -> Any:
            # All non-REMOTE serve() paths otherwise default to RuntimeLaunchSpec's
            # port 8000, so every concurrent serve would collide there; the
            # allocated ``chosen`` (record + socket pre-filtered) is bound instead.
            launch = reachable_launch(
                RuntimeLaunchSpec(
                    runtime=runtime_kind,
                    model=model,
                    alias=deployment_name,
                    port=chosen,
                    max_tokens=max_tokens,
                ),
                bind_host=bind_host,
                advertise_host=advertise_host,
            )
            # Mirror serve_store_model's high threshold: await_ready re-probes a
            # still-loading runtime (vLLM weights load slowly) and the default 3
            # would degrade-and-kill it mid-load. A bind collision still exits fast
            # (FAILED+exited), so reallocation stays responsive.
            spec = DeploymentSpec(
                name=deployment_name, launch=launch, health_failure_threshold=60
            )
            self.backend.deploy(spec)
            return self.backend.await_ready(spec.name)

        return self._deploy_with_port_reallocation(
            launch_and_await=_launch_and_await,
            bind_host=bind_host,
            exclude_name=deployment_name,
            port_override=None,
        )

    def serve_store_model(
        self,
        name: str,
        *,
        port: int | None = None,
        context_length: int = DEFAULT_DEPLOY_CONTEXT_LENGTH,
        max_tokens: int | None = None,
        deployment_name: str | None = None,
    ) -> object:
        """Deploy a store model. ``deployment_name`` overrides the record name so
        the SAME store model can run as several deployments (scale): the weights
        + family are looked up by ``name`` (the store entry), but the record —
        and its port — are keyed on ``deployment_name or name``. The llama-server
        ``--alias`` stays the base store ``name`` so every replica answers to the
        same model id (the placement row records ``model_name=name``), which is
        what a future load-balancer routes ``store:<name>`` across."""
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
        record_name = deployment_name or entry.name
        bind_host, advertise_host = self._reachability_hosts()
        # Store models always run the in-worker LLAMACPP subprocess (never REMOTE),
        # so guard the advertise host unconditionally here.
        self._guard_deterministic_advertise(advertise_host)

        # Analyzer (encoder) entries are a safetensors snapshot served by the
        # ENCODER runtime — not llama.cpp. The snapshot path goes in as the
        # model and the family's encoder_backend rides in as --backend (no
        # name-based guessing), so the checkpoint loads locally with zero
        # network at boot.
        from docie_bench.serving.model_store import get_family

        contract = get_family(entry.family)
        if contract.analyzer:
            runtime_kind = RuntimeKind.ENCODER
            launch_extra_args: tuple[str, ...] = (
                ("--backend", contract.encoder_backend)
                if contract.encoder_backend
                else ()
            )
        elif contract.transformers_runtime:
            # LAST-RESORT: a safetensors snapshot served by the TRANSFORMERS
            # runtime (no GGUF / arch llama.cpp can't load). The snapshot dir is
            # the model; --trust-remote-code rides in only when the family opts
            # in (a custom-code checkpoint), never by default.
            runtime_kind = RuntimeKind.TRANSFORMERS
            launch_extra_args = (
                ("--trust-remote-code",) if contract.trust_remote_code else ()
            )
        elif contract.multi_vector:
            # A ColBERT / late-interaction snapshot served by the MULTI-VECTOR
            # runtime (sentence-transformers MultiVectorEncoder), answered on
            # /v1/rerank. The snapshot dir is the model; no extra flags.
            runtime_kind = RuntimeKind.MULTI_VECTOR
            launch_extra_args = ()
        else:
            runtime_kind = RuntimeKind.LLAMACPP
            launch_extra_args = store.family_launch_args(name)

        def _launch_and_await(chosen: int) -> Any:
            launch = reachable_launch(
                RuntimeLaunchSpec(
                    runtime=runtime_kind,
                    model=entry.model_path.as_posix(),
                    # --alias stays the BASE store name so every replica answers
                    # to the same model id (load-balancing key); the RECORD name
                    # is what makes the deployment individually addressable.
                    alias=entry.name,
                    port=chosen,
                    context_length=context_length,
                    max_tokens=max_tokens,
                    extra_args=launch_extra_args,
                ),
                bind_host=bind_host,
                advertise_host=advertise_host,
            )
            spec = DeploymentSpec(
                name=record_name,
                launch=launch,
                # A large GGUF can take a while to load; keep the readiness poll from
                # tripping reconcile's degrade-and-kill while the model is still
                # coming up (there is no background reconcile, so this threshold only
                # matters during await_ready). OPPOSITE case from a bind collision
                # (which exits immediately -> FAILED, not slow-loading).
                health_failure_threshold=60,
            )
            # Spawn, then wait for the model to finish loading. Without this the sole
            # post-spawn probe sees "Connection refused" and the record freezes at
            # STARTING forever; await_ready re-probes until it is honestly serving.
            self.backend.deploy(spec)
            return self.backend.await_ready(spec.name)

        record = self._deploy_with_port_reallocation(
            launch_and_await=_launch_and_await,
            bind_host=bind_host,
            exclude_name=record_name,
            port_override=port,
        )
        # Record the placement at the same seam stop()/remove() clear it, so
        # every deploy surface — host-native `docie up` and the worker's Inngest
        # job alike — makes store:<name> resolvable, not just the worker path.
        # ``name`` (the base store name) is the placement's model_name, so all
        # replicas share one routable model id.
        record_placement(name, record)
        return record

    def start(self, name: str) -> object:
        from docie_bench.serving.supervisor import DesiredState

        record = self.backend.get(name)
        return self.backend.deploy(replace(record.spec, desired_state=DesiredState.RUNNING))

    def repair(self, name: str, *, port: int | None = None) -> object:
        """Redeploy an existing deployment on a (re)allocated port.

        Recovery without delete+recreate: the SAME launch (runtime, model,
        extra_args) is rebuilt on a different port and redeployed. Because the
        launch changes, ``backend.deploy`` kills the old pid and resets the
        restart budget / failure counters — so a deployment stuck FAILED (e.g.
        an EADDRINUSE bind-loop against an orphan holding its old port) gets a
        genuinely fresh chance. ``port=None`` auto-reallocates (the allocator
        excludes reserved + socket-bound ports, so it steps around the orphan);
        an explicit ``port`` is honored verbatim, letting the real bind
        arbitrate.
        """
        from docie_bench.serving.runtime import RuntimeKind
        from docie_bench.serving.supervisor import DeploymentSpec, DesiredState

        record = self.backend.get(name)
        old_launch = record.spec.launch
        bind_host, advertise_host = self._reachability_hosts()
        if old_launch.runtime != RuntimeKind.REMOTE:
            self._guard_deterministic_advertise(advertise_host)

        def _launch_and_await(chosen: int) -> Any:
            launch = reachable_launch(
                replace(old_launch, port=chosen),
                bind_host=bind_host,
                advertise_host=advertise_host,
            )
            spec = DeploymentSpec(
                name=record.spec.name,
                launch=launch,
                desired_state=DesiredState.RUNNING,
                health_failure_threshold=60,
            )
            self.backend.deploy(spec)
            return self.backend.await_ready(spec.name)

        result = self._deploy_with_port_reallocation(
            launch_and_await=_launch_and_await,
            bind_host=bind_host,
            exclude_name=name,
            port_override=port,
        )
        # Keep store:<name> resolvable after the port move (endpoint changed).
        record_placement(name, result)
        return result

    def reconfigure(
        self,
        name: str,
        *,
        context_length: int,
        max_tokens: int | None,
    ) -> object:
        """Apply editable launch defaults to an existing deployment in place.

        The launch equality check in ``PersistentSupervisor.deploy`` performs
        the actual process replacement. Reusing the existing ``DeploymentSpec``
        preserves restart policy and desired state; only the two operator-facing
        defaults change. Running deployments wait for the replacement to become
        ready, while stopped/offloaded deployments remain stopped.
        """
        from docie_bench.serving.runtime import LifecycleState
        from docie_bench.serving.supervisor import DesiredState

        record = self.backend.get(name)
        old_spec = record.spec
        old_launch = old_spec.launch
        launch = replace(
            old_launch,
            context_length=context_length,
            max_tokens=max_tokens,
        )
        spec = replace(old_spec, launch=launch)
        try:
            result = self.backend.deploy(spec)
            if spec.desired_state == DesiredState.RUNNING:
                result = self.backend.await_ready(spec.name)
                if result.state == LifecycleState.FAILED:
                    raise RuntimeError(
                        result.last_error
                        or f"deployment {name!r} rejected the updated launch settings"
                    )
        except Exception:
            # Context support varies by model, and a larger KV cache may also
            # exceed current RAM. A failed edit must not strand a previously
            # healthy deployment on the rejected launch: restore its exact old
            # spec and bring it back to the state it had before the edit.
            logger.exception(
                "deployment %r rejected reconfiguration; restoring previous launch",
                name,
            )
            rollback = self.backend.deploy(old_spec)
            if old_spec.desired_state == DesiredState.RUNNING:
                rollback = self.backend.await_ready(old_spec.name)
            record_placement(str(old_launch.alias or name), rollback)
            raise

        # The placement key is the deployment record name; the routed model id
        # is the shared alias for scaled replicas, or the record name otherwise.
        record_placement(str(old_launch.alias or name), result)
        return result

    def resize_store_model(self, name: str, *, context_length: int) -> object:
        """Change a live store deployment's context window with zero downtime.

        llama-server cannot resize its KV cache in place — the context length
        is fixed at process start — so a HOT deployment (running, READY, with
        a live endpoint) is resized by drain-and-relaunch: a shadow instance
        is spawned at ``context_length`` on its OWN port, awaited to READY,
        and only THEN is routing flipped to it and the old process reaped.
        The old instance keeps serving every request until the new one has
        proven itself healthy — routing is never pointed at nothing. A
        stopped/offloaded deployment has no process to drain, so its launch is
        just updated in place (mirrors ``reconfigure``).

        Admission reuses the SAME fit policy every load-shaped decision
        shares (``lifecycle.assess_fit``): the candidate is priced at the NEW
        context length and checked against live measured free RAM, because
        the old and new processes briefly coexist during the handoff. A
        rejection raises :class:`ResizeAdmissionError` carrying the
        FitDecision's numbers and touches NOTHING — no shadow is spawned, no
        port is allocated, the running deployment is completely unaffected.

        Only the LLAMACPP runtime honors ``--ctx-size``; every other runtime
        raises :class:`ResizeUnsupportedError` before anything is touched.
        """
        from docie_bench.serving import lifecycle
        from docie_bench.serving.runtime import LifecycleState, RuntimeKind
        from docie_bench.serving.supervisor import DeploymentSpec, DesiredState

        record = self.backend.get(name)
        old_spec = record.spec
        old_launch = old_spec.launch
        if old_launch.runtime != RuntimeKind.LLAMACPP:
            raise ResizeUnsupportedError(
                f"deployment {name!r} runs the {old_launch.runtime.value!r} runtime, "
                f"which does not accept a context-length override (only llama.cpp "
                f"honors --ctx-size)"
            )

        new_launch = replace(old_launch, context_length=context_length)
        is_hot = (
            old_spec.desired_state == DesiredState.RUNNING
            and record.state == LifecycleState.READY
            and record.endpoint is not None
        )
        if not is_hot:
            # Nothing running to drain: edit in place and stay stopped, same
            # as reconfigure's offloaded/stopped branch.
            return self.backend.deploy(replace(old_spec, launch=new_launch))

        candidate = replace(record, spec=replace(old_spec, launch=new_launch))
        decision = lifecycle.assess_fit(candidate)
        if not decision.fits:
            available = (decision.free_bytes or 0) - decision.margin_bytes
            raise ResizeAdmissionError(
                f"resizing {name!r} to {context_length} tokens of context needs "
                f"~{decision.needed_bytes} bytes but only {available} bytes are "
                f"available ({decision.free_bytes} free minus a "
                f"{decision.margin_bytes}-byte safety margin): {decision.reason}"
            )

        bind_host, advertise_host = self._reachability_hosts()
        # Store models always run the in-worker LLAMACPP subprocess, so guard
        # the advertise host unconditionally (mirrors serve_store_model).
        self._guard_deterministic_advertise(advertise_host)
        # Deliberately NOT excluding ``name``'s own port: the old instance is
        # still hot on it throughout the handoff (that is the whole point of
        # zero downtime), so the shadow must land on a genuinely different one.
        reserved = self._reserved_ports()
        chosen = self._port_allocator().allocate(bind_host=bind_host, reserved=reserved)
        shadow_launch = reachable_launch(
            replace(new_launch, port=chosen),
            bind_host=bind_host,
            advertise_host=advertise_host,
        )
        shadow_name = f"{name}__resize-{uuid.uuid4().hex[:8]}"
        self.backend.deploy(
            DeploymentSpec(name=shadow_name, launch=shadow_launch, health_failure_threshold=60)
        )
        shadow = self.backend.await_ready(
            shadow_name, timeout_s=lifecycle.load_timeout_s(old_launch.model)
        )
        if shadow.state != LifecycleState.READY:
            error = shadow.last_error or "the resized instance did not become ready"
            self.backend.remove(shadow_name)
            # The reconciler creates a placement row for ANY record it
            # observes mid-await (design: "the Board converges instead of
            # staying blind forever") — best-effort clean it up so a failed
            # resize never leaves a permanent orphan row behind.
            clear_placement(shadow_name)
            raise RuntimeError(f"resize of {name!r} was rejected by the runtime: {error}")

        try:
            # Flip routing to the proven-healthy new instance BEFORE touching
            # the old one: the old process keeps serving until this UPDATE
            # lands, so store:<name> is never pointed at nothing.
            mark_placement_ready(name, endpoint=str(shadow.endpoint or ""))
            previous = self.backend.adopt(name, shadow)
        except Exception:
            # ``name`` vanished (a concurrent delete) mid-handoff: the shadow
            # is still a live, unowned process. Reap it here too — otherwise
            # it leaks exactly like the orphan-process class this repo has
            # already had to fix once (an undeleted record holding RAM
            # forever), just introduced by resize instead of delete.
            self.backend.remove(shadow_name)
            clear_placement(shadow_name)
            raise
        self.backend.discard(shadow_name)
        # The shadow's own placement row (if the reconciler created one while
        # it was still a distinct record) is superseded by the flip above —
        # drop it so it never lingers as a phantom deployment.
        clear_placement(shadow_name)
        self.backend.reap(previous)
        return self.backend.get(name)

    def stop(self, name: str) -> object:
        result = self.backend.stop(name)
        # PR-1 (design fix #3): a stop RETAINS the placement row and UPDATEs it
        # (state=stopped, endpoint="", phase=cold) so display + future
        # auto-reload metadata survive; endpoint "" stops routing immediately.
        # Deleting the row is remove()'s job ONLY.
        mark_placement_stopped(name)
        return result

    def remove(self, name: str) -> object:
        """The one true delete: kill process, drop record + port, DELETE row."""
        result = self.backend.remove(name)
        clear_placement(name)
        return result if result is not None else {"name": name, "removed": True}

    # ------------------------------------------------------- lifecycle (PR-4)
    def _coordinator(self) -> Any:
        """The shared LoadCoordinator (lazy; one per supervisor => one lock map).

        Its eviction executor is THIS wrapper's ``unload`` so every victim
        goes through the placement-row UPDATE, exactly like a direct unload.
        Double-checked under ``_coordinator_lock``: the un-synchronized
        check-then-set was a classic race — two concurrent FIRST loads (the
        cold-start pileup the coordinator exists for) could each build a
        coordinator with its own per-deployment lock map.
        """
        if self._load_coordinator is None:
            with self._coordinator_lock:
                if self._load_coordinator is None:
                    from docie_bench.serving.lifecycle import LoadCoordinator

                    self._load_coordinator = LoadCoordinator(
                        self.backend, unload=self.unload
                    )
        return self._load_coordinator

    def load(self, name: str) -> object:
        """Cold-start a deployment (idempotent; may evict LRU victims, §4)."""
        record = self._coordinator().load(name)
        # Refresh the existing placement row so store:<name> resolution sees
        # the live endpoint immediately instead of one reconcile cycle later
        # (the row itself was RETAINED by unload — that is the whole point).
        mark_placement_ready(name, endpoint=str(record.endpoint or ""))
        return record

    def unload(self, name: str) -> object:
        """The DISTINCT unload path (design fix #3): UPDATE to evicted, never
        clear. Record and port reservation are retained by the backend."""
        result = self.backend.unload(name)
        mark_placement_stopped(name, phase="evicted")
        return result

    def pin(self, name: str, *, pinned: bool) -> object:
        return self.backend.pin(name, pinned=pinned)


def replica_deployment_name(base: str, index: int) -> str:
    """The deployment (record) name for the ``index``-th replica of ``base``.

    The first instance (index 1) IS the bare store name so a single deploy is
    unchanged; subsequent replicas get a ``-2``/``-3``/… suffix. This is the
    naming convention scaling and ``count_replica_deployments`` both key on."""
    return base if index <= 1 else f"{base}-{index}"


def _replica_re(base: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(base)}(?:-(\d+))?$")


def count_replica_deployments(base: str, existing_names: Iterable[str]) -> int:
    """How many live deployments are replicas of ``base`` (``base`` or ``base-N``)."""
    pattern = _replica_re(base)
    return sum(1 for n in existing_names if pattern.match(n))


def replica_names_to_add(
    base: str, existing_names: Iterable[str], target_total: int
) -> list[str]:
    """The deployment names to CREATE to bring ``base`` up to ``target_total``
    replicas, skipping any name already taken (replica of ``base`` or not).

    Returns fewer than the delta only if the search is exhausted (bounded). An
    already-satisfied target returns ``[]`` — scaling is idempotent."""
    existing = set(existing_names)
    current = count_replica_deployments(base, existing)
    delta = max(0, target_total - current)
    names: list[str] = []
    index = 1
    while len(names) < delta and index <= 10_000:
        candidate = replica_deployment_name(base, index)
        if candidate not in existing:
            names.append(candidate)
            existing.add(candidate)
        index += 1
    return names


def to_data(value: object) -> object:
    """Recursively convert common backend values to deterministic JSON-safe data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return to_data(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        # Shallow public-field dict (recursion below re-descends into nested
        # dataclasses), honoring ``metadata={"serialize": False}`` so internal-only
        # fields — e.g. DeploymentRecord.exited_after_start / log_offset — never
        # leak into the /deployments API payload or the deploy result. asdict()
        # would deep-convert them in; fields() lets us drop them by intent.
        public = {
            f.name: getattr(value, f.name)
            for f in fields(value)
            if f.metadata.get("serialize", True)
        }
        return to_data(public)
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
