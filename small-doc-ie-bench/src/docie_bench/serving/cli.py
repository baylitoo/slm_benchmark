"""Ollama-like operations CLI for the serving control plane.

Run with ``python -m docie_bench.serving.cli``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import textwrap
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from docie_bench.serving.control_plane import ControlPlane, to_data
from docie_bench.serving.drain_wait import (
    DEFAULT_WAIT_TIMEOUT_S,
    assess_node,
    find_node,
    wait_for_drain,
)
from docie_bench.serving.resources import DEFAULT_DEPLOY_CONTEXT_LENGTH

#: Where the central api lives, as whatever runs these commands reaches
#: it, and the key to call it with. Only the `fleet` commands read these;
#: every other command in this CLI is node-local and needs neither.
API_URL_ENV_VAR = "DOCIE_API_URL"
API_KEY_ENV_VAR = "DOCIE_API_KEY"


@dataclass
class _Context:
    plane_factory: Callable[[], ControlPlane]
    json_output: bool = False
    plane: ControlPlane | None = None

    def get_plane(self) -> ControlPlane:
        if self.plane is None:
            self.plane = self.plane_factory()
        return self.plane


def create_app(
    control_plane: ControlPlane | None = None,
    *,
    plane_factory: Callable[[], ControlPlane] = ControlPlane.from_defaults,
) -> typer.Typer:
    """Create an embeddable CLI, optionally backed by an injected control plane."""
    app = typer.Typer(
        name="docie-serving",
        help="Acquire, inspect, plan, and operate local model deployments.",
        no_args_is_help=True,
        pretty_exceptions_show_locals=False,
    )
    model_app = typer.Typer(help="Manage local model artifacts.", no_args_is_help=True)
    runtime_app = typer.Typer(help="Inspect available inference runtimes.", no_args_is_help=True)
    store_app = typer.Typer(
        help="Seed and inspect THIS node's blob store.", no_args_is_help=True
    )
    fleet_app = typer.Typer(
        help="Drain and resume nodes through the CENTRAL api (needs a key).",
        no_args_is_help=True,
    )
    app.add_typer(model_app, name="model")
    app.add_typer(runtime_app, name="runtime")
    app.add_typer(store_app, name="store")
    app.add_typer(fleet_app, name="fleet")
    state = _Context(
        plane_factory=(lambda: control_plane) if control_plane is not None else plane_factory
    )

    @app.callback()
    def main(
        ctx: typer.Context,
        json_output: bool = typer.Option(
            False,
            "--json",
            help="Emit stable, compact JSON for automation.",
        ),
    ) -> None:
        state.json_output = json_output
        ctx.obj = state

    # ------------------------------------------------------- node-local store
    # Seeding is otherwise reachable only through `serving/seed.requested`, an
    # ordinary event on the shared `docie-serving` app -- so it lands on
    # whichever serving node Inngest picks and a specific node cannot be
    # targeted (#562). The blob store is node-local, so "deploy the same model
    # on two nodes" needs the GGUF on both, which until node-addressed
    # lifecycle lands (#535) means seeding each node directly. These commands
    # are that path: they run against the store of whichever node executes
    # them, and never emit an event.

    def _store() -> Any:
        from docie_bench.serving.model_store import ModelStore
        from docie_bench.serving.paths import model_store_root

        return ModelStore(model_store_root())

    @store_app.command("list")
    def store_list(ctx: typer.Context) -> None:
        """What THIS node actually holds — the input a deploy here needs."""
        store = _store()
        _emit(
            ctx,
            [
                {
                    "name": entry.name,
                    "family": entry.family,
                    "source": entry.source,
                    "model_path": entry.model_path.as_posix(),
                }
                for entry in store.list()
            ],
        )

    def _publish_entry(ctx: typer.Context, name: str) -> dict[str, Any]:
        """Publish one local store entry to the shared repository.

        Separated from the seed commands because a local seed succeeding and a
        remote publication succeeding are different outcomes (#595). Seeding
        populates THIS node; publishing makes the model available to the
        fleet, and either can fail without the other being wrong.
        """
        from docie_bench.serving.model_repository import (
            iter_store_files,
            repository_from_settings,
        )

        repository = repository_from_settings()
        if repository is None:
            raise RuntimeError(
                "no shared model repository is configured; set "
                "DOCIE_MODEL_STORE_S3_BUCKET (and the matching credentials) "
                "on the node that publishes"
            )
        entry = _store().entry(name)
        from docie_bench.serving.node_identity import resolve_node_id

        kind = "snapshot" if Path(entry.model_path).is_dir() else "gguf"
        result = repository.publish(
            name=entry.name,
            family=entry.family,
            kind=kind,
            files=list(iter_store_files(entry)),
            source=entry.source,
            published_by=resolve_node_id(),
        )
        return {
            "name": result.name,
            "manifest": result.manifest_digest,
            "uploaded": list(result.uploaded),
            "skipped": list(result.skipped),
            "already_complete": result.already_complete,
        }

    @store_app.command("publish")
    def store_publish(
        ctx: typer.Context,
        name: str = typer.Argument(..., help="Store name to publish."),
    ) -> None:
        """Publish a model this node holds to the shared repository.

        Uploads every artifact that is not already there, writes the immutable
        manifest, and moves the pointer LAST -- so an interrupted publication
        leaves orphaned blobs and no deployable model, never a half-published
        one. A model whose artifacts are all present re-publishes without
        uploading anything.
        """
        try:
            _emit(ctx, _publish_entry(ctx, name))
        except Exception as exc:
            _emit_error(ctx, exc)
            raise typer.Exit(code=1) from exc

    @store_app.command("seed-ollama")
    def store_seed_ollama(
        ctx: typer.Context,
        reference: str = typer.Argument(..., help="Ollama reference, e.g. lfm2.5:350m."),
        name: str = typer.Option(..., "--name", help="Store name to register it under."),
        family: str = typer.Option(..., "--family", help="Model family contract."),
        mmproj: str | None = typer.Option(None, help="Explicit vision projector path."),
        copy: bool = typer.Option(
            False, "--copy", help="Copy instead of hard-linking the blob."
        ),
        publish: bool = typer.Option(
            False,
            "--publish",
            help="Also publish to the shared repository once the seed succeeds.",
        ),
    ) -> None:
        """Seed from a model Ollama already pulled on THIS machine.

        Idempotent: a store that already holds a complete copy re-registers it
        rather than re-reading the blob.
        """
        try:
            entry = _store().seed_from_ollama(
                reference,
                name=name,
                family=family,
                link=not copy,
                mmproj_source=mmproj,
            )
            payload = {
                "name": entry.name,
                "model_path": entry.model_path.as_posix(),
            }
        except Exception as exc:
            _emit_error(ctx, exc)
            raise typer.Exit(code=1) from exc
        # The seed has SUCCEEDED by here. A publication failure is reported as
        # its own outcome and must not make the local result look wrong -- the
        # weights are on this node either way.
        if publish:
            try:
                payload["published"] = _publish_entry(ctx, entry.name)
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                payload["published"] = None
                payload["publish_error"] = str(exc)
        _emit(ctx, payload)

    @store_app.command("add-gguf")
    def store_add_gguf(
        ctx: typer.Context,
        model_gguf: str = typer.Argument(..., help="Path to a GGUF already on disk."),
        name: str = typer.Option(..., "--name", help="Store name to register it under."),
        family: str = typer.Option(..., "--family", help="Model family contract."),
        mmproj: str | None = typer.Option(None, help="Vision projector path."),
        copy: bool = typer.Option(
            False, "--copy", help="Copy instead of hard-linking the file."
        ),
        publish: bool = typer.Option(
            False,
            "--publish",
            help="Also publish to the shared repository once the import succeeds.",
        ),
    ) -> None:
        """Register a GGUF already on THIS node's disk.

        The path an operator uses for a model downloaded onto this machine,
        so it carries ``--publish`` for the same reason ``seed-ollama`` does:
        otherwise the one route that handles an arbitrary local GGUF is also
        the one that cannot share it with the fleet.
        """
        try:
            entry = _store().add_gguf(
                name=name,
                family=family,
                model_gguf=model_gguf,
                mmproj=mmproj,
                source="cli",
                link=not copy,
            )
            payload = {
                "name": entry.name,
                "model_path": entry.model_path.as_posix(),
            }
        except Exception as exc:
            _emit_error(ctx, exc)
            raise typer.Exit(code=1) from exc
        # The import has SUCCEEDED by here; a publication failure is its own
        # outcome and must not make the local result look wrong.
        if publish:
            try:
                payload["published"] = _publish_entry(ctx, entry.name)
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                payload["published"] = None
                payload["publish_error"] = str(exc)
        _emit(ctx, payload)

    @model_app.command("list")
    def model_list(ctx: typer.Context) -> None:
        _execute(ctx, lambda plane: plane.list_models())

    @model_app.command("show")
    def model_show(
        ctx: typer.Context,
        model: str = typer.Argument(..., help="Model identity or alias."),
    ) -> None:
        _execute(ctx, lambda plane: plane.show_model(model))

    @model_app.command("pull")
    def model_pull(
        ctx: typer.Context,
        model: str = typer.Argument(..., help="Canonical model identity."),
        runtime: str | None = typer.Option(None, help="Validate for this runtime."),
        revision: str | None = typer.Option(None, help="Pinned source revision."),
        trust_remote_code: bool = typer.Option(
            False,
            "--trust-remote-code",
            help="Explicitly allow model-provided code.",
        ),
    ) -> None:
        _execute(
            ctx,
            lambda plane: plane.pull_model(
                model,
                runtime=runtime,
                revision=revision,
                trust_remote_code=trust_remote_code,
            ),
        )

    @model_app.command("remove")
    def model_remove(
        ctx: typer.Context,
        model: str = typer.Argument(..., help="Model identity or alias."),
    ) -> None:
        _execute(ctx, lambda plane: plane.remove_model(model))

    @runtime_app.command("list")
    def runtime_list(ctx: typer.Context) -> None:
        _execute(ctx, lambda plane: plane.list_runtimes())

    @runtime_app.command("probe")
    def runtime_probe(
        ctx: typer.Context,
        runtime: str = typer.Argument(..., help="Runtime adapter name."),
    ) -> None:
        _execute(ctx, lambda plane: plane.probe_runtime(runtime))

    @app.command("list")
    def deployment_list(ctx: typer.Context) -> None:
        """List managed deployments."""
        _execute(ctx, lambda plane: plane.list_deployments())

    @app.command()
    def status(
        ctx: typer.Context,
        name: str = typer.Argument(..., help="Deployment name."),
    ) -> None:
        """Show deployment status and health."""
        _execute(ctx, lambda plane: plane.deployment_status(name))

    @app.command()
    def serve(
        ctx: typer.Context,
        model: str = typer.Argument(..., help="Model identity or alias."),
        name: str | None = typer.Option(None, "--name", "--alias", help="Stable deployment name."),
        runtime: str | None = typer.Option(
            None,
            help="Runtime adapter; planner chooses by default.",
        ),
        replicas: int = typer.Option(1, min=1, help="Desired local replica count."),
    ) -> None:
        """Create and start a model deployment."""
        _execute(
            ctx,
            lambda plane: plane.serve(model, name=name, runtime=runtime, replicas=replicas),
        )

    @app.command()
    def up(
        ctx: typer.Context,
        name: str = typer.Argument(..., help="Model name in the canonical GGUF store."),
        port: int | None = typer.Option(
            None,
            min=1,
            max=65535,
            help="Bind port. Omit to auto-allocate the first free port in "
            "DOCIE_SERVING_PORT_RANGE_* (8088-8188 by default).",
        ),
        ctx_size: int = typer.Option(
            DEFAULT_DEPLOY_CONTEXT_LENGTH, "--ctx-size", min=1, help="llama.cpp context size."
        ),
        deployment_name: str | None = typer.Option(
            None,
            "--deployment-name",
            help="Record name. Defaults to the store name, which is NOT unique "
            "across nodes -- pass a distinct name per node in a fleet.",
        ),
    ) -> None:
        """Serve a store model in the background with its family's llama-server flags.

        FLEET NOTE: the record name defaults to the store name, so running this
        on two nodes for the same model creates two records with the same name.
        `ModelPlacement.name` is the primary key, so the second node's publish
        overwrites the first's and one replica becomes unroutable while still
        holding its RAM. Pass ``--deployment-name`` to give each node's record a
        distinct name. The scale path mints names for this reason (#530); this
        direct path keeps the store name so single-node use is unchanged.
        """
        _execute(
            ctx,
            lambda plane: plane.up(
                name,
                port=port,
                context_length=ctx_size,
                deployment_name=deployment_name,
            ),
            # Read the actual port back off the record: with no --port the control
            # plane allocates one, so a hardcoded 8088 hint would be wrong.
            hint=lambda record: (
                f"\nServing '{name}' in the background on {_deploy_endpoint(record)}\n"
                f"The model is still loading; once it answers, run:\n"
                f"docie-bench benchmark run --model-profile {name}"
            ),
        )

    @app.command()
    def start(ctx: typer.Context, name: str = typer.Argument(..., help="Deployment name.")) -> None:
        """Start an existing stopped deployment."""
        _execute(ctx, lambda plane: plane.start(name))

    @app.command()
    def stop(ctx: typer.Context, name: str = typer.Argument(..., help="Deployment name.")) -> None:
        """Stop a deployment while retaining its specification."""
        _execute(ctx, lambda plane: plane.stop(name))

    @app.command()
    def plan(
        ctx: typer.Context,
        model: str = typer.Argument(..., help="Model identity or alias."),
        runtime: str | None = typer.Option(None, help="Runtime adapter; compare all by default."),
        replicas: int = typer.Option(1, min=1, help="Replica count to assess."),
    ) -> None:
        """Assess compatibility and resources without launching."""
        _execute(ctx, lambda plane: plane.plan(model, runtime=runtime, replicas=replicas))

    @app.command()
    def doctor(
        ctx: typer.Context,
        probe_ports: bool = typer.Option(
            False,
            help=(
                "Try binding the deploy port window. Only meaningful INSIDE the "
                "serving container -- on the host it probes a different network "
                "namespace."
            ),
        ),
        probe_inngest: bool = typer.Option(
            True, help="Open a TCP connection to INNGEST_BASE_URL."
        ),
    ) -> None:
        """Check whether THIS machine is correctly configured as a fleet node.

        Run it before a new node takes traffic, and again after changing its
        .env. Exits non-zero when a check FAILS, so a bringup script can gate
        on it:

            docie-serving doctor && docker compose -f docker-compose.serving-node.yml up -d

        Every check corresponds to a misconfiguration that is otherwise silent
        or presents as a wrong answer rather than an error -- see
        serving/preflight.py for why each one is worth a probe.
        """
        from docie_bench.serving.preflight import Severity, collect_facts, evaluate, worst

        # An unreachable database should be a quick FAIL, not a doctor that
        # hangs on the driver's default connect timeout.
        os.environ.setdefault("PGCONNECT_TIMEOUT", "5")
        checks = evaluate(collect_facts(probe_ports=probe_ports, probe_inngest=probe_inngest))
        if _state(ctx).json_output:
            _emit(ctx, [check.as_dict() for check in checks])
        else:
            _render_checks(checks)
        if worst(checks) is Severity.FAIL:
            raise typer.Exit(code=1)

    @app.command()
    def gateway(
        host: str = typer.Option("127.0.0.1", help="Bind address."),
        port: int = typer.Option(8080, min=1, max=65535, help="Port for the unified endpoint."),
        models_config: Path = typer.Option(
            Path("configs/models.yaml"),
            exists=True,
            readable=True,
            help="Routing table: profiles -> upstream runtimes.",
        ),
    ) -> None:
        """Serve one OpenAI-compatible endpoint over every configured profile.

        Clients point base_url at http://<host>:<port>/v1 and request a model by
        profile name (or its upstream id); the gateway routes to the right runtime.
        """
        import uvicorn

        from docie_bench.serving.gateway import create_gateway_app

        typer.echo(
            f"docie gateway -> http://{host}:{port}/v1  (routing from {models_config})"
        )
        uvicorn.run(create_gateway_app(models_config), host=host, port=port)

    @app.command()
    def encoder(
        host: str = typer.Option("127.0.0.1", help="Bind address."),
        port: int = typer.Option(8090, min=1, max=65535, help="Port for the encoder endpoint."),
        model: str = typer.Option(
            "urchade/gliner_multi_pii-v1",
            help="Encoder model id (GLiNER or GLiNER2 checkpoint; zero-shot labels).",
        ),
        backend: str = typer.Option(
            "auto",
            help="auto | gliner | gliner2 (auto picks GLiNER2 for GLiNER2 model ids).",
        ),
        threshold: float = typer.Option(0.5, min=0.0, max=1.0, help="Default min confidence."),
    ) -> None:
        """Serve an encoder (PII/NER token classifier) behind the OpenAI surface.

        Text in as a user message, detected entities out as the assistant's
        JSON content — the analyzer contract the security proxy agent's
        `guard_model` consumes. Requires the `encoders` extra (GLiNER).
        """
        import uvicorn

        from docie_bench.encoders.server import create_encoder_app

        typer.echo(f"docie encoder [{model}] -> http://{host}:{port}/v1")
        uvicorn.run(
            create_encoder_app(
                model_id=model, backend_kind=backend, default_threshold=threshold
            ),
            host=host,
            port=port,
        )

    @app.command()
    def transformers(
        host: str = typer.Option("127.0.0.1", help="Bind address."),
        port: int = typer.Option(
            8091, min=1, max=65535, help="Port for the transformers endpoint."
        ),
        model: str = typer.Option(
            ...,
            help="Model snapshot dir or HF id served via AutoModel (last resort).",
        ),
        trust_remote_code: bool = typer.Option(
            False,
            "--trust-remote-code",
            help="Execute the repo's custom Python (needed for custom-code "
            "checkpoints). SECURITY: arbitrary code on the serving node — off by default.",
        ),
    ) -> None:
        """Serve a Hugging Face model behind the OpenAI surface via transformers.

        The LAST-RESORT path: unquantized AutoModel weights use ~2-3x the RAM of
        a GGUF Q4 and CPU inference is slower — prefer a GGUF repo whenever one
        exists. Chat + vision (image_url data URIs). Requires torch + transformers
        (the `ocr`/`encoders` extras).
        """
        import uvicorn

        from docie_bench.transformers_server.server import create_transformers_app

        typer.echo(f"docie transformers [{model}] -> http://{host}:{port}/v1")
        uvicorn.run(
            create_transformers_app(
                model_id=model, trust_remote_code=trust_remote_code
            ),
            host=host,
            port=port,
        )

    @app.command("multi-vector")
    def multi_vector(
        host: str = typer.Option("127.0.0.1", help="Bind address."),
        port: int = typer.Option(
            8092, min=1, max=65535, help="Port for the multi-vector rerank endpoint."
        ),
        model: str = typer.Option(
            ...,
            help="ColBERT / PyLate snapshot dir served via sentence-transformers "
            "MultiVectorEncoder (late-interaction, MaxSim scoring).",
        ),
    ) -> None:
        """Serve a multi-vector (ColBERT) retriever on /v1/rerank.

        Query + documents in, MaxSim relevance scores out -- the same rerank
        surface llama-server --reranking exposes for GGUF rerankers, here for
        the safetensors late-interaction shape. Requires sentence-transformers
        >= 6 (the `encoders` extra).
        """
        import uvicorn

        from docie_bench.multi_vector_server.server import create_multi_vector_app

        typer.echo(f"docie multi-vector [{model}] -> http://{host}:{port}/v1")
        uvicorn.run(create_multi_vector_app(model_id=model), host=host, port=port)

    @app.command("dense-embedding")
    def dense_embedding(
        host: str = typer.Option("127.0.0.1", help="Bind address."),
        port: int = typer.Option(
            8093, min=1, max=65535, help="Port for the text embeddings endpoint."
        ),
        model: str = typer.Option(
            ..., help="EmbeddingGemma 2 snapshot directory or Hugging Face model id."
        ),
        check_runtime: bool = typer.Option(False, "--check-runtime", hidden=True),
    ) -> None:
        """Serve text-only /v1/embeddings (requires the isolated embeddings extra)."""
        import uvicorn

        from docie_bench.embedding_server.server import create_embedding_app, dependency_error

        if check_runtime:
            import importlib.metadata

            reason = dependency_error()
            version = None
            if reason is None:
                version = (
                    f"sentence-transformers {importlib.metadata.version('sentence-transformers')}; "
                    f"transformers {importlib.metadata.version('transformers')}"
                )
            typer.echo(json.dumps({"reason": reason, "version": version}))
            raise typer.Exit(code=1 if reason else 0)

        typer.echo(f"docie dense-embedding [{model}] -> http://{host}:{port}/v1")
        uvicorn.run(create_embedding_app(model_id=model), host=host, port=port)

    # ----------------------------------------------------------------- fleet
    # The ONE group in this CLI that is not node-local. Every other command
    # operates on the control plane of whichever machine runs it; these speak
    # HTTP to the central api, because draining is a central decision -- it
    # has to refuse a drain that would strand a model, which only the catalog
    # can see -- and #634 settled that the DEPLOYMENT RUNNER calls the api
    # rather than the node writing its own drain straight to the catalog.
    #
    # So a serving node needs no new credential for this. Whatever replaces a
    # container does, and it already has one.

    @fleet_app.command("drain")
    def fleet_drain(
        ctx: typer.Context,
        node_id: str = typer.Argument(..., help="Node to take out of rotation."),
        wait: bool = typer.Option(
            False,
            "--wait",
            help="Block until the node holds no live runtime, or the timeout.",
        ),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT_S,
            "--timeout",
            help="Seconds to wait with --wait. Exceeding it exits non-zero.",
        ),
        acknowledge_outage: bool = typer.Option(
            False,
            "--acknowledge-outage",
            help=(
                "Drain even though this node holds a model's only live "
                "placement, taking that model offline."
            ),
        ),
    ) -> None:
        """Take a node out of the rotation, optionally waiting until it is free.

        With --wait this is the whole of a pre-stop hook: it exits 0 only once
        the node holds no live runtime, and non-zero on a timeout, an
        unreachable node, or a node that is not draining at all.

        WHAT EXIT 0 CLAIMS, precisely. The node waited out the longest request
        timeout its deployments declared -- capped at the drain ceiling -- and
        then stopped them. It does NOT prove every in-flight request finished.
        Raise the deployments' timeout_seconds if that bound is too tight, or
        accept the residual risk knowingly; serving/drain_wait.py says why no
        stronger claim is available.

        --acknowledge-outage is never added automatically, by this command or
        anything else. A refusal means a model is about to lose its last live
        replica, which is a decision rather than an obstacle.
        """
        result = _fleet_post(
            f"/v1/serving/nodes/{node_id}/drain",
            {"acknowledge_outage": acknowledge_outage},
        )
        if not wait:
            _emit(ctx, result)
            return

        outcome = wait_for_drain(_fleet_nodes, node_id, timeout_s=timeout)
        _emit(
            ctx,
            {
                **result,
                "readiness": str(outcome.readiness),
                "safe_to_replace": outcome.safe_to_replace,
                "waited_s": round(outcome.elapsed_s, 1),
                "polls": outcome.polls,
                "detail": outcome.detail,
            },
        )
        if not outcome.safe_to_replace:
            raise typer.Exit(code=1)

    @fleet_app.command("resume")
    def fleet_resume(
        ctx: typer.Context,
        node_id: str = typer.Argument(..., help="Node to admit again."),
    ) -> None:
        """Admit a node again, undoing a drain.

        Deliberately has no --wait: there is nothing to block on that would
        mean more than "the flag is cleared". The node restores the
        deployments the drain stopped on its own next cycle, and `fleet status`
        is how to watch that happen.
        """
        _emit(ctx, _fleet_post(f"/v1/serving/nodes/{node_id}/resume", None))

    @fleet_app.command("status")
    def fleet_status(
        ctx: typer.Context,
        node_id: str = typer.Argument("", help="One node, or omit for the fleet."),
    ) -> None:
        """What the central api says about the fleet, with drain readiness."""
        rows: list[Mapping[str, Any]] = list(_fleet_nodes())
        if node_id:
            row = find_node(rows, node_id)
            if row is None:
                typer.echo(f"no node {node_id!r} in the fleet", err=True)
                raise typer.Exit(code=1)
            rows = [row]
        _emit(
            ctx,
            [
                {
                    "node_id": row.get("node_id"),
                    "reporting": row.get("reporting"),
                    "deployments": row.get("deployments"),
                    "drain_state": row.get("drain_state"),
                    "readiness": str(assess_node(row)),
                }
                for row in rows
            ],
        )

    return app


def _deploy_endpoint(record: object) -> str:
    """Best-effort endpoint string from a to_data'd deployment record for hints."""
    if isinstance(record, Mapping):
        endpoint = record.get("endpoint")
        if isinstance(endpoint, str) and endpoint:
            return endpoint
        launch = record.get("spec")
        if isinstance(launch, Mapping):
            inner = launch.get("launch")
            if isinstance(inner, Mapping):
                port = inner.get("port")
                if port is not None:
                    return f"http://127.0.0.1:{port}/v1"
    return "the allocated port (see the record above)"


def _execute(
    ctx: typer.Context,
    operation: Callable[[ControlPlane], Awaitable[object]],
    *,
    hint: Callable[[object], str] | None = None,
) -> None:
    state = _state(ctx)

    async def invoke() -> object:
        return await operation(state.get_plane())

    try:
        result: object = asyncio.run(invoke())
    except Exception as exc:
        if state.json_output:
            payload = _json({"error": {"message": str(exc), "type": type(exc).__name__}})
            typer.echo(payload, err=True)
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _render(result, json_output=state.json_output)
    if hint is not None and not state.json_output:
        typer.echo(hint(result))


_SEVERITY_MARKS = {"pass": "ok  ", "info": "note", "warn": "WARN", "fail": "FAIL"}


def _render_checks(checks: Sequence[object]) -> None:
    """One line per verdict, with the consequence indented under the bad ones.

    Consequences are sentences rather than codes on purpose: the operator
    reading this is usually bringing up their first second node, and "WARN
    desired-state apply" on its own tells them nothing about what will not
    happen.
    """
    width = max((len(str(getattr(check, "name", ""))) for check in checks), default=0)
    for check in checks:
        severity = getattr(getattr(check, "severity", None), "value", "info")
        mark = _SEVERITY_MARKS.get(str(severity), "note")
        name = str(getattr(check, "name", ""))
        typer.echo(f"{mark}  {name:<{width}}  {getattr(check, 'detail', '')}")
        consequence = str(getattr(check, "consequence", "") or "")
        if consequence and severity in {"warn", "fail"}:
            for line in textwrap.wrap(consequence, width=84):
                typer.echo(f"      {' ' * width}  {line}")


def _emit(ctx: typer.Context, value: object) -> None:
    """Render a NODE-LOCAL result (no control plane, no event loop).

    Same rendering and the same error shape as :func:`_execute`, so `--json`
    behaves identically for the store commands.
    """
    _render(value, json_output=_state(ctx).json_output)


def _emit_error(ctx: typer.Context, exc: Exception) -> None:
    if _state(ctx).json_output:
        typer.echo(
            _json({"error": {"message": str(exc), "type": type(exc).__name__}}), err=True
        )
    else:
        typer.echo(f"Error: {exc}", err=True)


def _state(ctx: typer.Context) -> _Context:
    if not isinstance(ctx.obj, _Context):
        raise RuntimeError("CLI context was not initialized")
    return ctx.obj


def _render(value: object, *, json_output: bool) -> None:
    data = to_data(value)
    if json_output:
        typer.echo(_json(data))
    elif isinstance(data, list):
        _render_rows(data)
    elif isinstance(data, Mapping):
        width = max((len(str(key)) for key in data), default=0)
        for key, item in data.items():
            typer.echo(f"{str(key).upper():<{width}}  {_cell(item)}")
    else:
        typer.echo(_cell(data))


def _fleet_base_url() -> str:
    url = os.environ.get(API_URL_ENV_VAR, "").strip().rstrip("/")
    if not url:
        raise typer.BadParameter(
            f"set {API_URL_ENV_VAR} to the central api's base url "
            f"(e.g. https://studio.example.com). The fleet commands talk to "
            f"the central api, unlike the rest of this CLI"
        )
    return url


def _fleet_headers() -> dict[str, str]:
    key = os.environ.get(API_KEY_ENV_VAR, "").strip()
    if not key:
        raise typer.BadParameter(
            f"set {API_KEY_ENV_VAR} to an api key the central api accepts. "
            f"Draining is an operator action and the api authenticates it"
        )
    return {"Authorization": f"Bearer {key}"}


def _fleet_post(path: str, body: dict[str, Any] | None) -> dict[str, Any]:
    """POST to the central api, turning its refusals into readable exits.

    A 409 is the stranded-model refusal and carries the api's own sentence
    naming the models, which IS the decision the operator has to make -- so it
    is printed verbatim rather than summarised into a generic failure.
    """
    import httpx

    try:
        response = httpx.post(
            f"{_fleet_base_url()}{path}",
            json=body if body is not None else {},
            headers=_fleet_headers(),
            timeout=30.0,
        )
    except httpx.HTTPError as exc:
        typer.echo(f"cannot reach the central api: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if response.status_code >= 400:
        detail = response.text
        with contextlib.suppress(ValueError, AttributeError):
            detail = str(response.json().get("detail") or detail)
        typer.echo(f"{response.status_code}: {detail}", err=True)
        raise typer.Exit(code=1)
    decoded = response.json()
    return decoded if isinstance(decoded, dict) else {"result": decoded}


def _fleet_nodes() -> list[dict[str, Any]]:
    """The fleet as the central api reports it.

    Raises on a failed read rather than returning nothing: ``wait_for_drain``
    treats an exception as a transient read failure and retries, where an empty
    list would look like a node that has vanished.
    """
    import httpx

    response = httpx.get(
        f"{_fleet_base_url()}/v1/serving/nodes",
        headers=_fleet_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    rows = response.json().get("nodes")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _render_rows(rows: list[object]) -> None:
    if not rows:
        typer.echo("No results.")
        return
    if not all(isinstance(row, Mapping) for row in rows):
        for row in rows:
            typer.echo(_cell(row))
        return
    mappings = [row for row in rows if isinstance(row, Mapping)]
    columns = sorted({str(key) for row in mappings for key in row})
    widths = {
        column: max(len(column.upper()), *(len(_cell(row.get(column))) for row in mappings))
        for column in columns
    }
    typer.echo("  ".join(column.upper().ljust(widths[column]) for column in columns))
    for row in mappings:
        typer.echo("  ".join(_cell(row.get(column)).ljust(widths[column]) for column in columns))


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, str):
        return _json(value)
    return str(value)


def _json(value: object) -> str:
    return json.dumps(to_data(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


app = create_app()


if __name__ == "__main__":
    app()
