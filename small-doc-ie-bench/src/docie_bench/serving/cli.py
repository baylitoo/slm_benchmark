"""Ollama-like operations CLI for the serving control plane.

Run with ``python -m docie_bench.serving.cli``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

import typer

from docie_bench.serving.control_plane import ControlPlane, to_data


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
    app.add_typer(model_app, name="model")
    app.add_typer(runtime_app, name="runtime")
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

    return app


def _execute(
    ctx: typer.Context,
    operation: Callable[[ControlPlane], Awaitable[object]],
) -> None:
    state = _state(ctx)

    async def invoke() -> object:
        return await operation(state.get_plane())

    try:
        result: object = asyncio.run(invoke())
    except Exception as exc:
        if state.json_output:
            typer.echo(_json({"error": {"message": str(exc), "type": type(exc).__name__}}))
        else:
            typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _render(result, json_output=state.json_output)


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
