"""Seeding a specific node's blob store from its own shell (#562).

Seeding was reachable only through `serving/seed.requested`, an ordinary event
on the shared `docie-serving` app -- so it is delivered to whichever serving
node Inngest picks and a chosen node cannot be targeted. The blob store is
node-local, so "deploy the same model on two nodes" needs the GGUF on both,
which until node-addressed lifecycle lands (#535) means seeding each node
directly.

These commands are that path. They act on the store of whichever node runs
them and emit no event, which is exactly what makes them deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docie_bench.serving.cli import create_app
from docie_bench.settings import get_settings

runner = CliRunner()


@pytest.fixture
def node_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    get_settings.cache_clear()
    try:
        yield home
    finally:
        get_settings.cache_clear()


def _gguf(tmp_path: Path, name: str = "weights.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"GGUF" + b"\0" * 64)
    return path


def test_add_gguf_seeds_this_nodes_store(node_home: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        create_app(),
        [
            "--json",
            "store",
            "add-gguf",
            str(_gguf(tmp_path)),
            "--name",
            "lfm2.5-350m",
            "--family",
            "openai_chat",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["name"] == "lfm2.5-350m"
    # It landed in THIS node's store, under its serving home.
    assert (node_home / "models" / "index.json").is_file()


def test_the_seeded_model_is_then_listed(node_home: Path, tmp_path: Path) -> None:
    app = create_app()
    runner.invoke(
        app,
        [
            "store",
            "add-gguf",
            str(_gguf(tmp_path)),
            "--name",
            "lfm2.5-350m",
            "--family",
            "openai_chat",
        ],
    )

    listed = runner.invoke(app, ["--json", "store", "list"])

    assert listed.exit_code == 0, listed.output
    assert [row["name"] for row in json.loads(listed.output)] == ["lfm2.5-350m"]


def test_an_empty_store_lists_nothing_rather_than_failing(node_home: Path) -> None:
    result = runner.invoke(create_app(), ["--json", "store", "list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == []


def test_a_missing_gguf_fails_with_the_shared_error_shape(
    node_home: Path, tmp_path: Path
) -> None:
    """`--json` must stay machine-readable on the failure path too, since this
    is a command an operator scripts across several nodes."""
    result = runner.invoke(
        create_app(),
        [
            "--json",
            "store",
            "add-gguf",
            str(tmp_path / "absent.gguf"),
            "--name",
            "lfm2.5-350m",
            "--family",
            "openai_chat",
        ],
    )

    assert result.exit_code == 1
    assert "error" in json.loads(result.output)


def test_seeding_emits_no_event(node_home: Path, tmp_path: Path) -> None:
    """The whole point: an event would land on an arbitrary node, which is the
    behaviour these commands exist to avoid."""
    import docie_bench.inngest.client as client_module

    sent: list[object] = []

    async def fake_send(event: object) -> list[str]:
        sent.append(event)
        return ["evt"]

    # Patch the shared client rather than a local import, so any indirect send
    # would still be caught.
    original = client_module.inngest_client.send
    client_module.inngest_client.send = fake_send  # type: ignore[method-assign]
    try:
        result = runner.invoke(
            create_app(),
            [
                "store",
                "add-gguf",
                str(_gguf(tmp_path)),
                "--name",
                "lfm2.5-350m",
                "--family",
                "openai_chat",
            ],
        )
    finally:
        client_module.inngest_client.send = original  # type: ignore[method-assign]

    assert result.exit_code == 0, result.output
    assert sent == []
