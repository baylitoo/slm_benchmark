from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import pytest

import docie_bench.asr.compose_smoke as compose_smoke
from docie_bench.asr.compose_smoke import (
    ComposeProject,
    SmokeConfig,
    SmokeError,
    _read_env_value,
    _write_env,
    generate_test_wav,
    validate_project_name,
)


def test_compose_project_requires_unique_smoke_scope() -> None:
    assert validate_project_name("docie-asr-smoke-20260826-a1b2c3")
    for unsafe in ("", "docie", "small-doc-ie-bench", "docie-asr-smoke", "PRODUCTION"):
        with pytest.raises(ValueError, match="Compose project"):
            validate_project_name(unsafe)


def test_compose_commands_and_destructive_cleanup_are_exactly_scoped(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")
    env_file = tmp_path / "smoke.env"
    env_file.write_text("POSTGRES_PASSWORD=unused\n", encoding="utf-8")
    project = ComposeProject(
        "docie-asr-smoke-safe-123456",
        compose_file,
        env_file,
        runner=runner,
    )

    project.command("config", "--quiet")
    project.cleanup()

    assert len(calls) == 2
    for command in calls:
        assert command[:4] == [
            "docker",
            "compose",
            "--project-name",
            "docie-asr-smoke-safe-123456",
        ]
        assert command[4:6] == ["--env-file", str(env_file.resolve())]
        assert command[6:8] == ["--file", str(compose_file.resolve())]
    assert calls[-1][-3:] == [
        "down",
        "--volumes",
        "--remove-orphans",
    ]


def test_generated_audio_is_bounded_valid_pcm(tmp_path: Path) -> None:
    output = tmp_path / "generated.wav"
    generate_test_wav(output, seconds=0.25, sample_rate=8_000)
    assert output.stat().st_size < 8_000
    with wave.open(str(output), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 8_000
        assert audio.getnframes() == 2_000


def test_smoke_env_is_isolated_authenticated_and_cpu_only(tmp_path: Path) -> None:
    path = tmp_path / "smoke.env"
    _write_env(
        path,
        api_key="smoke-secret",
        ports={"api": 18080, "postgres": 15432, "inngest": 18288, "inngest_connect": 18289},
    )
    content = path.read_text(encoding="utf-8")
    assert 'API_KEYS={"smoke-secret":"asr-smoke-tenant"}' in content
    assert "AUTH_REQUIRED=true" in content
    assert "ASR_DEVICE=cpu" in content
    assert "ASR_COMPUTE_TYPE=int8" in content
    assert _read_env_value(path, "API_PORT") == "18080"
    assert _read_env_value(path, "DOCIE_ENV_FILE") == path.resolve().as_posix()


def test_compose_services_support_an_operator_supplied_env_file() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert compose.count("env_file: ${DOCIE_ENV_FILE:-.env}") == 4


def test_runbook_keeps_state_and_requires_tree_verification() -> None:
    runbook = Path("docs/asr-operations.md").read_text(encoding="utf-8")
    for invariant in (
        "## Release-gate smoke",
        "## Rollout checklist",
        "## Rollback procedure",
        "commit and tree",
        "serving-state",
        "artifact-store",
        "hf-cache",
        "Do not drop ASR tables",
    ):
        assert invariant in runbook


def test_release_gate_refuses_dirty_source_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compose_file = tmp_path / "docker-compose.yml"
    compose_file.write_text("services: {}\n", encoding="utf-8")

    def git(command: list[str], _cwd: Path) -> str:
        return " M src/file.py" if command == ["status", "--porcelain"] else "abc123"

    monkeypatch.setattr(compose_smoke, "_git", git)
    config = SmokeConfig(
        project="docie-asr-smoke-dirty-123456",
        compose_file=compose_file,
        evidence_path=tmp_path / "evidence.json",
    )
    with pytest.raises(SmokeError, match="source is dirty"):
        compose_smoke.run_compose_smoke(config)
