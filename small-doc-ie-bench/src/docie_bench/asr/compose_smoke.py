"""Isolated end-to-end Compose smoke for the managed CPU ASR path.

This is an operator release gate, not a pytest integration test. It creates a
uniquely named Compose project, seeds a small converted Whisper checkpoint via
the Studio API, deploys it through the serving control plane, and transcribes a
generated WAV through the public API. Exact-project volumes are removed during
cleanup; existing Compose projects are never addressed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MODEL_REPO = "Systran/faster-whisper-tiny.en"
DEFAULT_MODEL_NAME = "asr-smoke-tiny-en"
DEFAULT_PROJECT_PREFIX = "docie-asr-smoke"
_PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{7,62}$")
_PROTECTED_PROJECTS = {"docie", "small-doc-ie-bench", "small_doc_ie_bench"}


class SmokeError(RuntimeError):
    """Release-gate failure with an actionable message."""


@dataclass(frozen=True)
class SmokeConfig:
    project: str
    compose_file: Path
    evidence_path: Path
    model_repo: str = DEFAULT_MODEL_REPO
    model_name: str = DEFAULT_MODEL_NAME
    seed_timeout_seconds: float = 900.0
    deploy_timeout_seconds: float = 600.0
    poll_seconds: float = 3.0
    build: bool = True
    allow_dirty: bool = False


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def validate_project_name(project: str) -> str:
    """Require an unmistakably smoke-scoped Compose project name."""

    if not _PROJECT_RE.fullmatch(project) or not project.startswith(f"{DEFAULT_PROJECT_PREFIX}-"):
        raise ValueError(
            f"Compose project must match {DEFAULT_PROJECT_PREFIX}-<unique-suffix> "
            "using lowercase letters, digits, '_' or '-'"
        )
    if project in _PROTECTED_PROJECTS:
        raise ValueError(f"refusing protected Compose project name {project!r}")
    return project


def default_project_name() -> str:
    return f"{DEFAULT_PROJECT_PREFIX}-{datetime.now(UTC):%Y%m%d%H%M%S}-{secrets.token_hex(3)}"


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _unique_local_ports(names: Sequence[str]) -> dict[str, int]:
    ports: dict[str, int] = {}
    allocated: set[int] = set()
    for name in names:
        port = _free_local_port()
        while port in allocated:
            port = _free_local_port()
        ports[name] = port
        allocated.add(port)
    return ports


def generate_test_wav(path: Path, *, seconds: float = 1.25, sample_rate: int = 16_000) -> None:
    """Generate deterministic, non-copyrighted speech-shaped tone audio."""

    frames = bytearray()
    for index in range(round(seconds * sample_rate)):
        at = index / sample_rate
        envelope = min(1.0, at * 12.0, max(0.0, (seconds - at) * 12.0))
        carrier = math.sin(2 * math.pi * 220 * at)
        modulation = 0.55 + 0.45 * math.sin(2 * math.pi * 4.0 * at)
        frames.extend(struct.pack("<h", round(7_000 * envelope * modulation * carrier)))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(frames)


class ComposeProject:
    """All Docker mutations scoped to one validated Compose project."""

    def __init__(
        self,
        project: str,
        compose_file: Path,
        env_file: Path,
        *,
        runner: CommandRunner = subprocess.run,
    ) -> None:
        self.project = validate_project_name(project)
        self.compose_file = compose_file.resolve()
        self.env_file = env_file.resolve()
        self.runner = runner

    def command(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [
            "docker",
            "compose",
            "--project-name",
            self.project,
            "--env-file",
            str(self.env_file),
            "--file",
            str(self.compose_file),
            *args,
        ]
        return self.runner(
            command,
            cwd=self.compose_file.parent,
            check=check,
            capture_output=True,
            text=True,
        )

    def cleanup(self) -> subprocess.CompletedProcess[str]:
        # Exact validated project only. Volumes contain smoke-only DB/model data.
        return self.command("down", "--volumes", "--remove-orphans", check=False)


def _write_env(path: Path, *, api_key: str, ports: dict[str, int]) -> None:
    values = {
        "APP_ENV": "asr-compose-smoke",
        "AUTH_REQUIRED": "true",
        "API_KEYS": json.dumps({api_key: "asr-smoke-tenant"}, separators=(",", ":")),
        "POSTGRES_PASSWORD": secrets.token_hex(24),
        "INNGEST_EVENT_KEY": secrets.token_hex(32),
        "INNGEST_SIGNING_KEY": secrets.token_hex(32),
        "INNGEST_DEV": "0",
        "API_PORT": str(ports["api"]),
        "POSTGRES_PORT": str(ports["postgres"]),
        "INNGEST_HOST_PORT": str(ports["inngest"]),
        "INNGEST_CONNECT_HOST_PORT": str(ports["inngest_connect"]),
        "NEXT_PUBLIC_INNGEST_BASE_URL": f"http://localhost:{ports['inngest']}",
        "DOCIE_SERVING_MEM_LIMIT": "4g",
        "DOCIE_SERVING_IDLE_TTL_SECONDS": "0",
        "ASR_DEVICE": "cpu",
        "ASR_COMPUTE_TYPE": "int8",
        "DOCIE_ENV_FILE": path.resolve().as_posix(),
    }
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")


def _wait_for(
    description: str,
    probe: Callable[[], Any | None],
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = probe()
            if result is not None:
                return result
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
        time.sleep(poll_seconds)
    suffix = f"; last probe error: {last_error}" if last_error else ""
    raise SmokeError(f"timed out waiting for {description}{suffix}")


def _hf_revision(client: httpx.Client, repo: str) -> str:
    response = client.get(f"https://huggingface.co/api/models/{repo}")
    response.raise_for_status()
    revision = str(response.json().get("sha") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SmokeError(f"Hugging Face returned no immutable revision for {repo!r}")
    return revision


def _git(command: Sequence[str], cwd: Path) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise SmokeError("git is required to record source provenance")
    result = subprocess.run(  # noqa: S603 - fixed executable, fixed internal arguments
        [executable, *command], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _deployment_name(record: dict[str, Any]) -> str | None:
    spec = record.get("spec")
    return str(spec.get("name")) if isinstance(spec, dict) and spec.get("name") else None


def _healthy(client: httpx.Client) -> dict[str, Any] | None:
    response = client.get("/healthz")
    if response.status_code != 200:
        return None
    payload = response.json()
    return payload if isinstance(payload, dict) else None


def _deployment_absent(client: httpx.Client, name: str) -> bool | None:
    response = client.get("/v1/serving/deployments")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise SmokeError("deployment listing did not return a list during cleanup")
    return True if all(_deployment_name(item) != name for item in payload) else None


def _run_gate(config: SmokeConfig, compose: ComposeProject, api_key: str) -> dict[str, Any]:
    started = time.monotonic()
    timings: dict[str, float] = {}
    api_port = int(_read_env_value(compose.env_file, "API_PORT"))
    base_url = f"http://127.0.0.1:{api_port}"

    compose.command("config", "--quiet")
    up_args = ["up", "--detach"]
    if config.build:
        up_args.append("--build")
    up_args.extend(["postgres", "redis", "inngest", "serving", "api"])
    at = time.monotonic()
    compose.command(*up_args)
    timings["compose_up_seconds"] = time.monotonic() - at

    client = httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        timeout=httpx.Timeout(30.0, connect=5.0),
    )
    hf_client = httpx.Client(timeout=30.0)
    try:
        _wait_for(
            "API health",
            lambda: _healthy(client),
            timeout_seconds=180,
            poll_seconds=config.poll_seconds,
        )
        revision_before = _hf_revision(hf_client, config.model_repo)

        at = time.monotonic()
        seed = client.post(
            "/v1/studio/seed-hf",
            json={
                "repo": config.model_repo,
                "name": config.model_name,
                "family": "asr_whisper",
            },
        )
        seed.raise_for_status()
        seed_trigger = seed.json()
        seed_event = str(seed_trigger["event_ids"][0])

        def seeded() -> dict[str, Any] | None:
            response = client.get("/v1/studio/seeds")
            response.raise_for_status()
            row = next((item for item in response.json() if item["event_id"] == seed_event), None)
            if row and row["status"] == "failed":
                raise SmokeError(f"ASR seed failed: {row.get('error')}")
            return row if row and row["status"] == "completed" else None

        seed_result = _wait_for(
            "ASR snapshot seed",
            seeded,
            timeout_seconds=config.seed_timeout_seconds,
            poll_seconds=config.poll_seconds,
        )
        timings["seed_seconds"] = time.monotonic() - at
        revision_after = _hf_revision(hf_client, config.model_repo)
        if revision_before != revision_after:
            raise SmokeError(
                "model repository moved during seed; rerun so evidence maps to one revision"
            )

        at = time.monotonic()
        deploy = client.post(
            "/v1/studio/deploy",
            json={"model": config.model_name, "name": config.model_name},
        )
        deploy.raise_for_status()
        deploy_trigger = deploy.json()

        def ready() -> dict[str, Any] | None:
            response = client.get("/v1/serving/deployments")
            response.raise_for_status()
            record = next(
                (item for item in response.json() if _deployment_name(item) == config.model_name),
                None,
            )
            if record and str(record.get("state")) == "failed":
                raise SmokeError(f"ASR deployment failed: {record.get('last_error')}")
            return record if record and str(record.get("state")) == "ready" else None

        deployment = _wait_for(
            "managed ASR deployment readiness",
            ready,
            timeout_seconds=config.deploy_timeout_seconds,
            poll_seconds=config.poll_seconds,
        )
        timings["deploy_seconds"] = time.monotonic() - at
        endpoint = str(deployment.get("endpoint") or "")
        if not endpoint.startswith("http://serving:"):
            raise SmokeError(f"runtime endpoint is not private Compose routing: {endpoint!r}")

        with tempfile.TemporaryDirectory(prefix="docie-asr-audio-") as audio_dir:
            audio_path = Path(audio_dir) / "generated-tone.wav"
            generate_test_wav(audio_path)
            audio_sha256 = __import__("hashlib").sha256(audio_path.read_bytes()).hexdigest()
            audio_bytes = audio_path.read_bytes()

        arbitrary = client.post(
            "/v1/audio/transcriptions",
            data={"model": "arbitrary/hub-model"},
            files={"file": ("generated-tone.wav", audio_bytes, "audio/wav")},
        )
        if arbitrary.status_code != 404:
            raise SmokeError(
                f"arbitrary model selector returned {arbitrary.status_code}, expected 404"
            )

        at = time.monotonic()
        transcription = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": config.model_name,
                "language": "en",
                "response_format": "verbose_json",
                "temperature": "0",
            },
            files={"file": ("generated-tone.wav", audio_bytes, "audio/wav")},
            timeout=config.deploy_timeout_seconds,
        )
        transcription.raise_for_status()
        transcript = transcription.json()
        for field in ("text", "duration", "processing_seconds", "model", "backend"):
            if field not in transcript:
                raise SmokeError(f"verbose transcription omitted required evidence field {field!r}")
        if not isinstance(transcript["text"], str):
            raise SmokeError("verbose transcription text is not a string")
        if float(transcript["duration"]) <= 0 or float(transcript["processing_seconds"]) < 0:
            raise SmokeError("verbose transcription returned invalid timing evidence")
        if transcript["backend"] != "faster-whisper":
            raise SmokeError(f"unexpected ASR backend {transcript['backend']!r}")
        timings["transcription_seconds"] = time.monotonic() - at

        delete = client.delete(f"/v1/serving/deployments/{config.model_name}")
        delete.raise_for_status()
        _wait_for(
            "managed deployment cleanup",
            lambda: _deployment_absent(client, config.model_name),
            timeout_seconds=120,
            poll_seconds=config.poll_seconds,
        )
    finally:
        hf_client.close()
        client.close()

    timings["total_gate_seconds"] = time.monotonic() - started
    return {
        "status": "passed",
        "model": {
            "repo": config.model_repo,
            "revision": revision_before,
            "name": config.model_name,
        },
        "audio": {
            "source": "generated deterministic amplitude-modulated tone",
            "copyrighted": False,
            "sha256": audio_sha256,
        },
        "seed": seed_result,
        "deploy_trigger": deploy_trigger,
        "deployment": deployment,
        "routing_guard": {
            "selector": "arbitrary/hub-model",
            "status_code": arbitrary.status_code,
            "body": arbitrary.json(),
        },
        "transcription": transcript,
        "timings": timings,
    }


def _read_env_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise KeyError(key)


def run_compose_smoke(config: SmokeConfig) -> dict[str, Any]:
    """Run the release gate, always collect evidence/logs, always clean up."""

    validate_project_name(config.project)
    repo_root = config.compose_file.resolve().parent
    config.evidence_path.parent.mkdir(parents=True, exist_ok=True)
    dirty = bool(_git(["status", "--porcelain"], repo_root))
    if dirty and not config.allow_dirty:
        raise SmokeError(
            "release-gate source is dirty; commit/stash changes or pass --allow-dirty "
            "for a non-release diagnostic run"
        )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "config": {
            **asdict(config),
            "compose_file": str(config.compose_file),
            "evidence_path": str(config.evidence_path),
        },
        "source": {
            "commit": _git(["rev-parse", "HEAD"], repo_root),
            "tree": _git(["rev-parse", "HEAD^{tree}"], repo_root),
            "dirty": dirty,
        },
    }
    api_key = secrets.token_urlsafe(24)
    caught: BaseException | None = None
    with tempfile.TemporaryDirectory(prefix="docie-asr-compose-") as temp_dir:
        env_file = Path(temp_dir) / "smoke.env"
        ports = _unique_local_ports(("api", "postgres", "inngest", "inngest_connect"))
        _write_env(env_file, api_key=api_key, ports=ports)
        compose = ComposeProject(config.project, config.compose_file, env_file)
        try:
            evidence["result"] = _run_gate(config, compose, api_key)
            evidence["compose_ps"] = compose.command("ps", "--format", "json").stdout
            evidence["compose_images"] = compose.command("images", "--format", "json").stdout
        except BaseException as exc:  # noqa: BLE001 - persist diagnostics, then re-raise
            caught = exc
            evidence["result"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            logs = compose.command("logs", "--no-color", "--tail", "300", check=False)
            log_path = config.evidence_path.with_suffix(".logs.txt")
            log_path.write_text(logs.stdout + logs.stderr, encoding="utf-8")
            cleanup = compose.cleanup()
            evidence["cleanup"] = {
                "command_scope": config.project,
                "exit_code": cleanup.returncode,
                "stderr": cleanup.stderr.strip(),
                "volumes_removed": True,
            }
            if cleanup.returncode != 0 and caught is None:
                caught = SmokeError(f"exact-project cleanup failed: {cleanup.stderr.strip()}")

    evidence["finished_at"] = datetime.now(UTC).isoformat()
    config.evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if caught is not None:
        raise caught
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=default_project_name())
    parser.add_argument("--compose-file", type=Path, default=Path("docker-compose.yml"))
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("artifacts/asr-compose-smoke/evidence.json"),
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seed-timeout", type=float, default=900.0)
    parser.add_argument("--deploy-timeout", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a non-release diagnostic run from a dirty worktree",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = SmokeConfig(
        project=args.project,
        compose_file=args.compose_file,
        evidence_path=args.evidence,
        model_repo=args.model_repo,
        model_name=args.model_name,
        seed_timeout_seconds=args.seed_timeout,
        deploy_timeout_seconds=args.deploy_timeout,
        poll_seconds=args.poll_seconds,
        build=not args.no_build,
        allow_dirty=args.allow_dirty,
    )
    try:
        result = run_compose_smoke(config)
    except (SmokeError, ValueError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        print(f"ASR Compose smoke failed: {exc}")
        return 1
    print(f"ASR Compose smoke passed; evidence: {config.evidence_path.resolve()}")
    print(json.dumps(result["result"]["timings"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
