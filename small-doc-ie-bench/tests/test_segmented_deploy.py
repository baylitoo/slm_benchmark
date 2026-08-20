from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_compose_assigns_llama_only_to_serving_target() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    for name in ("api", "worker", "bench"):
        assert services[name]["build"]["target"] == "api-runtime"
    assert services["serving"]["build"]["target"] == (
        "${DOCIE_LLAMA_SERVER_TARGET:-serving-runtime-prebuilt}"
    )


def test_serving_encoder_extra_is_build_time_selectable() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    extras = compose["services"]["serving"]["build"]["args"]["PIP_EXTRAS"]

    assert extras == "${DOCIE_SERVING_EXTRAS:-ocr,encoders}"


def test_lightweight_target_does_not_depend_on_llama_builder() -> None:
    dockerfile = (ROOT / "infra" / "docker" / "Dockerfile.api").read_text(encoding="utf-8")
    lightweight = dockerfile.split("FROM app-runtime AS api-runtime", maxsplit=1)[1].split(
        "FROM app-runtime AS serving-runtime-source", maxsplit=1
    )[0]
    source = dockerfile.split("FROM app-runtime AS serving-runtime-source", maxsplit=1)[1]

    assert "llama-builder" not in lightweight
    assert "COPY --from=llama-builder" in source


def test_prebuilt_serving_target_uses_official_server_image() -> None:
    dockerfile = (ROOT / "infra" / "docker" / "Dockerfile.api").read_text(encoding="utf-8")
    prebuilt = dockerfile.split("FROM app-runtime AS serving-runtime-prebuilt", maxsplit=1)[1]

    assert "FROM ${LLAMA_SERVER_IMAGE} AS llama-prebuilt" in dockerfile
    assert "COPY --from=llama-prebuilt /app /opt/llama.cpp" in prebuilt
    assert "llama-server --version" in prebuilt
    assert "FROM python:3.11-slim-trixie AS app-runtime" in dockerfile


def test_makefile_exposes_segment_flag() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "SEGMENTS ?= api serving worker web" in makefile
    assert "docker compose up -d --build $(SEGMENTS)" in makefile


def test_python_build_context_excludes_local_heavy_artifacts() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".env", ".venv", "data", "models", "runs", "graphify-out"} <= ignored
