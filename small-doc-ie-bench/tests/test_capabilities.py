"""What this deployment accepts, read from its own configuration."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from docie_bench.ocr.factory import available_ocr_backends
from docie_bench.settings import get_settings


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    get_settings.cache_clear()
    from docie_bench.api import app

    with TestClient(app) as client:
        yield client
    get_settings.cache_clear()


def test_the_payload_answers_what_a_consumer_has_to_hard_code(client: TestClient) -> None:
    body = client.get("/v1/capabilities").json()
    assert set(body) == {"upload", "text", "metadata", "ocr_backends"}
    assert set(body["upload"]) == {
        "max_bytes",
        "max_request_body_bytes",
        "allowed_mime_types",
    }
    assert set(body["text"]) == {"max_chars", "max_ocr_blocks", "max_ocr_block_chars"}


def test_the_values_come_from_this_deployment_not_from_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole point: a consumer copying the documented defaults is wrong on
    # the first instance that customises them, and nothing tells either side.
    monkeypatch.setenv("AUTH_REQUIRED", "false")
    monkeypatch.setenv("ALLOWED_UPLOAD_MIME_TYPES", "application/pdf,image/webp")
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("MAX_OCR_BLOCKS", "250")
    get_settings.cache_clear()
    from docie_bench.api import app

    with TestClient(app) as client:
        body = client.get("/v1/capabilities").json()
    get_settings.cache_clear()

    assert body["upload"]["allowed_mime_types"] == ["application/pdf", "image/webp"]
    assert body["upload"]["max_bytes"] == 7 * 1024 * 1024
    assert body["text"]["max_ocr_blocks"] == 250


def test_the_mime_types_are_sorted_so_a_consumer_can_compare_them(
    client: TestClient,
) -> None:
    types = client.get("/v1/capabilities").json()["upload"]["allowed_mime_types"]
    assert types == sorted(types)


def test_every_ocr_backend_the_factory_accepts_is_listed(client: TestClient) -> None:
    listed = {entry["name"] for entry in client.get("/v1/capabilities").json()["ocr_backends"]}
    assert listed == {"liteparse", "tesseract", "paddleocr"}


def test_the_canonical_backend_carries_its_legacy_alias(client: TestClient) -> None:
    backends = {e["name"]: e for e in client.get("/v1/capabilities").json()["ocr_backends"]}
    assert backends["liteparse"]["aliases"] == ["pdf_text"]
    assert backends["liteparse"]["available"] is True


def test_an_optional_backend_reports_whether_its_import_is_present() -> None:
    # Reported from the module being importable, not by constructing one: the
    # optional backends raise only when built, and a GET must not pay for that.
    by_name = {entry["name"]: entry for entry in available_ocr_backends()}
    assert isinstance(by_name["tesseract"]["available"], bool)
    assert isinstance(by_name["paddleocr"]["available"], bool)


def test_nothing_about_models_keys_or_paths_is_exposed(client: TestClient) -> None:
    body: Any = client.get("/v1/capabilities").json()
    flat = repr(body).lower()
    for leaked in (
        "api_key",
        "database_url",
        "password",
        "token",
        "model_profile",
        "/home",
        "c:\\\\",
    ):
        assert leaked not in flat


def test_authentication_matches_the_other_v1_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.setenv("API_KEYS", "secret")
    get_settings.cache_clear()
    from docie_bench.api import app

    with TestClient(app, raise_server_exceptions=False) as client:
        schemas = client.get("/v1/schemas").status_code
        capabilities = client.get("/v1/capabilities").status_code
    get_settings.cache_clear()
    assert capabilities == schemas
