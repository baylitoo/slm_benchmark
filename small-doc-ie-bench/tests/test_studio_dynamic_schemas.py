"""GET/POST/DELETE /v1/studio/schemas/dynamic -- the missing "define once,
reuse by name" persistence layer for DynamicSchemaSpec (schemas/dynamic.py),
which was already fully validated and already compiled into a working
extraction schema, but only ever request-scoped until now."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import docie_bench.api as api
from docie_bench.storage.db import dispose_engine, init_engine


@pytest.fixture(autouse=True)
def schema_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'schemas.db'}")
    yield
    dispose_engine()


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def _payload(document_type: str = "invoice_custom") -> dict:
    return {
        "document_type": document_type,
        "fields": [
            {"name": "vendor_name", "type": "string"},
            {"name": "total", "type": "money"},
        ],
    }


def test_creates_and_lists_a_dynamic_schema(client: TestClient) -> None:
    created = client.post("/v1/studio/schemas/dynamic", json=_payload())
    assert created.status_code == 201
    assert created.json()["name"] == "invoice_custom"

    listed = client.get("/v1/studio/schemas/dynamic")
    assert listed.status_code == 200
    names = {s["name"] for s in listed.json()}
    assert names == {"invoice_custom"}


def test_fetches_one_schema_by_name(client: TestClient) -> None:
    client.post("/v1/studio/schemas/dynamic", json=_payload())

    resp = client.get("/v1/studio/schemas/dynamic/invoice_custom")

    assert resp.status_code == 200
    assert resp.json()["spec"]["document_type"] == "invoice_custom"


def test_fetch_unknown_name_is_404(client: TestClient) -> None:
    resp = client.get("/v1/studio/schemas/dynamic/does_not_exist")

    assert resp.status_code == 404
    assert resp.headers.get("X-Docie-Error") == "not_found"


def test_duplicate_document_type_is_409(client: TestClient) -> None:
    client.post("/v1/studio/schemas/dynamic", json=_payload())

    resp = client.post("/v1/studio/schemas/dynamic", json=_payload())

    assert resp.status_code == 409


def test_invalid_spec_is_422(client: TestClient) -> None:
    resp = client.post(
        "/v1/studio/schemas/dynamic",
        json={"document_type": "Not Snake Case", "fields": []},
    )

    assert resp.status_code == 422


def test_deletes_a_schema(client: TestClient) -> None:
    client.post("/v1/studio/schemas/dynamic", json=_payload())

    resp = client.delete("/v1/studio/schemas/dynamic/invoice_custom")

    assert resp.status_code == 200
    assert client.get("/v1/studio/schemas/dynamic/invoice_custom").status_code == 404


def test_delete_unknown_name_is_404(client: TestClient) -> None:
    resp = client.delete("/v1/studio/schemas/dynamic/does_not_exist")

    assert resp.status_code == 404
    assert resp.headers.get("X-Docie-Error") == "not_found"


def test_no_database_returns_empty_list_not_500(client: TestClient) -> None:
    dispose_engine()

    resp = client.get("/v1/studio/schemas/dynamic")

    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Worker-side resolution: _run_extraction resolves dynamic_schema_name and
# passes the saved spec through to ExtractionService, which already accepts
# schema_mode="dynamic" + an inline spec (extract/service.py) -- no changes
# needed there, only the by-name lookup this module adds.
# ---------------------------------------------------------------------------


def test_extraction_resolves_saved_schema_by_name(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post("/v1/studio/schemas/dynamic", json=_payload())

    from docie_bench.inngest import functions
    from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation

    captured: dict = {}

    class _FakeService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def extract_from_text(self, **kwargs) -> ExtractionResponse:
            captured.update(kwargs)
            return ExtractionResponse(
                request_id="req-1",
                schema_name="invoice_custom",
                model_profile="test-profile",
                document_hash="deadbeef",
                result={},
                validation=ExtractionValidation(valid=True, errors=[], warnings=[]),
                latency_ms=1,
            )

    monkeypatch.setattr(functions, "ExtractionService", _FakeService)
    monkeypatch.setattr("docie_bench.storage.audit.save_extraction_audit", lambda *a, **k: None)

    result = asyncio.run(
        functions._run_extraction(
            {"text": "some text", "tenant_id": "t1", "dynamic_schema_name": "invoice_custom"}
        )
    )

    assert result["schema_name"] == "invoice_custom"
    assert captured["schema_mode"] == "dynamic"
    assert captured["dynamic_schema"]["document_type"] == "invoice_custom"
    assert captured["schema_name"] == "invoice_custom"


def test_extraction_with_unknown_dynamic_schema_name_raises(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docie_bench.inngest import functions

    with pytest.raises(ValueError, match="not found"):
        asyncio.run(
            functions._run_extraction(
                {"text": "some text", "tenant_id": "t1", "dynamic_schema_name": "does_not_exist"}
            )
        )


# ---------------------------------------------------------------------------
# #462: "test this schema" -- an UNSAVED candidate spec run through the exact
# same extraction path inline, nothing written to the registry.
# ---------------------------------------------------------------------------


def test_extraction_accepts_an_inline_unsaved_dynamic_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docie_bench.inngest import functions
    from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation

    captured: dict = {}

    class _FakeService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def extract_from_text(self, **kwargs) -> ExtractionResponse:
            captured.update(kwargs)
            return ExtractionResponse(
                request_id="req-1",
                schema_name="invoice_draft",
                model_profile="test-profile",
                document_hash="deadbeef",
                result={},
                validation=ExtractionValidation(valid=True, errors=[], warnings=[]),
                latency_ms=1,
            )

    monkeypatch.setattr(functions, "ExtractionService", _FakeService)
    monkeypatch.setattr("docie_bench.storage.audit.save_extraction_audit", lambda *a, **k: None)

    result = asyncio.run(
        functions._run_extraction(
            {
                "text": "some text",
                "tenant_id": "t1",
                "dynamic_schema": _payload("invoice_draft"),
            }
        )
    )

    assert result["schema_name"] == "invoice_draft"
    assert captured["schema_mode"] == "dynamic"
    assert captured["dynamic_schema"]["document_type"] == "invoice_draft"
    assert captured["schema_name"] == "invoice_draft"


def test_inline_dynamic_schema_wins_over_a_schema_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller testing an edited-but-unsaved schema must see the EDITED
    version, not whatever happens to already be saved under the same name."""
    from docie_bench.inngest import functions
    from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation

    captured: dict = {}

    class _FakeService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def extract_from_text(self, **kwargs) -> ExtractionResponse:
            captured.update(kwargs)
            return ExtractionResponse(
                request_id="req-1",
                schema_name="invoice_custom",
                model_profile="test-profile",
                document_hash="deadbeef",
                result={},
                validation=ExtractionValidation(valid=True, errors=[], warnings=[]),
                latency_ms=1,
            )

    monkeypatch.setattr(functions, "ExtractionService", _FakeService)
    monkeypatch.setattr("docie_bench.storage.audit.save_extraction_audit", lambda *a, **k: None)

    edited_spec = _payload("invoice_custom")
    edited_spec["fields"].append({"name": "notes", "type": "string"})

    asyncio.run(
        functions._run_extraction(
            {
                "text": "some text",
                "tenant_id": "t1",
                "dynamic_schema_name": "invoice_custom",
                "dynamic_schema": edited_spec,
            }
        )
    )

    field_names = {f["name"] for f in captured["dynamic_schema"]["fields"]}
    assert "notes" in field_names


def test_trigger_extract_rejects_an_invalid_inline_dynamic_schema(client: TestClient) -> None:
    resp = client.post(
        "/v1/studio/extract",
        json={
            "text": "some text",
            "dynamic_schema": {"document_type": "Not Snake Case", "fields": []},
        },
    )

    assert resp.status_code == 422


def test_edits_saved_schema_and_compiles_variable_nested_lists(client: TestClient) -> None:
    from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder

    client.post("/v1/studio/schemas/dynamic", json=_payload("resume"))
    original = client.get("/v1/studio/schemas/dynamic/resume").json()
    spec = {"document_type": "resume", "fields": [
        {"name": "experience", "type": "list", "fields": [
            {"name": "company", "type": "string"},
            {"name": "projects", "type": "list", "fields": [
                {"name": "title", "type": "string"},
            ]},
        ]},
        {"name": "education", "type": "list", "fields": [
            {"name": "school", "type": "object", "fields": [
                {"name": "name", "type": "string"},
            ]},
        ]},
    ]}
    response = client.put("/v1/studio/schemas/dynamic/resume", json=spec)
    assert response.status_code == 200
    saved = response.json()
    assert saved["created_at"] == original["created_at"]
    assert saved["updated_at"] != original["updated_at"]
    assert len(client.get("/v1/studio/schemas/dynamic").json()) == 1
    fetched = client.get("/v1/studio/schemas/dynamic/resume").json()
    assert fetched["spec"] == saved["spec"]
    model = DynamicTemplateBuilder.build_model(DynamicSchemaSpec.model_validate(fetched["spec"]))
    # The same item definition accepts empty, single, and many repeated items.
    for size in (0, 1, 5):
        parsed = model.model_validate({
            "experience": [{"company": {"value": "Acme"}, "projects": [
                {"title": {"value": "A"}}, {"title": {"value": "B"}},
            ]} for _ in range(size)],
            "education": [{"school": {"name": {"value": "University"}}}],
        })
        assert len(parsed.experience) == size
        assert len(parsed.education) == 1


def test_update_unknown_schema_is_404(client: TestClient) -> None:
    response = client.put("/v1/studio/schemas/dynamic/invoice_custom", json=_payload())
    assert response.status_code == 404
    assert response.headers["X-Docie-Error"] == "not_found"


def test_invalid_update_preserves_original(client: TestClient) -> None:
    client.post("/v1/studio/schemas/dynamic", json=_payload())
    original = client.get("/v1/studio/schemas/dynamic/invoice_custom").json()
    for invalid in (_payload("different_name"), {"document_type": "invoice_custom", "fields": []}):
        response = client.put("/v1/studio/schemas/dynamic/invoice_custom", json=invalid)
        assert response.status_code == 422
        assert client.get("/v1/studio/schemas/dynamic/invoice_custom").json() == original


def test_update_without_database_is_503(client: TestClient) -> None:
    dispose_engine()
    response = client.put("/v1/studio/schemas/dynamic/invoice_custom", json=_payload())
    assert response.status_code == 503
