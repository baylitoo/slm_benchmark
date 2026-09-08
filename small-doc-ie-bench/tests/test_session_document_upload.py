"""POST /v1/studio/session-documents: upload a file for docs-search to see
during one conversation only (#296)."""

from __future__ import annotations

import asyncio
import base64

import pytest
from fastapi import HTTPException

from docie_bench.inngest.studio_api import (
    UploadSessionDocumentRequest,
    upload_session_document,
)
from docie_bench.settings import get_settings


@pytest.fixture
def session_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_SESSION_DOCUMENTS_ROOT", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _upload(b64: str, filename: str, session_id: str | None = None) -> dict:
    result = asyncio.run(
        upload_session_document(
            UploadSessionDocumentRequest(
                content_b64=b64, filename=filename, session_id=session_id
            ),
            tenant=None,  # type: ignore[arg-type]
        )
    )
    return result.model_dump()


def test_upload_mints_a_session_id_and_writes_the_file(session_root) -> None:
    b64 = base64.b64encode(b"%PDF-fake").decode("ascii")
    result = _upload(b64, "invoice.pdf")
    assert result["session_id"]
    assert result["stored_name"].endswith(".pdf")
    stored = session_root / result["session_id"] / result["stored_name"]
    assert stored.read_bytes() == b"%PDF-fake"


def test_second_upload_with_the_same_session_id_lands_in_the_same_directory(
    session_root,
) -> None:
    b64 = base64.b64encode(b"one").decode("ascii")
    first = _upload(b64, "a.pdf")
    second = _upload(base64.b64encode(b"two").decode("ascii"), "b.txt", first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert len(list((session_root / first["session_id"]).glob("*"))) == 2


def test_bad_base64_is_400(session_root) -> None:
    with pytest.raises(HTTPException) as exc:
        _upload("!!!not-base64!!!", "a.pdf")
    assert exc.value.status_code == 400


def test_unsupported_suffix_is_400(session_root) -> None:
    b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(HTTPException) as exc:
        _upload(b64, "a.docx")
    assert exc.value.status_code == 400


def test_unknown_session_id_is_400(session_root) -> None:
    b64 = base64.b64encode(b"x").decode("ascii")
    with pytest.raises(HTTPException) as exc:
        _upload(b64, "a.pdf", "a" * 32)
    assert exc.value.status_code == 400
