"""Session-scoped docs-search uploads (docie_bench.mcp_session_documents, #296)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from docie_bench import mcp_session_documents as sd
from docie_bench.settings import get_settings


@pytest.fixture
def session_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("MCP_SESSION_DOCUMENTS_ROOT", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_save_document_mints_a_fresh_session_id_when_none_given(session_root: Path) -> None:
    session_id, stored_name = sd.save_document(None, "invoice.pdf", b"%PDF-fake")
    assert sd._SESSION_ID_RE.match(session_id)
    assert stored_name.endswith(".pdf")
    assert (session_root / session_id / stored_name).read_bytes() == b"%PDF-fake"


def test_save_document_adds_a_second_file_to_the_same_session(session_root: Path) -> None:
    session_id, first = sd.save_document(None, "a.pdf", b"one")
    same_id, second = sd.save_document(session_id, "b.txt", b"two")
    assert same_id == session_id
    assert first != second
    assert len(list((session_root / session_id).glob("*"))) == 2


def test_save_document_rejects_an_unissued_session_id(session_root: Path) -> None:
    with pytest.raises(sd.SessionDocumentError, match="unknown session id"):
        sd.save_document("a" * 32, "a.pdf", b"x")


def test_save_document_rejects_a_malformed_session_id(session_root: Path) -> None:
    with pytest.raises(sd.SessionDocumentError, match="invalid session id"):
        sd.save_document("not-a-real-id", "a.pdf", b"x")


def test_save_document_rejects_an_unsupported_suffix(session_root: Path) -> None:
    with pytest.raises(sd.SessionDocumentError, match="unsupported file type"):
        sd.save_document(None, "a.docx", b"x")


def test_save_document_rejects_a_file_over_the_upload_cap(
    session_root: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    try:
        with pytest.raises(sd.SessionDocumentError, match="too large"):
            sd.save_document(None, "a.pdf", b"x" * (2 * 1024 * 1024))
    finally:
        get_settings.cache_clear()


def test_save_document_enforces_the_per_session_file_count_cap(
    session_root: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MCP_SESSION_DOCUMENTS_MAX_FILES", "2")
    get_settings.cache_clear()
    try:
        session_id, _ = sd.save_document(None, "a.pdf", b"1")
        sd.save_document(session_id, "b.pdf", b"2")
        with pytest.raises(sd.SessionDocumentError, match="already has 2 documents"):
            sd.save_document(session_id, "c.pdf", b"3")
    finally:
        get_settings.cache_clear()


def test_session_documents_dir_resolves_an_issued_session(session_root: Path) -> None:
    session_id, _ = sd.save_document(None, "a.pdf", b"x")
    assert sd.session_documents_dir(session_id) == session_root / session_id


def test_session_documents_dir_rejects_an_unknown_session(session_root: Path) -> None:
    with pytest.raises(sd.SessionDocumentError, match="unknown session id"):
        sd.session_documents_dir("b" * 32)


def test_gc_stale_sessions_removes_only_directories_past_the_ttl(session_root: Path) -> None:
    fresh_id, fresh_name = sd.save_document(None, "fresh.pdf", b"x")
    stale_id, stale_name = sd.save_document(None, "stale.pdf", b"x")
    old = time.time() - 3600 * 999
    os.utime(session_root / stale_id / stale_name, (old, old))
    os.utime(session_root / stale_id, (old, old))

    removed = sd.gc_stale_sessions(max_age_hours=24)

    assert removed == 1
    assert not (session_root / stale_id).exists()
    assert (session_root / fresh_id / fresh_name).exists()
