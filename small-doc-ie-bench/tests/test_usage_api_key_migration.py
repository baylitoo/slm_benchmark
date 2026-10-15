"""Startup against a database that predates ``usage_records.api_key_id``.

This is the shape that broke production: every test database is created fresh
from the current metadata, so the table always had the column and the ordering
bug was invisible. The guard has to build the OLD table explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy import text as sa_text

from docie_bench.storage.db import dispose_engine, init_engine

# usage_records exactly as it was before api_key_id was added. Every other
# indexed column has to be present, or the CreateIndex loop fails on a
# different column and the test would pass for the wrong reason.
LEGACY_USAGE_RECORDS = """
CREATE TABLE usage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment VARCHAR(200) NOT NULL,
    surface VARCHAR(16) NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER NOT NULL,
    status VARCHAR(16),
    tenant_id VARCHAR(128),
    tool_calls_json JSON,
    created_at DATETIME
)
"""


@pytest.fixture
def legacy_database(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(url)
    with engine.begin() as connection:
        connection.execute(sa_text(LEGACY_USAGE_RECORDS))
    engine.dispose()
    return url


def test_startup_adds_the_column_before_indexing_it(legacy_database: str) -> None:
    # Before the fix this raised UndefinedColumn from inside lifespan startup:
    # ensure_usage_record_table's CreateIndex loop includes
    # ix_usage_records_api_key_id, and CREATE TABLE IF NOT EXISTS is a no-op on
    # an existing table, so the column was never added.
    init_engine(legacy_database)
    try:
        engine = create_engine(legacy_database)
        columns = {c["name"] for c in inspect(engine).get_columns("usage_records")}
        engine.dispose()
    finally:
        dispose_engine()

    assert "api_key_id" in columns


def test_startup_is_idempotent_on_an_already_migrated_database(
    legacy_database: str,
) -> None:
    init_engine(legacy_database)
    dispose_engine()

    # Second boot must be a no-op, not a duplicate-column or duplicate-index
    # error -- every process that calls init_engine runs this at startup.
    init_engine(legacy_database)
    dispose_engine()


def test_a_fresh_database_still_gets_the_column(tmp_path: Path) -> None:
    # The column add returns early when the table is absent, so the table (and
    # its indexes) must still come out complete from the table-creation call.
    init_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    try:
        engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
        columns = {c["name"] for c in inspect(engine).get_columns("usage_records")}
        engine.dispose()
    finally:
        dispose_engine()

    assert "api_key_id" in columns
