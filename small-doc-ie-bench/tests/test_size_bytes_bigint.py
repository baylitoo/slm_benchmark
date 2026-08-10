"""size_bytes must be BIGINT wherever it holds a model/artifact byte count:
Postgres INTEGER caps at ~2.147 GB and a multi-GB GGUF overflows it on insert."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import BigInteger, create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.dialects import postgresql

from docie_bench.orchestrator.models import RunArtifact
from docie_bench.serving.catalog import ModelStoreEntry, ensure_size_bytes_bigint
from docie_bench.studio.models import StudioRunArtifact


def test_orm_size_bytes_columns_are_bigint() -> None:
    # Fresh databases get their schema from create_all, so the ORM column type is
    # what protects them — it must be BIGINT for every size_bytes.
    for model in (ModelStoreEntry, RunArtifact, StudioRunArtifact):
        column = model.__table__.c.size_bytes
        assert isinstance(column.type, BigInteger), f"{model.__name__}.size_bytes"
        assert column.type.compile(dialect=postgresql.dialect()) == "BIGINT"


def test_widen_is_noop_on_sqlite(tmp_path: Path) -> None:
    # SQLite INTEGER is a dynamic up-to-8-byte type with no 2 GB cap, so the
    # PostgreSQL-only migration must leave it untouched (returns nothing widened).
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(
            sa_text(
                "CREATE TABLE model_store_entry (name TEXT PRIMARY KEY, size_bytes INTEGER)"
            )
        )
    assert ensure_size_bytes_bigint(engine) == []
