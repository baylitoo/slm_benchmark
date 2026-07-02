"""Postgres-backed model-store catalog: large blob sizes must not overflow.

``ModelStoreEntry.size_bytes`` must be a 64-bit column. GGUF blobs routinely
exceed the ~2.147 GB Postgres INTEGER cap (a 7B Q4 is ~4 GB), which would
overflow on insert and wedge the seed job in a retry loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import BigInteger

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ModelStoreEntry
from docie_bench.serving.model_store import StoreEntry

# Larger than a signed 32-bit int (Postgres INTEGER max ~2.147 GB) — the size
# that used to overflow. Roughly a 3 GB blob.
_BIG_SIZE = 3_000_000_000


def test_size_bytes_column_is_bigint() -> None:
    # The load-bearing guard: BigInteger subclasses Integer, so this is False
    # for the old (overflowing) Integer column and True only after the fix.
    # (SQLite stores all ints as 64-bit, so a round-trip alone can't catch it.)
    assert isinstance(ModelStoreEntry.__table__.c.size_bytes.type, BigInteger)
    assert _BIG_SIZE > 2**31


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


def test_large_size_bytes_round_trips_through_catalog(_sqlite_catalog: None) -> None:
    catalog = ModelCatalog()
    entry = StoreEntry(
        name="qwen2.5-7b-q4",
        family="openai_chat",
        model_path=Path("/models/qwen2.5-7b-q4.gguf"),
    )

    upserted = catalog.upsert(entry, size_bytes=_BIG_SIZE)
    assert upserted["size_bytes"] == _BIG_SIZE

    listed = catalog.list()
    assert [row["size_bytes"] for row in listed] == [_BIG_SIZE]
    assert catalog.get("qwen2.5-7b-q4")["size_bytes"] == _BIG_SIZE
