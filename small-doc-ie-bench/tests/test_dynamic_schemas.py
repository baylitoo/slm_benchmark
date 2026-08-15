"""``docie_bench.studio.dynamic_schemas`` -- the save/list/fetch/delete side of
named, reusable ``DynamicSchemaSpec``s. The spec itself is already fully
validated and already compiles into a working pydantic model + NuExtract
template (schemas/dynamic.py) -- this module is purely persistence, so these
tests focus on the CRUD contract, not schema-compilation behavior (already
covered elsewhere)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text as sa_text

from docie_bench.schemas.dynamic import DynamicFieldSpec, DynamicSchemaSpec
from docie_bench.storage.db import dispose_engine, get_session_factory, init_engine
from docie_bench.studio.dynamic_schemas import (
    DynamicSchemaConflictError,
    DynamicSchemaNotFoundError,
    DynamicSchemaUnavailableError,
    create_dynamic_schema,
    delete_dynamic_schema,
    ensure_dynamic_schema_table,
    get_dynamic_schema,
    list_dynamic_schemas,
)


@pytest.fixture(autouse=True)
def schema_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'schemas.db'}")
    yield
    dispose_engine()


def _spec(document_type: str = "invoice_custom") -> DynamicSchemaSpec:
    return DynamicSchemaSpec(
        document_type=document_type,
        fields=[
            DynamicFieldSpec(name="vendor_name", type="string"),
            DynamicFieldSpec(name="total", type="money"),
        ],
    )


def test_create_then_get_round_trips_the_spec() -> None:
    saved = create_dynamic_schema(_spec())

    assert saved["name"] == "invoice_custom"
    fetched = get_dynamic_schema("invoice_custom")
    assert fetched is not None
    assert fetched["spec"]["document_type"] == "invoice_custom"
    assert [f["name"] for f in fetched["spec"]["fields"]] == ["vendor_name", "total"]


def test_get_unknown_name_returns_none() -> None:
    assert get_dynamic_schema("does_not_exist") is None


def test_duplicate_document_type_raises_conflict() -> None:
    create_dynamic_schema(_spec())

    with pytest.raises(DynamicSchemaConflictError):
        create_dynamic_schema(_spec())


def test_list_returns_all_saved_schemas_sorted_by_name() -> None:
    create_dynamic_schema(_spec("zzz_schema"))
    create_dynamic_schema(_spec("aaa_schema"))

    names = [s["name"] for s in list_dynamic_schemas()]

    assert names == ["aaa_schema", "zzz_schema"]


def test_delete_removes_the_schema() -> None:
    create_dynamic_schema(_spec())

    delete_dynamic_schema("invoice_custom")

    assert get_dynamic_schema("invoice_custom") is None
    assert list_dynamic_schemas() == []


def test_delete_unknown_name_raises_not_found() -> None:
    with pytest.raises(DynamicSchemaNotFoundError):
        delete_dynamic_schema("does_not_exist")


def test_migration_actually_creates_the_unique_index_not_just_the_bare_table() -> None:
    # Real bug caught while writing this test: mapped_column(unique=True)
    # compiles to a SEPARATE Index object, not part of CreateTable's own
    # single statement -- and Base.metadata.create_all() (run right after
    # ensure_dynamic_schema_table in init_engine) skips a table's indexes
    # entirely once checkfirst sees the table already exists. Without an
    # explicit CreateIndex step in the migration itself, `name` uniqueness
    # was never actually enforced at the DB level (test_duplicate_document_
    # type_raises_conflict above would pass anyway, via the migration's own
    # index -- this test locks in the index's existence directly instead of
    # only inferring it from application-level behavior).
    factory = get_session_factory()
    assert factory is not None
    engine = factory.kw["bind"]
    ensure_dynamic_schema_table(engine)  # idempotent; init_engine already ran it once

    with engine.connect() as connection:
        indexes = connection.execute(
            sa_text(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='dynamic_schemas'"
            )
        ).fetchall()

    assert any("UNIQUE" in (row[0] or "") for row in indexes)


def test_no_database_degrades_to_unavailable_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # Matches GET /datasets / GET /model-profiles' own "missing DB -> empty list"
    # contract for reads; writes need a real DB so they raise explicitly instead
    # of silently pretending to succeed.
    dispose_engine()

    assert list_dynamic_schemas() == []
    assert get_dynamic_schema("invoice_custom") is None
    with pytest.raises(DynamicSchemaUnavailableError):
        create_dynamic_schema(_spec())
    with pytest.raises(DynamicSchemaUnavailableError):
        delete_dynamic_schema("invoice_custom")
