"""Persistence for named, reusable dynamic extraction schemas.

``DynamicSchemaSpec`` (schemas.dynamic) is already fully validated and already
compiles into a working pydantic model + NuExtract template -- but it is
request-scoped only: a caller has always had to supply the full spec inline on
every extraction call, with no way to define one once and reference it by name
afterward. This module is the save/list/fetch/delete side of that gap; the
Studio API routes and the extraction-request wiring are separate call sites.

Shared operator config, like ``models.yaml``/``data/datasets.yaml`` -- no
tenant scoping, no version/lifecycle field, create-only for v1 (a mistake is
cheap to fix with delete + recreate, since this is a real table, not a text
file to splice). See ``DynamicSchema`` (studio/models.py) for the storage
shape and why each of those cuts was made.
"""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy import Table, delete, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

from docie_bench.schemas.dynamic import DynamicSchemaSpec
from docie_bench.storage.db import session_scope
from docie_bench.studio.models import DynamicSchema

# Arbitrary-but-stable advisory-lock key ("docie dynamic schemas v1"), distinct
# from every other migration's key so they never serialize against each other
# needlessly (see serving/catalog.py's identical pattern for the sibling
# serving_node/model_activity tables).
_DYNAMIC_SCHEMA_LOCK_KEY = 0x0D0C1E06


def ensure_dynamic_schema_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``dynamic_schemas`` if missing.

    ``CREATE TABLE IF NOT EXISTS`` under ``pg_advisory_xact_lock`` on
    PostgreSQL so concurrently starting processes (api, worker, N replicas)
    can't race each other into a duplicate-table abort -- mirrors
    ``ensure_serving_node_table``/``ensure_model_activity_table``
    (serving/catalog.py). No added-columns step: this table is new, so there
    is no pre-existing shape to migrate yet.

    Unlike those two siblings, ``DynamicSchema.name`` has ``unique=True`` --
    SQLAlchemy compiles a ``mapped_column(unique=True)`` as a SEPARATE
    ``Index`` object, not part of the single ``CREATE TABLE`` statement
    ``CreateTable`` emits, and ``Base.metadata.create_all()`` (called right
    after this in ``init_engine``) skips a table's indexes entirely once
    ``checkfirst`` sees the table already exists -- so without this explicit
    step, ``name``'s own uniqueness would silently never be enforced at the
    DB level on any process that ISN'T the one that races through
    ``create_all`` on a still-nonexistent table. ``CreateIndex(..., if_not_
    exists=True)`` is idempotent the same way ``CreateTable`` is, so this is
    safe under the same advisory lock, run every time regardless of `existed`.
    """
    schema_table = DynamicSchema.__table__
    assert isinstance(schema_table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _DYNAMIC_SCHEMA_LOCK_KEY},
            )
        existed = sa_inspect(connection).has_table("dynamic_schemas")
        connection.execute(CreateTable(schema_table, if_not_exists=True))
        for index in schema_table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


class DynamicSchemaError(RuntimeError):
    """Base error for the dynamic-schema store."""


class DynamicSchemaUnavailableError(DynamicSchemaError):
    """Raised when the store is used without a configured database."""


class DynamicSchemaConflictError(DynamicSchemaError):
    """A schema with this name already exists."""


class DynamicSchemaNotFoundError(DynamicSchemaError):
    """No schema exists under this name."""


def _to_dict(row: DynamicSchema) -> dict[str, Any]:
    return {
        "name": row.name,
        "spec": row.spec_json,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def create_dynamic_schema(spec: DynamicSchemaSpec) -> dict[str, Any]:
    """Save `spec` under its own ``document_type`` as the lookup name.

    A saved schema's ``document_type`` IS its identifier (both already share the
    same validated snake_case pattern -- see DynamicSchemaSpec/DynamicFieldSpec),
    so there is no separate name to keep in sync.
    """
    with session_scope() as session:
        if session is None:
            raise DynamicSchemaUnavailableError("Dynamic schemas require DATABASE_URL")
        row = DynamicSchema(name=spec.document_type, spec_json=spec.model_dump(mode="json"))
        session.add(row)
        try:
            session.flush()
        except IntegrityError as exc:
            raise DynamicSchemaConflictError(
                f"schema {spec.document_type!r} already exists"
            ) from exc
        return _to_dict(row)


def list_dynamic_schemas() -> list[dict[str, Any]]:
    with session_scope() as session:
        if session is None:
            return []
        rows = session.execute(select(DynamicSchema).order_by(DynamicSchema.name)).scalars()
        return [_to_dict(row) for row in rows]


def get_dynamic_schema(name: str) -> dict[str, Any] | None:
    with session_scope() as session:
        if session is None:
            return None
        row = session.execute(
            select(DynamicSchema).where(DynamicSchema.name == name)
        ).scalar_one_or_none()
        return _to_dict(row) if row is not None else None


def delete_dynamic_schema(name: str) -> None:
    with session_scope() as session:
        if session is None:
            raise DynamicSchemaUnavailableError("Dynamic schemas require DATABASE_URL")
        result = cast(
            CursorResult[Any],
            session.execute(delete(DynamicSchema).where(DynamicSchema.name == name)),
        )
        if result.rowcount == 0:
            raise DynamicSchemaNotFoundError(f"schema {name!r} does not exist")
