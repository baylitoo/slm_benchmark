"""Persistence for saved evaluators (``evaluators.spec.EvaluatorSpec``)."""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy import Table, delete, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

from docie_bench.evaluators.spec import EvaluatorSpec
from docie_bench.storage.db import session_scope
from docie_bench.studio.models import EvaluatorRecord, isoformat

# Arbitrary-but-stable advisory-lock key ("docie evaluators v1").
_EVALUATOR_LOCK_KEY = 0x0D0C1E0E


def ensure_evaluator_table(engine: Engine) -> bool:
    """Race-safe forward migration, same shape as ``ensure_routing_policy_table``."""
    table = EvaluatorRecord.__table__
    assert isinstance(table, Table)
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _EVALUATOR_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("evaluators")
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


class EvaluatorStoreError(RuntimeError):
    pass


class EvaluatorUnavailableError(EvaluatorStoreError):
    pass


class EvaluatorConflictError(EvaluatorStoreError):
    pass


class EvaluatorNotFoundError(EvaluatorStoreError):
    pass


def _view(row: EvaluatorRecord) -> dict[str, Any]:
    return {
        **row.spec_json,
        "created_at": isoformat(row.created_at),
        "updated_at": isoformat(row.updated_at),
    }


def _write(row: EvaluatorRecord, spec: EvaluatorSpec) -> None:
    row.spec_json = spec.model_dump(mode="json")
    row.enabled = spec.enabled
    row.agents_json = list(spec.agents)


def create_evaluator(spec: EvaluatorSpec) -> dict[str, Any]:
    with session_scope() as session:
        if session is None:
            raise EvaluatorUnavailableError("Evaluators require DATABASE_URL")
        row = EvaluatorRecord(name=spec.name)
        _write(row, spec)
        session.add(row)
        try:
            session.flush()
        except IntegrityError as exc:
            raise EvaluatorConflictError(f"evaluator {spec.name!r} already exists") from exc
        return _view(row)


def update_evaluator(name: str, spec: EvaluatorSpec) -> dict[str, Any]:
    if spec.name != name:
        raise EvaluatorStoreError("an evaluator cannot be renamed")
    with session_scope() as session:
        if session is None:
            raise EvaluatorUnavailableError("Evaluators require DATABASE_URL")
        row = session.execute(
            select(EvaluatorRecord).where(EvaluatorRecord.name == name)
        ).scalar_one_or_none()
        if row is None:
            raise EvaluatorNotFoundError(f"evaluator {name!r} does not exist")
        _write(row, spec)
        session.flush()
        return _view(row)


def list_evaluators() -> list[dict[str, Any]]:
    with session_scope() as session:
        if session is None:
            return []
        rows = session.execute(select(EvaluatorRecord).order_by(EvaluatorRecord.name)).scalars()
        return [_view(row) for row in rows]


def get_evaluator(name: str) -> EvaluatorSpec | None:
    with session_scope() as session:
        if session is None:
            return None
        row = session.execute(
            select(EvaluatorRecord).where(EvaluatorRecord.name == name)
        ).scalar_one_or_none()
        return EvaluatorSpec.model_validate(row.spec_json) if row is not None else None


def evaluators_for_agent(agent: str) -> list[EvaluatorSpec]:
    """Enabled evaluators that score ``agent``'s runs. Empty without a database."""
    with session_scope() as session:
        if session is None:
            return []
        rows = session.execute(
            select(EvaluatorRecord).where(EvaluatorRecord.enabled.is_(True))
        ).scalars()
        return [
            EvaluatorSpec.model_validate(row.spec_json)
            for row in rows
            if agent in (row.agents_json or [])
        ]


def delete_evaluator(name: str) -> None:
    with session_scope() as session:
        if session is None:
            raise EvaluatorUnavailableError("Evaluators require DATABASE_URL")
        result = cast(
            CursorResult[Any],
            session.execute(delete(EvaluatorRecord).where(EvaluatorRecord.name == name)),
        )
        if result.rowcount == 0:
            raise EvaluatorNotFoundError(f"evaluator {name!r} does not exist")
