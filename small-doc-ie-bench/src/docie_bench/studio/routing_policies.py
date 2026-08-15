"""Persistence for named, reusable multi-stage routing policies.

``RoutingPolicy`` (extract.routing) is already fully validated and already
drives benchmark runs via ``--routing-policy``/``routing_policy_path`` -- but
only from a server-side filesystem path (``configs/routing-policy.example.yaml``
and friends), which the Studio UI exposed as a raw text field with no
discoverability, versioning, or validation before submit. This module is the
save/list/fetch/delete side of a named registry; the Studio API routes and the
benchmark-trigger wiring (resolving a saved name to a policy at run time) are
separate call sites.

Shared operator config, like ``models.yaml``/``data/datasets.yaml`` -- no
tenant scoping, no lifecycle field, create-only for v1. See
``RoutingPolicyRecord`` (studio/models.py) for the storage shape and why a
saved policy's registry ``name`` is kept separate from the policy's own
``version`` field.
"""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy import Table, delete, select
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import CursorResult, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex, CreateTable

from docie_bench.extract.routing import RoutingPolicy
from docie_bench.storage.db import session_scope
from docie_bench.studio.models import RoutingPolicyRecord

# Arbitrary-but-stable advisory-lock key ("docie routing policies v1"),
# distinct from every other migration's key -- see dynamic_schemas.py's
# identical pattern for the sibling dynamic_schemas table.
_ROUTING_POLICY_LOCK_KEY = 0x0D0C1E07


def ensure_routing_policy_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``routing_policies`` if missing.

    Same shape and reasoning as ``ensure_dynamic_schema_table``
    (studio/dynamic_schemas.py): ``CREATE TABLE IF NOT EXISTS`` under
    ``pg_advisory_xact_lock`` on PostgreSQL, plus an explicit
    ``CreateIndex(..., if_not_exists=True)`` loop for ``name``'s
    ``unique=True`` index, which ``Base.metadata.create_all()`` would
    otherwise silently skip on any process that isn't the one racing through
    table creation.
    """
    policy_table = RoutingPolicyRecord.__table__
    assert isinstance(policy_table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _ROUTING_POLICY_LOCK_KEY},
            )
        existed = sa_inspect(connection).has_table("routing_policies")
        connection.execute(CreateTable(policy_table, if_not_exists=True))
        for index in policy_table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


class RoutingPolicyStoreError(RuntimeError):
    """Base error for the routing-policy store."""


class RoutingPolicyUnavailableError(RoutingPolicyStoreError):
    """Raised when the store is used without a configured database."""


class RoutingPolicyConflictError(RoutingPolicyStoreError):
    """A policy with this name already exists."""


class RoutingPolicyNotFoundError(RoutingPolicyStoreError):
    """No policy exists under this name."""


def _to_dict(row: RoutingPolicyRecord) -> dict[str, Any]:
    return {
        "name": row.name,
        "policy": row.spec_json,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def create_routing_policy(name: str, policy: RoutingPolicy) -> dict[str, Any]:
    """Save an already-validated `policy` under the registry key `name`.

    Storing the *validated* dump (not the caller's raw input) means every
    later read round-trips through ``RoutingPolicy.model_validate`` cleanly --
    a policy that fails validation never makes it into the table at all.
    """
    with session_scope() as session:
        if session is None:
            raise RoutingPolicyUnavailableError("Routing policies require DATABASE_URL")
        row = RoutingPolicyRecord(name=name, spec_json=policy.model_dump(mode="json"))
        session.add(row)
        try:
            session.flush()
        except IntegrityError as exc:
            raise RoutingPolicyConflictError(f"routing policy {name!r} already exists") from exc
        return _to_dict(row)


def list_routing_policies() -> list[dict[str, Any]]:
    with session_scope() as session:
        if session is None:
            return []
        rows = session.execute(
            select(RoutingPolicyRecord).order_by(RoutingPolicyRecord.name)
        ).scalars()
        return [_to_dict(row) for row in rows]


def get_routing_policy(name: str) -> dict[str, Any] | None:
    with session_scope() as session:
        if session is None:
            return None
        row = session.execute(
            select(RoutingPolicyRecord).where(RoutingPolicyRecord.name == name)
        ).scalar_one_or_none()
        return _to_dict(row) if row is not None else None


def delete_routing_policy(name: str) -> None:
    with session_scope() as session:
        if session is None:
            raise RoutingPolicyUnavailableError("Routing policies require DATABASE_URL")
        result = cast(
            CursorResult[Any],
            session.execute(delete(RoutingPolicyRecord).where(RoutingPolicyRecord.name == name)),
        )
        if result.rowcount == 0:
            raise RoutingPolicyNotFoundError(f"routing policy {name!r} does not exist")
