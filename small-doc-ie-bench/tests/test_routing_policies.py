"""``docie_bench.studio.routing_policies`` -- the save/list/fetch/delete side of
named, reusable ``RoutingPolicy``s. The policy itself is already fully
validated by ``RoutingPolicy`` (extract/routing.py) -- this module is purely
persistence, so these tests focus on the CRUD contract, not routing behavior
(already covered elsewhere)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text as sa_text

from docie_bench.extract.routing import RoutingPolicy, RoutingRule, RuleCondition, StagePolicy
from docie_bench.storage.db import dispose_engine, get_session_factory, init_engine
from docie_bench.studio.routing_policies import (
    RoutingPolicyConflictError,
    RoutingPolicyNotFoundError,
    RoutingPolicyUnavailableError,
    create_routing_policy,
    delete_routing_policy,
    ensure_routing_policy_table,
    get_routing_policy,
    list_routing_policies,
)


@pytest.fixture(autouse=True)
def policy_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'routing_policies.db'}")
    yield
    dispose_engine()


def _policy(version: str = "cpu-cascade-1") -> RoutingPolicy:
    return RoutingPolicy(
        version=version,
        stages=[
            StagePolicy(
                name="ollama_qwen25_1b",
                rules=[
                    RoutingRule(
                        when=RuleCondition(status="success", validation_valid=True),
                        decision="accept",
                        reason="cleared the quality gate",
                    )
                ],
            )
        ],
    )


def test_create_then_get_round_trips_the_policy() -> None:
    saved = create_routing_policy("cpu-cascade", _policy())

    assert saved["name"] == "cpu-cascade"
    fetched = get_routing_policy("cpu-cascade")
    assert fetched is not None
    assert fetched["policy"]["version"] == "cpu-cascade-1"
    assert [s["name"] for s in fetched["policy"]["stages"]] == ["ollama_qwen25_1b"]


def test_get_unknown_name_returns_none() -> None:
    assert get_routing_policy("does_not_exist") is None


def test_duplicate_name_raises_conflict() -> None:
    create_routing_policy("cpu-cascade", _policy())

    with pytest.raises(RoutingPolicyConflictError):
        create_routing_policy("cpu-cascade", _policy())


def test_name_is_a_separate_key_from_the_policy_version_field() -> None:
    # Two different registry names can save policies sharing the same internal
    # `version` label -- name and version are deliberately decoupled (see
    # RoutingPolicyRecord's own docstring for why).
    create_routing_policy("cascade-a", _policy(version="v1"))
    create_routing_policy("cascade-b", _policy(version="v1"))

    assert {p["name"] for p in list_routing_policies()} == {"cascade-a", "cascade-b"}


def test_list_returns_all_saved_policies_sorted_by_name() -> None:
    create_routing_policy("zzz_policy", _policy())
    create_routing_policy("aaa_policy", _policy())

    names = [p["name"] for p in list_routing_policies()]

    assert names == ["aaa_policy", "zzz_policy"]


def test_delete_removes_the_policy() -> None:
    create_routing_policy("cpu-cascade", _policy())

    delete_routing_policy("cpu-cascade")

    assert get_routing_policy("cpu-cascade") is None
    assert list_routing_policies() == []


def test_delete_unknown_name_raises_not_found() -> None:
    with pytest.raises(RoutingPolicyNotFoundError):
        delete_routing_policy("does_not_exist")


def test_migration_actually_creates_the_unique_index_not_just_the_bare_table() -> None:
    # Same real gap PR #195's equivalent test caught for dynamic_schemas:
    # mapped_column(unique=True) compiles to a SEPARATE Index object, not part
    # of CreateTable's own single statement, and Base.metadata.create_all()
    # skips a table's indexes once checkfirst sees the table already exists.
    factory = get_session_factory()
    assert factory is not None
    engine = factory.kw["bind"]
    ensure_routing_policy_table(engine)  # idempotent; init_engine already ran it once

    with engine.connect() as connection:
        indexes = connection.execute(
            sa_text(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='routing_policies'"
            )
        ).fetchall()

    assert any("UNIQUE" in (row[0] or "") for row in indexes)


def test_no_database_degrades_to_unavailable_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    dispose_engine()

    assert list_routing_policies() == []
    assert get_routing_policy("cpu-cascade") is None
    with pytest.raises(RoutingPolicyUnavailableError):
        create_routing_policy("cpu-cascade", _policy())
    with pytest.raises(RoutingPolicyUnavailableError):
        delete_routing_policy("cpu-cascade")
