"""Every Studio endpoint serialises a stored timestamp the same way."""

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from docie_bench.storage.db import Base
from docie_bench.studio.dynamic_schemas import _to_dict as schema_row
from docie_bench.studio.models import DynamicSchema, RoutingPolicyRecord, isoformat
from docie_bench.studio.routing_policies import _to_dict as policy_row
from docie_bench.studio.seed_store import _to_dict as seed_row


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_a_naive_stored_timestamp_is_serialised_as_utc() -> None:
    # SQLite drops the offset on write and hands the value back naive, whatever
    # DateTime(timezone=True) says.
    naive = dt.datetime(2026, 9, 11, 15, 45, 30)
    assert isoformat(naive) == "2026-09-11T15:45:30+00:00"


def test_an_aware_timestamp_keeps_its_own_offset() -> None:
    aware = dt.datetime(2026, 9, 11, 15, 45, 30, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert isoformat(aware) == "2026-09-11T15:45:30+02:00"


def test_a_missing_timestamp_stays_null() -> None:
    assert isoformat(None) is None


def test_schemas_and_policies_agree_with_the_stores_on_the_wire_format(
    session: Session,
) -> None:
    # Before: these two called .isoformat() directly and emitted a bare
    # local-looking string, while the seed/batch/result stores emitted the
    # offset. Same database, same column type, two formats.
    session.add(DynamicSchema(name="resume", spec_json={}))
    session.add(RoutingPolicyRecord(name="cheap-first", spec_json={}))
    session.commit()

    schema = schema_row(session.query(DynamicSchema).one())
    policy = policy_row(session.query(RoutingPolicyRecord).one())
    for row in (schema, policy):
        for field in ("created_at", "updated_at"):
            assert row[field].endswith("+00:00"), (field, row[field])


def test_every_serialised_timestamp_round_trips_to_an_aware_datetime(
    session: Session,
) -> None:
    session.add(DynamicSchema(name="resume", spec_json={}))
    session.commit()
    stamp = schema_row(session.query(DynamicSchema).one())["created_at"]
    assert dt.datetime.fromisoformat(stamp).tzinfo is not None


def test_the_seed_store_is_unchanged(session: Session) -> None:
    from docie_bench.studio.models import SeedRun

    session.add(
        SeedRun(
            event_id="e1",
            channel="hf",
            kind="model",
            reference="r",
            name="lfm2.5-2.6b",
            status="ok",
        )
    )
    session.commit()
    row = seed_row(session.query(SeedRun).one())
    assert row["created_at"].endswith("+00:00")
