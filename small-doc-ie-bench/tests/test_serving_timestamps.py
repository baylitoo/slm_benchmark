"""The serving catalog serialises timestamps the way the Studio stores do."""

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import docie_bench.serving.catalog as catalog
from docie_bench.storage.db import isoformat
from docie_bench.studio.models import isoformat as studio_isoformat


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    catalog.Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_a_stored_entry_is_serialised_with_its_offset(session: Session) -> None:
    # SQLite drops the offset on write and hands the value back naive, whatever
    # DateTime(timezone=True) says, so a bare .isoformat() emitted a string a
    # browser reads as local time.
    session.add(catalog.ModelStoreEntry(name="m1", family="lfm2", model_path="/x"))
    session.commit()
    row = session.query(catalog.ModelStoreEntry).one()
    assert row.created_at.tzinfo is None
    assert isoformat(row.created_at).endswith("+00:00")


def test_no_view_in_the_catalog_serialises_a_bare_timestamp() -> None:
    source = catalog.__file__ or ""
    with open(source, encoding="utf-8") as handle:
        offenders = [line.strip() for line in handle if ".isoformat()" in line]
    assert offenders == []


def test_the_studio_and_the_serving_catalog_share_one_serialiser() -> None:
    assert studio_isoformat is isoformat


def test_an_aware_value_keeps_its_own_offset() -> None:
    aware = dt.datetime(2026, 9, 11, 15, 45, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert isoformat(aware) == "2026-09-11T15:45:00+02:00"


def test_a_missing_timestamp_stays_null() -> None:
    assert isoformat(None) is None
