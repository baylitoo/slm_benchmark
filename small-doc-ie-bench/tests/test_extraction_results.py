"""``docie_bench.studio.extraction_results`` -- the durable extraction-outcome
store. Before this module, GET /v1/studio/runs/{event_id} depended entirely
on proxying Inngest's own run-status API for an extraction's output, which
does not reliably carry it on this project's self-hosted Inngest server (a
real integration test surfaced the gap -- see ExtractionRunResult's
docstring)."""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.storage.db import dispose_engine, init_engine
from docie_bench.studio.extraction_results import (
    get_extraction_result,
    record_extraction_result,
)


@pytest.fixture(autouse=True)
def extraction_database(tmp_path: Path):
    init_engine(f"sqlite:///{tmp_path / 'extractions.db'}")
    yield
    dispose_engine()


def test_record_then_get_completed_round_trips() -> None:
    record_extraction_result(
        event_id="ev1",
        tenant_id="t1",
        status="completed",
        output={"invoice_number": {"value": "42"}},
    )

    result = get_extraction_result("ev1", tenant_id="t1")

    assert result is not None
    assert result["status"] == "completed"
    assert result["output"] == {"invoice_number": {"value": "42"}}
    assert result["error"] is None


def test_record_failed_carries_error_not_output() -> None:
    record_extraction_result(
        event_id="ev1", tenant_id="t1", status="failed", error="model timed out"
    )

    result = get_extraction_result("ev1", tenant_id="t1")

    assert result is not None
    assert result["status"] == "failed"
    assert result["output"] is None
    assert result["error"] == "model timed out"


def test_redelivery_overwrites_instead_of_raising() -> None:
    """Inngest is at-least-once: a redelivered doc/extract.requested re-runs
    extract_document, which calls record_extraction_result again for the same
    event_id. Must upsert, not raise on the duplicate primary key."""
    record_extraction_result(event_id="ev1", tenant_id="t1", status="failed", error="transient 503")
    record_extraction_result(event_id="ev1", tenant_id="t1", status="completed", output={"a": 1})

    result = get_extraction_result("ev1", tenant_id="t1")

    assert result is not None
    assert result["status"] == "completed"
    assert result["output"] == {"a": 1}
    assert result["error"] is None


def test_cross_tenant_read_is_none_not_an_error() -> None:
    """Never confirm another tenant's event id -- mirrors RunStore.get_run's
    404-not-403 contract."""
    record_extraction_result(event_id="ev1", tenant_id="tenant-a", status="completed", output={})

    assert get_extraction_result("ev1", tenant_id="tenant-b") is None


def test_unknown_event_id_is_none() -> None:
    assert get_extraction_result("nope", tenant_id="t1") is None


def test_no_database_configured_degrades_without_raising() -> None:
    dispose_engine()  # undo the autouse fixture's init_engine for this one test

    record_extraction_result(event_id="ev1", tenant_id="t1", status="completed", output={"a": 1})

    assert get_extraction_result("ev1", tenant_id="t1") is None
