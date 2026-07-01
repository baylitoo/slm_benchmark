"""Validity gate: a below-threshold valid_rate must fail loudly, not score zero."""

from __future__ import annotations

import pytest

from docie_bench.benchmark.metrics import ValidityGateError, evaluate_validity_gate


def _row(profile: str, valid_rate: float | None) -> dict[str, object]:
    return {"model_profile": profile, "valid_rate": valid_rate}


def test_gate_disabled_when_threshold_non_positive() -> None:
    rows = [_row("a", 0.0), _row("b", 0.0)]
    assert evaluate_validity_gate(rows, 0.0) == []
    assert evaluate_validity_gate(rows, -1.0) == []


def test_gate_flags_below_threshold_profiles() -> None:
    rows = [_row("good", 0.9), _row("bad", 0.1)]
    failures = evaluate_validity_gate(rows, 0.5)
    assert [row["model_profile"] for row in failures] == ["bad"]


def test_gate_at_threshold_passes() -> None:
    assert evaluate_validity_gate([_row("edge", 0.5)], 0.5) == []


def test_gate_ignores_rows_without_numeric_valid_rate() -> None:
    assert evaluate_validity_gate([_row("missing", None)], 0.5) == []


def test_validity_gate_error_message_lists_failures() -> None:
    failures = [{"model_profile": "bad", "valid_rate": 0.0}]
    err = ValidityGateError(0.5, failures)
    assert err.threshold == 0.5
    assert err.failures == failures
    assert "bad" in str(err)
    assert "0.500" in str(err)


def test_raising_the_gate_is_a_runtime_error() -> None:
    with pytest.raises(ValidityGateError):
        raise ValidityGateError(0.5, [{"model_profile": "bad", "valid_rate": 0.0}])
