"""Integration tests for routing wired through the benchmark runner.

The routing *engine* is unit-tested in ``test_routing.py``. These tests cover the
benchmark integration added for issue #14: a ``--routing-policy`` run must route
each document through the multi-stage router, populate ``response.routing``, and
surface routing metrics in the summary (which otherwise render ``N/A``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docie_bench.benchmark.runner import run_benchmark
from docie_bench.schemas.common import ExtractionResponse, ExtractionValidation

_TWO_STAGE_POLICY = """
version: test-1
stages:
  - name: fast
    rules:
      - when: {status: success, validation_valid: true, min_confidence: 0.85}
        decision: accept
        reason: fast cleared the confidence gate
    default_decision: fallback
    default_reason: fast below the confidence gate
  - name: accurate
    rules:
      - when: {status: success, validation_valid: true}
        decision: accept
        reason: accurate returned a valid extraction
budget:
  max_stages: 2
"""


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    document = tmp_path / "doc-0.txt"
    document.write_text("Invoice INV-0", encoding="utf-8")
    dataset = tmp_path / "manifest.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "doc_id": "doc-0",
                "file_path": document.name,
                "schema_name": "invoice",
                "ground_truth": {"invoice_number": "INV-0"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    models = tmp_path / "models.yaml"
    models.write_text(
        """
profiles:
  fast:
    model: fast-model
    base_url: http://fast/v1
  accurate:
    model: accurate-model
    base_url: http://accurate/v1
""",
        encoding="utf-8",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(_TWO_STAGE_POLICY, encoding="utf-8")
    return dataset, models, policy


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failing: tuple[str, ...] = (),
) -> None:
    """Stub ExtractionService so 'fast' fails the confidence gate and 'accurate' clears it.

    Profiles named in ``failing`` raise on extraction to exercise the stage-failure path.
    """

    class FakeService:
        def __init__(self, profile: Any) -> None:
            self.profile = profile

        async def extract_from_file(self, **kwargs: Any) -> ExtractionResponse:
            if self.profile.name in failing:
                raise RuntimeError(f"{self.profile.name} runtime exploded")
            confidence = 0.5 if self.profile.name == "fast" else 0.95
            return ExtractionResponse(
                request_id=f"req-{self.profile.name}",
                schema_name="invoice",
                model_profile=self.profile.name,
                document_hash=None,
                result={
                    "invoice_number": {
                        "value": "INV-0",
                        "evidence_ids": ["b1"],
                        "confidence": confidence,
                    }
                },
                validation=ExtractionValidation(valid=True),
                latency_ms=5,
            )

    monkeypatch.setattr(
        "docie_bench.benchmark.runner.get_settings",
        lambda: SimpleNamespace(default_ocr_backend="pdf_text", runs_dir=tmp_path / "runs"),
    )
    # build_extraction_router resolves ExtractionService from the routing_config namespace.
    monkeypatch.setattr(
        "docie_bench.benchmark.routing_config.ExtractionService", FakeService
    )


def _predictions(result: Any) -> list[dict[str, Any]]:
    text = result.predictions_path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_routing_run_populates_audit_and_lights_up_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, models, policy = _inputs(tmp_path)
    _install_fakes(monkeypatch, tmp_path)

    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            output_dir=tmp_path / "run",
            routing_policy_path=policy,
        )
    )

    # The routed document falls back from 'fast' to 'accurate', which accepts.
    audit = _predictions(result)[0]["routing"]
    assert audit is not None
    assert audit["terminal_decision"] == "accept"
    assert audit["selected_stage"] == "accurate"
    assert audit["fallback_count"] == 1
    assert [stage["stage"] for stage in audit["stages"]] == ["fast", "accurate"]

    # Routing metrics must be present and non-N/A now that the audit is populated.
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert len(metrics["summary"]) == 1
    entry = metrics["summary"][0]
    assert entry["model_profile"] == "routed:test-1"
    assert entry["ingestion_path"] == "routed"
    assert entry["routing_accept_rate"] == 1.0
    assert entry["routing_fallback_rate"] == 1.0
    assert entry["routing_escalation_rate"] == 0.0

    # Provenance records the policy so routed runs are reproducible.
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["inputs"]["routing_policy"]["version"] == "test-1"
    assert manifest["invocation"]["routing_policy_path"] is not None


def test_routing_budget_exhaustion_escalates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, models, _ = _inputs(tmp_path)
    _install_fakes(monkeypatch, tmp_path)
    policy = tmp_path / "budget.yaml"
    policy.write_text(
        """
version: budget-1
stages:
  - name: fast
    rules:
      - when: {status: success, validation_valid: true, min_confidence: 0.85}
        decision: accept
        reason: fast cleared the gate
    default_decision: fallback
    default_reason: fast below the gate
  - name: accurate
    rules:
      - when: {status: success, validation_valid: true}
        decision: accept
        reason: accurate accepted
budget:
  max_stages: 1
""",
        encoding="utf-8",
    )

    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            output_dir=tmp_path / "run",
            routing_policy_path=policy,
        )
    )

    audit = _predictions(result)[0]["routing"]
    assert audit["terminal_decision"] == "escalate"
    assert audit["budget_exhausted"] is True
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["summary"][0]["routing_budget_exhaustion_rate"] == 1.0


def test_routing_policy_unknown_profile_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, models, _ = _inputs(tmp_path)
    _install_fakes(monkeypatch, tmp_path)
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
version: bad
stages:
  - name: does_not_exist
    rules: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown model profiles"):
        asyncio.run(
            run_benchmark(
                dataset_path=dataset,
                models_config_path=models,
                output_dir=tmp_path / "run",
                routing_policy_path=bad,
            )
        )


def test_routing_policy_conflicts_with_model_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset, models, policy = _inputs(tmp_path)
    _install_fakes(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="cannot be combined"):
        asyncio.run(
            run_benchmark(
                dataset_path=dataset,
                models_config_path=models,
                model_profile="fast",
                output_dir=tmp_path / "run",
                routing_policy_path=policy,
            )
        )


def test_routing_stage_failure_recovers_and_aggregates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 'fast' raises; the route falls back to 'accurate', which accepts. The errored
    # stage must be recorded and surfaced in routing_stage_failure_rate.
    dataset, models, policy = _inputs(tmp_path)
    _install_fakes(monkeypatch, tmp_path, failing=("fast",))

    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            output_dir=tmp_path / "run",
            routing_policy_path=policy,
        )
    )

    audit = _predictions(result)[0]["routing"]
    assert audit["terminal_decision"] == "accept"
    assert audit["selected_stage"] == "accurate"
    stages = {stage["stage"]: stage["status"] for stage in audit["stages"]}
    assert stages == {"fast": "error", "accurate": "success"}
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["summary"][0]["routing_stage_failure_rate"] == 0.5


def test_routing_all_stages_fail_escalates_with_routed_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Every stage raises and the first escalates with no prior response, so the router
    # returns response=None. The runner must surface that as a failed, routed-labelled row.
    dataset, models, _ = _inputs(tmp_path)
    _install_fakes(monkeypatch, tmp_path, failing=("fast", "accurate"))
    policy = tmp_path / "escalate.yaml"
    policy.write_text(
        """
version: escalate-1
stages:
  - name: fast
    rules:
      - when: {status: success, validation_valid: true}
        decision: accept
        reason: never reached because the stage errors
    default_decision: escalate
    default_reason: fast errored with nothing to fall back to
  - name: accurate
    rules:
      - when: {status: success, validation_valid: true}
        decision: accept
        reason: also unreachable
""",
        encoding="utf-8",
    )

    result = asyncio.run(
        run_benchmark(
            dataset_path=dataset,
            models_config_path=models,
            output_dir=tmp_path / "run",
            routing_policy_path=policy,
        )
    )

    row = _predictions(result)[0]
    assert row["task_state"] == "failed"
    assert row["model_profile"] == "routed:escalate-1"
    assert row["ingestion_path"] == "routed"
    assert "Routing terminated without an accepted response" in row["error"]
