"""Cost guard: a paid profile refuses to run without a ceiling or dry-run.

No live stack, no paid API. `estimate_run_cost` is pure (no upstream call), so
the guard aborts BEFORE any task runs. Also asserts the api_key value never
appears in cost log lines (profile name + $ do).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from docie_bench.benchmark.cost import (
    DEFAULT_EST_INPUT_TOKENS,
    CostCeilingExceeded,
    CostGuardError,
    enforce_cost_guard,
    estimate_run_cost,
)
from docie_bench.benchmark.runner import run_benchmark
from docie_bench.llm.model_profiles import ModelProfile, Pricing

_SECRET_KEY = "sk-super-secret-value-123"  # noqa: S105 - fake key for a leak-check test


def _paid(name: str = "hosted", max_tokens: int = 900) -> ModelProfile:
    return ModelProfile(
        name=name,
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key=_SECRET_KEY,
        max_tokens=max_tokens,
        pricing=Pricing(input_usd_per_1k=0.00015, output_usd_per_1k=0.0006),
    )


def _local(name: str = "local") -> ModelProfile:
    return ModelProfile(name=name, model="m", base_url="http://x/v1", api_key="k")


# ── pure estimate ────────────────────────────────────────────────────────────


def test_estimate_is_conservative_upper_bound() -> None:
    est = estimate_run_cost([_paid(max_tokens=900)], doc_count=10)
    item = est.per_profile[0]
    assert item.est_input_tokens == DEFAULT_EST_INPUT_TOKENS * 10
    assert item.est_output_tokens == 900 * 10  # worst-case max_tokens output
    # 20000/1k*0.00015 + 9000/1k*0.0006 = 0.003 + 0.0054 = 0.0084
    assert est.total_usd == 0.0084


def test_estimate_ignores_unpriced_profiles() -> None:
    est = estimate_run_cost([_local()], doc_count=100)
    assert est.total_usd == 0.0
    assert est.per_profile == ()


# ── guard decisions ──────────────────────────────────────────────────────────


def test_guard_raises_without_ceiling_for_paid() -> None:
    with pytest.raises(CostGuardError):
        enforce_cost_guard([_paid()], doc_count=5, cost_ceiling=None)


def test_guard_aborts_when_estimate_exceeds_ceiling() -> None:
    with pytest.raises(CostCeilingExceeded):
        enforce_cost_guard([_paid()], doc_count=5, cost_ceiling=0.00001)


def test_guard_passes_when_ceiling_covers_estimate() -> None:
    est = enforce_cost_guard([_paid()], doc_count=1, cost_ceiling=100.0)
    assert est is not None
    assert est.total_usd > 0


def test_guard_noop_when_no_paid_profile() -> None:
    assert enforce_cost_guard([_local()], doc_count=1000, cost_ceiling=None) is None


# ── run_benchmark integration (guard fires before any upstream call) ──────────


def _paid_config(tmp_path) -> object:  # noqa: ANN001
    cfg = tmp_path / "models.yaml"
    cfg.write_text(
        "profiles:\n"
        "  paid_api:\n"
        "    model: gpt-4o-mini\n"
        "    base_url: https://api.openai.com/v1\n"
        f"    api_key: {_SECRET_KEY}\n"
        "    max_tokens: 900\n"
        "    pricing:\n"
        "      input_usd_per_1k: 0.00015\n"
        "      output_usd_per_1k: 0.0006\n",
        encoding="utf-8",
    )
    doc = tmp_path / "doc.txt"
    doc.write_text("invoice body", encoding="utf-8")
    return cfg, doc


def test_run_benchmark_paid_without_ceiling_raises(tmp_path) -> None:  # noqa: ANN001
    cfg, doc = _paid_config(tmp_path)
    with pytest.raises(CostGuardError):
        asyncio.run(
            run_benchmark(
                dataset_path=None,
                document_path=doc,
                models_config_path=cfg,
                model_profile="paid_api",
                cost_ceiling=None,
            )
        )


def test_run_benchmark_paid_below_ceiling_aborts_preflight(tmp_path) -> None:  # noqa: ANN001
    cfg, doc = _paid_config(tmp_path)
    with pytest.raises(CostCeilingExceeded):
        asyncio.run(
            run_benchmark(
                dataset_path=None,
                document_path=doc,
                models_config_path=cfg,
                model_profile="paid_api",
                cost_ceiling=0.00001,
            )
        )


# ── key never logged ─────────────────────────────────────────────────────────


def test_api_key_never_appears_in_cost_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="docie_bench.benchmark.cost"):
        estimate_run_cost([_paid(name="hosted")], doc_count=3)
    assert _SECRET_KEY not in caplog.text
    assert "hosted" in caplog.text  # profile name IS logged
    assert "usd=" in caplog.text  # dollar amount IS logged


def test_paid_profile_snapshot_excludes_api_key() -> None:
    # The manifest embeds profile snapshots and the report embeds the manifest,
    # so the snapshot is the report/manifest key-leak path for a paid profile.
    from docie_bench.benchmark.reproducibility import profile_snapshot

    snapshot = profile_snapshot(_paid())
    assert "api_key" not in snapshot
    assert _SECRET_KEY not in repr(snapshot)
    # pricing (non-secret) survives so paid profiles remain identifiable in provenance.
    assert snapshot.get("pricing") is not None
