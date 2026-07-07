"""Donut end-to-end DL competitor adapter (kind='donut').

Stubs the DL inference (`_run_donut`) so no torch/transformers/weights are needed.
Proves: build_solution dispatches the kind; the adapter maps Donut's FIXED output
keys onto benchmark schema fields via field_map and emits valid JSON; and a schema
field the model cannot produce is declared-not-emitted (never a null in output).
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.serving.solutions import DonutSolution, SolutionError, build_solution


def _png_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(b"not-a-real-png").decode()


def _image_request(model: str) -> dict:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": _png_data_uri()}}],
            }
        ],
    }


def _donut_profile(**options: object) -> ModelProfile:
    base = {
        "field_map": {"nm": "vendor_name", "total_price": "total"},
        "supported_fields": ["vendor_name", "total"],
    }
    base.update(options)
    return ModelProfile(
        name="donut_cord", model="", base_url="", api_key="", kind="donut", options=base
    )


def test_build_solution_dispatches_donut() -> None:
    assert isinstance(build_solution(_donut_profile()), DonutSolution)


def test_donut_maps_fixed_keys_and_emits_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    # Donut's fixed CORD schema — note `extra_key` has no field_map entry.
    monkeypatch.setattr(
        "docie_bench.serving.solutions._run_donut",
        lambda model_name, task_prompt, raw: {
            "nm": "ACME CORP",
            "total_price": "42.00",
            "extra_key": "ignored",
        },
    )
    solution = build_solution(_donut_profile())
    completion = asyncio.run(solution.complete(_image_request("donut_cord")))
    content = completion["choices"][0]["message"]["content"]
    parsed = json.loads(content)  # must be valid JSON the benchmark can parse
    assert parsed == {
        "vendor_name": {"value": "ACME CORP"},
        "total": {"value": "42.00"},
    }
    assert completion["model"] == "donut_cord"


def test_donut_omits_unsupported_field_no_null(monkeypatch: pytest.MonkeyPatch) -> None:
    # `invoice_number` is a schema field Donut cannot emit — it must be ABSENT,
    # not present as null (which would be scored wrong instead of unsupported).
    monkeypatch.setattr(
        "docie_bench.serving.solutions._run_donut",
        lambda model_name, task_prompt, raw: {"nm": "ACME", "total_price": "1.00"},
    )
    solution = build_solution(_donut_profile())
    completion = asyncio.run(solution.complete(_image_request("donut_cord")))
    parsed = json.loads(completion["choices"][0]["message"]["content"])
    assert "invoice_number" not in parsed
    assert None not in parsed.values()


def test_donut_requires_field_map() -> None:
    profile = ModelProfile(name="d", model="", base_url="", api_key="", kind="donut", options={})
    with pytest.raises(SolutionError):
        build_solution(profile)


def test_donut_supported_field_without_mapping_is_construction_error() -> None:
    # `invoice_number` declared supported but absent from field_map => always null;
    # caught at construction, not as a silent runtime null.
    with pytest.raises(SolutionError):
        build_solution(
            _donut_profile(supported_fields=["vendor_name", "total", "invoice_number"])
        )
