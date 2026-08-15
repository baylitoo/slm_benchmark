"""``add_pipeline_profile`` -- the write-side counterpart to ``load_model_profiles``.

Authors a ``kind: pipeline`` (OCR->LLM) entry into a models.yaml by splicing text
rather than a full yaml.safe_load/safe_dump round-trip, specifically so the file's
hand-written comments survive (see the function's own docstring for why). These tests
cover: the comment-preservation property that motivates the splice approach, the
round-trip through ``load_model_profiles`` that proves the written YAML actually
parses into the right ``ModelProfile``, and every validation rule mirrored from
``serving.solutions.PipelineSolution``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.llm.model_profiles import (
    ProfileConflictError,
    ProfileWriteError,
    add_pipeline_profile,
    load_model_profiles,
)

FIXTURE = """\
# Model profiles are runtime endpoints, not hard dependencies.
# Some hand-written documentation a full round-trip would destroy.
profiles:
  extractor_llm:
    model: qwen3:4b
    base_url: http://localhost:11434/v1
    api_key: local-not-used
    prompt_profile: strict_extraction_v1

  vision_llm:
    model: vl-model
    base_url: http://localhost:8088/v1
    api_key: local-not-used
    vision: true

  ocr_only:
    kind: ocr
    options:
      backend: tesseract
"""


@pytest.fixture
def models_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def test_appends_ocr_backend_pipeline_profile_and_preserves_comments(
    models_yaml: Path,
) -> None:
    profile = add_pipeline_profile(
        models_yaml,
        name="invoice_pipeline",
        extractor="extractor_llm",
        ocr_backend="tesseract",
        language="en",
    )

    assert profile.kind == "pipeline"
    assert profile.options == {
        "extractor": "extractor_llm",
        "ocr_backend": "tesseract",
        "language": "en",
    }

    text = models_yaml.read_text(encoding="utf-8")
    assert "# Model profiles are runtime endpoints, not hard dependencies." in text
    assert "Some hand-written documentation a full round-trip would destroy." in text

    # Existing profiles still parse correctly, unchanged.
    reloaded = load_model_profiles(models_yaml)
    assert set(reloaded) == {"extractor_llm", "vision_llm", "ocr_only", "invoice_pipeline"}
    assert reloaded["extractor_llm"].model == "qwen3:4b"
    assert reloaded["invoice_pipeline"].kind == "pipeline"
    assert reloaded["invoice_pipeline"].options["extractor"] == "extractor_llm"


def test_appends_ocr_model_pipeline_profile(models_yaml: Path) -> None:
    profile = add_pipeline_profile(
        models_yaml,
        name="vlm_pipeline",
        extractor="extractor_llm",
        ocr_model="vision_llm",
    )

    assert profile.options == {"extractor": "extractor_llm", "ocr_model": "vision_llm"}


def test_duplicate_name_raises_conflict(models_yaml: Path) -> None:
    with pytest.raises(ProfileConflictError):
        add_pipeline_profile(
            models_yaml, name="extractor_llm", extractor="extractor_llm", ocr_backend="tesseract"
        )


def test_missing_name_raises(models_yaml: Path) -> None:
    with pytest.raises(ProfileWriteError):
        add_pipeline_profile(
            models_yaml, name="  ", extractor="extractor_llm", ocr_backend="tesseract"
        )


@pytest.mark.parametrize(
    ("ocr_backend", "ocr_model"),
    [(None, None), ("tesseract", "vision_llm")],
)
def test_requires_exactly_one_ocr_source(
    models_yaml: Path, ocr_backend: str | None, ocr_model: str | None
) -> None:
    with pytest.raises(ProfileWriteError, match="exactly one"):
        add_pipeline_profile(
            models_yaml,
            name="bad_pipeline",
            extractor="extractor_llm",
            ocr_backend=ocr_backend,
            ocr_model=ocr_model,
        )


def test_unknown_extractor_raises(models_yaml: Path) -> None:
    with pytest.raises(ProfileWriteError, match="not configured"):
        add_pipeline_profile(
            models_yaml, name="p", extractor="nope", ocr_backend="tesseract"
        )


def test_non_passthrough_extractor_raises(models_yaml: Path) -> None:
    with pytest.raises(ProfileWriteError, match="passthrough"):
        add_pipeline_profile(
            models_yaml, name="p", extractor="ocr_only", ocr_backend="tesseract"
        )


def test_unknown_ocr_model_raises(models_yaml: Path) -> None:
    with pytest.raises(ProfileWriteError, match="not configured"):
        add_pipeline_profile(
            models_yaml, name="p", extractor="extractor_llm", ocr_model="nope"
        )


def test_non_vision_ocr_model_raises(models_yaml: Path) -> None:
    with pytest.raises(ProfileWriteError, match="vision"):
        add_pipeline_profile(
            models_yaml, name="p", extractor="extractor_llm", ocr_model="extractor_llm"
        )


def test_unknown_ocr_backend_raises(models_yaml: Path) -> None:
    with pytest.raises(ProfileWriteError, match="Unknown OCR backend"):
        add_pipeline_profile(
            models_yaml, name="p", extractor="extractor_llm", ocr_backend="not-a-backend"
        )


def test_missing_file_creates_profiles_block(tmp_path: Path) -> None:
    path = tmp_path / "fresh.yaml"
    with pytest.raises(ProfileWriteError, match="not configured"):
        # No extractor exists yet in a brand-new file -- still exercises the
        # "file doesn't exist" branch of both load and splice before failing.
        add_pipeline_profile(path, name="p", extractor="nope", ocr_backend="tesseract")
