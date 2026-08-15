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
    "name", ["has space", "has/slash", "has:colon", "", "-leading-dash", "über"]
)
def test_invalid_name_charset_raises(models_yaml: Path, name: str) -> None:
    with pytest.raises(ProfileWriteError):
        add_pipeline_profile(
            models_yaml, name=name, extractor="extractor_llm", ocr_backend="tesseract"
        )


@pytest.mark.parametrize("name", ["yes", "No", "TRUE", "off", "null", "~"])
def test_yaml_boolean_like_name_raises(models_yaml: Path, name: str) -> None:
    # PyYAML's implicit resolver coerces these to bool/None, not str -- silently
    # defeating the duplicate-name check (a second write with the same "name" a
    # human would type would key differently in load_model_profiles and not collide).
    with pytest.raises(ProfileWriteError):
        add_pipeline_profile(
            models_yaml, name=name, extractor="extractor_llm", ocr_backend="tesseract"
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


def test_splice_profile_creates_profiles_header_for_a_brand_new_file(tmp_path: Path) -> None:
    # add_pipeline_profile can never actually reach this branch of _splice_profile:
    # a missing file means existing={}, so the extractor lookup always fails first
    # (see test_missing_file_creates_profiles_block above). Exercised directly so the
    # branch itself -- not just its unreachability through the one caller -- is proven
    # to produce valid YAML.
    from docie_bench.llm.model_profiles import _splice_profile

    path = tmp_path / "fresh.yaml"
    _splice_profile(path, "p", {"kind": "pipeline", "options": {"extractor": "e"}})

    reloaded = load_model_profiles(path)
    assert reloaded["p"].kind == "pipeline"
    assert reloaded["p"].options == {"extractor": "e"}


def test_concurrent_write_losing_the_race_raises_a_clear_error_not_keyerror(
    models_yaml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two callers racing add_pipeline_profile for DIFFERENT names against the same
    # file, with no lock, can have one writer's insert clobbered by the other's
    # os.replace landing after it -- the loser's own name is then missing on re-read.
    # Simulated directly (rather than an actual thread race, which would be flaky)
    # by monkeypatching the post-splice re-read to omit the just-written name, exactly
    # what a lost update looks like from add_pipeline_profile's point of view.
    import docie_bench.llm.model_profiles as model_profiles_module

    real_load = model_profiles_module.load_model_profiles
    calls = {"n": 0}

    def flaky_load(path: Path) -> dict[str, object]:
        calls["n"] += 1
        result = real_load(path)
        if calls["n"] == 2:
            result.pop("racer", None)
        return result

    monkeypatch.setattr(model_profiles_module, "load_model_profiles", flaky_load)

    with pytest.raises(ProfileWriteError, match="concurrent write"):
        add_pipeline_profile(
            models_yaml, name="racer", extractor="extractor_llm", ocr_backend="tesseract"
        )
