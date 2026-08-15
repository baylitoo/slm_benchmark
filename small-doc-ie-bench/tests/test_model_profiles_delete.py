"""``delete_pipeline_profile`` -- the DELETE counterpart to ``add_pipeline_profile``.

Removes a ``kind: pipeline``/``kind: ocr`` entry by locating its exact line range in
the text and cutting it out, same text-splice philosophy as the INSERT side (a full
yaml.safe_load/safe_dump round-trip would destroy configs/models.yaml's hand-written
comments). The interesting risk here is different from INSERT's: this file's real
per-profile documentation comments sit at the SAME 2-space indent as the profile keys
(see configs/models.yaml, e.g. the comment block above `nuextract3:`), not at column
0 -- so the block-end scan has to be stricter than `_splice_profile`'s own "comments
don't count" rule, or a delete could silently swallow a NEIGHBORING profile's
documentation. These tests cover that directly, plus the edge cases a line-range
delete can get wrong: only/first/last profile in the block, EOF blank-line handling
across a create/delete cycle, a nonexistent name, refusing to delete a passthrough,
and a duplicate top-level key (which must refuse rather than guess).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.llm.model_profiles import (
    ProfileNotFoundError,
    ProfileWriteError,
    _remove_profile_block,
    add_pipeline_profile,
    delete_pipeline_profile,
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

  # This comment documents pipeline_b, the NEXT profile below -- it must survive
  # a delete of pipeline_a untouched, and stay attached to pipeline_b.
  pipeline_b:
    kind: pipeline
    options:
      extractor: extractor_llm
      ocr_backend: tesseract
"""


@pytest.fixture
def models_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(FIXTURE, encoding="utf-8")
    return path


def _insert_pipeline_a_before_pipeline_b(models_yaml: Path) -> None:
    """Splice a comment-less `pipeline_a` block in right before the comment+
    pipeline_b block, matching how add_pipeline_profile actually appends (always
    comment-less) -- this is the shape that would trigger the comment-swallowing
    bug if the delete-side block-end scan reused the insert-side's rule."""
    text = models_yaml.read_text(encoding="utf-8")
    marker = "  # This comment documents pipeline_b"
    idx = text.index(marker)
    block = (
        "  pipeline_a:\n"
        "    kind: pipeline\n"
        "    options:\n"
        "      extractor: extractor_llm\n"
        "      ocr_backend: tesseract\n"
        "\n"
    )
    models_yaml.write_text(text[:idx] + block + text[idx:], encoding="utf-8")


def test_deletes_pipeline_profile_and_preserves_next_profiles_own_comment(
    models_yaml: Path,
) -> None:
    _insert_pipeline_a_before_pipeline_b(models_yaml)
    assert set(load_model_profiles(models_yaml)) >= {"pipeline_a", "pipeline_b"}

    delete_pipeline_profile(models_yaml, name="pipeline_a")

    text = models_yaml.read_text(encoding="utf-8")
    assert "  pipeline_a:\n" not in text
    assert "# This comment documents pipeline_b, the NEXT profile below" in text

    reloaded = load_model_profiles(models_yaml)
    assert "pipeline_a" not in reloaded
    assert reloaded["pipeline_b"].kind == "pipeline"
    assert reloaded["pipeline_b"].options["extractor"] == "extractor_llm"
    # File-level comments untouched either.
    assert "# Model profiles are runtime endpoints, not hard dependencies." in text


def test_deletes_ocr_backend_profile_in_the_middle_of_the_file(models_yaml: Path) -> None:
    delete_pipeline_profile(models_yaml, name="ocr_only")

    reloaded = load_model_profiles(models_yaml)
    assert "ocr_only" not in reloaded
    assert set(reloaded) == {"extractor_llm", "vision_llm", "pipeline_b"}
    # Neighbors on both sides survive with their own fields intact.
    assert reloaded["vision_llm"].vision is True
    assert reloaded["pipeline_b"].options["ocr_backend"] == "tesseract"


def test_deletes_last_profile_in_file(models_yaml: Path) -> None:
    delete_pipeline_profile(models_yaml, name="pipeline_b")

    reloaded = load_model_profiles(models_yaml)
    assert set(reloaded) == {"extractor_llm", "vision_llm", "ocr_only"}
    text = models_yaml.read_text(encoding="utf-8")
    assert not text.rstrip("\n").endswith(":")  # no dangling empty block


def test_deletes_only_profile_leaves_empty_but_loadable_registry(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "profiles:\n  solo:\n    kind: ocr\n    options:\n      backend: tesseract\n",
        encoding="utf-8",
    )

    delete_pipeline_profile(path, name="solo")

    # profiles: with nothing under it parses to {"profiles": None} -- must degrade
    # to an empty registry, not AttributeError on .items().
    assert load_model_profiles(path) == {}


def test_create_then_delete_last_profile_does_not_accumulate_blank_lines(
    models_yaml: Path,
) -> None:
    before = models_yaml.read_text(encoding="utf-8")

    add_pipeline_profile(
        models_yaml, name="roundtrip", extractor="extractor_llm", ocr_backend="tesseract"
    )
    delete_pipeline_profile(models_yaml, name="roundtrip")

    after = models_yaml.read_text(encoding="utf-8")
    assert after == before

    # And a second cycle doesn't drift either (guards against a one-time-only fix).
    add_pipeline_profile(
        models_yaml, name="roundtrip", extractor="extractor_llm", ocr_backend="tesseract"
    )
    delete_pipeline_profile(models_yaml, name="roundtrip")
    assert models_yaml.read_text(encoding="utf-8") == before


def test_deleting_nonexistent_name_raises_not_found(models_yaml: Path) -> None:
    with pytest.raises(ProfileNotFoundError):
        delete_pipeline_profile(models_yaml, name="does_not_exist")


def test_deleting_from_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(ProfileNotFoundError):
        delete_pipeline_profile(tmp_path / "nope.yaml", name="anything")


def test_deleting_passthrough_profile_is_refused(models_yaml: Path) -> None:
    before = models_yaml.read_text(encoding="utf-8")

    with pytest.raises(ProfileWriteError, match="pipeline/ocr"):
        delete_pipeline_profile(models_yaml, name="extractor_llm")

    # Refused before any write -- file untouched.
    assert models_yaml.read_text(encoding="utf-8") == before
    assert "extractor_llm" in load_model_profiles(models_yaml)


def test_duplicate_top_level_key_refuses_to_guess(tmp_path: Path) -> None:
    # A hand-edited file with the same key twice: load_model_profiles (plain
    # yaml.safe_load) silently resolves to whichever PyYAML parses last (here,
    # the pipeline one) -- but the text scan must refuse rather than delete the
    # FIRST (passthrough) occurrence out from under that.
    path = tmp_path / "models.yaml"
    path.write_text(
        "profiles:\n"
        "  dupe:\n"
        "    model: qwen3:4b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: local-not-used\n"
        "\n"
        "  dupe:\n"
        "    kind: pipeline\n"
        "    options:\n"
        "      extractor: dupe\n"
        "      ocr_backend: tesseract\n",
        encoding="utf-8",
    )
    assert load_model_profiles(path)["dupe"].kind == "pipeline"  # confirms the setup
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ProfileWriteError, match="appears 2 times"):
        delete_pipeline_profile(path, name="dupe")

    assert path.read_text(encoding="utf-8") == before


def test_remove_profile_block_raises_if_name_not_found(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("profiles:\n  a:\n    model: x\n", encoding="utf-8")

    with pytest.raises(ProfileWriteError, match="not found"):
        _remove_profile_block(path, "b")
