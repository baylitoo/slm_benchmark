"""The arch registry's tables must agree with each other.

`resolve_family` reads six tables, and adding a model family touches four or
five of them. Every disagreement between them is silent rather than loud:

* a vision upgrade whose target is not actually a vision family serves a
  multimodal checkpoint through a text contract -- the images are simply
  dropped, and the extraction comes back plausibly wrong rather than failing
* a ``RUNTIME_NOTES`` key that no table can produce is a note that can never
  be shown, so the operator never learns their llama-server is too old and
  reads the load failure as a broken model
* a family name that exists in a mapping but not in ``FAMILIES`` fails at
  seed time, long after the detection that chose it looked fine

One of these (``ARCH_TO_FAMILY`` -> ``FAMILIES``) was already covered in
test_arch_detect.py. These are the rest.
"""

from __future__ import annotations

from docie_bench.serving.arch_registry import (
    ARCH_TO_FAMILY,
    DEFAULT_VISION_FAMILY,
    RUNTIME_NOTES,
    TEXT_ARCH_TO_VISION_FAMILY,
    TRANSFORMERS_FAMILY,
    VISION_ARCHS,
    VISION_FAMILIES,
)
from docie_bench.serving.model_store import FAMILIES


def test_every_vision_family_is_a_real_family() -> None:
    assert not VISION_FAMILIES - set(FAMILIES)


def test_every_vision_upgrade_target_is_a_real_family() -> None:
    assert not set(TEXT_ARCH_TO_VISION_FAMILY.values()) - set(FAMILIES)


def test_every_vision_upgrade_target_is_actually_multimodal() -> None:
    """The one that matters most.

    An mmproj makes a repo multimodal whatever arch the backbone reports, so
    the text family is upgraded. Upgrading to a family that is not in
    ``VISION_FAMILIES`` would hand a multimodal checkpoint a text contract:
    the images are dropped and the extraction answers from the prompt alone.
    """
    assert not set(TEXT_ARCH_TO_VISION_FAMILY.values()) - VISION_FAMILIES


def test_the_default_vision_family_is_multimodal() -> None:
    """The fallback for an mmproj on an unrecognised backbone, so it carries
    the same requirement as the per-backbone targets above."""
    assert DEFAULT_VISION_FAMILY in VISION_FAMILIES
    assert DEFAULT_VISION_FAMILY in FAMILIES


def test_every_vision_upgrade_key_is_an_arch_the_registry_knows() -> None:
    """A key no detection can produce is dead weight: the upgrade never fires
    and the repo keeps whatever family the backbone's arch chose."""
    assert not set(TEXT_ARCH_TO_VISION_FAMILY) - set(ARCH_TO_FAMILY)


def test_every_runtime_note_is_reachable() -> None:
    """A note keyed by an arch no table produces can never be shown, so an
    operator on an older llama-server reads an arch it cannot load as a broken
    model rather than an image that needs rebuilding."""
    reachable = set(ARCH_TO_FAMILY) | VISION_ARCHS | set(TEXT_ARCH_TO_VISION_FAMILY)
    assert not set(RUNTIME_NOTES) - reachable


def test_the_transformers_fallback_family_exists() -> None:
    """The last-resort family every unmatched safetensors repo lands on."""
    assert TRANSFORMERS_FAMILY in FAMILIES
