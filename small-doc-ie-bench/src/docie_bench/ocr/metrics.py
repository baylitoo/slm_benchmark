from __future__ import annotations

import re
from typing import Any

from rapidfuzz.distance import Levenshtein

from docie_bench.schemas.common import OCRBlock


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def error_rate(reference: list[str], hypothesis: list[str]) -> float | None:
    if not reference:
        return 0.0 if not hypothesis else None
    return Levenshtein.distance(reference, hypothesis) / len(reference)


def score_ocr(
    reference_text: str,
    blocks: list[OCRBlock],
    reference_blocks: list[OCRBlock] | None = None,
) -> dict[str, Any]:
    hypothesis_text = "\n".join(block.text for block in blocks)
    reference_chars = list(normalize_text(reference_text))
    hypothesis_chars = list(normalize_text(hypothesis_text))
    reference_words = normalize_text(reference_text).split()
    hypothesis_words = normalize_text(hypothesis_text).split()
    cer = error_rate(reference_chars, hypothesis_chars)
    wer = error_rate(reference_words, hypothesis_words)
    expected_layout = (
        "\n".join(block.text for block in reference_blocks)
        if reference_blocks is not None
        else reference_text
    )
    return {
        "character_error_rate": cer,
        "word_error_rate": wer,
        "character_accuracy": max(0.0, 1.0 - cer) if cer is not None else None,
        "word_accuracy": max(0.0, 1.0 - wer) if wer is not None else None,
        "layout_preservation": layout_preservation(expected_layout, hypothesis_text),
    }


def layout_preservation(reference: str, hypothesis: str) -> float | None:
    """Compare line-boundary placement while tolerating backend block segmentation."""
    reference_boundaries = _line_boundaries(reference)
    hypothesis_boundaries = _line_boundaries(hypothesis)
    if not reference_boundaries and not hypothesis_boundaries:
        return 1.0
    if not reference_boundaries or not hypothesis_boundaries:
        return 0.0
    overlap = len(reference_boundaries & hypothesis_boundaries)
    precision = overlap / len(hypothesis_boundaries)
    recall = overlap / len(reference_boundaries)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _line_boundaries(text: str) -> set[int]:
    boundaries: set[int] = set()
    word_count = 0
    lines = [line for line in text.splitlines() if normalize_text(line)]
    for line in lines[:-1]:
        word_count += len(re.findall(r"\S+", normalize_text(line)))
        boundaries.add(word_count)
    return boundaries
