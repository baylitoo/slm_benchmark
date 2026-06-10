from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any

from rapidfuzz import fuzz

from docie_bench.schemas.common import OCRBlock

DEFAULT_MATCH_THRESHOLD = 0.7


def ground_evidence(
    payload: dict[str, Any],
    blocks: list[OCRBlock],
    *,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> dict[str, Any]:
    """Link typed extraction fields to the best matching OCR block."""
    grounded = _copy_and_ground(payload, blocks, match_threshold)
    return grounded if isinstance(grounded, dict) else payload


def _copy_and_ground(obj: Any, blocks: list[OCRBlock], match_threshold: float) -> Any:
    if isinstance(obj, list):
        return [_copy_and_ground(item, blocks, match_threshold) for item in obj]
    if not isinstance(obj, dict):
        return obj

    result = {
        key: _copy_and_ground(value, blocks, match_threshold)
        for key, value in obj.items()
    }
    candidate = _field_candidate(result)
    if candidate is None:
        return result

    evidence_ids, score = _best_match(candidate, blocks)
    if evidence_ids and score >= match_threshold:
        result["evidence_ids"] = evidence_ids
        result["confidence"] = round(score, 4)
    else:
        result["evidence_ids"] = []
        result["confidence"] = 0.0
    return result


def _field_candidate(field: dict[str, Any]) -> str | None:
    value = field.get("value", field.get("amount"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _best_match(candidate: str, blocks: list[OCRBlock]) -> tuple[list[str], float]:
    variants = _candidate_variants(candidate)
    best_ids: list[str] = []
    best_score = 0.0
    for index, block in enumerate(blocks):
        spans = [([block.id], block.text)]
        if index + 1 < len(blocks) and blocks[index + 1].page == block.page:
            next_block = blocks[index + 1]
            spans.append(([block.id, next_block.id], f"{block.text} {next_block.text}"))
        for evidence_ids, text in spans:
            block_text = _normalize(text)
            if not block_text:
                continue
            score = max(_match_score(variant, block_text) for variant in variants)
            if score > best_score or (
                score == best_score and (not best_ids or len(evidence_ids) < len(best_ids))
            ):
                best_ids = evidence_ids
                best_score = score
    return best_ids, best_score


def _match_score(candidate: str, block_text: str) -> float:
    if candidate == block_text or candidate in block_text:
        return 1.0
    if len(candidate) < 4:
        return 0.0
    coverage = min(1.0, len(block_text) / len(candidate))
    return (fuzz.partial_ratio(candidate, block_text) / 100) * coverage


def _candidate_variants(value: str) -> set[str]:
    variants = {_normalize(value)}
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return {variant for variant in variants if variant}
    variants.update(
        {
            _normalize(parsed.strftime("%d/%m/%Y")),
            _normalize(parsed.strftime("%m/%d/%Y")),
            _normalize(parsed.strftime("%d %B %Y")),
        }
    )
    return {variant for variant in variants if variant}


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", ascii_text)
