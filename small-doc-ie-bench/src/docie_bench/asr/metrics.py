from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

_WHITESPACE = re.compile(r"\s+")


def normalize_transcript(value: str) -> str:
    """Apply a deterministic, language-neutral ASR scoring normalization.

    NFKC and case-folding make Unicode/case differences non-errors. Punctuation
    and symbols become word boundaries rather than disappearing and joining
    words. Language-specific number expansion is intentionally out of scope.
    """

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return _WHITESPACE.sub(" ", normalized).strip()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance using O(min(n, m)) memory."""

    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_value in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_value in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_value != hyp_value),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class TranscriptScore:
    normalized_reference: str
    normalized_hypothesis: str
    reference_words: int
    word_errors: int
    wer: float
    reference_characters: int
    character_errors: int
    cer: float


def score_transcript(reference: str, hypothesis: str) -> TranscriptScore:
    normalized_reference = normalize_transcript(reference)
    normalized_hypothesis = normalize_transcript(hypothesis)
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    # CER excludes normalized whitespace. WER already measures word boundaries,
    # while CER should show recognition errors inside the words themselves.
    reference_characters = list(normalized_reference.replace(" ", ""))
    hypothesis_characters = list(normalized_hypothesis.replace(" ", ""))
    word_errors = edit_distance(reference_words, hypothesis_words)
    character_errors = edit_distance(reference_characters, hypothesis_characters)
    return TranscriptScore(
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        reference_words=len(reference_words),
        word_errors=word_errors,
        wer=_rate(word_errors, len(reference_words)),
        reference_characters=len(reference_characters),
        character_errors=character_errors,
        cer=_rate(character_errors, len(reference_characters)),
    )


def _rate(errors: int, reference_units: int) -> float:
    if reference_units:
        return errors / reference_units
    return 0.0 if errors == 0 else 1.0
