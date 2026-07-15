"""Regex-based PII analyzer + placeholder anonymizer for the security proxy.

Deterministic and dependency-free by design: this is the baseline detector the
``proxy_security`` agent ships with. It is deliberately conservative (checksum
validation on cards/IBANs, octet checks on IPs) so numeric-heavy business
documents — invoices, orders — are not shredded by false positives. A future
``options.guard_model`` (an encoder specialized in PII/NER) plugs in above this
as a higher-recall analyzer; the placeholder/restore mechanics stay the same.

Placeholders are stable per value within one request (``[EMAIL_1]`` everywhere
the same address appears), so the backing model can still reason about entity
identity without ever seeing the raw value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Detector priority: earlier types win when spans overlap (an IBAN's digit run
# must not be re-claimed as a phone number).
PII_TYPES = ("EMAIL", "IBAN", "CREDIT_CARD", "NATIONAL_ID", "PHONE", "IP_ADDRESS")

_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "EMAIL": [re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")],
    "IBAN": [re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")],
    "CREDIT_CARD": [re.compile(r"\b(?:\d[ -]?){12,18}\d\b")],
    "NATIONAL_ID": [
        # US SSN (dashed form only — the undashed form collides with too much).
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        # French NIR: sex(1/2) yy mm dept commune order key = 15 digits.
        re.compile(r"\b[12]\d{2}(?:0[1-9]|1[0-2])(?:\d{2}|2A|2B)\d{6}\d{2}\b"),
    ],
    "PHONE": [
        re.compile(r"\+\d{1,3}[ .-]?\d(?:[ .-]?\d){7,12}"),
        # French national grouped form: 06 12 34 56 78.
        re.compile(r"\b0\d(?:[ .]\d{2}){4}\b"),
        # North-American forms: (555) 123-4567 / 555-123-4567.
        re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b"),
    ],
    "IP_ADDRESS": [re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")],
}


@dataclass(frozen=True)
class PiiEntity:
    """One detected PII span (value retained in-process only, never reported)."""

    type: str
    value: str
    start: int
    end: int


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _iban_ok(candidate: str) -> bool:
    compact = candidate.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    remainder = 0
    for char in rearranged:
        value = str(int(char, 36))  # digits stay, letters -> 10..35
        remainder = int(str(remainder) + value) % 97
    return remainder == 1


def _ip_ok(candidate: str) -> bool:
    return all(0 <= int(octet) <= 255 for octet in candidate.split("."))


def _validated(pii_type: str, value: str) -> bool:
    if pii_type == "CREDIT_CARD":
        digits = re.sub(r"[ -]", "", value)
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    if pii_type == "IBAN":
        return _iban_ok(value)
    if pii_type == "IP_ADDRESS":
        return _ip_ok(value)
    return True


def analyze(text: str, entities: list[str] | None = None) -> list[PiiEntity]:
    """Detect PII spans in ``text``, restricted to ``entities`` when given.

    Returns non-overlapping spans in document order; on overlap the earlier
    type in ``PII_TYPES`` wins (then the longer match).
    """
    wanted = [t for t in PII_TYPES if entities is None or t in entities]
    candidates: list[PiiEntity] = []
    for pii_type in wanted:
        for pattern in _PATTERNS[pii_type]:
            for match in pattern.finditer(text):
                if _validated(pii_type, match.group()):
                    candidates.append(
                        PiiEntity(pii_type, match.group(), match.start(), match.end())
                    )

    # Priority resolution: type order first, longest span next; claimed
    # character ranges block lower-priority overlaps.
    candidates.sort(key=lambda e: (PII_TYPES.index(e.type), -(e.end - e.start), e.start))
    claimed: list[tuple[int, int]] = []
    kept: list[PiiEntity] = []
    for entity in candidates:
        if any(entity.start < end and start < entity.end for start, end in claimed):
            continue
        claimed.append((entity.start, entity.end))
        kept.append(entity)
    kept.sort(key=lambda e: e.start)
    return kept


def anonymize(
    text: str,
    detected: list[PiiEntity],
    *,
    placeholders: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Replace detected spans with stable placeholders.

    ``placeholders`` maps placeholder -> original value; pass the same dict
    across the messages of one request so a value repeated in two messages
    gets ONE placeholder. Returns ``(masked_text, placeholders)``.
    """
    mapping: dict[str, str] = placeholders if placeholders is not None else {}
    value_to_placeholder = {value: key for key, value in mapping.items()}
    counters: dict[str, int] = {}
    for key in mapping:
        pii_type, _, index = key.strip("[]").rpartition("_")
        if pii_type and index.isdigit():
            counters[pii_type] = max(counters.get(pii_type, 0), int(index))

    pieces: list[str] = []
    cursor = 0
    for entity in detected:
        placeholder = value_to_placeholder.get(entity.value)
        if placeholder is None:
            counters[entity.type] = counters.get(entity.type, 0) + 1
            placeholder = f"[{entity.type}_{counters[entity.type]}]"
            mapping[placeholder] = entity.value
            value_to_placeholder[entity.value] = placeholder
        pieces.append(text[cursor : entity.start])
        pieces.append(placeholder)
        cursor = entity.end
    pieces.append(text[cursor:])
    return "".join(pieces), mapping


def deanonymize(text: str, placeholders: dict[str, str]) -> str:
    """Restore original values for every placeholder present in ``text``."""
    for placeholder, value in placeholders.items():
        text = text.replace(placeholder, value)
    return text
