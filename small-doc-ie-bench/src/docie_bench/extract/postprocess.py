"""Model-independent clean-up of a rehydrated extraction.

Small extractors copy dates as written ("Mars 2022", "2023-2025",
"Aujourd'hui"), write placeholders ("N/A", "-") where the schema wants null,
and occasionally emit the same list element twice. All three are fixed here
deterministically, from the schema's own types, so every model and every
schema benefits the same way.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PLACEHOLDERS = {"", "null", "none", "n/a", "na", "-", "–", "—", "nan"}
_CURRENT = {
    "aujourd'hui",
    "aujourd’hui",
    "aujourdhui",
    "present",
    "présent",
    "presente",
    "actuel",
    "actuellement",
    "en cours",
    "en poste",
    "current",
    "currently",
    "now",
    "ongoing",
    "today",
    "à ce jour",
    "a ce jour",
}
_MONTHS = {
    "janvier": 1, "janv": 1, "jan": 1, "january": 1,
    "fevrier": 2, "février": 2, "fev": 2, "fév": 2, "feb": 2, "february": 2,
    "mars": 3, "mar": 3, "march": 3,
    "avril": 4, "avr": 4, "apr": 4, "april": 4,
    "mai": 5, "may": 5,
    "juin": 6, "jun": 6, "june": 6,
    "juillet": 7, "juil": 7, "jul": 7, "july": 7,
    "aout": 8, "août": 8, "aug": 8, "august": 8,
    "septembre": 9, "sept": 9, "sep": 9, "september": 9,
    "octobre": 10, "oct": 10, "october": 10,
    "novembre": 11, "nov": 11, "november": 11,
    "decembre": 12, "décembre": 12, "dec": 12, "déc": 12, "december": 12,
}
_END_HINTS = ("end", "fin", "until", "to_", "_to", "jusqu")
_RANGE_SPLIT = re.compile(r"\s+(?:-|–|—|->|→|à|au|to|jusqu'à)\s+|\s*(?:–|—|->|→)\s*")
_YEAR_RANGE = re.compile(r"^\s*(\d{4})\s*[-/–—]\s*(\d{4})\s*$")
_ISO = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")
_NUMERIC = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$|^(\d{1,2})[./-](\d{4})$")
_TEXTUAL = re.compile(
    r"^(?:(\d{1,2})(?:er|e|st|nd|rd|th)?\s+)?([a-zéûèê.]+)\.?\s+(\d{4})$"
)


def is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDERS


def normalize_date(value: str, *, field_name: str = "") -> str | None:
    """ISO date (YYYY, YYYY-MM or YYYY-MM-DD) for anything recognisable,
    None for 'current' words and placeholders, the original text otherwise."""
    text = value.strip()
    lowered = text.lower()
    if lowered in _PLACEHOLDERS or lowered.rstrip(".") in _CURRENT:
        return None
    if range_match := _YEAR_RANGE.match(text):
        return range_match.group(2 if _wants_end(field_name) else 1)
    parts = [p for p in _RANGE_SPLIT.split(text) if p and p.strip()]
    if len(parts) == 2:
        picked = parts[1] if _wants_end(field_name) else parts[0]
        return normalize_date(picked, field_name=field_name)
    if iso := _ISO.match(text):
        year, month, day = iso.groups()
        return _assemble(year, month, day) or text
    if numeric := _NUMERIC.match(text):
        day, month, year, month_only, year_only = numeric.groups()
        if year_only:
            return _assemble(year_only, month_only, None) or text
        return _assemble(year, month, day) or text
    if textual := _TEXTUAL.match(lowered):
        day, month_name, year = textual.groups()
        month = _MONTHS.get(month_name.rstrip("."))
        if month is not None:
            return _assemble(year, str(month), day) or text
    return text


def _wants_end(field_name: str) -> bool:
    name = field_name.lower()
    return any(hint in name for hint in _END_HINTS)


def _assemble(year: str, month: str | None, day: str | None) -> str | None:
    if not (1000 <= int(year) <= 2999):
        return None
    if month is None:
        return year
    m = int(month)
    if not 1 <= m <= 12:
        return None
    if day is None:
        return f"{year}-{m:02d}"
    d = int(day)
    if not 1 <= d <= 31:
        return None
    return f"{year}-{m:02d}-{d:02d}"


def normalize_placeholders(value: Any) -> Any:
    """Scalar placeholder strings ("N/A", "-", "null", "") become None."""
    if isinstance(value, dict):
        return {k: normalize_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_placeholders(v) for v in value]
    if isinstance(value, str) and is_placeholder(value):
        return None
    return value


def dedupe_lists(value: Any) -> Any:
    """Drop exact duplicate elements from every list, keeping first occurrences."""
    if isinstance(value, dict):
        return {k: dedupe_lists(v) for k, v in value.items()}
    if isinstance(value, list):
        seen: set[str] = set()
        out: list[Any] = []
        for item in value:
            cleaned = dedupe_lists(item)
            key = json.dumps(cleaned, sort_keys=True, ensure_ascii=False, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out
    return value


def normalize_dates(payload: Any, root: dict[str, Any]) -> Any:
    """Walk a rehydrated extraction with its rich Pydantic schema and
    normalise every DateField's ``value``."""
    defs = root.get("$defs", {})

    def is_date_ref(node: Any) -> bool:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.rsplit("/", 1)[-1] == "DateField":
                return True
            for choice in node.get("anyOf", []) or []:
                if is_date_ref(choice):
                    return True
        return False

    def resolve(node: Any) -> Any:
        seen = 0
        while isinstance(node, dict) and "$ref" in node and seen < 100:
            node = defs.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
            seen += 1
        return node

    def pick(node: dict[str, Any], value: Any) -> Any:
        for key in ("anyOf", "oneOf"):
            choices = node.get(key)
            if not isinstance(choices, list):
                continue
            for choice in choices:
                resolved = resolve(choice)
                if isinstance(resolved, dict) and resolved.get("type") != "null":
                    return resolved
        return node

    def walk(value: Any, node: Any, name: str) -> Any:
        if is_date_ref(node) and isinstance(value, dict) and isinstance(value.get("value"), str):
            return {**value, "value": normalize_date(value["value"], field_name=name)}
        node = resolve(node)
        if not isinstance(node, dict):
            return value
        node = resolve(pick(node, value))
        if isinstance(value, list) and isinstance(node.get("items"), dict | list):
            items = node["items"]
            return [walk(v, items, name) for v in value]
        props = node.get("properties")
        if isinstance(value, dict) and isinstance(props, dict):
            return {k: walk(v, props.get(k, {}), k) for k, v in value.items()}
        return value

    return walk(payload, root, "")
