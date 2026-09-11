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
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
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
    "janvier": 1,
    "janv": 1,
    "jan": 1,
    "january": 1,
    "fevrier": 2,
    "février": 2,
    "fev": 2,
    "fév": 2,
    "feb": 2,
    "february": 2,
    "mars": 3,
    "mar": 3,
    "march": 3,
    "avril": 4,
    "avr": 4,
    "apr": 4,
    "april": 4,
    "mai": 5,
    "may": 5,
    "juin": 6,
    "jun": 6,
    "june": 6,
    "juillet": 7,
    "juil": 7,
    "jul": 7,
    "july": 7,
    "aout": 8,
    "août": 8,
    "aug": 8,
    "august": 8,
    "septembre": 9,
    "sept": 9,
    "sep": 9,
    "september": 9,
    "octobre": 10,
    "oct": 10,
    "october": 10,
    "novembre": 11,
    "nov": 11,
    "november": 11,
    "decembre": 12,
    "décembre": 12,
    "dec": 12,
    "déc": 12,
    "december": 12,
}
_END_HINTS = ("end", "fin", "until", "to_", "_to", "jusqu")
_RANGE_SPLIT = re.compile(r"\s+(?:-|–|—|->|→|à|au|to|jusqu'à)\s+|\s*(?:–|—|->|→)\s*")
_YEAR_RANGE = re.compile(r"^\s*(\d{4})\s*[-/–—]\s*(\d{4})\s*$")
_ISO = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")
_NUMERIC = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$|^(\d{1,2})[./-](\d{4})$")
_TEXTUAL = re.compile(r"^(?:(\d{1,2})(?:er|e|st|nd|rd|th)?\s+)?([a-zéûèê.]+)\.?\s+(\d{4})$")


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


_KEEP = object()

_CURRENCY_BY_SYMBOL = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY"}
_CURRENCY_CODES = {"EUR", "USD", "GBP", "CHF", "CAD", "JPY", "AUD", "SEK", "NOK", "DKK"}
_NUMBER_CHARS = re.compile(r"^[+-]?[\d .,'  ]+$")
_GROUPED = re.compile(r"^\d{1,3}(?:[ .,'  ]\d{3})+$")
_THOUSANDS = re.compile(r"[ .,'  ]")


def _refs(node: Any) -> set[str]:
    """Every ``$ref`` name reachable from a schema node without resolving it."""
    names: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            names.add(ref.rsplit("/", 1)[-1])
        for choice in node.get("anyOf", []) or []:
            names |= _refs(choice)
    return names


def _walk_schema(payload: Any, root: dict[str, Any], leaf: Callable[..., Any]) -> Any:
    """Walk a rehydrated extraction next to its rich Pydantic schema.

    ``leaf(value, node, refs, name)`` sees every value with its resolved schema
    node, the ``$ref`` names that led to it (``DateField``, ``MoneyField``, ...)
    and its field name. Returning ``_KEEP`` descends into the value as usual.
    """
    defs = root.get("$defs", {})

    def resolve(node: Any) -> Any:
        seen = 0
        while isinstance(node, dict) and "$ref" in node and seen < 100:
            node = defs.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
            seen += 1
        return node

    def pick(node: dict[str, Any]) -> Any:
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
        refs = _refs(node)
        resolved = resolve(node)
        if not isinstance(resolved, dict):
            return value
        resolved = resolve(pick(resolved))
        if not isinstance(resolved, dict):
            resolved = {}
        replacement = leaf(value, resolved, refs, name)
        if replacement is not _KEEP:
            return replacement
        if isinstance(value, list) and isinstance(resolved.get("items"), dict | list):
            items = resolved["items"]
            return [walk(item, items, name) for item in value]
        props = resolved.get("properties")
        if isinstance(value, dict) and isinstance(props, dict):
            return {k: walk(v, props.get(k, {}), k) for k, v in value.items()}
        return value

    return walk(payload, root, "")


def normalize_by_schema(payload: Any, root: dict[str, Any]) -> Any:
    """Every DateField ``value`` is normalised, every null list becomes ``[]``."""

    def leaf(value: Any, node: dict[str, Any], refs: set[str], name: str) -> Any:
        if "DateField" in refs and isinstance(value, dict) and isinstance(value.get("value"), str):
            return {**value, "value": normalize_date(value["value"], field_name=name)}
        if value is None and node.get("type") == "array":
            return []
        return _KEEP

    return _walk_schema(payload, root, leaf)


def split_currency(text: str) -> tuple[str, str | None]:
    """Split a leading or trailing currency marker off an amount."""
    text = text.strip()
    for marker, code in _CURRENCY_BY_SYMBOL.items():
        if text.startswith(marker):
            return text[len(marker) :].strip(), code
        if text.endswith(marker):
            return text[: -len(marker)].strip(), code
    head, _, tail = text.rpartition(" ")
    if head and tail.upper() in _CURRENCY_CODES:
        return head.strip(), tail.upper()
    head, _, tail = text.partition(" ")
    if tail and head.upper() in _CURRENCY_CODES:
        return tail.strip(), head.upper()
    return text, None


def parse_number(text: str) -> Decimal | None:
    """Parse a human-written amount, or ``None`` when the text is not one.

    Deliberately strict: only digits, a sign and separators are accepted, and
    every thousands group must hold exactly three digits. Stripping the other
    characters instead would read ``"12 rue de la Paix"`` as ``12`` and a phone
    number as a nine-digit amount.
    """
    text = text.strip().replace(" ", " ").replace(" ", " ")
    if not text or not _NUMBER_CHARS.match(text) or not any(c.isdigit() for c in text):
        return None
    sign = "-" if text.startswith("-") else ""
    text = text.lstrip("+-").strip()

    decimals = ""
    last = max(",.", key=text.rfind)
    if last in text:
        head, _, tail = text.rpartition(last)
        if last in head:
            # "1,234,567": the separator repeats, so all of them group thousands.
            pass
        elif tail.isdigit() and (len(tail) != 3 or _THOUSANDS.search(head)):
            # Three digits behind the only separator is a thousands group
            # ("1,234"); anything else is the decimal part ("3,5", "1.234,56").
            text, decimals = head, tail

    integer = text.strip() or "0"
    if not integer.isdigit() and not _GROUPED.match(integer):
        return None
    digits = _THOUSANDS.sub("", integer)
    if not digits.isdigit():
        return None
    try:
        return Decimal(f"{sign}{digits}.{decimals}" if decimals else f"{sign}{digits}")
    except InvalidOperation:
        return None


def normalize_currency(value: Any) -> str | None:
    """Map a currency symbol to its ISO 4217 code, keeping any other code.

    Unlike :func:`split_currency`, which needs a whitelist to tell a currency
    from a word, this reads an already typed currency leaf: every three-letter
    code is kept, so a client billing in MAD or XOF is not silently nulled.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text in _CURRENCY_BY_SYMBOL:
        return _CURRENCY_BY_SYMBOL[text]
    return text.upper() if len(text) == 3 and text.isalpha() else None


def coerce_scalars(payload: Any, root: dict[str, Any]) -> tuple[Any, list[str]]:
    """Coerce number and money leaves the model wrote as text.

    Small extractors copy amounts as they appear ("1 234,56 EUR", "3,5"), which
    a Decimal-typed field rejects, and one rejected leaf used to invalidate the
    whole document. Runs after grounding, which matches the value as the model
    wrote it against the page text.
    """
    warnings: list[str] = []

    def coerce(value: dict[str, Any], name: str, key: str) -> dict[str, Any]:
        raw = value.get(key)
        if not isinstance(raw, str):
            return value
        text, currency = split_currency(raw)
        number = parse_number(text)
        if number is None:
            warnings.append(f"{name or key}: {raw!r} is not a number; value dropped")
            return {**value, key: None}
        out = {**value, key: str(number)}
        if currency and key == "amount" and not out.get("currency"):
            out["currency"] = currency
        return out

    def leaf(value: Any, node: dict[str, Any], refs: set[str], name: str) -> Any:
        if not isinstance(value, dict):
            return _KEEP
        if "NumberField" in refs:
            return coerce(value, name, "value")
        if "MoneyField" in refs:
            money = coerce(value, name, "amount")
            written = money.get("currency")
            if isinstance(written, str):
                code = normalize_currency(written)
                if code is None:
                    warnings.append(
                        f"{name or 'currency'}: {written!r} is not a currency; value dropped"
                    )
                money = {**money, "currency": code}
            return money
        return _KEEP

    return _walk_schema(payload, root, leaf), warnings
