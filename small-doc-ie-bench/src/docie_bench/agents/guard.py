"""Encoder-backed analyzer for the security proxy (``options.guard_model``).

Talks the encoder convention (see ``docie_bench.encoders.server``): one
OpenAI chat request per text, entities back as JSON. The guard endpoint is
resolved like any backing model (profile / live deployment / ``store:<name>``),
so a GLiNER deployment on this CPU server is addressed exactly like an SLM.

Everything returned here feeds the SAME placeholder/anonymize machinery as the
regex analyzer — the guard only changes *what gets found*, never what happens
to it.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from docie_bench.agents.pii import PiiEntity
from docie_bench.llm.model_profiles import ModelProfile


class GuardAnalysisError(Exception):
    """The guard encoder could not produce a usable analysis."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def labels_from_entities(entities: list[str] | None) -> list[str] | None:
    """Derive zero-shot labels from regex entity types when no explicit
    ``guard_labels`` are configured (``CREDIT_CARD`` -> ``credit card``)."""
    if not entities:
        return None
    return [str(entity).replace("_", " ").lower() for entity in entities]


def _placeholder_type(label: str) -> str:
    """Encoder label -> placeholder type (``phone number`` -> ``PHONE_NUMBER``)."""
    cleaned = "".join(c if c.isalnum() else "_" for c in label.strip().upper())
    return "_".join(part for part in cleaned.split("_") if part) or "PII"


def _parse_entities(payload: dict[str, Any], text: str) -> list[PiiEntity]:
    """Validate the encoder's entities against the analyzed text.

    Spans are trusted only when ``text[start:end] == value``; otherwise the
    value is re-located with ``str.find`` and dropped if absent — a guard that
    hallucinates offsets must degrade to "not found", never to masking the
    wrong characters. Overlaps keep the higher-score (then longer) span.
    """
    raw = payload.get("entities")
    if not isinstance(raw, list):
        raise GuardAnalysisError("guard response has no 'entities' list")

    scored: list[tuple[float, PiiEntity]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", ""))
        label = str(item.get("type", "")).strip()
        if not value or not label:
            continue
        start = item.get("start")
        end = item.get("end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or text[start:end] != value
        ):
            start = text.find(value)
            if start < 0:
                continue
            end = start + len(value)
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scored.append((score, PiiEntity(_placeholder_type(label), value, start, end)))

    scored.sort(key=lambda pair: (-pair[0], -(pair[1].end - pair[1].start), pair[1].start))
    claimed: list[tuple[int, int]] = []
    kept: list[PiiEntity] = []
    for _, entity in scored:
        if any(entity.start < end and start < entity.end for start, end in claimed):
            continue
        claimed.append((entity.start, entity.end))
        kept.append(entity)
    kept.sort(key=lambda entity: entity.start)
    return kept


async def guard_analyze(
    text: str,
    *,
    guard: ModelProfile,
    http_client: httpx.AsyncClient,
    labels: list[str] | None = None,
    threshold: float | None = None,
) -> list[PiiEntity]:
    """Analyze one text through the guard encoder endpoint."""
    body: dict[str, Any] = {
        "model": guard.model,
        "messages": [{"role": "user", "content": text}],
    }
    if labels:
        body["labels"] = labels
    if threshold is not None:
        body["threshold"] = threshold
    url = f"{guard.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {guard.api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = await http_client.post(
            url, json=body, headers=headers, timeout=guard.timeout_seconds
        )
    except httpx.RequestError as exc:
        raise GuardAnalysisError(
            f"guard encoder {guard.base_url} is unreachable: {exc}"
        ) from exc
    if response.status_code >= 400:
        raise GuardAnalysisError(
            f"guard encoder returned {response.status_code}: {response.text[:300]}"
        )
    try:
        completion = response.json()
    except ValueError as exc:
        raise GuardAnalysisError("guard encoder returned a non-JSON response") from exc
    if not isinstance(completion, dict):
        raise GuardAnalysisError("guard encoder returned an unexpected payload shape")

    # Prefer the mirrored extension key; fall back to parsing the assistant
    # content so any endpoint honouring the convention (or an SLM prompted to
    # emit it) works too.
    payload = completion.get("docie_encoder")
    if not isinstance(payload, dict):
        try:
            choices = completion.get("choices") or []
            content = choices[0]["message"]["content"]
            payload = json.loads(content)
        except (LookupError, TypeError, ValueError) as exc:
            raise GuardAnalysisError(
                "guard encoder response carries no parseable entities payload"
            ) from exc
    if not isinstance(payload, dict):
        raise GuardAnalysisError("guard encoder entities payload is not a JSON object")
    return _parse_entities(payload, text)
