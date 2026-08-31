"""Generation-time model confidence from llama-server's per-token logprobs (#335).

Distinct from `extract/grounding.py`'s `confidence` (a POST-HOC score: does the
extracted text actually appear in the OCR evidence). This is the model's OWN
certainty while generating each token -- a field can be perfectly grounded
(verbatim in the document) while the model was genuinely torn between two
candidate values, which grounding can't see and logprobs can. Surfaced as a
sibling field, `model_confidence`, never overwriting `confidence`.

Design (locked in via issue #335's design-decision comment):
- Aggregation: per-field confidence = MIN token logprob over the tokens
  spanning that field's value in the raw generated JSON (pessimistic -- one
  uncertain token drags the whole field down). This is a natural-log
  probability (<= 0, closer to 0 = more confident), NOT a 0-1 score like
  `confidence` -- deliberately not renormalized, so don't compare the two
  directly.
- Scope: llama.cpp-backed deployments only (gated by the caller via
  `ModelProfile.runtime == "llamacpp"` + `ModelProfile.logprob_confidence`).

Mechanics and KNOWN LIMITATIONS (see the PR description for the authoritative
list -- restated briefly here since they shape this module's behavior):
- Token pieces from `logprobs.content[]` are concatenated in generation order
  to reconstruct the model's raw output text. If that reconstruction doesn't
  reproduce the API's own `message.content` EXACTLY, every field for that
  extraction gets `model_confidence: None` rather than a guessed offset --
  some tokenizers/servers report pieces that don't literally concatenate back
  to the exact string, and this module never trusts a mismatched offset.
- Only TOP-LEVEL scalar (string/number/bool) field values are resolved. A
  field whose raw JSON value is a nested object (e.g. a MoneyField's
  `{amount, currency}`) or a list (e.g. `line_items`) is NOT descended into
  and always reports `model_confidence: None` -- attempting to attribute a
  span to one of several sibling leaves (which token range is "amount" vs
  "currency"?) risks silently mis-scoring rather than a clean unresolved.
- Value-span lookup is a plain string search over a handful of common JSON-
  literal representations (quoted string; a few numeric reformattings), tried
  first alongside the field's own JSON key (`"key":value` / `"key": value`)
  and only falling back to a bare, key-independent search of the value alone.
  A repeated short literal (e.g. a currency code, or `0`) can still match the
  wrong occurrence in the bare fallback -- not bulletproof, by design (see
  issue #335: "doesn't need to be bulletproof for every possible JSON value
  shape, just handle the common ones").
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

# logprobs.content[] token count llama-server reports per generated token; 1 is
# enough since only the CHOSEN token's own logprob is needed for min-aggregation,
# never the runner-up alternatives `top_logprobs` would otherwise cost extra
# payload for.
TOP_LOGPROBS_N = 1


def reconstruct_generated_text(
    logprobs_content: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, list[tuple[int, int, float]]] | None:
    """Concatenate `logprobs.content[]` token pieces into the generated text.

    Returns `(text, spans)` where each span is `(start, end, logprob)` in that
    reconstructed text's character offsets -- or `None` if `logprobs_content`
    is empty or any entry is missing a usable `token`/`logprob` pair. Callers
    read `None` as "no per-field confidence available for this extraction",
    never as a reason to crash.
    """
    if not logprobs_content:
        return None
    text_parts: list[str] = []
    spans: list[tuple[int, int, float]] = []
    offset = 0
    for entry in logprobs_content:
        if not isinstance(entry, Mapping):
            return None
        token = entry.get("token")
        logprob = entry.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, (int, float)):
            return None
        start = offset
        offset += len(token)
        spans.append((start, offset, float(logprob)))
        text_parts.append(token)
    return "".join(text_parts), spans


def min_logprob_in_span(
    spans: Sequence[tuple[int, int, float]], start: int, end: int
) -> float | None:
    """MIN logprob among tokens whose char range overlaps `[start, end)`."""
    covering = [logprob for t_start, t_end, logprob in spans if t_start < end and t_end > start]
    return min(covering) if covering else None


def _numeric_candidates(value: int | float | Decimal) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    _add(json.dumps(value) if not isinstance(value, Decimal) else str(value))
    try:
        dec = Decimal(str(value))
    except InvalidOperation:
        return candidates
    _add(format(dec, "f"))
    _add(format(dec.normalize(), "f"))
    if dec == dec.to_integral_value():
        _add(str(int(dec)))
    _add(f"{dec:.2f}")
    return candidates


def _value_search_candidates(value: Any) -> list[str]:
    """JSON-literal text forms to search for a scalar field's value.

    Only string/number/bool scalars are handled; containers (dict/list) and
    `None` return no candidates, which callers read as "unresolved"."""
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, str):
        return [json.dumps(value, ensure_ascii=False)]
    if isinstance(value, (int, float, Decimal)):
        return _numeric_candidates(value)
    return []


def find_value_span(text: str, key: str, value: Any) -> tuple[int, int] | None:
    """Locate a scalar field's JSON-literal value substring within `text`.

    Tries the value alongside its own JSON key first (`"key":value` /
    `"key": value`), then falls back to a bare, key-independent search of the
    value alone. Returns the FIRST match's `(start, end)`, or `None` if no
    candidate representation is found anywhere in `text`.
    """
    candidates = _value_search_candidates(value)
    if not candidates:
        return None
    key_literal = json.dumps(key)
    for candidate in candidates:
        for joiner in (f"{key_literal}:{candidate}", f"{key_literal}: {candidate}"):
            idx = text.find(joiner)
            if idx != -1:
                start = idx + len(joiner) - len(candidate)
                return start, start + len(candidate)
    for candidate in candidates:
        idx = text.find(candidate)
        if idx != -1:
            return idx, idx + len(candidate)
    return None


def extract_logprobs_content(
    raw_response: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Pull `choices[0].logprobs.content` out of a chat/completions response.

    Defensive against any shape mismatch (missing keys, wrong types) -- always
    `None` on anything unexpected, never a raised exception.
    """
    if not isinstance(raw_response, Mapping):
        return None
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, Mapping):
        return None
    content = logprobs.get("content")
    return content if isinstance(content, list) else None


def extract_message_content(raw_response: Mapping[str, Any] | None) -> str | None:
    """Pull the UNMUTATED `choices[0].message.content` out of a response.

    Deliberately reads straight off the API response dict, not the
    prefill-restored / mojibake-fixed string `openai_client.chat_json` derives
    from it -- token offsets from `logprobs.content[]` only line up with what
    the server actually returned.
    """
    if not isinstance(raw_response, Mapping):
        return None
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def compute_field_confidences(
    flat_result: Mapping[str, Any],
    raw_response: Mapping[str, Any] | None,
) -> dict[str, float | None]:
    """Per-TOP-LEVEL-field MIN-aggregated logprob confidence.

    `flat_result` must be the FLAT dict as generated by the model (the
    `chat_json` return value, before NuExtract normalization or schema
    rehydration reshape it) -- its scalar values are literally what appears in
    the raw JSON text, which is what makes offset lookup possible at all.

    Every key of `flat_result` is present in the result, `None` when: there
    were no usable logprobs, the reconstructed token text didn't exactly
    reproduce the server's response content, the value is a nested
    object/list, or its literal span couldn't be found.
    """
    confidences: dict[str, float | None] = dict.fromkeys(flat_result)
    reconstructed = reconstruct_generated_text(extract_logprobs_content(raw_response))
    raw_content = extract_message_content(raw_response)
    if reconstructed is None or raw_content is None:
        return confidences
    token_text, spans = reconstructed
    if token_text != raw_content:
        # Reconstruction-fidelity guard (#335): don't compute an offset
        # against text the tokens don't exactly reproduce.
        return confidences
    for key, value in flat_result.items():
        if isinstance(value, (dict, list)):
            continue
        span = find_value_span(token_text, key, value)
        if span is None:
            continue
        confidences[key] = min_logprob_in_span(spans, span[0], span[1])
    return confidences


def _is_field_wrapper(node: Any) -> bool:
    return isinstance(node, dict) and "evidence_ids" in node and "confidence" in node


def attach_model_confidence(
    normalized: dict[str, Any], confidences: Mapping[str, float | None]
) -> dict[str, Any]:
    """Set `model_confidence` on every `{..., evidence_ids, confidence}`
    wrapper found in `normalized` (an `ExtractionResponse.result`-shaped
    dict), mutating and returning it in place.

    Top-level fields resolved by `compute_field_confidences` get that value;
    everything else -- nested wrappers inside a MoneyField-shaped object or a
    `line_items`-style list -- gets `None` (see module docstring: those value
    shapes are not resolved this round).
    """

    def _walk(node: Any, *, resolved: float | None) -> None:
        if _is_field_wrapper(node):
            node["model_confidence"] = resolved
            return
        if isinstance(node, dict):
            for value in node.values():
                _walk(value, resolved=None)
        elif isinstance(node, list):
            for item in node:
                _walk(item, resolved=None)

    for key, value in normalized.items():
        _walk(value, resolved=confidences.get(key))
    return normalized
