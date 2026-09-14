"""OpenAI-compatible shim server for encoder (token-classification) models.

Contract (the "encoder convention" the security proxy's guard analyzer and any
external platform rely on):

* ``POST /v1/chat/completions`` — the text to analyze is the LAST user
  message's text content. Optional extra body fields (ignored by OpenAI SDKs,
  honoured here): ``labels`` (list of entity labels to detect — zero-shot
  models take them verbatim) and ``threshold`` (min confidence). The response
  is a normal chat completion whose assistant content is a JSON object::

      {"entities": [{"type": "email", "value": "a@b.fr",
                     "start": 8, "end": 14, "score": 0.97}, ...]}

  ``start``/``end`` are character offsets into the analyzed text. The same
  payload is mirrored under the top-level ``docie_encoder`` key so callers can
  skip content parsing.
* ``GET /v1/models`` / ``GET /healthz`` — the usual discovery/liveness pair.

The default backend is GLiNER (``pip install .[encoders]``; default model
``urchade/gliner_multi_pii-v1``) — zero-shot, so one served encoder covers
PII, IP/confidentiality terms, or any label set the caller sends. ``backend``
is an injection seam: tests (and future encoder integrations) pass any object
with the same ``predict`` signature.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from docie_bench.extract.postprocess import dedupe_lists
from docie_bench.openai_protocol import openai_error
from docie_bench.schemas.dynamic import (
    document_from_gliformer_records,
    gliformer_model_from_json_schema,
    gliformer_records_plan,
)

logger = logging.getLogger("docie_bench.encoders.server")

DEFAULT_ENCODER_MODEL = "urchade/gliner_multi_pii-v1"
DEFAULT_GLIFORMER_MODEL = "knowledgator/gliformer-base-v1"

# A practical default label set for the PII/confidentiality use case; callers
# override per request via `labels`.
DEFAULT_LABELS = [
    "person",
    "organization",
    "email",
    "phone number",
    "address",
    "credit card number",
    "iban",
    "passport number",
    "social security number",
    "date of birth",
    "ip address",
]
DEFAULT_THRESHOLD = 0.5


class EncoderBackend(Protocol):
    """One synchronous prediction over one text (runs in a worker thread)."""

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]: ...


class GlinerBackend:
    """GLiNER zero-shot NER backend (lazy import — optional dependency)."""

    def __init__(self, model_id: str = DEFAULT_ENCODER_MODEL) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the GLiNER encoder backend requires the 'encoders' extra: "
                "pip install 'small-doc-ie-bench[encoders]'"
            ) from exc
        self.model_id = model_id
        self._model = GLiNER.from_pretrained(model_id)

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        raw = self._model.predict_entities(text, labels, threshold=threshold)
        return [
            {
                "type": str(entity["label"]),
                "value": str(entity["text"]),
                "start": int(entity["start"]),
                "end": int(entity["end"]),
                "score": float(entity.get("score", 0.0)),
            }
            for entity in raw
        ]


# A document longer than the encoder's input window has to be structured in
# pieces. The window is read off the model rather than guessed; this is only the
# fallback for a checkpoint that declares none.
# Only for a checkpoint that declares nothing. Conservative on purpose: chunking
# a document that would have fitted costs time, reading past the real window
# truncates it silently.
DEFAULT_STRUCTURE_WINDOW_TOKENS = 512
# Lines repeated at the seam, so a record split across a boundary is seen whole
# by at least one chunk.
STRUCTURE_OVERLAP_LINES = 2


def model_input_window(model: Any) -> int | None:
    """The model's own input limit, when it declares one.

    Order matters, and the obvious field is the wrong one.
    ``max_position_embeddings`` describes the BACKBONE's absolute position
    table; GLiFormer's DeBERTa reports 512 there while the checkpoint declares
    ``max_len`` of 8192 (large) and 16384 (base), because DeBERTa's positions
    are relative and the encoder is not bound by that table.

    Reading it first chunked a document into sixteen to thirty-two times more
    pieces than needed: every extra seam is another inference call and another
    place a record can be cut in half. So ``max_len`` is asked first, the
    tokenizer's own limit second, and the backbone's table only as a last
    resort for a checkpoint that says nothing else.
    """
    config = getattr(model, "config", None)
    candidates = (
        (config, "max_len"),
        (model, "max_len"),
        (getattr(model, "tokenizer", None), "model_max_length"),
        (config, "max_position_embeddings"),
    )
    for owner, attribute in candidates:
        value = getattr(owner, attribute, None)
        # transformers writes a sentinel in the billions when a tokenizer
        # declares no limit; that is "unknown", not "unbounded".
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000:
            return value
    return None


# The prompt marks page boundaries as "[page N]" when a document spans several
# pages (see llm/prompts.render_ocr_blocks), so the structure the OCR found
# survives into the text this server receives.
_PAGE_MARKER = re.compile(r"^\[page \d+\]$")


def split_into_pages(text: str) -> list[str]:
    """The document's pages, marker included, or one piece when unmarked."""
    pages: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _PAGE_MARKER.match(line.strip()) and current:
            pages.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        pages.append("\n".join(current))
    return pages


def _split_oversized_page(
    page: str, count_tokens: Callable[[str], int], budget: int
) -> list[str]:
    """Cut one page that cannot be kept whole, repeating its marker.

    Every fragment carries the "[page N]" line, so a piece taken from the
    middle of a page still tells the model where it came from instead of
    arriving as unplaced text.
    """
    lines = page.splitlines()
    marker = lines[0] if lines and _PAGE_MARKER.match(lines[0].strip()) else None
    if marker is None:
        return _split_lines_for_window(page, count_tokens, budget)
    # The marker is prepended to every fragment, so its own cost comes out
    # of the budget first: the ceiling is the invariant, and a fragment that
    # fitted only before the marker was added would break it.
    body = "\n".join(lines[1:])
    room = budget - count_tokens(marker)
    if room <= 0:
        return _split_lines_for_window(page, count_tokens, budget)
    fragments = _split_lines_for_window(body, count_tokens, room)
    return [marker + "\n" + fragment for fragment in fragments]


def pack_pages(
    pages: list[str],
    count_tokens: Callable[[str], int],
    budget: int,
) -> list[str]:
    """Group whole pages into pieces under ``budget``.

    A page is the unit because it is the one the document itself declares: a
    record rarely straddles a page break, while an arbitrary line cut lands
    mid-record by construction. Grouping greedily is also what handles a sparse
    page — a cover sheet, a page holding one table — since it simply joins the
    pages after it until the budget is reached, rather than spending a whole
    inference call on four lines.

    A single page over budget is the one case a page cannot be kept whole; it
    falls through to the line splitter, and only that page does.
    """
    pieces: list[str] = []
    current: list[str] = []
    for page in pages:
        if count_tokens(page) > budget:
            if current:
                pieces.append("\n".join(current))
                current = []
            pieces.extend(_split_oversized_page(page, count_tokens, budget))
            continue
        candidate = [*current, page]
        if current and count_tokens("\n".join(candidate)) > budget:
            pieces.append("\n".join(current))
            current = [page]
            continue
        current = candidate
    if current:
        pieces.append("\n".join(current))
    return pieces


def split_for_window(
    text: str,
    count_tokens: Callable[[str], int],
    budget: int,
    *,
    overlap_lines: int = STRUCTURE_OVERLAP_LINES,
) -> list[str]:
    """Divide a document into pieces the model can read, page by page.

    A document that fits goes through whole. Past the window it is divided on
    the boundaries the document itself declares — its pages — and whole pages
    are grouped up to the budget, so a sparse page rides along with the next one
    instead of costing its own inference call. Only a page that is itself too
    large is cut on line boundaries, and only that page.

    Before this, every document past the window was cut on an arbitrary line,
    which lands mid-record by construction: the text either side of a seam is
    what the overlap exists to repair, and not cutting there at all is better
    than repairing it.
    """
    if budget <= 0 or count_tokens(text) <= budget:
        return [text]
    pages = split_into_pages(text)
    if len(pages) > 1:
        return [piece for piece in pack_pages(pages, count_tokens, budget) if piece.strip()]
    return _split_lines_for_window(text, count_tokens, budget, overlap_lines=overlap_lines)


def _split_lines_for_window(
    text: str,
    count_tokens: Callable[[str], int],
    budget: int,
    *,
    overlap_lines: int = STRUCTURE_OVERLAP_LINES,
) -> list[str]:
    """Split one page on line boundaries so no piece exceeds ``budget`` tokens.

    The last resort, for a page too large to keep whole. Lines are the unit
    because OCR emits them.
    A few lines are repeated at each seam so a record written across a boundary
    is seen whole by one side, and the overlap is shortened when carrying it
    would push the next piece over budget: the ceiling is the invariant, the
    overlap is best effort. A single line over budget is split on whitespace
    rather than dropped.
    """
    if budget <= 0 or count_tokens(text) <= budget:
        return [text]

    def fits(lines: list[str]) -> bool:
        return count_tokens("\n".join(lines)) <= budget

    pieces: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if count_tokens(line) > budget:
            if current:
                pieces.append("\n".join(current))
                current = []
            pieces.extend(_split_long_line(line, count_tokens, budget))
            continue
        if not current or fits([*current, line]):
            current.append(line)
            continue
        pieces.append("\n".join(current))
        carried = current[-overlap_lines:] if overlap_lines else []
        while carried and not fits([*carried, line]):
            carried = carried[1:]
        current = [*carried, line]
    if current:
        pieces.append("\n".join(current))
    return [piece for piece in pieces if piece.strip()] or [text]


def _split_long_line(line: str, count_tokens: Callable[[str], int], budget: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    for word in line.split():
        candidate = [*current, word]
        if current and count_tokens(" ".join(candidate)) > budget:
            pieces.append(" ".join(current))
            current = [word]
            continue
        current = candidate
    if current:
        pieces.append(" ".join(current))
    return pieces


def merge_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold the records read from each chunk into the one the document describes.

    A list field concatenates, because chunk two holds experiences chunk one
    never saw. A scalar takes the first chunk that answered, because the same
    name repeated on every page is one name, not four.

    Exact duplicates are dropped afterwards by ``dedupe_lists``, which the
    extraction pipeline already uses for this.
    """
    merged: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if value is None or value == "" or value == []:
                continue
            existing = merged.get(key)
            if isinstance(value, list):
                merged[key] = [*(existing if isinstance(existing, list) else []), *value]
            elif key not in merged:
                merged[key] = value
    return cast(dict[str, Any], dedupe_lists(merged))


class GliformerBackend:
    """GLiFormer backend: zero-shot NER plus schema-driven structuring.

    ``knowledgator/gliformer-base-v1`` and ``-large-v1`` are one DeBERTa encoder
    under several task heads. ``predict_entities`` carries the same signature and
    result keys as GLiNER's, so the analyzer surface is unchanged; ``structure``
    is the extra head this family brings (see :meth:`structure`).
    """

    # A class attribute so an instance built without ``__init__`` still has a
    # window: ``structure`` must never fail on a missing attribute.
    input_window: int = DEFAULT_STRUCTURE_WINDOW_TOKENS

    def __init__(self, model_id: str = DEFAULT_GLIFORMER_MODEL) -> None:
        try:
            from gliformer import GLiFormer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the GLiFormer encoder backend requires the 'encoders' extra: "
                "pip install 'small-doc-ie-bench[encoders]'"
            ) from exc
        self.model_id = model_id
        self._model = GLiFormer.from_pretrained(model_id, load_tokenizer=True)
        self.input_window = model_input_window(self._model) or DEFAULT_STRUCTURE_WINDOW_TOKENS

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        raw = self._model.predict_entities(text, labels, threshold=threshold)
        return [
            {
                "type": str(entity["label"]),
                "value": str(entity["text"]),
                "start": int(entity["start"]),
                "end": int(entity["end"]),
                "score": float(entity.get("score", 0.0)),
            }
            for entity in raw
        ]

    def structure(self, text: str, schema: Any, *, validate_output: bool = True) -> Any:
        """Extract records for ``schema``.

        ``schema`` is whatever GLiFormer accepts: ``{"employee": ["name",
        "company"]}`` for the plain field-list form, or ``{"company": Company}``
        with a Pydantic model for nested records.
        :class:`~docie_bench.schemas.dynamic.DynamicTemplateBuilder` builds
        either one from a saved schema.

        A document past the encoder's input window is structured in pieces and
        the pieces folded back into one record: without this a three-page CV
        returns only what fits in the first window, with no error to say the
        rest was never read.
        """
        chunks = split_for_window(text, self.count_tokens, self.input_window)
        if len(chunks) == 1:
            return self._model.structure(text, schema, validate_output=validate_output)
        found: dict[str, list[dict[str, Any]]] = {key: [] for key in schema}
        for chunk in chunks:
            answer = self._model.structure(chunk, schema, validate_output=validate_output)
            if not isinstance(answer, dict):
                continue
            for key, rows in found.items():
                chunk_rows = answer.get(key)
                if isinstance(chunk_rows, dict):
                    chunk_rows = [chunk_rows]
                rows.extend(row for row in (chunk_rows or []) if isinstance(row, dict))
        if not all(isinstance(value, list) for value in schema.values()):
            # The nested form: one model per key, and the document is one
            # record of it, so the chunks fold into that record.
            key = next(iter(found))
            return {key: [merge_records(found[key])]}
        # The plain record form: every key is a record type the caller folds by
        # what the schema says it is, so each keeps its own rows.
        return found

    def count_tokens(self, text: str) -> int:
        """Length in the model's own tokens, or a character estimate without one."""
        tokenizer = getattr(self._model, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        if encode is not None:
            try:
                return len(encode(text))
            except Exception as exc:  # noqa: BLE001 - never fail a request on this
                logger.debug("tokenizer_count_failed", extra={"docie_error": str(exc)})
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Moderation task presets (GLiNER2 guardrails — e.g.
# fastino/GLiNER2-Guardrails-PII-Multi). Callers request tasks BY NAME; the
# label schemas live here so every client shares one calibrated config.
# ---------------------------------------------------------------------------

_SAFETY_LABELS = ["safe", "unsafe"]
_REFUSAL_LABELS = ["refusal", "compliance"]
_TOXICITY_LABELS = [
    "violence_and_weapons",
    "non_violent_crime",
    "sexual_content",
    "hate_and_discrimination",
    "self_harm_and_suicide",
    "pii_exposure",
    "misinformation",
    "copyright_violation",
    "child_safety",
    "political_manipulation",
    "unethical_conduct",
    "regulated_advice",
    "privacy_violation",
    "other",
    "benign",
]
_JAILBREAK_LABELS = [
    "prompt_injection",
    "jailbreak_attempt",
    "policy_evasion",
    "instruction_override",
    "system_prompt_exfiltration",
    "data_exfiltration",
    "roleplay_bypass",
    "hypothetical_bypass",
    "obfuscated_attack",
    "multi_step_attack",
    "social_engineering",
    "benign",
]

MODERATION_TASKS: dict[str, Any] = {
    "prompt_safety": _SAFETY_LABELS,
    "response_safety": _SAFETY_LABELS,
    "response_refusal": _REFUSAL_LABELS,
    "prompt_toxicity": {"labels": _TOXICITY_LABELS, "multi_label": True, "cls_threshold": 0.4},
    "response_toxicity": {"labels": _TOXICITY_LABELS, "multi_label": True, "cls_threshold": 0.4},
    "jailbreak_detection": {"labels": _JAILBREAK_LABELS, "multi_label": True, "cls_threshold": 0.4},
}


class Gliner2Backend:
    """GLiNER2 schema-conditioned backend: PII extraction + safety moderation.

    One checkpoint, two heads (e.g. ``fastino/GLiNER2-Guardrails-PII-Multi``):
    ``predict`` normalizes ``extract_entities`` into the shim's entity shape,
    ``classify`` runs the guardrail tasks. Spans are validated downstream by
    the guard analyzer either way, so a value-only return (no offsets) still
    works — it is relocated against the text there.
    """

    def __init__(self, model_id: str) -> None:
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the GLiNER2 encoder backend requires the gliner2 package "
                "(shipped in the 'encoders' extra): pip install 'small-doc-ie-bench[encoders]'"
            ) from exc
        self.model_id = model_id
        self._model = GLiNER2.from_pretrained(model_id)

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        raw = self._model.extract_entities(
            text, labels, threshold=threshold, include_spans=True, include_confidence=True
        )
        entities: list[dict[str, Any]] = []
        for label, values in (raw.get("entities") or {}).items():
            for value in values or []:
                if isinstance(value, dict):
                    text_value = str(value.get("text") or value.get("value") or "")
                    start = value.get("start")
                    end = value.get("end")
                    score = value.get("confidence", value.get("score", 0.0))
                else:
                    text_value = str(value)
                    start = end = None
                    score = 0.0
                if not text_value:
                    continue
                if not isinstance(start, int) or not isinstance(end, int):
                    start = text.find(text_value)
                    if start < 0:
                        continue
                    end = start + len(text_value)
                entities.append(
                    {
                        "type": str(label),
                        "value": text_value,
                        "start": start,
                        "end": end,
                        "score": float(score or 0.0),
                    }
                )
        return entities

    def classify(self, text: str, tasks: dict[str, Any]) -> dict[str, Any]:
        return dict(self._model.classify_text(text, tasks, threshold=0.5))


def _one_record(records: Any, record_name: str) -> dict[str, Any]:
    """GLiFormer returns a LIST of records per schema key; an extraction of one
    document wants one object. The first record is the document."""
    if isinstance(records, dict):
        rows = records.get(record_name)
        if isinstance(rows, list):
            first = rows[0] if rows else {}
            return first if isinstance(first, dict) else {}
        if isinstance(rows, dict):
            return rows
        return dict(records)
    return {}


def _completion(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-encoder",
        "object": "chat.completion",
        "created": 0,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_backend(model_id: str, kind: str = "auto") -> Any:
    """Instantiate the right backend for ``model_id``.

    ``auto`` keys on the model id, case-insensitive, so a deploy needs no extra
    plumbing — ``fastino/GLiNER2-…`` and ``knowledgator/gliformer-…`` just work.
    ``gliformer`` is tested first: it is its own library, and neither substring
    is a prefix of the other, but ordering makes that independent of how the
    names read.
    """
    normalized = kind.strip().lower()
    if normalized == "auto":
        lowered = model_id.lower()
        if "gliformer" in lowered:
            normalized = "gliformer"
        elif "gliner2" in lowered:
            normalized = "gliner2"
        else:
            normalized = "gliner"
    if normalized == "gliformer":
        return GliformerBackend(model_id)
    if normalized == "gliner2":
        return Gliner2Backend(model_id)
    if normalized == "gliner":
        return GlinerBackend(model_id)
    raise ValueError(
        f"unknown encoder backend {kind!r} (expected auto, gliner, gliner2, or gliformer)"
    )


def _last_user_text(messages: list[Any]) -> str | None:
    """The last user message's text (joining text parts of a multimodal list)."""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(parts)
    return None


def create_encoder_app(
    *,
    model_id: str = DEFAULT_ENCODER_MODEL,
    backend: EncoderBackend | None = None,
    backend_kind: str = "auto",
    default_labels: list[str] | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> FastAPI:
    """Build the encoder shim app. ``backend=None`` loads one at startup
    (``backend_kind="auto"`` picks GLiNER2 for GLiNER2 model ids)."""
    labels_default = list(default_labels or DEFAULT_LABELS)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load at startup (not first request) so a missing extra/weights fails
        # the deploy immediately instead of 500ing the first caller.
        if app.state.backend is None:
            app.state.backend = build_backend(model_id, backend_kind)
        yield

    app = FastAPI(
        title="docie encoder",
        summary="Encoder (token-classification) model behind the OpenAI chat surface.",
        lifespan=lifespan,
    )
    app.state.backend = backend
    app.state.model_id = model_id

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "model": app.state.model_id, "kind": "encoder"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "docie-encoders",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            return openai_error(
                "request body must be valid JSON",
                status_code=400,
                error_type="invalid_request_error",
            )
        if not isinstance(body, dict):
            return openai_error(
                "request body must be a JSON object",
                status_code=400,
                error_type="invalid_request_error",
            )
        text = _last_user_text(body.get("messages") or [])
        if text is None:
            return openai_error(
                "an encoder request needs at least one user message with text content",
                status_code=400,
                error_type="invalid_request_error",
            )

        # A schema on the request turns this into a structured extraction: the
        # rest of the platform already sends one as
        # response_format.json_schema, so a served GLiFormer is reached through
        # the ordinary extraction path with no channel of its own.
        schema_spec = (
            (body.get("response_format") or {})
            if isinstance(body.get("response_format"), dict)
            else {}
        )
        json_schema = schema_spec.get("json_schema") or {}
        schema_root = json_schema.get("schema") if isinstance(json_schema, dict) else None
        if isinstance(schema_root, dict) and schema_root:
            structure = getattr(app.state.backend, "structure", None)
            if structure is None:
                return openai_error(
                    "this encoder backend has no structuring head — serve a "
                    "GLiFormer checkpoint (e.g. knowledgator/gliformer-base-v1) "
                    "to extract records from a schema",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            record_name = str(json_schema.get("name") or "record")
            # "records" asks for the plain {record: [field, ...]} form the model
            # card documents first; the default builds the nested Pydantic one.
            # Sent as an extra body field by the gliformer_records style.
            wants_records = str(body.get("structure_mode") or "").strip().lower() == "records"
            plan = None
            try:
                if wants_records:
                    plan = gliformer_records_plan(schema_root, name=record_name)
                    schema_for_model: Any = plan.fields
                else:
                    schema_for_model = {
                        record_name: gliformer_model_from_json_schema(
                            schema_root, name=record_name
                        )
                    }
            except ValueError as exc:
                return openai_error(
                    f"schema cannot be structured: {exc}",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            records = await asyncio.to_thread(structure, text, schema_for_model)
            document = (
                document_from_gliformer_records(records, plan, name=record_name)
                if plan is not None
                else _one_record(records, record_name)
            )
            return JSONResponse(_completion(app.state.model_id, document))

        labels_raw = body.get("labels")
        labels = (
            [str(label) for label in labels_raw]
            if isinstance(labels_raw, list) and labels_raw
            else labels_default
        )
        try:
            threshold = float(body.get("threshold", default_threshold))
        except (TypeError, ValueError):
            return openai_error(
                "'threshold' must be a number",
                status_code=400,
                error_type="invalid_request_error",
            )

        backend_impl: EncoderBackend = app.state.backend
        entities = await asyncio.to_thread(backend_impl.predict, text, labels, threshold)

        # Optional guardrail tasks (GLiNER2 checkpoints): `tasks` is a list of
        # preset names (see MODERATION_TASKS) or a raw schema dict passed
        # through verbatim.
        moderation: dict[str, Any] | None = None
        tasks_raw = body.get("tasks")
        if tasks_raw:
            if isinstance(tasks_raw, list):
                unknown = [t for t in tasks_raw if t not in MODERATION_TASKS]
                if unknown:
                    return openai_error(
                        f"unknown moderation task(s): {', '.join(map(str, unknown))} "
                        f"(available: {', '.join(sorted(MODERATION_TASKS))})",
                        status_code=400,
                        error_type="invalid_request_error",
                    )
                schema = {str(t): MODERATION_TASKS[str(t)] for t in tasks_raw}
            elif isinstance(tasks_raw, dict):
                schema = tasks_raw
            else:
                return openai_error(
                    "'tasks' must be a list of preset names or a schema object",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            classify = getattr(backend_impl, "classify", None)
            if classify is None:
                return openai_error(
                    "this encoder backend has no moderation head — serve a "
                    "GLiNER2 guardrails checkpoint (e.g. "
                    "fastino/GLiNER2-Guardrails-PII-Multi) for 'tasks'",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            moderation = await asyncio.to_thread(classify, text, schema)

        payload: dict[str, Any] = {"entities": entities}
        if moderation is not None:
            payload["moderation"] = moderation
        return JSONResponse(
            {
                "id": "chatcmpl-encoder",
                "object": "chat.completion",
                "created": 0,
                "model": app.state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "docie_encoder": payload,
            }
        )

    return app
