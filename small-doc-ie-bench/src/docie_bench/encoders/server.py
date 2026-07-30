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
import json
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DEFAULT_ENCODER_MODEL = "urchade/gliner_multi_pii-v1"

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

    def predict(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]: ...


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

    def predict(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Moderation task presets (GLiNER2 guardrails — e.g.
# fastino/GLiNER2-Guardrails-PII-Multi). Callers request tasks BY NAME; the
# label schemas live here so every client shares one calibrated config.
# ---------------------------------------------------------------------------

_SAFETY_LABELS = ["safe", "unsafe"]
_REFUSAL_LABELS = ["refusal", "compliance"]
_TOXICITY_LABELS = [
    "violence_and_weapons", "non_violent_crime", "sexual_content",
    "hate_and_discrimination", "self_harm_and_suicide", "pii_exposure",
    "misinformation", "copyright_violation", "child_safety",
    "political_manipulation", "unethical_conduct", "regulated_advice",
    "privacy_violation", "other", "benign",
]
_JAILBREAK_LABELS = [
    "prompt_injection", "jailbreak_attempt", "policy_evasion",
    "instruction_override", "system_prompt_exfiltration", "data_exfiltration",
    "roleplay_bypass", "hypothetical_bypass", "obfuscated_attack",
    "multi_step_attack", "social_engineering", "benign",
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

    def predict(
        self, text: str, labels: list[str], threshold: float
    ) -> list[dict[str, Any]]:
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


def build_backend(model_id: str, kind: str = "auto") -> Any:
    """Instantiate the right backend for ``model_id``.

    ``auto`` keys on the model id ("gliner2" anywhere, case-insensitive) so a
    deploy needs no extra plumbing — ``fastino/GLiNER2-…`` just works.
    """
    normalized = kind.strip().lower()
    if normalized == "auto":
        normalized = "gliner2" if "gliner2" in model_id.lower() else "gliner"
    if normalized == "gliner2":
        return Gliner2Backend(model_id)
    if normalized == "gliner":
        return GlinerBackend(model_id)
    raise ValueError(f"unknown encoder backend {kind!r} (expected auto, gliner, or gliner2)")


def _openai_error(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
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

    app = FastAPI(
        title="docie encoder",
        summary="Encoder (token-classification) model behind the OpenAI chat surface.",
    )
    app.state.backend = backend
    app.state.model_id = model_id

    @app.on_event("startup")
    def load_model() -> None:
        # Load at startup (not first request) so a missing extra/weights fails
        # the deploy immediately instead of 500ing the first caller.
        if app.state.backend is None:
            app.state.backend = build_backend(model_id, backend_kind)

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
            return _openai_error(
                "request body must be valid JSON",
                status_code=400,
                error_type="invalid_request_error",
            )
        if not isinstance(body, dict):
            return _openai_error(
                "request body must be a JSON object",
                status_code=400,
                error_type="invalid_request_error",
            )
        text = _last_user_text(body.get("messages") or [])
        if text is None:
            return _openai_error(
                "an encoder request needs at least one user message with text content",
                status_code=400,
                error_type="invalid_request_error",
            )

        labels_raw = body.get("labels")
        labels = (
            [str(label) for label in labels_raw]
            if isinstance(labels_raw, list) and labels_raw
            else labels_default
        )
        try:
            threshold = float(body.get("threshold", default_threshold))
        except (TypeError, ValueError):
            return _openai_error(
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
                    return _openai_error(
                        f"unknown moderation task(s): {', '.join(map(str, unknown))} "
                        f"(available: {', '.join(sorted(MODERATION_TASKS))})",
                        status_code=400,
                        error_type="invalid_request_error",
                    )
                schema = {str(t): MODERATION_TASKS[str(t)] for t in tasks_raw}
            elif isinstance(tasks_raw, dict):
                schema = tasks_raw
            else:
                return _openai_error(
                    "'tasks' must be a list of preset names or a schema object",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            classify = getattr(backend_impl, "classify", None)
            if classify is None:
                return _openai_error(
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
