"""OpenAI-compatible shim server over a Hugging Face ``transformers`` model.

The LAST-RESORT serving path (see the package docstring). ``AutoProcessor`` +
``AutoModelForImageTextToText`` / ``AutoModelForCausalLM`` load a checkpoint,
its chat template and its (multimodal) processor straight from the repo, so a
model with no GGUF — or an architecture llama.cpp cannot serve — is still
servable with NO per-model family contract.

Contract (mirrors ``docie_bench.encoders.server`` so every deploy surface,
health probe and reconciler overlay is inherited unchanged):

* ``POST /v1/chat/completions`` — standard OpenAI chat body. ``messages`` may
  carry multimodal ``content`` parts (``{"type":"text",...}`` and
  ``{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}``);
  the backend decodes image parts to PIL images and feeds them through the
  processor's chat template. Honoured extras: ``max_tokens``/
  ``max_completion_tokens`` and ``temperature``. Response is a normal
  (non-streaming) chat completion.
* ``GET /v1/models`` / ``GET /healthz`` — the usual discovery/liveness pair.

``backend`` is an injection seam: tests pass any object with the same
``generate`` signature, so the server's request/response shaping is covered
without importing torch. The real backend (:class:`HfTransformersBackend`) is
lazy — a missing ``transformers``/``torch`` install fails the deploy at
startup with an actionable message, never 500s the first caller.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import time
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0


class TransformersBackend(Protocol):
    """One synchronous generation over an OpenAI ``messages`` list.

    ``messages`` is the raw OpenAI list (text or multimodal content parts);
    the backend owns the transformers-specific conversion. Runs in a worker
    thread so the event loop is never blocked by CPU generation.
    """

    def generate(
        self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float
    ) -> str: ...


# ---------------------------------------------------------------------------
# OpenAI multimodal <-> PIL helpers (pure, unit-tested).
# ---------------------------------------------------------------------------


def _decode_data_uri(url: str) -> bytes:
    """Bytes from a ``data:...;base64,<payload>`` URI. Raises ValueError otherwise."""
    if not url.startswith("data:"):
        raise ValueError(
            "transformers serving accepts only inline base64 data: image URLs "
            "(the serving node does not fetch remote image URLs)"
        )
    _, _, payload = url.partition(",")
    if not payload:
        raise ValueError("data: image URL has no base64 payload")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"data: image URL is not valid base64: {exc}") from exc


def split_prompt(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate OpenAI messages and collect image data-URIs (server-side check).

    Returns ``(messages, image_urls)`` — ``image_urls`` are the raw data: URIs
    in document order. Used by the server for EAGER validation (a bad image
    fails as a 400, not a 500 deep in generation). The messages are returned
    unchanged; the transformers-specific conversion is the backend's job
    (:func:`to_transformers_messages`). Raises ValueError on a malformed part.
    """
    image_urls: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if not isinstance(url, str) or not url:
                    raise ValueError("an image_url part is missing its 'url'")
                _decode_data_uri(url)
                image_urls.append(url)
    return messages, image_urls


def to_transformers_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert OpenAI messages to the shape a transformers chat template expects.

    OpenAI multimodal parts are ``{"type":"image_url","image_url":{"url":...}}``;
    a transformers processor's ``apply_chat_template`` instead expects an image
    PLACEHOLDER part ``{"type":"image"}`` in the message (the actual pixels are
    passed via the ``images=`` kwarg, in document order). This pure function
    does that rewrite — image parts become ``{"type":"image"}`` and their
    data-URIs are collected — so the (untestable-here) backend conversion is
    covered by a plain unit test. Text parts and string content pass through;
    the returned ``image_urls`` align positionally with the placeholders.
    """
    out: list[dict[str, Any]] = []
    image_urls: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        new_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if not isinstance(url, str) or not url:
                    raise ValueError("an image_url part is missing its 'url'")
                image_urls.append(url)
                new_parts.append({"type": "image"})
            else:
                new_parts.append(part)
        out.append({**message, "content": new_parts})
    return out, image_urls


class HfTransformersBackend:
    """AutoModel backend (lazy import — the heavyweight last-resort path).

    Loads the processor + model once at construction. Vision is auto-detected:
    a checkpoint that loads under ``AutoModelForImageTextToText`` is served
    multimodal; everything else falls back to ``AutoModelForCausalLM`` (text).
    ``trust_remote_code`` is OFF by default — a custom-code checkpoint
    (``config.json`` ``auto_map``) must opt in explicitly, since it executes
    arbitrary repo Python on the serving node.
    """

    def __init__(self, model_id: str, *, trust_remote_code: bool = False) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "the transformers serving backend requires torch + transformers "
                "(shipped in the 'ocr'/'encoders' extras): "
                "pip install 'small-doc-ie-bench[encoders]'"
            ) from exc

        import torch

        # The image-text-to-text auto class is optional: an older transformers
        # may not ship it. Resolve it defensively so TEXT serving never depends
        # on the vision class being present (a hard import would kill it too).
        image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)

        self.model_id = model_id
        self._torch = torch
        self.vision = False
        self._processor = None
        # Prefer a full processor (carries the multimodal chat template); fall
        # back to a bare tokenizer for text-only checkpoints without one.
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
        except (ValueError, OSError, KeyError):
            self._processor = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )

        # Try the image-text-to-text head first (VLMs); fall back to causal LM.
        self._model = None
        if image_text_cls is not None:
            try:
                self._model = image_text_cls.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype="auto",
                )
                self.vision = True
            except (ValueError, OSError, KeyError):
                self._model = None
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype="auto",
            )
        self._model.eval()

    def _images(self, image_urls: list[str]) -> list[Any]:
        if not image_urls:
            return []
        import io

        from PIL import Image  # transformers pulls Pillow for VLMs

        return [
            Image.open(io.BytesIO(_decode_data_uri(url))).convert("RGB")
            for url in image_urls
        ]

    def generate(
        self, messages: list[dict[str, Any]], *, max_tokens: int, temperature: float
    ) -> str:
        # Rewrite OpenAI image_url parts to transformers image PLACEHOLDERS and
        # collect the pixels for the images= kwarg (see to_transformers_messages).
        tf_messages, image_urls = to_transformers_messages(messages)
        images = self._images(image_urls)

        # apply_chat_template with tokenize=True builds the model inputs
        # (input_ids + pixel_values for VLMs) straight from the repo's own
        # template — the whole reason this path needs no per-model contract.
        inputs = self._processor.apply_chat_template(
            tf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **({"images": images} if images else {}),
        )
        inputs = inputs.to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **gen_kwargs)
        # Decode ONLY the newly generated tail (strip the prompt tokens).
        new_tokens = output[0][input_len:]
        decoder = getattr(self._processor, "batch_decode", None)
        if decoder is not None:
            text = self._processor.batch_decode(
                [new_tokens], skip_special_tokens=True
            )[0]
        else:  # pragma: no cover - all processors/tokenizers expose decode
            text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


def _openai_error(message: str, *, status_code: int, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": error_type}},
    )


def create_transformers_app(
    *,
    model_id: str,
    backend: TransformersBackend | None = None,
    trust_remote_code: bool = False,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    default_temperature: float = DEFAULT_TEMPERATURE,
) -> FastAPI:
    """Build the transformers shim app. ``backend=None`` loads one at startup."""
    app = FastAPI(
        title="docie transformers",
        summary="Hugging Face transformers model behind the OpenAI chat surface "
        "(last-resort serving; unquantized weights ~2-3x a GGUF's RAM).",
    )
    app.state.backend = backend
    app.state.model_id = model_id
    app.state.trust_remote_code = trust_remote_code

    @app.on_event("startup")
    def load_model() -> None:
        # Load at startup (not first request) so missing weights/extra fail the
        # deploy immediately instead of 500ing the first caller. The multi-GB
        # download happens here — the deploy's readiness window is sized for it.
        if app.state.backend is None:
            app.state.backend = HfTransformersBackend(
                model_id, trust_remote_code=trust_remote_code
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "model": app.state.model_id, "kind": "transformers"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "docie-transformers",
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
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return _openai_error(
                "a chat request needs a non-empty 'messages' array",
                status_code=400,
                error_type="invalid_request_error",
            )
        try:
            split_prompt(messages)  # eager image validation -> 400 not 500
        except ValueError as exc:
            return _openai_error(
                str(exc), status_code=400, error_type="invalid_request_error"
            )

        # OpenAI names the cap max_tokens (legacy) or max_completion_tokens.
        max_tokens_raw = body.get("max_completion_tokens", body.get("max_tokens"))
        try:
            max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else default_max_tokens
            temperature = float(body.get("temperature", default_temperature))
        except (TypeError, ValueError):
            return _openai_error(
                "'max_tokens' must be an integer and 'temperature' a number",
                status_code=400,
                error_type="invalid_request_error",
            )
        if max_tokens < 1:
            return _openai_error(
                "'max_tokens' must be positive",
                status_code=400,
                error_type="invalid_request_error",
            )

        backend_impl: TransformersBackend = app.state.backend
        try:
            content = await asyncio.to_thread(
                backend_impl.generate,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except ValueError as exc:
            return _openai_error(
                str(exc), status_code=400, error_type="invalid_request_error"
            )

        return JSONResponse(
            {
                "id": "chatcmpl-transformers",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": app.state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )

    return app
