from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

from docie_bench.extract.grounding import ground_evidence
from docie_bench.extract.logprob_confidence import (
    attach_model_confidence,
    compute_field_confidences,
)
from docie_bench.extract.validators import validate_extraction
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.mojibake import fix_mojibake
from docie_bench.llm.openai_client import OpenAICompatibleClient
from docie_bench.llm.prompts import (
    OCR_TRANSCRIPTION_SYSTEM_PROMPT,
    OCR_TRANSCRIPTION_USER_PROMPT,
    SCHEMA_PROPOSER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    VISION_SYSTEM_PROMPT,
    build_nuextract3_prompts,
    build_nuextract_prompts,
    build_schema_proposer_prompt,
    build_user_prompt,
    build_vision_user_prompt,
)
from docie_bench.ocr.base import text_to_blocks
from docie_bench.ocr.service import processor_from_settings
from docie_bench.schemas.common import ExtractionResponse, OCRBlock, Usage
from docie_bench.schemas.dynamic import DynamicSchemaSpec, DynamicTemplateBuilder
from docie_bench.schemas.extraction import (
    flatten_schema_json,
    rehydrate_extraction_result,
    schema_json,
)
from docie_bench.security import redact_fields
from docie_bench.serving.solutions import _PIPELINE_UPSTREAM_TIMEOUT_S
from docie_bench.settings import get_settings
from docie_bench.vision import DocumentImage, load_document_images

logger = logging.getLogger(__name__)


_CURRENCY_MAP = {"€": "EUR", "£": "GBP", "$": "USD", "¥": "JPY", "₣": "CHF"}
_DATE_FIELD_NAMES = {"issue_date", "due_date", "birth_date", "expiry_date"}
_DECIMAL_FIELD_NAMES = {"vat_rate", "quantity", "tax_rate"}


_COUNTRY_ISO: dict[str, str] = {
    "france": "FRA", "française": "FRA", "francaise": "FRA",
    "germany": "DEU", "deutschland": "DEU", "allemagne": "DEU",
    "spain": "ESP", "espagne": "ESP", "españa": "ESP",
    "united kingdom": "GBR", "uk": "GBR",
    "united states": "USA", "usa": "USA",
    "italy": "ITA", "italie": "ITA",
}


def _norm_amount(raw: str) -> str:
    """Normalize a locale-formatted amount when a model ignores the type hint."""
    s = re.sub(r"[€£$¥₣a-zA-Z]", "", raw).strip()
    if "," in s and "." in s:
        # Ambiguous: detect thousands vs decimal by position
        comma_pos = s.rfind(",")
        dot_pos = s.rfind(".")
        s = s.replace(",", "") if dot_pos > comma_pos else s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(" ", "").replace(",", ".")
    else:
        s = s.replace(" ", "")
    return s


def _norm_date(raw: str) -> str:
    """Fallback date normalisation for formats the model ignores the 'date' type hint on."""
    s = raw.strip()
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY (European numeric)
    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", s)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    # "28 Feb 2026" / "28 February 2026" (written English month)
    try:
        from dateutil import parser as _dp
        dt = _dp.parse(s, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s


def _derive_invoice_subtotal(result: dict[str, Any]) -> bool:
    """Fill a missing subtotal from total_ttc - vat_amount when both are
    present. Returns whether it actually derived a value, so the caller can
    flag it as computed rather than extracted (see the derived-subtotal
    warning in _extract_blocks) -- without that, a synthesized number is
    indistinguishable from one the model actually read off the page."""
    if result.get("subtotal") is not None:
        return False

    total = result.get("total_ttc")
    vat = result.get("vat_amount")
    if not isinstance(total, dict) or not isinstance(vat, dict):
        return False

    try:
        subtotal = Decimal(str(total["amount"])) - Decimal(str(vat["amount"]))
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return False
    if not subtotal.is_finite() or subtotal < 0:
        return False

    total_currency = total.get("currency")
    vat_currency = vat.get("currency")
    if total_currency and vat_currency and total_currency != vat_currency:
        return False

    result["subtotal"] = {
        "amount": format(subtotal, "f"),
        "currency": total_currency or vat_currency,
    }
    return True


def _normalize_nuextract_raw(raw: dict[str, Any], schema_name: str) -> tuple[dict[str, Any], bool]:
    """Post-process NuExtract3 output: enforce document_type, strip IBAN spaces,
    null-out empty MoneyFields, and apply fallback normalization for any values the
    model returned in locale format despite type hints. Derive a missing invoice
    subtotal when total and VAT provide an unambiguous fallback.

    Returns (result, derived_subtotal) -- the caller turns a True flag into a
    validation warning (see _extract_blocks)."""
    result: dict[str, Any] = {"document_type": schema_name}
    for key, val in raw.items():
        if key == "document_type":
            continue  # already set above
        if isinstance(val, list):
            result[key] = [_normalize_nested_nuextract(item) for item in val]
            continue
        if not isinstance(val, dict):
            result[key] = val
            continue
        sub = dict(val)

        # MoneyField
        if "amount" in sub:
            amt = sub.get("amount")
            if amt is None or amt == "":
                result[key] = None
                continue
            if isinstance(amt, str):
                sub["amount"] = _norm_amount(amt)
            if "currency" in sub and isinstance(sub.get("currency"), str):
                currency = sub["currency"].strip()
                sub["currency"] = _CURRENCY_MAP.get(currency, currency) or None

        # Date fallback
        if key in _DATE_FIELD_NAMES and isinstance(sub.get("value"), str) and sub["value"]:
            sub["value"] = _norm_date(sub["value"])

        # NumberField fallback (strip "%" etc.)
        if key in _DECIMAL_FIELD_NAMES and isinstance(sub.get("value"), str):
            sub["value"] = re.sub(r"[%\s]", "", sub["value"]).replace(",", ".")

        # IBAN spaces
        if key == "iban" and isinstance(sub.get("value"), str):
            sub["value"] = sub["value"].replace(" ", "")

        # country: normalize full country name → ISO-3166-1 alpha-3
        if key == "country" and isinstance(sub.get("value"), str):
            iso = _COUNTRY_ISO.get(sub["value"].lower().strip())
            if iso:
                sub["value"] = iso

        # document_number: strip leading "N° " prefix if present
        if key == "document_number" and isinstance(sub.get("value"), str):
            sub["value"] = re.sub(r"^N[°o][\s\.]*", "", sub["value"]).strip()

        # Empty-string value → null
        if sub.get("value") == "":
            result[key] = None
            continue

        result[key] = sub
    derived_subtotal = False
    if schema_name == "invoice":
        derived_subtotal = _derive_invoice_subtotal(result)
    return result, derived_subtotal


def _normalize_nested_nuextract(obj: Any, field_name: str | None = None) -> Any:
    if isinstance(obj, list):
        return [_normalize_nested_nuextract(item) for item in obj]
    if not isinstance(obj, dict):
        return obj
    normalized = {
        key: _normalize_nested_nuextract(value, key)
        for key, value in obj.items()
    }
    if isinstance(normalized.get("amount"), str):
        normalized["amount"] = _norm_amount(normalized["amount"])
    if isinstance(normalized.get("currency"), str):
        currency = normalized["currency"].strip()
        normalized["currency"] = _CURRENCY_MAP.get(currency, currency) or None
    if field_name in _DECIMAL_FIELD_NAMES and isinstance(normalized.get("value"), str):
        normalized["value"] = re.sub(r"[%\s]", "", normalized["value"]).replace(",", ".")
    if field_name in _DATE_FIELD_NAMES and isinstance(normalized.get("value"), str):
        normalized["value"] = _norm_date(normalized["value"])
    if normalized.get("value") == "":
        return None
    return normalized


def hash_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _data_urls(images: list[DocumentImage]) -> list[str]:
    """Base64-encode every page image -- synchronous, CPU-bound for a
    many-page document, so callers run this through asyncio.to_thread rather
    than call DocumentImage.data_url() directly in a loop on the event loop."""
    return [image.data_url() for image in images]


def _is_array_field(field_schema: dict[str, Any]) -> bool:
    """True for a top-level field whose flattened JSON-schema type is
    ``array`` -- a static schema's list field is a bare
    ``{"type": "array", ...}``; a dynamic schema's is Optional by
    construction (every field defaults to None) and wraps that same shape in
    ``anyOf`` alongside a ``null`` branch. Both are checked."""
    if field_schema.get("type") == "array":
        return True
    for branch in field_schema.get("anyOf", ()):
        if isinstance(branch, dict) and branch.get("type") == "array":
            return True
    return False


def _split_schema_into_groups(generation_schema: dict[str, Any]) -> list[list[str]] | None:
    """Split a flattened generation schema's top-level fields into
    independently-extractable groups (design #444): every list-typed field
    becomes its own group; every remaining field (scalars, nested objects)
    forms one shared base group. ``document_type``/``extraction_notes``
    never appear here -- flatten_schema_json already strips them, they're
    reattached during post-processing (_normalize_nuextract_raw /
    rehydrate_extraction_result), so no group needs to "own" them.

    Returns None when the result would be a single group -- nothing to
    parallelize, the caller keeps the existing one-call path unchanged.
    """
    properties = generation_schema.get("properties")
    if not isinstance(properties, dict) or len(properties) < 2:
        return None
    list_fields = [
        name
        for name, spec in properties.items()
        if isinstance(spec, dict) and _is_array_field(spec)
    ]
    other_fields = [name for name in properties if name not in list_fields]
    groups: list[list[str]] = []
    if other_fields:
        groups.append(other_fields)
    groups.extend([name] for name in list_fields)
    if len(groups) < 2:
        return None
    return groups


def _subset_schema(generation_schema: dict[str, Any], field_names: list[str]) -> dict[str, Any]:
    """Narrow a flattened generation schema down to just ``field_names`` --
    the per-group schema a split extraction's chat_json call is compiled
    against. Keeps every other top-level schema key (additionalProperties,
    type, ...) unchanged; only properties/required are subset."""
    subset = {**generation_schema, "properties": {
        name: generation_schema["properties"][name] for name in field_names
    }}
    required = generation_schema.get("required")
    if isinstance(required, list):
        subset["required"] = [name for name in required if name in field_names]
    return subset


class ExtractionService:
    def __init__(
        self,
        profile: ModelProfile,
        proposer_profile: ModelProfile | None = None,
        profiles: Mapping[str, ModelProfile] | None = None,
        disable_thinking: bool = False,
        max_tokens: int | None = None,
        # Additive, opt-in live preview (#397), threaded straight through to
        # OpenAICompatibleClient.chat_json for the extraction call ONLY (not
        # the dynamic-schema proposer call, a different JSON shape the
        # Playground buffer must never be confused with). Default None means
        # every existing caller (Benchmark, Batch, Review, the sync /v1/extract
        # routes) keeps taking chat_json's unchanged blocking path.
        on_delta: Callable[[str], None] | None = None,
        on_reset: Callable[[], None] | None = None,
    ) -> None:
        self.profile = profile
        self.proposer_profile = proposer_profile
        # Needed only for kind="pipeline" profiles, to resolve options.extractor
        # (and, later, options.ocr_model) by name -- the benchmark runner
        # already loads the full profile map for routing, so this just
        # threads it through instead of loading it again.
        self.profiles = profiles or {}
        self.disable_thinking = disable_thinking
        self.max_tokens = max_tokens
        self.on_delta = on_delta
        self.on_reset = on_reset

    async def extract_from_text(
        self,
        *,
        text: str | None,
        ocr_blocks: list[OCRBlock] | None,
        schema_name: str,
        schema_mode: str = "static",
        dynamic_schema: dict[str, Any] | DynamicSchemaSpec | None = None,
        language: str | None = None,
        document_hash: str | None = None,
        metadata: dict[str, str] | None = None,
        parallel_extraction: bool = False,
    ) -> ExtractionResponse:
        blocks = (
            ocr_blocks
            if ocr_blocks is not None
            else text_to_blocks(text or "", source="manual")
        )
        logger.debug(
            "ocr_complete",
            extra={
                "docie_step": "ocr",
                "docie_backend": "manual",
                "docie_block_count": len(blocks),
                **(
                    {"docie_blocks": [{"id": b.id, "text": b.text} for b in blocks]}
                    if getattr(get_settings(), "log_document_content", False)
                    else {}
                ),
            },
        )
        return await self._extract_blocks(
            blocks=blocks,
            schema_name=schema_name,
            schema_mode=schema_mode,
            dynamic_schema=dynamic_schema,
            language=language,
            document_hash=document_hash,
            metadata=metadata or {},
            parallel_extraction=parallel_extraction,
        )

    async def extract_from_file(
        self,
        *,
        path: Path,
        ocr_backend_name: str,
        schema_name: str,
        schema_mode: str = "static",
        dynamic_schema: dict[str, Any] | DynamicSchemaSpec | None = None,
        language: str | None = None,
        metadata: dict[str, str] | None = None,
        parallel_extraction: bool = False,
    ) -> ExtractionResponse:
        # kind="ocr" (no extractor -- just OCR text as the "completion", per
        # serving.solutions.OcrSolution) is NOT handled here: it falls through
        # to the branches below and hits the same bug this method fixes for
        # "pipeline" -- OCRs with the wrong backend, then tries calling the
        # OCR profile's placeholder base_url as if it were a real LLM. Left
        # alone deliberately: a pure-OCR profile has no schema-shaped output,
        # so there's nothing for the benchmark to score it against in the
        # first place. Flagging here so this isn't silently rediscovered.
        if getattr(self.profile, "kind", "passthrough") == "pipeline":
            return await self._extract_pipeline(
                path=path,
                schema_name=schema_name,
                schema_mode=schema_mode,
                dynamic_schema=dynamic_schema,
                language=language,
                metadata=metadata or {},
                parallel_extraction=parallel_extraction,
            )
        if self.profile.vision:
            t0 = time.perf_counter()
            images = await asyncio.to_thread(
                load_document_images,
                path,
                max_pages=self.profile.vision_max_pages,
                pdf_dpi=self.profile.vision_pdf_dpi,
            )
            logger.debug(
                "vision_ingestion_complete",
                extra={
                    "docie_step": "vision_ingestion",
                    "docie_path": str(path),
                    "docie_page_count": len(images),
                    "docie_ingestion_latency_ms": int((time.perf_counter() - t0) * 1000),
                },
            )
            return await self._extract_blocks(
                blocks=[],
                images=images,
                schema_name=schema_name,
                schema_mode=schema_mode,
                dynamic_schema=dynamic_schema,
                language=language,
                document_hash=await asyncio.to_thread(hash_file, path),
                metadata=metadata or {},
                parallel_extraction=parallel_extraction,
            )
        if ocr_backend_name.lower().strip() == "vision":
            raise ValueError("ocr_backend='vision' requires a model profile with vision: true")
        t0 = time.perf_counter()
        ocr_result = await asyncio.to_thread(
            processor_from_settings(get_settings()).process,
            path,
            backend_name=ocr_backend_name,
            language=language,
        )
        blocks = ocr_result.artifact.blocks
        ocr_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(
            "ocr_complete",
            extra={
                "docie_step": "ocr",
                "docie_backend": ocr_backend_name,
                "docie_path": str(path),
                "docie_block_count": len(blocks),
                "docie_ocr_latency_ms": ocr_ms,
                **(
                    {"docie_blocks": [{"id": b.id, "text": b.text} for b in blocks]}
                    if getattr(get_settings(), "log_document_content", False)
                    else {}
                ),
            },
        )
        return await self._extract_blocks(
            blocks=blocks,
            images=None,
            schema_name=schema_name,
            schema_mode=schema_mode,
            dynamic_schema=dynamic_schema,
            language=language,
            document_hash=ocr_result.artifact.document_hash,
            metadata=metadata or {},
            parallel_extraction=parallel_extraction,
        )

    async def _extract_pipeline(
        self,
        *,
        path: Path,
        schema_name: str,
        schema_mode: str,
        dynamic_schema: dict[str, Any] | DynamicSchemaSpec | None,
        language: str | None,
        metadata: dict[str, str],
        parallel_extraction: bool = False,
    ) -> ExtractionResponse:
        """kind="pipeline": OCR the document, then extract with the configured
        `options.extractor` profile -- the benchmark-side counterpart to
        `serving.solutions.PipelineSolution`, which does the same OCR step for
        the gateway but returns a raw completion (fine for the Studio Agents
        chat surface, not enough to score: no schema validation, grounding, or
        nuextract normalization).

        Delegating to a fresh ExtractionService(extractor_profile) for the
        actual extraction reuses ALL of that machinery unchanged -- the only
        difference from a normal text profile is that the input came from OCR
        instead of the caller. The OCR step is EITHER a built-in backend
        (`options.ocr_backend`, default tesseract) or a deployed vision model
        doing VLM-as-OCR (`options.ocr_model`), mirroring the gateway's own
        two pipeline modes.
        """
        options = self.profile.options
        extractor_name = options.get("extractor")
        if not extractor_name:
            raise ValueError(
                f"pipeline profile {self.profile.name!r} requires options.extractor "
                "(the name of a passthrough LLM profile)"
            )
        extractor = self.profiles.get(str(extractor_name))
        if extractor is None:
            raise ValueError(
                f"pipeline extractor profile {extractor_name!r} is not configured "
                f"(known profiles: {sorted(self.profiles)})"
            )
        if extractor.kind != "passthrough":
            raise ValueError(
                f"pipeline extractor {extractor_name!r} must be a passthrough LLM "
                f"profile, got kind={extractor.kind!r}"
            )

        t0 = time.perf_counter()
        ocr_model_name = options.get("ocr_model")
        if ocr_model_name:
            vision = self.profiles.get(str(ocr_model_name))
            if vision is None:
                raise ValueError(
                    f"pipeline ocr_model {ocr_model_name!r} is not configured "
                    f"(known profiles: {sorted(self.profiles)})"
                )
            if vision.kind != "passthrough" or not vision.vision:
                raise ValueError(
                    f"pipeline ocr_model {ocr_model_name!r} must be a passthrough "
                    f"vision deployment profile, got kind={vision.kind!r} "
                    f"vision={vision.vision!r}"
                )
            ocr_backend_label = f"vlm:{vision.name}"
            text = await self._vlm_ocr_text(vision, path)
            # "manual" -- same convention extract_from_text uses for text that
            # didn't come through a registered OCR backend (see its own
            # text_to_blocks(..., source="manual") call).
            blocks = text_to_blocks(text, source="manual")
            document_hash = await asyncio.to_thread(hash_file, path)
        else:
            backend_name = str(options.get("ocr_backend", "tesseract"))
            ocr_language = options.get("language") or language
            ocr_result = await asyncio.to_thread(
                processor_from_settings(get_settings()).process,
                path,
                backend_name=backend_name,
                language=ocr_language,
            )
            ocr_backend_label = backend_name
            blocks = ocr_result.artifact.blocks
            document_hash = ocr_result.artifact.document_hash
        logger.debug(
            "ocr_complete",
            extra={
                "docie_step": "ocr",
                "docie_backend": ocr_backend_label,
                "docie_path": str(path),
                "docie_block_count": len(blocks),
                "docie_ocr_latency_ms": int((time.perf_counter() - t0) * 1000),
            },
        )

        extractor_service = ExtractionService(
            extractor,
            proposer_profile=self.proposer_profile,
            profiles=self.profiles,
            disable_thinking=self.disable_thinking or bool(options.get("no_think")),
            max_tokens=self.max_tokens,
            # The actual JSON-producing model call happens on the INNER
            # service, not this OCR-only pipeline profile -- the live
            # preview belongs on that call.
            on_delta=self.on_delta,
            on_reset=self.on_reset,
        )
        response = await extractor_service.extract_from_text(
            text=None,
            ocr_blocks=blocks,
            schema_name=schema_name,
            schema_mode=schema_mode,
            dynamic_schema=dynamic_schema,
            language=language,
            document_hash=document_hash,
            metadata=metadata,
            parallel_extraction=parallel_extraction,
        )
        # Report as the pipeline profile the caller asked to benchmark, not
        # the inner extractor -- matches what predictions.jsonl/metrics key
        # on elsewhere in the runner (task.profile.name).
        return response.model_copy(update={"model_profile": self.profile.name})

    async def _vlm_ocr_text(self, vision: ModelProfile, path: Path) -> str:
        """Transcribe a document to text with a vision deployment
        (options.ocr_model). Mirrors serving.solutions.PipelineSolution.
        _vlm_ocr: an unconstrained plain-text chat completion, deliberately
        NOT going through OpenAICompatibleClient.chat_json -- that wrapper
        negotiates a JSON response_format style, retries, and a circuit
        breaker, all built for structured extraction; asking a VLM to wrap a
        verbatim transcription in JSON is unneeded indirection for a task
        that already IS plain text. A raw httpx POST, same as the gateway.

        load_document_images (already used by the vision extraction path
        above) reads directly from the file path -- simpler here than the
        gateway's own route, which reconstructs images from an inline
        request data URI because its input is already an HTTP request, not
        a file on disk.
        """
        images = await asyncio.to_thread(
            load_document_images,
            path,
            max_pages=vision.vision_max_pages,
            pdf_dpi=vision.vision_pdf_dpi,
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": OCR_TRANSCRIPTION_USER_PROMPT}
        ]
        content += [
            {"type": "image_url", "image_url": {"url": url}}
            for url in await asyncio.to_thread(_data_urls, images)
        ]
        request = {
            "model": vision.model,
            "messages": [
                {"role": "system", "content": OCR_TRANSCRIPTION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
        }
        if self.disable_thinking or bool(self.profile.options.get("no_think")):
            request["chat_template_kwargs"] = {"enable_thinking": False}
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{vision.base_url}/chat/completions",
                    json=request,
                    headers={
                        "Authorization": f"Bearer {vision.api_key}",
                        "Content-Type": "application/json",
                    },
                    # Floored, not raw vision.timeout_seconds: a CPU-bound
                    # multi-page transcription at single-digit tok/s
                    # routinely outlasts a model's default 180s chat
                    # timeout mid-response -- the exact scenario this
                    # benchmark exists to measure. Matches the gateway's
                    # own floor on this same call (PipelineSolution._vlm_ocr).
                    timeout=max(vision.timeout_seconds, _PIPELINE_UPSTREAM_TIMEOUT_S),
                )
            except httpx.RequestError as exc:
                raise ValueError(
                    f"pipeline ocr_model {vision.name!r} upstream is unreachable: {exc}"
                ) from exc
        if resp.status_code >= 400:
            raise ValueError(
                f"pipeline ocr_model {vision.name!r} returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        try:
            text = resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            text = ""
        if isinstance(text, list):  # multimodal parts -> concatenate text
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return fix_mojibake(str(text or "")) or ""

    async def _extract_group(
        self,
        *,
        field_names: list[str] | None,
        generation_schema: dict[str, Any],
        nuextract_template: dict[str, Any] | None,
        blocks: list[OCRBlock],
        images: list[DocumentImage] | None,
        image_urls: list[str] | None,
        schema_name: str,
        language: str | None,
        metadata: dict[str, str],
        want_logprobs: bool,
    ) -> tuple[
        dict[str, Any], dict[str, Any] | None, str | None, int | None, dict[str, float | None]
    ]:
        """Run one chat_json call over ``generation_schema`` -- the unit of
        work fanned out across a split extraction's groups (design #444), or
        called once directly for an unsplit one. ``field_names=None`` marks
        the unsplit path (every field, streaming allowed); a split call's
        caller has already narrowed ``generation_schema``/``nuextract_template``
        to just its group's fields (see _split_schema_into_groups /
        _subset_schema) -- nothing here re-derives the split.

        Returns (raw, usage_dict, effective_style, queue_wait_ms,
        field_confidences) -- everything _extract_blocks needs to merge
        across groups without this method knowing whether it's one of many.
        """
        if self.profile.prompt_profile == "nuextract3":
            # NuExtract3 gets the template out-of-band via chat_template_kwargs
            # (the `nuextract3` response style), so the prompt carries only the
            # document — as the page image (vision) or OCR text. This must win
            # over the generic vision branch below.
            system_prompt, user_prompt = build_nuextract3_prompts(
                blocks=blocks,
                has_images=bool(images),
            )
        elif images:
            system_prompt = VISION_SYSTEM_PROMPT
            user_prompt = build_vision_user_prompt(
                schema_name=schema_name,
                schema=generation_schema,
                page_count=len(images),
                language=language,
                metadata=metadata,
            )
        elif self.profile.prompt_profile == "nuextract_v1":
            system_prompt, user_prompt = build_nuextract_prompts(
                schema_name=schema_name,
                blocks=blocks,
                language=language,
                template=nuextract_template,
            )
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = build_user_prompt(
                schema_name=schema_name,
                schema=generation_schema,
                blocks=blocks,
                language=language,
                metadata=metadata,
            )
        client = OpenAICompatibleClient(self.profile)
        # build_response_format's own "nuextract3" branch resolves a template
        # PURELY from schema_name, via a static lookup (llm.prompts._NUEXTRACT_TEMPLATES)
        # that only ever knew about the built-in schemas -- a dynamic schema's
        # freshly-built nuextract_template never reached it, so every
        # dynamic-schema extraction through nuextract3 silently sent an EMPTY
        # template (nothing to extract -> every field comes back null).
        # chat_json's chat_template_kwargs is MERGED on top of
        # build_response_format's own extra_body (see build_payload), so
        # overriding "template" here for this one case is enough -- the
        # static-schema path (nuextract_template is None) is untouched.
        extra_template_kwargs: dict[str, Any] = {"enable_thinking": False}
        if self.profile.prompt_profile == "nuextract3" and nuextract_template is not None:
            extra_template_kwargs["template"] = json.dumps(
                nuextract_template, ensure_ascii=False
            )
        try:
            raw, usage_dict, raw_response = await client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name=schema_name,
                schema=generation_schema,
                image_urls=image_urls,
                chat_template_kwargs=extra_template_kwargs or None,
                max_tokens=self.max_tokens,
                # LFM2.5's bundled template opens <think> unconditionally and
                # ignores enable_thinking/reasoning_effort. Continuing an
                # assistant JSON turn bypasses that generation prompt while
                # retaining the kwargs for templates that do honor them.
                assistant_prefill=(
                    "{"
                    if self.disable_thinking
                    and self.profile.prompt_profile == "strict_extraction_v1"
                    else None
                ),
                request_logprobs=want_logprobs,
                # A split extraction's groups run concurrently -- N interleaved
                # streams would garble one live preview, so streaming only
                # ever wires up on the unsplit path (field_names is None);
                # _extract_blocks itself also refuses to split when a
                # streaming callback is present, so this is a second,
                # independent guard, not the only one.
                on_delta=self.on_delta if field_names is None else None,
                on_reset=self.on_reset if field_names is None else None,
            )
            effective_style = getattr(client, "last_response_format_style", None)
            queue_wait_ms = getattr(client, "last_queue_wait_ms", None)
        finally:
            await client.aclose()
        # Snapshot BEFORE any reshaping: `raw` here is exactly the FLAT dict
        # the model generated (matches `generation_schema`'s unwrapped shape),
        # which is what field-value substrings must be located against.
        field_confidences: dict[str, float | None] = (
            compute_field_confidences(raw, raw_response) if want_logprobs else {}
        )
        return raw, usage_dict, effective_style, queue_wait_ms, field_confidences

    async def _extract_blocks(
        self,
        *,
        blocks: list[OCRBlock],
        images: list[DocumentImage] | None = None,
        schema_name: str,
        schema_mode: str,
        dynamic_schema: dict[str, Any] | DynamicSchemaSpec | None,
        language: str | None,
        document_hash: str | None,
        metadata: dict[str, str],
        parallel_extraction: bool = False,
    ) -> ExtractionResponse:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        dynamic_spec: DynamicSchemaSpec | None = None
        model_cls = None
        nuextract_template = None
        if schema_mode == "dynamic":
            if isinstance(dynamic_schema, DynamicSchemaSpec):
                dynamic_spec = dynamic_schema
            elif dynamic_schema is not None:
                dynamic_spec = DynamicSchemaSpec.model_validate(dynamic_schema)
            else:
                if not blocks:
                    raise ValueError(
                        "Dynamic schema inference requires OCR text; supply a reusable "
                        "dynamic_schema for vision-only extraction"
                    )
                dynamic_spec = await self._propose_schema(blocks=blocks, language=language)
            schema_name = dynamic_spec.document_type
            model_cls = DynamicTemplateBuilder.build_model(dynamic_spec)
            schema = model_cls.model_json_schema()
            nuextract_template = DynamicTemplateBuilder.build_nuextract_template(dynamic_spec)
        elif schema_mode == "static":
            if dynamic_schema is not None:
                raise ValueError("dynamic_schema can only be supplied when schema_mode='dynamic'")
            schema = schema_json(schema_name)
        else:
            raise ValueError("schema_mode must be 'static' or 'dynamic'")
        # Keep two explicit contracts. The rich Pydantic schema is the internal
        # validation/audit shape. The compact value schema is what the model
        # sees and what structured decoders compile; the rich schema contains
        # regexes and wrapper defaults that llama.cpp cannot turn into GBNF.
        generation_schema = flatten_schema_json(schema)
        # Opt-in (#335): llama.cpp-only per-token logprob confidence. Both the
        # request flag AND the declared runtime must agree -- an unlabeled or
        # non-llama.cpp profile never gets `logprobs` on the wire, regardless
        # of the flag (see ModelProfile.runtime/.logprob_confidence).
        want_logprobs = self.profile.logprob_confidence and self.profile.runtime == "llamacpp"
        image_urls = await asyncio.to_thread(_data_urls, images) if images else None

        # Design #444: split a schema's list-typed top-level fields into
        # independent groups and fan them out concurrently, each through the
        # SAME gateway semaphore (no new limiter). Opt-in per request
        # (parallel_extraction) -- most real schemas have a list field, so an
        # always-on split would silently change the call pattern (and
        # benchmark timing) for nearly every extraction. Streaming also
        # forces the unsplit path -- N concurrent groups would interleave
        # into one garbled live preview.
        groups = (
            None
            if not parallel_extraction or self.on_delta is not None or self.on_reset is not None
            else _split_schema_into_groups(generation_schema)
        )
        if groups is None:
            raw, usage_dict, effective_style, queue_wait_ms, field_confidences = (
                await self._extract_group(
                    field_names=None,
                    generation_schema=generation_schema,
                    nuextract_template=nuextract_template,
                    blocks=blocks,
                    images=images,
                    image_urls=image_urls,
                    schema_name=schema_name,
                    language=language,
                    metadata=metadata,
                    want_logprobs=want_logprobs,
                )
            )
        else:

            async def _run_group(
                field_names: list[str],
            ) -> tuple[
                dict[str, Any], dict[str, Any] | None, str | None, int | None,
                dict[str, float | None],
            ]:
                return await self._extract_group(
                    field_names=field_names,
                    generation_schema=_subset_schema(generation_schema, field_names),
                    nuextract_template=(
                        {k: v for k, v in nuextract_template.items() if k in field_names}
                        if nuextract_template is not None
                        else None
                    ),
                    blocks=blocks,
                    images=images,
                    image_urls=image_urls,
                    schema_name=schema_name,
                    language=language,
                    metadata=metadata,
                    want_logprobs=want_logprobs,
                )

            slots = self.profile.deployment_slot_count or 1
            fanout = asyncio.Semaphore(max(1, min(slots, self.profile.max_concurrency)))

            async def _bounded(field_names: list[str]) -> Any:
                async with fanout:
                    return await _run_group(field_names)

            try:
                async with asyncio.TaskGroup() as tg:
                    tasks = [tg.create_task(_bounded(group)) for group in groups]
            except* Exception as eg:
                raise eg.exceptions[0] from None
            group_results = [task.result() for task in tasks]

            raw = {}
            field_confidences = {}
            effective_style = None
            usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            usage_seen = False
            queue_waits: list[int] = []
            for group_raw, group_usage, group_style, group_wait, group_conf in group_results:
                raw.update(group_raw)
                field_confidences.update(group_conf)
                if effective_style is None:
                    effective_style = group_style
                if group_wait is not None:
                    queue_waits.append(group_wait)
                if isinstance(group_usage, dict):
                    usage_seen = True
                    for key in usage_totals:
                        usage_totals[key] += group_usage.get(key) or 0
            usage_dict = usage_totals if usage_seen else None
            # Groups run concurrently -- their waits overlap, so the total
            # queued time a caller actually experienced is the LONGEST one
            # waited, not the sum (summing would double-count overlapping time).
            queue_wait_ms = max(queue_waits) if queue_waits else None

        derived_subtotal = False
        if self.profile.prompt_profile in {"nuextract_v1", "nuextract3"}:
            raw, derived_subtotal = _normalize_nuextract_raw(raw, schema_name)
        raw = rehydrate_extraction_result(raw, schema)
        raw = ground_evidence(raw, blocks)
        normalized, validation = validate_extraction(schema_name, raw, blocks, model_cls=model_cls)
        if field_confidences:
            # Attached to the plain validated dict, not a Pydantic field on
            # TextField/MoneyField/etc: `model_confidence` is ad-hoc metadata
            # this round, so any path that re-validates a stored result back
            # through those wrapper models (extra="ignore" by default) will
            # silently drop it again. Acceptable for this opt-in signal.
            attach_model_confidence(normalized, field_confidences)
        if derived_subtotal:
            # The model didn't report a subtotal; it was computed here from
            # total_ttc - vat_amount, not read off the document. Without this,
            # a synthesized value is indistinguishable from a genuine
            # extraction to every downstream consumer (API response, review
            # queue, benchmark scoring).
            validation.warnings.append(
                "subtotal.amount was derived from total_ttc - vat_amount "
                "(the model did not extract it directly)"
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = Usage.model_validate(usage_dict) if isinstance(usage_dict, dict) else None

        logger.debug(
            "extraction_complete",
            extra={
                "docie_step": "extraction_complete",
                "docie_schema_name": schema_name,
                "docie_model_profile": self.profile.name,
                "docie_doc_id": metadata.get("doc_id"),
                "docie_latency_ms": latency_ms,
                # Time spent waiting for the ModelGateway's per-(base_url,
                # model) semaphore before the HTTP call to the model even
                # started -- separates queueing from generation. Everything
                # else in docie_latency_ms not accounted for here is prompt
                # construction, the model call itself, and post-processing
                # (rehydration/grounding/validation), which stay lumped
                # together as "generation" -- not worth a further split.
                "docie_queue_wait_ms": queue_wait_ms,
                "docie_generation_ms": (
                    max(latency_ms - queue_wait_ms, 0) if queue_wait_ms is not None else None
                ),
                "docie_valid": validation.valid,
                "docie_errors": validation.errors,
                "docie_warnings": validation.warnings,
                "docie_normalized_result": redact_fields(
                    normalized, get_settings().audit_redaction_fields
                ),
            },
        )

        return ExtractionResponse(
            request_id=request_id,
            schema_name=schema_name,
            model_profile=self.profile.name,
            document_hash=document_hash,
            result=normalized,
            validation=validation,
            usage=usage,
            latency_ms=latency_ms,
            dynamic_schema=(
                dynamic_spec.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
                if dynamic_spec
                else None
            ),
            response_format_style=effective_style,
            queue_wait_ms=queue_wait_ms,
            ocr_blocks=blocks or None,
        )

    async def _propose_schema(
        self,
        *,
        blocks: list[OCRBlock],
        language: str | None,
    ) -> DynamicSchemaSpec:
        profile = self.proposer_profile or self.profile
        if profile.prompt_profile == "nuextract_v1":
            raise ValueError(
                "Dynamic schema inference requires an instruction-following proposer profile "
                "or a reusable dynamic_schema"
            )
        client = OpenAICompatibleClient(profile)
        try:
            raw, _usage, _response = await client.chat_json(
                system_prompt=SCHEMA_PROPOSER_SYSTEM_PROMPT,
                user_prompt=build_schema_proposer_prompt(blocks=blocks, language=language),
                schema_name="dynamic_schema_spec",
                schema=DynamicSchemaSpec.model_json_schema(),
            )
        finally:
            await client.aclose()
        return DynamicSchemaSpec.model_validate(raw)
