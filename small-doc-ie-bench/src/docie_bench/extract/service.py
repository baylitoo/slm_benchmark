from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from docie_bench.extract.validators import validate_extraction

logger = logging.getLogger(__name__)
from docie_bench.llm.model_profiles import ModelProfile
from docie_bench.llm.openai_client import OpenAICompatibleClient
from docie_bench.llm.prompts import SYSTEM_PROMPT, build_nuextract_prompts, build_user_prompt
from docie_bench.ocr.base import text_to_blocks
from docie_bench.ocr.factory import get_ocr_backend
from docie_bench.schemas.common import ExtractionResponse, OCRBlock, Usage
from docie_bench.schemas.extraction import schema_json


_CURRENCY_MAP = {"€": "EUR", "£": "GBP", "$": "USD", "¥": "JPY", "₣": "CHF"}
_DATE_FIELD_NAMES = {"issue_date", "due_date", "birth_date", "expiry_date"}
_DECIMAL_FIELD_NAMES = {"vat_rate"}


_COUNTRY_ISO: dict[str, str] = {
    "france": "FRA", "française": "FRA", "francaise": "FRA",
    "germany": "DEU", "deutschland": "DEU", "allemagne": "DEU",
    "spain": "ESP", "espagne": "ESP", "españa": "ESP",
    "united kingdom": "GBR", "uk": "GBR",
    "united states": "USA", "usa": "USA",
    "italy": "ITA", "italie": "ITA",
}


def _norm_amount(raw: str) -> str:
    """Fallback: '4 026,00 €' → '4026.00'. With typed templates the model should output clean numbers."""
    s = re.sub(r"[€£$¥₣a-zA-Z]", "", raw).strip()
    if "," in s and "." in s:
        # Ambiguous: detect thousands vs decimal by position
        comma_pos = s.rfind(",")
        dot_pos = s.rfind(".")
        if dot_pos > comma_pos:
            # e.g. '1,234.56' — dot is decimal, comma is thousands
            s = s.replace(",", "")
        else:
            # e.g. '1.234,56' — comma is decimal, dot is thousands
            s = s.replace(".", "").replace(",", ".")
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


def _normalize_nuextract_raw(raw: dict[str, Any], schema_name: str) -> dict[str, Any]:
    """Post-process NuExtract3 output: enforce document_type, strip IBAN spaces,
    null-out empty MoneyFields, and apply fallback normalization for any values the
    model returned in locale format despite type hints."""
    result: dict[str, Any] = {"document_type": schema_name}
    for key, val in raw.items():
        if key == "document_type":
            continue  # already set above
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
                sub["currency"] = _CURRENCY_MAP.get(sub["currency"].strip(), sub["currency"].strip()) or None

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
    return result


def hash_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


class ExtractionService:
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    async def extract_from_text(
        self,
        *,
        text: str | None,
        ocr_blocks: list[OCRBlock] | None,
        schema_name: str,
        language: str | None = None,
        document_hash: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ExtractionResponse:
        blocks = ocr_blocks if ocr_blocks is not None else text_to_blocks(text or "", source="manual")
        logger.debug(
            "ocr_complete",
            extra={
                "docie_step": "ocr",
                "docie_backend": "manual",
                "docie_block_count": len(blocks),
                "docie_blocks": [{"id": b.id, "text": b.text} for b in blocks],
            },
        )
        return await self._extract_blocks(
            blocks=blocks,
            schema_name=schema_name,
            language=language,
            document_hash=document_hash,
            metadata=metadata or {},
        )

    async def extract_from_file(
        self,
        *,
        path: Path,
        ocr_backend_name: str,
        schema_name: str,
        language: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> ExtractionResponse:
        backend = get_ocr_backend(ocr_backend_name, language=language)
        t0 = time.perf_counter()
        blocks = backend.extract(path)
        ocr_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(
            "ocr_complete",
            extra={
                "docie_step": "ocr",
                "docie_backend": ocr_backend_name,
                "docie_path": str(path),
                "docie_block_count": len(blocks),
                "docie_ocr_latency_ms": ocr_ms,
                "docie_blocks": [{"id": b.id, "text": b.text} for b in blocks],
            },
        )
        return await self._extract_blocks(
            blocks=blocks,
            schema_name=schema_name,
            language=language,
            document_hash=hash_file(path),
            metadata=metadata or {},
        )

    async def _extract_blocks(
        self,
        *,
        blocks: list[OCRBlock],
        schema_name: str,
        language: str | None,
        document_hash: str | None,
        metadata: dict[str, str],
    ) -> ExtractionResponse:
        request_id = str(uuid.uuid4())
        schema = schema_json(schema_name)
        if self.profile.prompt_profile == "nuextract_v1":
            system_prompt, user_prompt = build_nuextract_prompts(
                schema_name=schema_name,
                blocks=blocks,
                language=language,
            )
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = build_user_prompt(
                schema_name=schema_name,
                schema=schema,
                blocks=blocks,
                language=language,
                metadata=metadata,
            )
        started = time.perf_counter()
        client = OpenAICompatibleClient(self.profile)
        try:
            raw, usage_dict, _raw_response = await client.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name=schema_name,
                schema=schema,
            )
        finally:
            await client.aclose()
        if self.profile.prompt_profile == "nuextract_v1":
            raw = _normalize_nuextract_raw(raw, schema_name)
        normalized, validation = validate_extraction(schema_name, raw, blocks)
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
                "docie_valid": validation.valid,
                "docie_errors": validation.errors,
                "docie_warnings": validation.warnings,
                "docie_normalized_result": normalized,
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
        )
