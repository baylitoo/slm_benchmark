from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_env: str = "local"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104 - container service bind default
    api_port: int = 8080
    max_request_body_mb: int = Field(default=26, ge=1, le=1_025)
    max_upload_mb: int = Field(default=25, ge=1, le=1_024)
    max_text_chars: int = Field(default=1_000_000, ge=1, le=100_000_000)
    max_ocr_blocks: int = Field(default=1_000, ge=1, le=100_000)
    max_ocr_block_chars: int = Field(default=20_000, ge=1, le=10_000_000)
    max_metadata_entries: int = Field(default=50, ge=0, le=10_000)
    raw_document_storage: Literal["disabled", "enabled"] = "disabled"
    allowed_upload_mime_types: str = (
        "application/pdf,text/plain,image/png,image/jpeg,image/tiff"
    )

    # Fail closed by default (B3). Local dev sets AUTH_REQUIRED=false in .env;
    # any networked deployment must populate API_KEYS and leave this on.
    auth_required: bool = True
    api_keys: SecretStr = Field(default=SecretStr(""))
    rate_limit_requests: int = Field(default=60, ge=0)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    tenant_max_concurrent_requests: int = Field(default=4, ge=0)
    enable_benchmark_api: bool = False

    redacted_response_fields: str = ""
    redacted_audit_fields: str = ""
    log_document_content: bool = False

    database_url: str | None = None
    review_claim_lease_seconds: int = Field(default=900, ge=30, le=86400)
    review_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    review_evidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    annotation_export_dir: Path = Path("annotations")

    default_schema_name: str = "invoice"
    # Studio-friendly default: strongest structured style declared, with the
    # serving negotiation ladder auto-downgrading per runtime so the Playground
    # returns valid JSON out-of-box even on small models.
    default_model_profile: str = "studio_default"
    default_ocr_backend: str = "pdf_text"
    # Validity gate: fail a benchmark run loudly when a profile's valid_rate is
    # below this threshold instead of silently scoring zeros. 0.0 disables it.
    valid_rate_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_cache_dir: Path = Path(".cache/ocr")
    ocr_cache_max_mb: int = Field(default=2048, ge=0)
    ocr_cache_enabled: bool = True
    runs_dir: Path = Path("runs")

    openai_compat_base_url: str = "http://llm-llamacpp:8000/v1"
    openai_compat_api_key: SecretStr = Field(default=SecretStr("local-not-used"))
    openai_compat_model: str = "local-model"
    openai_compat_timeout_seconds: float = 180.0
    openai_compat_response_format_style: str = "openai_json_schema"

    prometheus_multiproc_dir: str | None = None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_request_body_bytes(self) -> int:
        return self.max_request_body_mb * 1024 * 1024

    @property
    def allowed_mime_types(self) -> set[str]:
        return {
            value.strip().lower()
            for value in self.allowed_upload_mime_types.split(",")
            if value.strip()
        }

    @property
    def response_redaction_fields(self) -> set[str]:
        return {
            value.strip()
            for value in self.redacted_response_fields.split(",")
            if value.strip()
        }

    @property
    def audit_redaction_fields(self) -> set[str]:
        return {value.strip() for value in self.redacted_audit_fields.split(",") if value.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
