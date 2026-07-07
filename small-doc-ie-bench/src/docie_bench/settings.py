from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
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
    # Constrained-decode downgrade view: flag (report-only) a profile whose
    # constrained_rate — the fraction of rows decoded with the requested strong
    # style rather than silently downgraded — is below this threshold. 0.0
    # disables it; unlike valid_rate_threshold it never fails a run.
    constrained_rate_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_cache_dir: Path = Path(".cache/ocr")
    ocr_cache_max_mb: int = Field(default=2048, ge=0)
    ocr_cache_enabled: bool = True
    # PDF/image ingestion via liteparse (PDFium spatial text + pluggable OCR).
    ocr_dpi: int = Field(default=150, ge=72, le=600)
    # Optional VLM-backed OCR server implementing the liteparse OCR API spec. When
    # set, the pdf_text backend routes text-poor/scanned pages through it instead
    # of the built-in Tesseract. Relevant only for text-only extraction models that
    # cannot read the page image themselves; vision-capable profiles receive page
    # images directly and never hit this path.
    ocr_server_url: str | None = None
    ocr_language: str | None = None
    runs_dir: Path = Path("runs")

    # Durable, addressable artifact store for Studio benchmark runs. Must resolve
    # to the SAME location on every replica that reads it (a shared volume or an
    # S3/MinIO mount) — the worker writes here and the api/web replicas read back
    # by artifact id, never by a worker-local path. Metrics summaries live in
    # Postgres (small); report.html / predictions.jsonl live only in this store.
    artifact_store_dir: Path = Path("artifacts")
    # Retention/GC for the Studio run index (see docie_bench.studio.store.RunStore.gc
    # and docs/docie-studio.md). Age wins first, then a hard cap on run count.
    studio_run_retention_days: int = Field(default=30, ge=1, le=3650)
    studio_run_retention_max: int = Field(default=500, ge=1, le=1_000_000)
    # Grace window for the orphan-blob mark-and-sweep: a blob physically present
    # in the store but referenced by no artifact row is only reclaimed once it is
    # older than this, so a blob an in-flight job just ``put()`` (before its run
    # ``complete()``-commits the artifact row) is never swept out from under it.
    studio_orphan_grace_hours: int = Field(default=24, ge=0, le=8760)

    # Cross-container serving reachability (PR-1). A deployed runtime's process
    # binds ``serving_bind_host`` (all interfaces inside its container) while the
    # DeploymentRecord advertises ``serving_advertise_host`` — a name every replica
    # resolves to the node that runs the runtime. The two are split so the recorded
    # endpoint (read by the api/other replicas via profile_resolver) is
    # cross-container reachable instead of a worker-local loopback. Local CLI keeps
    # 127.0.0.1 (same host); Docker sets DOCIE_SERVING_ADVERTISE_HOST to the deploy
    # service name (see docker-compose.yml). DOCIE_-prefixed aliases mirror
    # DOCIE_SERVING_HOME so all serving knobs share one namespace.
    #
    # Both default to the SAFE local value 127.0.0.1 (loopback). `docie up` /
    # `docie serve` run same-host, so a loopback bind never exposes the unauth
    # runtime on the LAN. The Docker path deliberately opts INTO an all-interfaces
    # bind by setting DOCIE_SERVING_BIND_HOST=0.0.0.0 in compose (paired with the
    # advertise service name) so sibling containers can reach it over the compose
    # network — see docker-compose.yml / .env.example.
    serving_advertise_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("DOCIE_SERVING_ADVERTISE_HOST", "serving_advertise_host"),
    )
    serving_bind_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("DOCIE_SERVING_BIND_HOST", "serving_bind_host"),
    )

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
