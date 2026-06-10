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
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    max_upload_mb: int = 25
    raw_document_storage: Literal["disabled", "enabled"] = "disabled"

    database_url: str | None = None

    default_schema_name: str = "invoice"
    default_model_profile: str = "local_llamacpp"
    default_ocr_backend: str = "pdf_text"
    ocr_cache_dir: Path = Path(".cache/ocr")
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
