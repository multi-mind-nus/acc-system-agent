from pathlib import Path
from urllib.parse import urlsplit
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    app_version: str = "dev"
    environment: str = "development"
    classification_provider: Literal["MOCK", "REMOTE", "DISABLED"] = "DISABLED"
    review_provider: Literal["MOCK", "REMOTE", "DISABLED"] = "DISABLED"
    document_path: Path = Path("/data/documents")
    max_document_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    model_api_url: str = ""
    model_health_url: str = ""
    model_api_key: SecretStr = SecretStr("")
    model_connect_timeout_seconds: float = Field(default=10, gt=0, le=30)
    model_request_timeout_seconds: float = Field(default=180, gt=0, le=300)

    @field_validator("model_api_url", "model_health_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if value:
            url = urlsplit(value)
            if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("Use an HTTP(S) URL without credentials, query or fragment")
        return value


settings = Settings()
