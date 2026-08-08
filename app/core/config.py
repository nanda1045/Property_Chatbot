from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["local", "test", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_host: str = Field(default="127.0.0.1", min_length=1)
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_reload: bool = True
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ]
    )

    mysql_host: str = Field(default="127.0.0.1", min_length=1)
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_user: str = Field(default="root", min_length=1)
    mysql_password: str = Field(default="root", repr=False)
    mysql_database: str = Field(default="aker_chatbot", min_length=1)
    mysql_connect_timeout_seconds: int = Field(default=5, ge=1, le=60)

    chroma_path: Path = Path("Data/chroma")
    chroma_collection: str = "property_chunks"
    bm25_path: Path = Path("Data/retrieval/bm25.sqlite3")
    unstructured_chunks_path: Path = Path("Data/unstructured/property_chunks.jsonl")
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_path: Path = Path("Data/models/sentence-transformers")

    default_property_code: str = "115r"
    default_llm_provider: str = "anthropic"
    default_llm_model: str = "claude-haiku-4-5-20251001"

    auth_mode: Literal["local", "entra"] = "local"
    local_auth_user_id: str = "local-user"
    local_auth_display_name: str = "Local Demo User"
    local_auth_email: str = "local-user@example.test"
    local_auth_role: Literal["Viewer", "Analyst", "PropertyManager"] = "PropertyManager"
    local_auth_allowed_properties: list[str] = Field(default_factory=lambda: ["*"])
    entra_tenant_id: str | None = None
    entra_api_audience: str | None = None
    entra_required_scope: str = "access_as_user"
    entra_authority_host: str = "https://login.microsoftonline.com"
    entra_jwks_cache_seconds: int = Field(default=3600, ge=60, le=86400)
    auth_property_access: dict[str, list[str]] = Field(default_factory=dict)

    agent_max_steps: int = Field(default=8, ge=1)
    agent_max_tool_calls: int = Field(default=12, ge=1)
    agent_max_planner_retries: int = Field(default=2, ge=0)
    agent_max_sql_approvals: int = Field(default=1, ge=0)
    agent_max_run_seconds: float = Field(default=60.0, gt=0)

    stream_queue_max_size: int = Field(default=128, ge=4, le=4096)
    stream_poll_interval_seconds: float = Field(default=0.25, gt=0, le=5)
    stream_heartbeat_seconds: float = Field(default=10.0, gt=0, le=60)
    stream_thread_join_seconds: float = Field(default=5.0, gt=0, le=30)

    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)

    @field_validator("app_env", mode="before")
    @classmethod
    def normalize_environment(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("auth_mode", mode="before")
    @classmethod
    def normalize_auth_mode(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("default_property_code", "local_auth_user_id")
    @classmethod
    def reject_blank_scope(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("trusted scope values cannot be blank")
        return normalized

    @field_validator("entra_authority_host")
    @classmethod
    def validate_authority_host(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized.startswith("https://"):
            raise ValueError("ENTRA_AUTHORITY_HOST must use HTTPS")
        return normalized

    @field_validator("entra_required_scope")
    @classmethod
    def validate_required_scope(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ENTRA_REQUIRED_SCOPE cannot be blank")
        return normalized

    @field_validator("local_auth_allowed_properties")
    @classmethod
    def normalize_local_properties(cls, value: list[str]) -> list[str]:
        normalized = sorted(
            {item.strip().lower() for item in value if item and item.strip()}
        )
        if not normalized:
            raise ValueError("LOCAL_AUTH_ALLOWED_PROPERTIES cannot be empty")
        return normalized

    @field_validator("auth_property_access")
    @classmethod
    def normalize_property_access(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for subject, properties in value.items():
            subject_key = subject.strip()
            property_codes = sorted(
                {item.strip().lower() for item in properties if item and item.strip()}
            )
            if not subject_key or not property_codes:
                raise ValueError("AUTH_PROPERTY_ACCESS entries require a subject and properties")
            normalized[subject_key] = property_codes
        return normalized

    @model_validator(mode="after")
    def validate_deployment_security(self) -> Settings:
        if self.app_env == "production" and self.app_reload:
            raise ValueError("APP_RELOAD must be false in production")
        if self.app_env == "production" and self.auth_mode != "entra":
            raise ValueError("AUTH_MODE must be entra in production")
        if self.auth_mode == "entra":
            if not self.entra_tenant_id or not self.entra_api_audience:
                raise ValueError(
                    "ENTRA_TENANT_ID and ENTRA_API_AUDIENCE are required in Entra mode"
                )
            tenant = self.entra_tenant_id.strip().lower()
            if tenant in {"common", "organizations", "consumers"}:
                raise ValueError("ENTRA_TENANT_ID must identify one tenant")
            self.entra_tenant_id = self.entra_tenant_id.strip()
            self.entra_api_audience = self.entra_api_audience.strip()
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
