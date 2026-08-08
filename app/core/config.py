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
    runtime_user_id: str = "local-user"
    default_llm_provider: str = "anthropic"
    default_llm_model: str = "claude-haiku-4-5-20251001"

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

    @field_validator("default_property_code", "runtime_user_id")
    @classmethod
    def reject_blank_scope(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("trusted scope values cannot be blank")
        return normalized

    @model_validator(mode="after")
    def reject_production_reload(self) -> Settings:
        if self.app_env == "production" and self.app_reload:
            raise ValueError("APP_RELOAD must be false in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
