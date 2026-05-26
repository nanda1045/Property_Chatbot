from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "root"
    mysql_database: str = "aker_chatbot"

    chroma_path: Path = Path("Data/chroma")
    chroma_collection: str = "property_chunks"
    bm25_path: Path = Path("Data/retrieval/bm25.sqlite3")
    unstructured_chunks_path: Path = Path("Data/unstructured/property_chunks.jsonl")
    embedding_provider: str = "sentence_transformer"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_path: Path = Path("Data/models/sentence-transformers")

    default_property_code: str = "115r"
    default_llm_provider: str = "mock"
    default_llm_model: str = "mock-property-assistant"

    openai_api_key: str | None = Field(default=None, repr=False)
    anthropic_api_key: str | None = Field(default=None, repr=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()
