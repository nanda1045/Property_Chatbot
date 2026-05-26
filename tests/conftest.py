from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings, get_settings
from app.db.mysql import MySQLDatabase
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid_store import HybridPropertyRetriever
from app.services.rent_roll_repository import RentRollRepository


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def mysql_db(settings: Settings) -> MySQLDatabase:
    db = MySQLDatabase(settings)
    try:
        db.fetch_one("SELECT 1 AS ok")
    except Exception as exc:
        pytest.skip(f"MySQL is not available: {exc}")
    return db


@pytest.fixture(scope="session")
def rent_roll_repository(mysql_db: MySQLDatabase) -> RentRollRepository:
    return RentRollRepository(mysql_db)


@pytest.fixture(scope="session")
def hybrid_retriever(settings: Settings) -> HybridPropertyRetriever:
    store = HybridPropertyRetriever(
        chroma_path=Path(settings.chroma_path),
        chroma_collection=settings.chroma_collection,
        bm25_path=Path(settings.bm25_path),
        embedder=build_embedder(settings),
    )
    counts = store.count()
    if counts["vector"] == 0 or counts["keyword"] == 0:
        pytest.skip(
            "Retrieval indexes are empty. Run scripts/ingest_unstructured.py --reset first."
        )
    return store
