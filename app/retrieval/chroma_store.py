from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrieval.chunks import PropertyChunk
from app.retrieval.embeddings import TextEmbedder


@dataclass(frozen=True)
class RetrievalResult:
    id: str
    content: str
    metadata: dict
    distance: float | None
    score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None


class ChromaPropertyStore:
    def __init__(
        self,
        persist_path: Path,
        collection_name: str,
        embedder: TextEmbedder,
    ) -> None:
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_provider": self.embedder.provider_name,
                "embedding_dimensions": self.embedder.dimensions,
            },
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_provider": self.embedder.provider_name,
                "embedding_dimensions": self.embedder.dimensions,
            },
        )

    def ingest(self, chunks: list[PropertyChunk], batch_size: int = 100) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [chunk.content for chunk in batch]
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=documents,
                metadatas=[chunk.metadata() for chunk in batch],
                embeddings=self.embedder.embed_many(documents),
            )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query: str,
        property_code: str,
        n_results: int = 5,
        page_type: str | None = None,
    ) -> list[RetrievalResult]:
        where: dict = {"property_code": property_code.lower()}
        if page_type:
            where = {
                "$and": [
                    {"property_code": property_code.lower()},
                    {"page_type": page_type},
                ]
            }

        results = self.collection.query(
            query_embeddings=[self.embedder.embed(query)],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            RetrievalResult(
                id=result_id,
                content=document,
                metadata=metadata or {},
                distance=distance,
            )
            for result_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
