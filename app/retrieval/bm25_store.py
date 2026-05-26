from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval.chroma_store import RetrievalResult
from app.retrieval.chunks import PropertyChunk

QUERY_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{1,}", re.IGNORECASE)


@dataclass(frozen=True)
class BM25SearchConfig:
    path: Path


class BM25PropertyStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()
        self._ensure_schema()

    def ingest(self, chunks: list[PropertyChunk]) -> int:
        rows = [self._chunk_row(chunk) for chunk in chunks]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO property_chunks (
                  id, property_code, property_name, address, source_url, page_type,
                  section_heading, section_index, section_split_index, chunk_index,
                  chunk_strategy, scraped_at, title, content
                )
                VALUES (
                  :id, :property_code, :property_name, :address, :source_url, :page_type,
                  :section_heading, :section_index, :section_split_index, :chunk_index,
                  :chunk_strategy, :scraped_at, :title, :content
                )
                """,
                rows,
            )
        return len(chunks)

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM property_chunks").fetchone()
            return int(row["count"])

    def search(
        self,
        query: str,
        property_code: str,
        n_results: int = 10,
        page_type: str | None = None,
    ) -> list[RetrievalResult]:
        fts_query = self._fts_query(query)
        if not fts_query:
            return []

        clauses = ["property_chunks MATCH ?", "property_code = ?"]
        params: list[Any] = [fts_query, property_code.lower()]
        if page_type:
            clauses.append("page_type = ?")
            params.append(page_type)
        params.append(n_results)

        sql = f"""
            SELECT
              id, property_code, property_name, address, source_url, page_type,
              section_heading, section_index, section_split_index, chunk_index,
              chunk_strategy, scraped_at, title, content,
              bm25(property_chunks, 0.8, 1.3, 2.0) AS bm25_score
            FROM property_chunks
            WHERE {" AND ".join(clauses)}
            ORDER BY bm25_score ASC
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        return [
            RetrievalResult(
                id=row["id"],
                content=row["content"],
                metadata=self._row_metadata(row),
                distance=float(row["bm25_score"]),
            )
            for row in rows
        ]

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS property_chunks USING fts5(
                  id UNINDEXED,
                  property_code UNINDEXED,
                  property_name UNINDEXED,
                  address UNINDEXED,
                  source_url UNINDEXED,
                  page_type UNINDEXED,
                  section_heading,
                  section_index UNINDEXED,
                  section_split_index UNINDEXED,
                  chunk_index UNINDEXED,
                  chunk_strategy UNINDEXED,
                  scraped_at UNINDEXED,
                  title,
                  content,
                  tokenize = 'porter unicode61'
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _chunk_row(chunk: PropertyChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "property_code": chunk.property_code,
            "property_name": chunk.property_name,
            "address": chunk.address,
            "source_url": chunk.source_url,
            "page_type": chunk.page_type,
            "section_heading": chunk.section_heading,
            "section_index": chunk.section_index,
            "section_split_index": chunk.section_split_index,
            "chunk_index": chunk.chunk_index,
            "chunk_strategy": chunk.chunk_strategy,
            "scraped_at": chunk.scraped_at,
            "title": chunk.title,
            "content": chunk.content,
        }

    @staticmethod
    def _row_metadata(row: sqlite3.Row) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "property_code": row["property_code"],
            "property_name": row["property_name"],
            "source_url": row["source_url"],
            "page_type": row["page_type"],
            "chunk_index": int(row["chunk_index"]),
            "scraped_at": row["scraped_at"],
        }
        for key in [
            "address",
            "title",
            "section_heading",
            "chunk_strategy",
        ]:
            if row[key] is not None:
                metadata[key] = row[key]
        for key in ["section_index", "section_split_index"]:
            if row[key] is not None:
                metadata[key] = int(row[key])
        return metadata

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = [
            match.group(0).lower().replace("'", "")
            for match in QUERY_TOKEN_RE.finditer(query)
        ]
        tokens = [token for token in tokens if len(token) > 1]
        if not tokens:
            return ""
        deduped = list(dict.fromkeys(tokens[:12]))
        return " OR ".join(f'"{token}"' for token in deduped)
