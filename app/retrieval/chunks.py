from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PropertyChunk:
    id: str
    property_code: str
    property_name: str
    address: str | None
    source_url: str
    page_type: str
    section_heading: str | None
    section_index: int | None
    section_split_index: int | None
    chunk_strategy: str | None
    chunk_index: int
    title: str | None
    content: str
    scraped_at: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PropertyChunk:
        return cls(
            id=str(row["id"]),
            property_code=str(row["property_code"]).lower(),
            property_name=str(row["property_name"]),
            address=row.get("address"),
            source_url=str(row["source_url"]),
            page_type=str(row["page_type"]),
            section_heading=row.get("section_heading"),
            section_index=int(row["section_index"]) if row.get("section_index") else None,
            section_split_index=(
                int(row["section_split_index"]) if row.get("section_split_index") else None
            ),
            chunk_strategy=row.get("chunk_strategy"),
            chunk_index=int(row["chunk_index"]),
            title=row.get("title"),
            content=str(row["content"]),
            scraped_at=str(row["scraped_at"]),
        )

    def metadata(self) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "property_code": self.property_code,
            "property_name": self.property_name,
            "source_url": self.source_url,
            "page_type": self.page_type,
            "chunk_index": self.chunk_index,
            "scraped_at": self.scraped_at,
        }
        if self.section_heading:
            metadata["section_heading"] = self.section_heading
        if self.section_index is not None:
            metadata["section_index"] = self.section_index
        if self.section_split_index is not None:
            metadata["section_split_index"] = self.section_split_index
        if self.chunk_strategy:
            metadata["chunk_strategy"] = self.chunk_strategy
        if self.address:
            metadata["address"] = self.address
        if self.title:
            metadata["title"] = self.title
        return metadata


def load_chunks(path: Path) -> list[PropertyChunk]:
    chunks = []
    with path.open("r", encoding="utf-8") as chunk_file:
        for line in chunk_file:
            if line.strip():
                chunks.append(PropertyChunk.from_row(json.loads(line)))
    return chunks
