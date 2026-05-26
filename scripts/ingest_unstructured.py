#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.core.config import get_settings
from app.retrieval.bm25_store import BM25PropertyStore
from app.retrieval.chroma_store import ChromaPropertyStore
from app.retrieval.chunks import load_chunks
from app.retrieval.embeddings import build_embedder


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest property chunks into Chroma and BM25.")
    parser.add_argument("--chunks-path", default=str(settings.unstructured_chunks_path))
    parser.add_argument("--chroma-path", default=str(settings.chroma_path))
    parser.add_argument("--collection", default=settings.chroma_collection)
    parser.add_argument("--bm25-path", default=str(settings.bm25_path))
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--codes", nargs="*", help="Optional subset of property codes to ingest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(Path(args.chunks_path))
    if args.codes:
        selected_codes = {code.lower() for code in args.codes}
        chunks = [chunk for chunk in chunks if chunk.property_code in selected_codes]

    store = ChromaPropertyStore(
        persist_path=Path(args.chroma_path),
        collection_name=args.collection,
        embedder=build_embedder(get_settings()),
    )
    keyword_store = BM25PropertyStore(Path(args.bm25_path))
    if args.reset:
        store.reset()
        keyword_store.reset()

    vector_inserted = store.ingest(chunks, batch_size=args.batch_size)
    keyword_inserted = keyword_store.ingest(chunks)
    counts = Counter(chunk.property_code for chunk in chunks)

    print(f"Ingested {vector_inserted} chunk(s) into Chroma collection `{args.collection}`.")
    print(f"Ingested {keyword_inserted} chunk(s) into BM25 index `{args.bm25_path}`.")
    print(f"Chroma count: {store.count()}")
    print(f"BM25 count: {keyword_store.count()}")
    print("Chunks by property:")
    for property_code, count in sorted(counts.items()):
        print(f"  {property_code}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
