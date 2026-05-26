#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from app.core.config import get_settings
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid_store import HybridPropertyRetriever


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run a property-scoped hybrid retrieval search.")
    parser.add_argument("query")
    parser.add_argument("--property-code", default=settings.default_property_code)
    parser.add_argument("--page-type")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--chroma-path", default=str(settings.chroma_path))
    parser.add_argument("--collection", default=settings.chroma_collection)
    parser.add_argument("--bm25-path", default=str(settings.bm25_path))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    store = HybridPropertyRetriever(
        chroma_path=Path(args.chroma_path),
        chroma_collection=args.collection,
        bm25_path=Path(args.bm25_path),
        embedder=build_embedder(settings),
    )
    results = store.search(
        query=args.query,
        property_code=args.property_code,
        page_type=args.page_type,
        n_results=args.n_results,
    )

    print(f"Query: {args.query}")
    print(f"Property scope: {args.property_code.lower()}")
    print(f"Results: {len(results)}")
    for index, result in enumerate(results, start=1):
        metadata = result.metadata
        print()
        print(
            f"{index}. {metadata.get('property_code')} "
            f"{metadata.get('page_type')} score={result.score:.4f} "
            f"vector_rank={result.vector_rank} keyword_rank={result.keyword_rank}"
        )
        if metadata.get("section_heading"):
            print(f"Section: {metadata.get('section_heading')}")
        print(metadata.get("title", ""))
        print(metadata.get("source_url", ""))
        preview = result.content[:500].replace("\n", " ")
        print(preview)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
