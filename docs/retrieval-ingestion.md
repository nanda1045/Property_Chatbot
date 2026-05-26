# Hybrid Retrieval Ingestion

The unstructured property website chunks are ingested into two local retrieval indexes:

- Chroma vector search, using a local sentence-transformer embedding model.
- SQLite FTS5 BM25 keyword search.

At query time, results from both indexes are merged with reciprocal rank fusion (RRF). Retrieval must always pass an active `property_code` filter to both Chroma and BM25 before ranking or fusion.

## Chunking

Scraped pages use `html_section_v1` chunking. Instead of flattening the whole page and splitting every N characters, the scraper uses HTML structure:

- Headings such as `h1`, `h2`, `h3`, and `h4` start sections.
- List and paragraph content is grouped under the nearest section heading.
- Common navigation/footer/cookie boilerplate is filtered out.
- Oversized sections still fall back to paragraph-preserving splits.

Each retrieval chunk includes:

```text
property_code
property_name
source_url
page_type
section_heading
chunk_strategy
content
```

This produces cleaner chunks such as `Community Features` or `Apartment Features`, which improves retrieval for amenity and floorplan questions.

## Ingest

```bash
uv run python scripts/ingest_unstructured.py --reset
```

This reads:

```text
Data/unstructured/property_chunks.jsonl
```

and writes persistent retrieval stores to:

```text
Data/chroma
Data/retrieval/bm25.sqlite3
```

## Search

```bash
uv run python scripts/search_unstructured.py "What amenities are available?" --property-code 115r
```

The retrieval wrapper always applies this Chroma metadata filter:

```python
where={"property_code": active_property_code}
```

BM25 applies the same scope as a SQL predicate:

```sql
property_code = ?
```

Optional page filtering is also supported:

```bash
uv run python scripts/search_unstructured.py "parking and EV charging" --property-code 115r --page-type amenities
```

## Embeddings

For local reproducibility without OpenAI embedding costs, the prototype uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model is downloaded once and cached under:

```text
Data/models/sentence-transformers
```

After the first download, the embedder loads from the local cache. The old deterministic hashing embedder remains available as `EMBEDDING_PROVIDER=local_hash` for offline debugging, but sentence-transformer embeddings are the default.

In a production version, this interface can be swapped to a hosted embedding model while keeping the same property filters and hybrid fusion layer.
