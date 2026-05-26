from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from app.retrieval.bm25_store import BM25PropertyStore
from app.retrieval.chroma_store import ChromaPropertyStore, RetrievalResult
from app.retrieval.embeddings import TextEmbedder

QUERY_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{1,}", re.IGNORECASE)

QUERY_STOPWORDS = {
    "about",
    "all",
    "also",
    "and",
    "are",
    "available",
    "can",
    "cite",
    "cited",
    "data",
    "does",
    "for",
    "from",
    "have",
    "has",
    "how",
    "into",
    "is",
    "list",
    "listed",
    "listing",
    "me",
    "mention",
    "mentioned",
    "mentions",
    "only",
    "please",
    "property",
    "selected",
    "show",
    "source",
    "sources",
    "that",
    "the",
    "this",
    "use",
    "website",
    "what",
    "which",
    "with",
}

SPECIFIC_QUERY_TERMS = {
    "amenity",
    "amenities",
    "bath",
    "baths",
    "bed",
    "bedroom",
    "bedrooms",
    "bike",
    "charging",
    "dog",
    "ev",
    "fee",
    "fees",
    "fitness",
    "floorplan",
    "floorplans",
    "garage",
    "parking",
    "pet",
    "pool",
    "pools",
    "rent",
    "sqft",
    "storage",
}

NOISY_SECTION_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bfollow us\b",
        r"\binstagram\b",
        r"\bpage overview\b",
        r"\bvirtual tours?\b",
        r"\bour address\b",
        r"\bcontact\b",
        r"\bschedule (?:a )?tour\b",
        r"\bprivacy\b",
        r"\bterms\b",
    ]
)
NOISY_SECTION_PATTERNS = tuple(NOISY_SECTION_PATTERNS)

QUERY_SYNONYMS = {
    "bike": {"bicycle", "bicycles", "locker", "lockers", "racks"},
    "bikes": {"bike", "bicycle", "bicycles", "locker", "lockers", "racks"},
    "charging": {"charger", "chargers", "stations"},
    "ev": {"electric", "charging", "station", "stations"},
    "fitness": {"gym", "workout", "yoga"},
    "floorplan": {"floorplans", "bed", "beds", "bedroom", "bath", "baths", "sqft"},
    "floorplans": {"floorplan", "bed", "beds", "bedroom", "bath", "baths", "sqft"},
    "pool": {"pools", "swimming"},
    "pools": {"pool", "swimming"},
    "pet": {"dog", "friendly", "pet-friendly", "spa"},
    "storage": {"locker", "lockers"},
}

GENERIC_QUERY_TERMS = {"amenity", "amenities", "feature", "features"}


class HybridPropertyRetriever:
    def __init__(
        self,
        chroma_path: Path,
        chroma_collection: str,
        bm25_path: Path,
        embedder: TextEmbedder,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = ChromaPropertyStore(
            persist_path=chroma_path,
            collection_name=chroma_collection,
            embedder=embedder,
        )
        self.keyword_store = BM25PropertyStore(bm25_path)
        self.rrf_k = rrf_k

    def count(self) -> dict[str, int]:
        return {
            "vector": self.vector_store.count(),
            "keyword": self.keyword_store.count(),
        }

    def search(
        self,
        query: str,
        property_code: str,
        n_results: int = 5,
        page_type: str | None = None,
        vector_k: int | None = None,
        keyword_k: int | None = None,
    ) -> list[RetrievalResult]:
        vector_results = self.vector_store.search(
            query=query,
            property_code=property_code,
            page_type=page_type,
            n_results=vector_k or max(n_results * 3, 10),
        )
        keyword_results = self.keyword_store.search(
            query=query,
            property_code=property_code,
            page_type=page_type,
            n_results=keyword_k or max(n_results * 3, 10),
        )

        merged: dict[str, RetrievalResult] = {}
        scores: dict[str, float] = {}
        vector_ranks: dict[str, int] = {}
        keyword_ranks: dict[str, int] = {}

        for rank, result in enumerate(vector_results, start=1):
            merged.setdefault(result.id, result)
            vector_ranks[result.id] = rank
            scores[result.id] = scores.get(result.id, 0.0) + self._rrf(rank)

        for rank, result in enumerate(keyword_results, start=1):
            merged.setdefault(result.id, result)
            keyword_ranks[result.id] = rank
            scores[result.id] = scores.get(result.id, 0.0) + self._rrf(rank)

        ranked_ids = sorted(scores, key=lambda result_id: scores[result_id], reverse=True)
        fused = []
        candidate_limit = max(n_results * 3, 10)
        for result_id in ranked_ids[:candidate_limit]:
            result = merged[result_id]
            fused.append(
                replace(
                    result,
                    score=scores[result_id],
                    vector_rank=vector_ranks.get(result_id),
                    keyword_rank=keyword_ranks.get(result_id),
                )
            )
        return self._post_filter_results(
            query=query,
            results=fused,
            n_results=n_results,
            page_type=page_type,
        )

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def _post_filter_results(
        self,
        query: str,
        results: list[RetrievalResult],
        n_results: int,
        page_type: str | None,
    ) -> list[RetrievalResult]:
        if len(results) <= 1 or page_type == "floorplans":
            return results[:n_results]

        query_terms = self._query_terms(query)
        if not query_terms or self._is_broad_amenity_question(query, query_terms):
            return self._drop_obvious_noise(results, n_results)

        scored_query_terms = self._target_query_terms(query_terms)
        has_specific_terms = bool(scored_query_terms & SPECIFIC_QUERY_TERMS)
        threshold = 2.0 if has_specific_terms else 1.0

        filtered: list[RetrievalResult] = []
        covered_query_terms: set[str] = set()
        for index, result in enumerate(results):
            result_terms = self._result_terms(result)
            result_query_terms = scored_query_terms & result_terms
            is_redundant = bool(result_query_terms) and result_query_terms <= covered_query_terms
            score = self._local_relevance_score(
                scored_query_terms,
                result,
                result_terms,
                page_type,
            )
            keep_best_result = index == 0 and bool(result_query_terms) and score > -2.0
            if (keep_best_result or score >= threshold) and not is_redundant:
                filtered.append(result)
                covered_query_terms.update(result_query_terms)

        return (filtered or results[:1])[:n_results]

    def _drop_obvious_noise(
        self,
        results: list[RetrievalResult],
        n_results: int,
    ) -> list[RetrievalResult]:
        filtered = [
            result
            for index, result in enumerate(results)
            if index == 0 or not self._is_noisy_section(result)
        ]
        return (filtered or results[:1])[:n_results]

    def _local_relevance_score(
        self,
        query_terms: set[str],
        result: RetrievalResult,
        result_terms: set[str],
        page_type: str | None,
    ) -> float:
        overlap = query_terms & result_terms
        specific_overlap = overlap & SPECIFIC_QUERY_TERMS
        score = float(len(overlap) + (2 * len(specific_overlap)))

        if result.vector_rank is not None and result.keyword_rank is not None:
            score += 1.0
        if page_type and result.metadata.get("page_type") == page_type:
            score += 0.5
        if self._is_noisy_section(result):
            score -= 4.0
        if self._is_image_heavy(result):
            score -= 1.0
        return score

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        terms: set[str] = set()
        for match in QUERY_TOKEN_RE.finditer(query):
            raw_token = match.group(0)
            terms.add(HybridPropertyRetriever._normalize_token(raw_token))
            if "-" in raw_token:
                terms.update(
                    HybridPropertyRetriever._normalize_token(part)
                    for part in raw_token.split("-")
                )
        terms = {
            term
            for term in terms
            if len(term) > 1 and term not in QUERY_STOPWORDS
        }
        expanded = set(terms)
        for term in terms:
            expanded.update(QUERY_SYNONYMS.get(term, set()))
        return expanded

    @staticmethod
    def _target_query_terms(query_terms: set[str]) -> set[str]:
        targeted_terms = query_terms - GENERIC_QUERY_TERMS
        return targeted_terms or query_terms

    @staticmethod
    def _result_terms(result: RetrievalResult) -> set[str]:
        content = "\n".join(
            line
            for line in result.content.splitlines()
            if not line.strip().lower().startswith("image:")
        )
        text = " ".join(
            str(value)
            for value in [
                content,
                result.metadata.get("section_heading", ""),
                result.metadata.get("page_type", ""),
            ]
            if value
        )
        return {
            HybridPropertyRetriever._normalize_token(match.group(0))
            for match in QUERY_TOKEN_RE.finditer(text)
        }

    @staticmethod
    def _normalize_token(token: str) -> str:
        normalized = token.lower().replace("'", "")
        if normalized in {"sq", "sf"}:
            return "sqft"
        if normalized.endswith("ies") and len(normalized) > 4:
            return f"{normalized[:-3]}y"
        if normalized.endswith("s") and len(normalized) > 3:
            return normalized[:-1]
        return normalized

    @staticmethod
    def _is_broad_amenity_question(query: str, query_terms: set[str]) -> bool:
        lowered = query.lower()
        asks_for_amenities = "amenit" in lowered
        targeted_terms = query_terms & (
            SPECIFIC_QUERY_TERMS
            - {"amenity", "amenities"}
        )
        return asks_for_amenities and not targeted_terms

    @staticmethod
    def _is_noisy_section(result: RetrievalResult) -> bool:
        section_text = " ".join(
            str(value)
            for value in [
                result.metadata.get("section_heading", ""),
                result.metadata.get("title", ""),
            ]
            if value
        )
        return any(pattern.search(section_text) for pattern in NOISY_SECTION_PATTERNS)

    @staticmethod
    def _is_image_heavy(result: RetrievalResult) -> bool:
        content = result.content.lower()
        image_count = content.count("image:")
        return image_count >= 3 or (content.startswith("image:") and image_count >= 1)
