from __future__ import annotations

import json
from pathlib import Path

from app.retrieval.hybrid_store import HybridPropertyRetriever


def load_cases() -> list[dict]:
    return json.loads(Path("evals/retrieval_cases.json").read_text(encoding="utf-8"))


def test_hybrid_indexes_have_expected_representative_sample(
    hybrid_retriever: HybridPropertyRetriever,
) -> None:
    expected_count = sum(1 for _ in Path("Data/unstructured/property_chunks.jsonl").open())
    assert hybrid_retriever.count() == {
        "vector": expected_count,
        "keyword": expected_count,
    }


def test_retrieval_cases_are_relevant_and_property_scoped(
    hybrid_retriever: HybridPropertyRetriever,
) -> None:
    for case in load_cases():
        results = hybrid_retriever.search(
            query=case["query"],
            property_code=case["property_code"],
            page_type=case.get("page_type"),
            n_results=case.get("n_results", 3),
        )

        assert results, case["name"]
        assert any(result.vector_rank is not None for result in results), case["name"]
        assert any(result.keyword_rank is not None for result in results), case["name"]
        assert {result.metadata["property_code"] for result in results} == {
            case["property_code"]
        }
        assert case["expected_page_type"] in {
            result.metadata["page_type"] for result in results
        }

        combined_content = "\n".join(result.content for result in results).lower()
        assert any(term.lower() in combined_content for term in case["expected_terms"]), (
            case["name"],
            case["expected_terms"],
        )


def test_targeted_amenity_filter_removes_generic_chunks(
    hybrid_retriever: HybridPropertyRetriever,
) -> None:
    results = hybrid_retriever.search(
        query="Does this property have bike storage or EV charging?",
        property_code="115r",
        page_type="amenities",
        n_results=5,
    )

    sections = {result.metadata.get("section_heading") for result in results}
    combined_content = "\n".join(result.content for result in results)

    assert sections == {"Community Features"}
    assert "EV Charging Stations" in combined_content
    assert "Bike Racks & Storage Lockers" in combined_content


def test_moderate_filter_drops_redundant_or_noisy_chunks(
    hybrid_retriever: HybridPropertyRetriever,
) -> None:
    results = hybrid_retriever.search(
        query="What amenities mention pool or fitness?",
        property_code="176r",
        page_type="amenities",
        n_results=5,
    )

    sections = {result.metadata.get("section_heading") for result in results}
    combined_content = "\n".join(result.content for result in results)

    assert "Brand New Amenities" in sections
    assert "Apartment Features" not in sections
    assert "Follow Us on Instagram" not in sections
    assert "Page Overview" not in sections
    assert "Fitness Center" in combined_content
    assert "Pool" in combined_content


def test_broad_amenity_question_keeps_community_and_apartment_sections(
    hybrid_retriever: HybridPropertyRetriever,
) -> None:
    results = hybrid_retriever.search(
        query="What amenities are listed on the property website?",
        property_code="115r",
        page_type="amenities",
        n_results=5,
    )

    sections = {result.metadata.get("section_heading") for result in results}

    assert "Community Features" in sections
    assert "Apartment Features" in sections
