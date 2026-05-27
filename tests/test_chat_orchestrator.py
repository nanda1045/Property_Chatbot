from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


def load_cases() -> list[dict]:
    return json.loads(Path("evals/chat_cases.json").read_text(encoding="utf-8"))


def test_chat_cases_return_grounded_markdown_components_and_sources(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    for case in load_cases():
        response = client.post(
            "/chat",
            json={
                "property_code": case["property_code"],
                "model": case["model"],
                "message": case["message"],
            },
        )
        assert response.status_code == 200, case["name"]
        body = response.json()

        assert body["property_code"] == case["property_code"]
        assert body["answer_markdown"]
        for term in case["expected_answer_terms"]:
            assert term in body["answer_markdown"], case["name"]

        component_types = {component["type"] for component in body["components"]}
        for component_type in case["expected_component_types"]:
            assert component_type in component_types, case["name"]

        for tool_key in case["expected_tool_keys"]:
            assert tool_key in body["tool_results"], case["name"]

        expected_source_code = case["expected_source_property_code"]
        if expected_source_code:
            assert body["sources"], case["name"]
            assert {source["property_code"] for source in body["sources"]} == {
                expected_source_code
            }


def test_unknown_property_returns_clear_message(mysql_db, hybrid_retriever) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "missing-code",
            "model": "mock:mock-property-assistant",
            "message": "What is the occupancy?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["property_code"] == "missing-code"
    assert "could not find property code" in body["answer_markdown"].lower()


def test_unavailable_requested_year_does_not_fall_back_to_latest(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "I want the breakdown for lease charges for the year 2024 only.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["property_code"] == "115r"
    assert "don't have rent-roll data for **2024**" in body["answer_markdown"]
    assert "loaded structured data" not in body["answer_markdown"].lower()
    assert "2025-12-01, the largest charge categories" not in body["answer_markdown"]
    assert body["components"] == []
    assert "report_periods" in body["tool_results"]
    assert "charge_breakdown" not in body["tool_results"]


def test_ambiguous_query_asks_for_clarification_before_expensive_tools(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Compare",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert body["property_code"] == "115r"
    assert "What would you like to look at" in answer
    assert "Rent-roll KPIs" in answer
    assert body["components"] == []
    assert body["sources"] == []
    assert set(body["tool_results"]) == {"property_profile"}


def test_single_word_charges_query_asks_charge_specific_clarification(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "charges",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert "What charge view would you like" in answer
    assert "Biggest charge categories" in answer
    assert "Rent vs lease charges comparison" in answer
    assert "As of **2025-12-01**, occupancy is" not in answer
    assert body["components"] == []
    assert body["sources"] == []
    assert set(body["tool_results"]) == {"property_profile"}


def test_reviews_query_does_not_attach_irrelevant_kpis_or_sources(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Can you provide the reviews for this property?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert "don't see matching website evidence for reviews" in answer
    assert "96.66%" not in answer
    assert body["components"] == []
    assert body["sources"] == []
    assert set(body["tool_results"]) == {"property_profile", "property_content"}
    assert body["tool_results"]["property_content"] == []


def test_unknown_website_fact_query_does_not_attach_irrelevant_kpis_or_sources(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Do they have package lockers?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert "don't see matching website evidence" in answer
    assert "96.66%" not in answer
    assert "Occupancy" not in answer
    assert body["components"] == []
    assert body["sources"] == []
    assert set(body["tool_results"]) == {"property_profile", "property_content"}
    assert body["tool_results"]["property_content"] == []
    assert "latest_kpis" not in body["tool_results"]


def test_mentioned_property_mismatch_adds_inline_scope_note(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "For The Alexander, does it have a pool? Use the selected property only.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert "You mentioned **The Alexander**" in answer
    assert "active property is **Canfield Park (`115r`)**" in answer
    assert "Pool with Cabanas" in answer
    assert "Indoor & Outdoor Swimming Pools" not in answer
    assert {source["property_code"] for source in body["sources"]} == {"115r"}


def test_retrieval_results_include_evidence_confidence_for_debugging(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Does this property have bike storage or EV charging?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    retrieval_results = body["tool_results"]["property_content"]
    assert retrieval_results
    evidence = retrieval_results[0]["evidence"]
    assert evidence["confidence"] in {"high", "medium"}
    assert "bike" in evidence["matched_terms"] or "ev" in evidence["matched_terms"]
    assert evidence["line_matches"]


def test_floorplan_answer_summarizes_categories_without_website_boilerplate(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "What floorplans are advertised on the website?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert "Studio" in answer
    assert "1-bedroom" in answer
    assert "2-bedroom" in answer
    assert "Monthly leasing prices" not in answer
    assert "artist" not in answer.lower()
    assert body["sources"]
    assert {source["property_code"] for source in body["sources"]} == {"115r"}
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_url"] == "https://canfield-park.com/floorplans/"
    assert "Source:" not in answer


def test_website_answer_uses_sources_section_without_duplicate_source_link(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "What is included in the apartment features?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert "Apartment Features" in answer
    assert "Washer & Dryer" in answer
    assert "Source:" not in answer
    assert len(body["sources"]) == 1
    assert body["sources"][0]["source_url"] == "https://canfield-park.com/amenities/"


def test_simple_location_question_uses_property_profile_not_random_retrieval(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Where is this property located?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    assert "306 Canfield Avenue" in answer
    assert "Bridgeport, CT 06605" in answer
    assert "Relevant website evidence" not in answer
    assert "A04" not in answer
    assert body["sources"] == []
    assert set(body["tool_results"]) == {"property_profile"}


@pytest.mark.parametrize(
    "message",
    [
        "Show vacant units and their unit types.",
        "I want only vacant units and their type",
        "give me all those vacant units",
    ],
)
def test_vacant_unit_detail_prompts_do_not_trigger_rent_by_type_charts(
    mysql_db,
    hybrid_retriever,
    message: str,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": message,
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    component_titles = [component["title"] for component in body["components"]]
    component_types = [component["type"] for component in body["components"]]

    assert "vacant units include" in answer
    assert "A105" in answer
    assert "115mxB01" in answer
    assert "2-bedroom" in answer
    assert component_titles == ["Vacant Units"]
    assert component_types == ["table"]
    assert body["components"][0]["data"][0]["bedroom_category"] == "2-bedroom"
    assert "vacant_units" in body["tool_results"]
    assert body["tool_results"]["vacant_units"][0]["bedroom_category"] == "2-bedroom"
    assert "rent_by_unit_type" not in body["tool_results"]
    assert "rent_by_bedroom_category" not in body["tool_results"]


def test_latest_month_rent_lease_comparison_does_not_trigger_occupancy_trend(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Compare rent and lease charges for the latest month.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    component_types = [component["type"] for component in body["components"]]
    component_titles = [component["title"] for component in body["components"]]

    assert "market rent is **$741,209**" in answer
    assert "lease charges are **$804,981**" in answer
    assert "higher than market rent" in answer
    assert "Occupancy stayed high" not in answer
    assert component_types == ["comparison_view"]
    assert component_titles == ["Market Rent vs Lease Charges"]
    assert "rent_lease_comparison" in body["tool_results"]
    assert "occupancy_trend" not in body["tool_results"]


def test_rent_vs_lease_charges_followup_routes_to_comparison(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Rent vs lease charges comparison",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    component_types = [component["type"] for component in body["components"]]

    assert "market rent is **$741,209**" in answer
    assert "lease charges are **$804,981**" in answer
    assert "higher than market rent" in answer
    assert "As of **2025-12-01**, occupancy is" not in answer
    assert component_types == ["comparison_view"]
    assert "rent_lease_comparison" in body["tool_results"]


def test_occupancy_change_across_available_months_routes_to_trend(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "What changed in occupancy across the available months?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    component_types = [component["type"] for component in body["components"]]

    assert "12-month trend window" in answer
    assert "percentage points" in answer
    assert "As of **2025-12-01**, occupancy is" not in answer
    assert component_types == ["line_chart"]
    assert "occupancy_trend" in body["tool_results"]


def test_executive_summary_is_interpretive_not_just_kpi_snapshot(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Give me a quick executive summary of this property.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    component_titles = [component["title"] for component in body["components"]]

    assert "Quick executive summary" in answer
    assert "highly occupied" in answer
    assert "$63,772" in answer
    assert "stabilized property" in answer
    assert "The latest summary shows" not in answer
    assert component_titles == ["Occupancy", "Lease Charges", "Vacant Units"]


def test_rent_by_unit_type_explains_bedroom_categories_and_floorplan_codes(
    mysql_db,
    hybrid_retriever,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": "Compare average market rent by unit type.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    component_titles = [component["title"] for component in body["components"]]
    bedroom_chart = next(
        component
        for component in body["components"]
        if component["title"] == "Average Market Rent by Bedroom Category"
    )

    assert "broader bedroom category averages" in answer
    assert "detailed floorplan/unit-type codes" in answer
    assert "1-bedroom" in answer
    assert "2-bedroom" in answer
    assert "Average Market Rent by Bedroom Category" in component_titles
    assert "Average Market Rent by Floorplan Code" in component_titles
    assert [row["label"] for row in bedroom_chart["data"]] == [
        "Studio",
        "1-bedroom",
        "2-bedroom",
    ]
    assert "rent_by_bedroom_category" in body["tool_results"]


@pytest.mark.parametrize(
    "message",
    [
        "What is the average balance by bedroom category for this property?",
        "Which bedroom category has the highest vacancy rate?",
        "Show median lease charges by unit type.",
        "Give me the top 10 market rents for this property.",
    ],
)
def test_unsupported_structured_aggregates_do_not_use_partial_tool_results(
    mysql_db,
    hybrid_retriever,
    message: str,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": message,
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]

    assert "can't calculate" in answer
    assert "partial rows" not in answer
    assert body["components"] == []
    assert body["sources"] == []
    assert set(body["tool_results"]) == {"property_profile"}
    assert "top_balances" not in body["tool_results"]
    assert "vacant_units" not in body["tool_results"]
    assert "charge_breakdown" not in body["tool_results"]
    assert "latest_kpis" not in body["tool_results"]


@pytest.mark.parametrize(
    ("message", "expected_component_type", "expected_tool_key", "expected_answer_term"),
    [
        (
            "How did occupancy move each month?",
            "line_chart",
            "occupancy_trend",
            "12-month trend window",
        ),
        (
            "Market rent against lease charges",
            "comparison_view",
            "rent_lease_comparison",
            "higher than market rent",
        ),
        (
            "Show the largest income buckets.",
            "bar_chart",
            "charge_breakdown",
            "largest charge categories",
        ),
        (
            "Give me a leadership overview.",
            "kpi_card",
            "latest_kpis",
            "Quick executive summary",
        ),
        (
            "Open units by layout",
            "table",
            "vacant_units",
            "vacant units include",
        ),
    ],
)
def test_semantic_intent_router_handles_common_paraphrases(
    mysql_db,
    hybrid_retriever,
    message: str,
    expected_component_type: str,
    expected_tool_key: str,
    expected_answer_term: str,
) -> None:
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={
            "property_code": "115r",
            "model": "mock:mock-property-assistant",
            "message": message,
        },
    )

    assert response.status_code == 200
    body = response.json()
    answer = body["answer_markdown"]
    component_types = [component["type"] for component in body["components"]]

    assert expected_answer_term in answer
    assert expected_component_type in component_types
    assert expected_tool_key in body["tool_results"]
    assert "Relevant website evidence" not in answer
