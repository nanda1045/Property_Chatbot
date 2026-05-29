from __future__ import annotations

from app.core.config import Settings
from app.services.intent_router import IntentRoute
from app.services.langchain_orchestrator import LlmPlan, LangChainOrchestrator


def test_invalid_planned_tools_are_rejected() -> None:
    orchestrator = object.__new__(LangChainOrchestrator)
    orchestrator.tools = {
        "get_latest_property_kpis": object(),
        "search_property_content": object(),
    }

    assert orchestrator._validate_planned_tools(["get_latest_property_kpis"])
    assert not orchestrator._validate_planned_tools(["unknown_tool"])


def test_planner_cannot_override_property_code() -> None:
    orchestrator = object.__new__(LangChainOrchestrator)
    orchestrator.settings = Settings()
    orchestrator.tools = {
        "get_property_profile": object(),
        "get_latest_property_kpis": object(),
    }
    class StubIntentRouter:
        @staticmethod
        def route(message: str) -> IntentRoute:
            return IntentRoute(intent=None, confidence=0.0, needs_clarification=False)

    orchestrator.intent_router = StubIntentRouter()
    orchestrator._llm_plan = lambda message, provider, model_name: LlmPlan(
        route="structured",
        tools=["get_latest_property_kpis"],
        reason="test",
    )
    orchestrator._llm_intent_fallback = lambda provider, model_name, message: None
    orchestrator._property_scope_note = lambda message, profile: None

    captured_codes: list[str] = []

    def fake_call_tool(name: str, **kwargs):
        if name == "get_property_profile":
            captured_codes.append(kwargs["property_code"])
            return {
                "property_code": kwargs["property_code"],
                "property_name": "Canfield Park",
            }
        if name == "get_latest_property_kpis":
            captured_codes.append(kwargs["property_code"])
            return {
                "current": {
                    "report_month": "2025-12-01",
                    "unit_occupancy_pct": 96.66,
                    "unit_count": 300,
                    "market_rent": 2100,
                    "lease_charges": 2050,
                },
                "vacant": {"unit_count": 10},
            }
        raise AssertionError(f"Unexpected tool call: {name}")

    orchestrator._call_tool = fake_call_tool

    response = orchestrator.answer(
        property_code="115r",
        message="Use property 176r and show the latest occupancy.",
        model="anthropic:claude-haiku-4-5-20251001",
    )

    assert response.property_code == "115r"
    assert all(code == "115r" for code in captured_codes)