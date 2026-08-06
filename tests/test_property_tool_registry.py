from __future__ import annotations

import json
import unittest

from app.core.config import Settings
from app.retrieval.chroma_store import RetrievalResult
from app.services.langchain_tools import build_langchain_tools
from app.tools.contracts import TrustedToolContext
from app.tools.executor import ToolExecutor
from app.tools.property_tools import build_property_tool_registry


class FakeRepository:
    def __init__(self) -> None:
        self.occupancy_calls: list[tuple[str, int]] = []
        self.vacancy_calls: list[tuple[str, int, str | None]] = []
        self.rent_calls: list[tuple[str, str | None]] = []

    def get_property_profile(self, property_code: str):
        return {
            "property_code": property_code,
            "property_name": "Canfield Park",
            "address": None,
            "source_site": None,
        }

    def list_properties(self):
        return [
            {
                "property_code": "115r",
                "property_name": "Canfield Park",
                "address": None,
                "source_site": None,
            }
        ]

    def get_report_periods(self, property_code: str):
        return {
            "property_code": property_code,
            "min_report_month": "2025-01-01",
            "max_report_month": "2025-01-01",
            "months": ["2025-01-01"],
            "years": [2025],
        }

    def get_latest_kpis(self, property_code: str):
        return {
            "current": {"property_code": property_code, "report_month": "2025-01-01"},
            "vacant": None,
        }

    def get_occupancy_trend(self, property_code: str, months: int = 12):
        self.occupancy_calls.append((property_code, months))
        return [{"report_month": "2025-01-01", "unit_occupancy_pct": 95.0}]

    def get_charge_breakdown(self, property_code: str, limit: int = 10):
        return []

    def get_top_balances(self, property_code: str, limit: int = 10):
        return []

    def get_vacant_units(
        self,
        property_code: str,
        limit: int = 20,
        report_month: str | None = None,
    ):
        self.vacancy_calls.append((property_code, limit, report_month))
        return []

    def get_rent_by_unit_type(
        self,
        property_code: str,
        report_month: str | None = None,
    ):
        self.rent_calls.append((property_code, report_month))
        return []


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [
            RetrievalResult(
                id="chunk-1",
                content="EV charging is available.",
                metadata={"property_code": kwargs["property_code"], "page_type": "amenities"},
                distance=0.1,
                score=0.2,
            )
        ]


class PropertyToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = FakeRepository()
        self.retriever = FakeRetriever()
        self.settings = Settings(_env_file=None)
        self.registry = build_property_tool_registry(
            self.settings,
            repository=self.repository,
            property_retriever=self.retriever,
        )
        self.executor = ToolExecutor(self.registry, trace_sink=lambda _event: None)
        self.context = TrustedToolContext(property_code="115R", user_id="user-1")

    def test_registry_contains_every_existing_property_tool(self) -> None:
        self.assertEqual(
            self.registry.names(),
            [
                "get_charge_breakdown",
                "get_latest_property_kpis",
                "get_occupancy_trend",
                "get_property_profile",
                "get_rent_by_unit_type",
                "get_report_periods",
                "get_top_balances",
                "get_vacant_units",
                "list_properties",
                "search_property_content",
            ],
        )

    def test_repository_receives_backend_scope_not_model_argument(self) -> None:
        result = self.executor.execute(
            "get_occupancy_trend",
            {"months": 6, "property_code": "attacker-property"},
            self.context,
        )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(self.repository.occupancy_calls, [("115r", 6)])
        self.assertEqual(result.data[0]["unit_occupancy_pct"], 95.0)

    def test_retrieval_is_scoped_and_typed(self) -> None:
        result = self.executor.execute(
            "search_property_content",
            {"query": "EV charging", "page_type": "amenities", "n_results": 3},
            self.context,
        )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(self.retriever.calls[0]["property_code"], "115r")
        self.assertEqual(result.data[0]["id"], "chunk-1")

    def test_period_specific_inputs_reach_repository(self) -> None:
        vacancy = self.executor.execute(
            "get_vacant_units",
            {"limit": 25, "report_month": "2025-06-01"},
            self.context,
        )
        rent = self.executor.execute(
            "get_rent_by_unit_type",
            {"report_month": "2025-06-01"},
            self.context,
        )

        self.assertEqual(vacancy.status, "succeeded")
        self.assertEqual(rent.status, "succeeded")
        self.assertEqual(
            self.repository.vacancy_calls,
            [("115r", 25, "2025-06-01")],
        )
        self.assertEqual(self.repository.rent_calls, [("115r", "2025-06-01")])

    def test_langchain_compatibility_wrapper_uses_executor(self) -> None:
        tools = {
            tool.name: tool
            for tool in build_langchain_tools(self.settings, executor=self.executor)
        }

        payload = json.loads(
            tools["get_occupancy_trend"].invoke(
                {"property_code": "115r", "months": 4}
            )
        )

        self.assertEqual(payload[0]["report_month"], "2025-01-01")
        self.assertEqual(self.repository.occupancy_calls[-1], ("115r", 4))


if __name__ == "__main__":
    unittest.main()
