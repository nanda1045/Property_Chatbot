from __future__ import annotations

import unittest
from typing import Any

from app.agents.loop import BoundedAgentLoop, ExecutionLimits
from app.agents.occupancy_investigation import (
    EvidenceVerificationError,
    OccupancyInvestigationPolicy,
    build_occupancy_investigation_report,
    is_occupancy_investigation_request,
    verify_occupancy_investigation_report,
)
from app.tools.contracts import ToolResult

DECLINING_TREND = [
    {
        "report_month": "2025-01-01",
        "unit_occupancy_pct": 95.0,
        "market_rent": 100000.0,
        "lease_charges": 90000.0,
    },
    {
        "report_month": "2025-02-01",
        "unit_occupancy_pct": 89.5,
        "market_rent": 101500.0,
        "lease_charges": 88750.0,
    },
    {
        "report_month": "2025-03-01",
        "unit_occupancy_pct": 91.0,
        "market_rent": 102000.0,
        "lease_charges": 90000.0,
    },
]


def tool_result(tool_name: str, data: Any, number: int) -> ToolResult:
    return ToolResult(
        invocation_id=f"tool-{number}",
        tool_name=tool_name,
        status="succeeded",
        attempt=1,
        duration_ms=number,
        data=data,
    )


class OccupancyInvestigationTests(unittest.TestCase):
    def test_trigger_requires_decline_and_investigation_language(self) -> None:
        self.assertTrue(
            is_occupancy_investigation_request(
                "Investigate why occupancy declined and prepare an executive brief."
            )
        )
        self.assertFalse(is_occupancy_investigation_request("Show occupancy trend"))
        self.assertFalse(is_occupancy_investigation_request("Prepare an executive brief"))

    def test_decline_trajectory_uses_observations_to_choose_period_tools(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits())
        policy = OccupancyInvestigationPolicy(months=12)
        invocations: dict[str, ToolResult] = {}
        vacancies = [
            {
                "unit": "101",
                "unit_type": "A1",
                "market_rent": 1500.0,
                "report_month": "2025-02-01",
            },
            {
                "unit": "102",
                "unit_type": "B1",
                "market_rent": 1800.0,
                "report_month": "2025-02-01",
            },
        ]
        rent_by_type = [
            {
                "unit_type": "A1",
                "unit_count": 10,
                "avg_market_rent": 1550.0,
                "report_month": "2025-02-01",
            }
        ]
        retrieval = [
            {
                "id": "chunk-1",
                "content": "Covered parking and resident amenities are advertised.",
                "metadata": {
                    "property_code": "115r",
                    "document_id": "doc-1",
                    "source_url": "https://example.test/amenities",
                },
            }
        ]

        def execute(action):
            outputs = {
                "get_occupancy_trend": DECLINING_TREND,
                "get_vacant_units": vacancies,
                "get_rent_by_unit_type": rent_by_type,
                "search_property_content": retrieval,
            }
            result = tool_result(action.tool_name, outputs[action.tool_name], len(invocations) + 1)
            invocations[action.key] = result
            return result.data

        result = loop.run(policy.decide, execute)
        report, markdown = build_occupancy_investigation_report(
            property_code="115r",
            property_name="Canfield Park",
            observations=loop.observations,
            invocations=invocations,
            loop_result=result,
            total_tool_calls=4,
        )

        self.assertEqual(
            report.trace_summary.tool_order,
            [
                "get_occupancy_trend",
                "get_vacant_units",
                "get_rent_by_unit_type",
                "search_property_content",
            ],
        )
        self.assertEqual(
            loop.actions[1].arguments["report_month"],
            "2025-02-01",
        )
        self.assertEqual(
            loop.actions[2].arguments["report_month"],
            "2025-02-01",
        )
        self.assertEqual(report.trace_summary.verification_status, "passed")
        self.assertIn("numerical_metrics_grounded", report.trace_summary.verification_checks[2])
        self.assertIn("does not prove", report.summary)
        self.assertIn("Executive Brief", markdown)
        self.assertTrue(any(item.source_type == "retrieval" for item in report.citations))

    def test_no_decline_stops_without_unnecessary_downstream_calls(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits())
        policy = OccupancyInvestigationPolicy()
        invocations: dict[str, ToolResult] = {}
        stable_trend = [
            {"report_month": "2025-01-01", "unit_occupancy_pct": 94.0},
            {"report_month": "2025-02-01", "unit_occupancy_pct": 95.0},
        ]

        def execute(action):
            result = tool_result(action.tool_name, stable_trend, 1)
            invocations[action.key] = result
            return result.data

        result = loop.run(policy.decide, execute)
        report, _ = build_occupancy_investigation_report(
            property_code="115r",
            property_name="Canfield Park",
            observations=loop.observations,
            invocations=invocations,
            loop_result=result,
            total_tool_calls=1,
        )

        self.assertEqual(result.steps, 1)
        self.assertEqual(result.reason, "no_occupancy_decline_found")
        self.assertEqual(report.trace_summary.tool_order, ["get_occupancy_trend"])
        self.assertIn("does not contain", report.summary)

    def test_verifier_rejects_unsupported_metric_and_cross_property_evidence(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits())
        policy = OccupancyInvestigationPolicy()
        invocations: dict[str, ToolResult] = {}

        # Empty vacancies causes the policy to request retrieval next.
        def evidence_executor(action):
            if action.tool_name == "get_occupancy_trend":
                data = DECLINING_TREND
            elif action.tool_name == "search_property_content":
                data = [
                    {
                        "id": "chunk-scope",
                        "content": "Resident amenities.",
                        "metadata": {
                            "property_code": "115r",
                            "source_url": "https://example.test/amenities",
                        },
                    }
                ]
            else:
                data = []
            result = tool_result(action.tool_name, data, len(invocations) + 1)
            invocations[action.key] = result
            return data

        result = loop.run(policy.decide, evidence_executor)
        report, _ = build_occupancy_investigation_report(
            property_code="115r",
            property_name="Canfield Park",
            observations=loop.observations,
            invocations=invocations,
            loop_result=result,
            total_tool_calls=3,
        )

        unsupported = report.model_copy(deep=True)
        unsupported.findings[0].metrics[0].value = 999.0
        with self.assertRaisesRegex(EvidenceVerificationError, "unsupported numerical"):
            verify_occupancy_investigation_report(unsupported, "115r")

        cross_property = report.model_copy(deep=True)
        cross_property.citations[0].property_code = "176r"
        with self.assertRaisesRegex(EvidenceVerificationError, "another property"):
            verify_occupancy_investigation_report(cross_property, "115r")

        retrieval_scope = report.model_copy(deep=True)
        retrieval_citation = next(
            citation
            for citation in retrieval_scope.citations
            if citation.source_type == "retrieval"
        )
        retrieval_citation.evidence["metadata"]["property_code"] = "176r"
        with self.assertRaisesRegex(EvidenceVerificationError, "another property"):
            verify_occupancy_investigation_report(retrieval_scope, "115r")


if __name__ == "__main__":
    unittest.main()
