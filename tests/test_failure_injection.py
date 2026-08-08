from __future__ import annotations

import hashlib
import json
import time
import unittest
from typing import Any

from pydantic import BaseModel, ConfigDict, RootModel

from app.agents.evaluation import (
    EvaluatedToolCall,
    TrajectoryExpectation,
    TrajectorySnapshot,
    evaluate_trajectory,
)
from app.agents.occupancy_investigation import (
    EvidenceVerificationError,
    verify_occupancy_investigation_report,
)
from app.agents.policies import property_scope_conflict
from app.schemas import OccupancyInvestigationReport
from app.tools.contracts import (
    ToolSpec,
    TransientToolError,
    TrustedToolContext,
)
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class QueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class RetrievalOutput(RootModel[list[dict[str, Any]]]):
    pass


def injected_executor(
    handler,
    *,
    max_attempts: int = 1,
    timeout_seconds: float = 1.0,
    output_model: type[BaseModel] = QueryOutput,
) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="injected_tool",
            description="Tool used for deterministic failure injection.",
            input_model=QueryInput,
            output_model=output_model,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        ),
        handler,
    )
    return ToolExecutor(
        registry,
        sleep=lambda _delay: None,
        random_value=lambda: 0.0,
    )


class FailureInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TrustedToolContext(
            property_code="115r",
            user_id="user-1",
            roles=("PropertyManager",),
            allowed_property_codes=("*",),
        )

    def test_tool_timeout_is_bounded_and_structured(self) -> None:
        def slow_handler(_input: QueryInput, _context: TrustedToolContext) -> dict:
            time.sleep(0.03)
            return {"value": 1}

        result = injected_executor(
            slow_handler,
            timeout_seconds=0.002,
        ).execute("injected_tool", {"value": 1}, self.context)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "timeout")
        self.assertEqual(result.attempt, 1)

    def test_malformed_tool_output_is_rejected(self) -> None:
        result = injected_executor(lambda _input, _context: {"unexpected": 1}).execute(
            "injected_tool", {"value": 1}, self.context
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "output_validation")

    def test_temporary_database_failure_retries_then_recovers(self) -> None:
        calls = 0

        def flaky_database(_input: QueryInput, _context: TrustedToolContext) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("database connection dropped")
            return {"value": 7}

        result = injected_executor(
            flaky_database,
            max_attempts=3,
        ).execute("injected_tool", {"value": 7}, self.context)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.attempt, 2)
        self.assertEqual(calls, 2)

    def test_empty_retrieval_cannot_pass_grounding(self) -> None:
        result = injected_executor(
            lambda _input, _context: [],
            output_model=RetrievalOutput,
        ).execute("injected_tool", {"value": 1}, self.context)
        evaluation = evaluate_trajectory(
            TrajectorySnapshot(
                property_code="115r",
                status="completed",
                tool_calls=[
                    EvaluatedToolCall(
                        tool_name="injected_tool",
                        property_code="115r",
                    )
                ],
                step_count=1,
                max_steps=2,
                max_tool_calls=2,
                final_answer="A factual claim with no evidence.",
            ),
            TrajectoryExpectation(
                required_tools=["injected_tool"],
                expected_tool_order=["injected_tool"],
            ),
        )

        self.assertEqual(result.data, [])
        grounded = next(
            check for check in evaluation.checks if check.name == "final_answer_grounded"
        )
        self.assertFalse(grounded.passed)

    def test_invalid_model_tool_arguments_never_reach_handler(self) -> None:
        calls = 0

        def handler(_input: QueryInput, _context: TrustedToolContext) -> dict:
            nonlocal calls
            calls += 1
            return {"value": 1}

        result = injected_executor(handler).execute(
            "injected_tool",
            {"value": "not-an-integer", "property_code": "176r"},
            self.context,
        )

        self.assertEqual(result.error.type, "input_validation")
        self.assertEqual(calls, 0)

    def test_duplicate_completion_is_idempotent(self) -> None:
        calls = 0

        def handler(tool_input: QueryInput, _context: TrustedToolContext) -> dict:
            nonlocal calls
            calls += 1
            return {"value": tool_input.value}

        executor = injected_executor(handler)
        first = executor.execute("injected_tool", {"value": 4}, self.context)
        duplicate = executor.execute("injected_tool", {"value": 4}, self.context)

        self.assertEqual(calls, 1)
        self.assertFalse(first.cached)
        self.assertTrue(duplicate.cached)
        self.assertEqual(first.invocation_id, duplicate.invocation_id)
        self.assertEqual(executor.budget.call_count, 1)

    def test_cross_property_request_is_rejected_before_data_access(self) -> None:
        conflict = property_scope_conflict(
            "Show occupancy for property 176r",
            {"property_code": "115r", "property_name": "Canfield Park"},
            [
                {"property_code": "115r", "property_name": "Canfield Park"},
                {"property_code": "176r", "property_name": "Alexander at Patroon"},
            ],
        )

        self.assertEqual(conflict["property_code"], "176r")

    def test_missing_citation_evidence_fails_verification(self) -> None:
        evidence = {"rows": [{"unit_occupancy_pct": 91.0}]}
        serialized = json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
        )
        report = OccupancyInvestigationReport.model_validate(
            {
                "summary": "Grounded summary.",
                "findings": [
                    {
                        "finding_id": "missing",
                        "title": "Missing evidence",
                        "narrative": "This finding references evidence not returned.",
                        "citation_ids": ["missing-citation"],
                    }
                ],
                "citations": [
                    {
                        "citation_id": "stored-citation",
                        "property_code": "115r",
                        "source_type": "structured_tool",
                        "source_name": "get_latest_property_kpis",
                        "tool_invocation_id": "tool-1",
                        "content_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
                        "retrieved_at": "2026-08-07T00:00:00+00:00",
                        "evidence": evidence,
                    }
                ],
                "artifacts": [],
                "trace_summary": {
                    "steps": 1,
                    "tool_calls": 1,
                    "duration_ms": 1,
                    "tool_order": ["get_latest_property_kpis"],
                    "stop_reason": "completed",
                    "verification_status": "passed",
                    "verification_checks": [],
                },
            }
        )

        with self.assertRaisesRegex(EvidenceVerificationError, "missing evidence"):
            verify_occupancy_investigation_report(report, "115r")

    def test_repeated_transient_failure_stops_at_retry_limit(self) -> None:
        calls = 0

        def unavailable(_input: QueryInput, _context: TrustedToolContext) -> dict:
            nonlocal calls
            calls += 1
            raise TransientToolError("database remains unavailable")

        result = injected_executor(
            unavailable,
            max_attempts=3,
        ).execute("injected_tool", {"value": 1}, self.context)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "transient_failure")
        self.assertEqual(result.attempt, 3)
        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
