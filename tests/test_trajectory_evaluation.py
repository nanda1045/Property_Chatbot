from __future__ import annotations

import unittest
from pathlib import Path

from app.agents.evaluation import (
    EvaluatedToolCall,
    TrajectoryExpectation,
    TrajectorySnapshot,
    evaluate_trajectory,
)
from scripts.run_trajectory_evals import load_cases, run_case


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_fixture_trajectories_pass_all_roadmap_checks(self) -> None:
        cases = load_cases(Path("evals/trajectory_cases.json"))
        results = [run_case(case) for case in cases]

        self.assertTrue(all(result["passed"] for result in results))
        self.assertEqual(
            {check["name"] for check in results[0]["checks"]},
            {
                "correct_tool_selected",
                "correct_tool_order",
                "no_unnecessary_tool_calls",
                "property_scope_maintained",
                "approval_requested_when_required",
                "run_stopped_within_limits",
                "final_answer_grounded",
            },
        )
        self.assertEqual(
            results[1]["tool_order"],
            ["get_occupancy_trend"],
        )

    def test_evaluator_reports_each_trajectory_violation(self) -> None:
        snapshot = TrajectorySnapshot(
            property_code="115r",
            status="completed",
            tool_calls=[
                EvaluatedToolCall(
                    tool_name="unnecessary_tool",
                    property_code="176r",
                ),
                EvaluatedToolCall(
                    tool_name="required_tool",
                    property_code="176r",
                ),
            ],
            step_count=3,
            max_steps=2,
            max_tool_calls=1,
            approval_requested=False,
            final_answer="Unsupported answer",
            citation_ids=["missing-citation"],
            evidence_ids=[],
        )
        result = evaluate_trajectory(
            snapshot,
            TrajectoryExpectation(
                required_tools=["required_tool", "missing_tool"],
                expected_tool_order=["required_tool"],
                forbidden_tools=["unnecessary_tool"],
                approval_required=True,
                grounded_answer_required=True,
            ),
        )

        failed = {check.name for check in result.checks if not check.passed}
        self.assertFalse(result.passed)
        self.assertEqual(
            failed,
            {
                "correct_tool_selected",
                "correct_tool_order",
                "no_unnecessary_tool_calls",
                "property_scope_maintained",
                "approval_requested_when_required",
                "run_stopped_within_limits",
                "final_answer_grounded",
            },
        )


if __name__ == "__main__":
    unittest.main()
