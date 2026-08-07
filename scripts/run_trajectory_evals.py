#!/usr/bin/env python3
"""Run deterministic agent trajectory evaluations without external APIs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.agents.evaluation import (
    EvaluatedToolCall,
    TrajectoryExpectation,
    TrajectorySnapshot,
    evaluate_trajectory,
)
from app.agents.loop import AgentAction, BoundedAgentLoop, ExecutionLimits
from app.agents.occupancy_investigation import (
    OccupancyInvestigationPolicy,
    build_occupancy_investigation_report,
)
from app.tools.contracts import ToolResult

DEFAULT_CASES_PATH = Path("evals/trajectory_cases.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output-json")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    """Execute one fixture through the real bounded loop, policy, and verifier."""
    limits = ExecutionLimits()
    loop = BoundedAgentLoop(limits)
    policy = OccupancyInvestigationPolicy(months=12)
    outputs = dict(case["outputs"])
    invocations: dict[str, ToolResult] = {}

    def execute(action: AgentAction) -> Any:
        if action.tool_name not in outputs:
            raise AssertionError(f"fixture has no output for {action.tool_name}")
        data = outputs[action.tool_name]
        invocation = ToolResult(
            invocation_id=str(uuid5(NAMESPACE_URL, f"{case['name']}:{action.key}")),
            tool_name=action.tool_name,
            status="succeeded",
            attempt=1,
            duration_ms=1,
            query_parameters=action.arguments,
            data_timestamp=_data_timestamp(data),
            completed_at="2026-08-07T00:00:00+00:00",
            data=data,
        )
        invocations[action.key] = invocation
        return data

    loop_result = loop.run(policy.decide, execute)
    report, markdown = build_occupancy_investigation_report(
        property_code=case["property_code"],
        property_name=case["property_name"],
        observations=loop.observations,
        invocations=invocations,
        loop_result=loop_result,
        total_tool_calls=len(loop.actions),
    )
    citation_ids = [citation.citation_id for citation in report.citations]
    snapshot = TrajectorySnapshot(
        property_code=case["property_code"],
        status=loop_result.status,
        tool_calls=[
            EvaluatedToolCall(
                tool_name=action.tool_name,
                property_code=case["property_code"],
            )
            for action in loop.actions
        ],
        step_count=loop_result.steps,
        max_steps=limits.max_steps,
        max_tool_calls=limits.max_tool_calls,
        approval_requested=False,
        final_answer=markdown,
        citation_ids=citation_ids,
        evidence_ids=citation_ids,
    )
    evaluation = evaluate_trajectory(
        snapshot,
        TrajectoryExpectation.model_validate(case["expectation"]),
    )
    return {
        "name": case["name"],
        "passed": evaluation.passed,
        "tool_order": [action.tool_name for action in loop.actions],
        "stop_reason": loop_result.reason,
        "verification_checks": report.trace_summary.verification_checks,
        "checks": [check.model_dump(mode="json") for check in evaluation.checks],
    }


def _data_timestamp(data: Any) -> str:
    timestamps: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"report_month", "scraped_at"} and item is not None:
                    timestamps.append(str(item))
                elif isinstance(item, dict | list):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(data)
    return max(timestamps) if timestamps else datetime.now(UTC).isoformat()


def main() -> int:
    args = parse_args()
    results = [run_case(case) for case in load_cases(Path(args.cases))]
    passed = sum(int(result["passed"]) for result in results)
    for result in results:
        marker = "PASS" if result["passed"] else "FAIL"
        print(f"{marker} {result['name']}")
        for check in result["checks"]:
            status = "ok" if check["passed"] else "!!"
            print(f"     {status} {check['name']}: {check['details']}")
    print(f"\nTrajectory pass rate: {passed}/{len(results)}")

    report = {
        "summary": {
            "passed": passed,
            "total": len(results),
            "pass_rate": passed / len(results) if results else 1.0,
        },
        "cases": results,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report to {output_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
