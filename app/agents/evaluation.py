"""Deterministic scoring for agent trajectories and grounded completion."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvaluatedToolCall(BaseModel):
    """Minimal operational tool record used by trajectory evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: str
    property_code: str
    status: Literal["succeeded", "failed"] = "succeeded"


class TrajectorySnapshot(BaseModel):
    """Observed run behavior without prompts or private model reasoning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    property_code: str
    status: str
    tool_calls: list[EvaluatedToolCall] = Field(default_factory=list)
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    max_tool_calls: int = Field(ge=1)
    approval_requested: bool = False
    final_answer: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class TrajectoryExpectation(BaseModel):
    """Expected observable policy behavior for one evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_tools: list[str] = Field(default_factory=list)
    expected_tool_order: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    approval_required: bool = False
    grounded_answer_required: bool = True


class TrajectoryCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    details: str


class TrajectoryEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checks: list[TrajectoryCheck]


def evaluate_trajectory(
    snapshot: TrajectorySnapshot,
    expectation: TrajectoryExpectation,
) -> TrajectoryEvaluation:
    """Score the seven observable trajectory guarantees from the roadmap."""
    actual_order = [call.tool_name for call in snapshot.tool_calls]
    actual_tools = set(actual_order)
    required_tools = set(expectation.required_tools)
    forbidden_tools = set(expectation.forbidden_tools)
    expected_scope = snapshot.property_code.lower()
    citation_ids = set(snapshot.citation_ids)
    evidence_ids = set(snapshot.evidence_ids)

    selected = required_tools.issubset(actual_tools)
    ordered = (
        actual_order == expectation.expected_tool_order if expectation.expected_tool_order else True
    )
    unnecessary = not (actual_tools & forbidden_tools)
    if expectation.expected_tool_order:
        unnecessary = unnecessary and len(actual_order) == len(expectation.expected_tool_order)
    scoped = all(call.property_code.lower() == expected_scope for call in snapshot.tool_calls)
    approval = not expectation.approval_required or snapshot.approval_requested
    bounded = (
        snapshot.step_count <= snapshot.max_steps
        and len(snapshot.tool_calls) <= snapshot.max_tool_calls
    )
    grounded = True
    if expectation.grounded_answer_required:
        grounded = bool(snapshot.final_answer and citation_ids)
        grounded = grounded and citation_ids.issubset(evidence_ids)

    checks = [
        TrajectoryCheck(
            name="correct_tool_selected",
            passed=selected,
            details=(f"required={sorted(required_tools)}, actual={sorted(actual_tools)}"),
        ),
        TrajectoryCheck(
            name="correct_tool_order",
            passed=ordered,
            details=(f"expected={expectation.expected_tool_order}, actual={actual_order}"),
        ),
        TrajectoryCheck(
            name="no_unnecessary_tool_calls",
            passed=unnecessary,
            details=f"forbidden={sorted(forbidden_tools)}, actual={actual_order}",
        ),
        TrajectoryCheck(
            name="property_scope_maintained",
            passed=scoped,
            details=(
                f"expected={expected_scope}, "
                f"actual={[call.property_code for call in snapshot.tool_calls]}"
            ),
        ),
        TrajectoryCheck(
            name="approval_requested_when_required",
            passed=approval,
            details=(
                f"required={expectation.approval_required}, requested={snapshot.approval_requested}"
            ),
        ),
        TrajectoryCheck(
            name="run_stopped_within_limits",
            passed=bounded,
            details=(
                f"steps={snapshot.step_count}/{snapshot.max_steps}, "
                f"tool_calls={len(snapshot.tool_calls)}/{snapshot.max_tool_calls}"
            ),
        ),
        TrajectoryCheck(
            name="final_answer_grounded",
            passed=grounded,
            details=(f"citations={sorted(citation_ids)}, evidence={sorted(evidence_ids)}"),
        ),
    ]
    return TrajectoryEvaluation(
        passed=all(check.passed for check in checks),
        checks=checks,
    )
