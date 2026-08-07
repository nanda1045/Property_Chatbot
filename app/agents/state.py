"""Typed, serializable state for durable agent runs."""

from __future__ import annotations

from typing import Any, Literal, TypedDict
from uuid import uuid4

RunStatus = Literal[
    "created",
    "planning",
    "running",
    "waiting_for_approval",
    "verifying",
    "completed",
    "failed",
    "cancelled",
]


class AgentState(TypedDict):
    run_id: str
    conversation_id: str
    user_id: str
    property_code: str
    user_goal: str
    status: RunStatus
    current_step: int
    max_steps: int
    plan: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    pending_approval: dict[str, Any] | None
    artifacts: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    tool_call_count: int
    max_tool_calls: int
    error: dict[str, Any] | None
    final_answer: str | None


TERMINAL_RUN_STATUSES: frozenset[RunStatus] = frozenset(
    {"completed", "failed", "cancelled"}
)

ALLOWED_STATUS_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    "created": frozenset({"planning", "cancelled", "failed"}),
    "planning": frozenset({"running", "waiting_for_approval", "failed", "cancelled"}),
    "running": frozenset(
        {"waiting_for_approval", "verifying", "completed", "failed", "cancelled"}
    ),
    "waiting_for_approval": frozenset({"running", "failed", "cancelled"}),
    "verifying": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def new_agent_state(
    *,
    conversation_id: str,
    user_id: str,
    property_code: str,
    user_goal: str,
    max_steps: int = 8,
    max_tool_calls: int = 12,
    run_id: str | None = None,
) -> AgentState:
    """Create a new state with backend-owned identity, scope, and budgets."""
    if not conversation_id.strip():
        raise ValueError("conversation_id is required")
    if not user_id.strip():
        raise ValueError("user_id is required")
    if not property_code.strip():
        raise ValueError("property_code is required")
    if not user_goal.strip():
        raise ValueError("user_goal is required")
    if max_steps < 1 or max_tool_calls < 1:
        raise ValueError("execution budgets must be positive")

    return AgentState(
        run_id=run_id or str(uuid4()),
        conversation_id=conversation_id,
        user_id=user_id,
        property_code=property_code.lower(),
        user_goal=user_goal,
        status="created",
        current_step=0,
        max_steps=max_steps,
        plan=[],
        observations=[],
        pending_approval=None,
        artifacts=[],
        citations=[],
        tool_call_count=0,
        max_tool_calls=max_tool_calls,
        error=None,
        final_answer=None,
    )


def transition_agent_state(state: AgentState, status: RunStatus) -> None:
    """Apply a valid lifecycle transition in place."""
    current = state["status"]
    if status == current:
        return
    if status not in ALLOWED_STATUS_TRANSITIONS[current]:
        raise ValueError(f"invalid agent run transition: {current} -> {status}")
    state["status"] = status
