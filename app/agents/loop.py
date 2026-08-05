"""Bounded plan-execute-observe-decide control for agent workflows."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionLimits(BaseModel):
    """Hard limits applied to one agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = Field(default=8, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    max_planner_retries: int = Field(default=2, ge=0)
    max_sql_approvals: int = Field(default=1, ge=0)
    max_run_seconds: float = Field(default=60.0, gt=0)


class AgentAction(BaseModel):
    """One server-validated tool action selected by the workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        """Return a stable signature used to reject repeated identical actions."""
        return json.dumps(
            {"tool_name": self.tool_name, "arguments": self.arguments},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


class AgentObservation(BaseModel):
    """Sanitized result visible to the workflow's next decision."""

    model_config = ConfigDict(extra="forbid")

    step: int
    action_key: str
    tool_name: str
    status: Literal["succeeded", "failed"]
    duration_ms: int = 0
    data: Any = None
    error: dict[str, Any] | None = None


class LoopDecision(BaseModel):
    """Choose one next action or explicitly stop the loop."""

    model_config = ConfigDict(extra="forbid")

    action: AgentAction | None = None
    complete: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> LoopDecision:
        if self.complete == (self.action is not None):
            raise ValueError("decision must contain one action or mark the loop complete")
        return self


class AgentLoopResult(BaseModel):
    """Serializable outcome of a completed bounded loop."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed"] = "completed"
    reason: str | None = None
    steps: int
    planner_attempts: int
    sql_approval_count: int
    duration_ms: int
    actions: list[AgentAction]
    observations: list[AgentObservation]


class AgentLoopError(RuntimeError):
    """Base error for a safely terminated agent loop."""


class AgentLimitExceeded(AgentLoopError):
    def __init__(self, limit_name: str, maximum: int | float) -> None:
        self.limit_name = limit_name
        self.maximum = maximum
        super().__init__(f"Agent {limit_name} limit was exceeded (maximum {maximum}).")


class RepeatedActionError(AgentLoopError):
    def __init__(self, action: AgentAction) -> None:
        self.action = action
        super().__init__(
            f"Repeated identical tool action was rejected: {action.tool_name}."
        )


DecisionFunction = Callable[[tuple[AgentObservation, ...]], LoopDecision]
ActionExecutor = Callable[[AgentAction], Any]
Clock = Callable[[], float]


class BoundedAgentLoop:
    """Execute one action at a time and bound every continuation decision."""

    def __init__(
        self,
        limits: ExecutionLimits,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        self.limits = limits
        self.clock = clock
        self._started_at = clock()
        self._actions: list[AgentAction] = []
        self._observations: list[AgentObservation] = []
        self._seen_action_signatures: set[str] = set()
        self._planner_attempts = 0
        self._sql_approval_count = 0

    @property
    def step_count(self) -> int:
        return len(self._observations)

    @property
    def planner_attempts(self) -> int:
        return self._planner_attempts

    @property
    def sql_approval_count(self) -> int:
        return self._sql_approval_count

    @property
    def actions(self) -> tuple[AgentAction, ...]:
        return tuple(self._actions)

    @property
    def observations(self) -> tuple[AgentObservation, ...]:
        return tuple(self._observations)

    def plan_with_retries(self, plan_once: Callable[[], Any | None]) -> Any | None:
        """Retry invalid planner responses within the configured planner budget."""
        maximum_attempts = self.limits.max_planner_retries + 1
        for _ in range(maximum_attempts):
            self.ensure_within_duration()
            self._planner_attempts += 1
            try:
                plan = plan_once()
            except Exception:
                plan = None
            self.ensure_within_duration()
            if plan is not None:
                return plan
        return None

    def request_sql_approval(self) -> None:
        """Consume the run's bounded SQL-approval allowance."""
        self.ensure_within_duration()
        if self._sql_approval_count >= self.limits.max_sql_approvals:
            raise AgentLimitExceeded(
                "SQL approval",
                self.limits.max_sql_approvals,
            )
        self._sql_approval_count += 1

    def run(
        self,
        decide: DecisionFunction,
        execute: ActionExecutor,
    ) -> AgentLoopResult:
        """Run until the decision function declares completion or a bound is hit."""
        while True:
            self.ensure_within_duration()
            decision = decide(tuple(self._observations))
            self.ensure_within_duration()

            if decision.complete:
                return AgentLoopResult(
                    reason=decision.reason,
                    steps=self.step_count,
                    planner_attempts=self._planner_attempts,
                    sql_approval_count=self._sql_approval_count,
                    duration_ms=self.elapsed_ms(),
                    actions=list(self._actions),
                    observations=list(self._observations),
                )

            action = decision.action
            if action is None:
                raise AssertionError("validated loop decision did not contain an action")
            if self.step_count >= self.limits.max_steps:
                raise AgentLimitExceeded("step", self.limits.max_steps)
            if len(self._actions) >= self.limits.max_tool_calls:
                raise AgentLimitExceeded("tool call", self.limits.max_tool_calls)

            signature = action.signature()
            if signature in self._seen_action_signatures:
                raise RepeatedActionError(action)

            self._seen_action_signatures.add(signature)
            self._actions.append(action)
            step = self.step_count + 1
            action_started = self.clock()

            try:
                data = execute(action)
                self.ensure_within_duration()
            except Exception as error:
                observation = AgentObservation(
                    step=step,
                    action_key=action.key,
                    tool_name=action.tool_name,
                    status="failed",
                    duration_ms=self._duration_ms(action_started),
                    error={"type": type(error).__name__, "message": str(error)},
                )
                self._observations.append(observation)
                raise

            self._observations.append(
                AgentObservation(
                    step=step,
                    action_key=action.key,
                    tool_name=action.tool_name,
                    status="succeeded",
                    duration_ms=self._duration_ms(action_started),
                    data=data,
                )
            )

    def ensure_within_duration(self) -> None:
        if self.clock() - self._started_at > self.limits.max_run_seconds:
            raise AgentLimitExceeded(
                "run duration",
                self.limits.max_run_seconds,
            )

    def elapsed_ms(self) -> int:
        return self._duration_ms(self._started_at)

    def _duration_ms(self, started_at: float) -> int:
        return max(0, round((self.clock() - started_at) * 1000))
