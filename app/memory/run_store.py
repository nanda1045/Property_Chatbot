"""MySQL-backed persistence for agent runs and execution steps."""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

from app.agents.state import AgentState, RunStatus

StepStatus = Literal["succeeded", "failed", "cancelled"]


class RunDatabase(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int: ...

    def fetch_one(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None: ...

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]: ...


def _json_dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


class AgentRunStore:
    """Persist and reload run state within trusted user/property scope."""

    def __init__(self, database: RunDatabase) -> None:
        self.database = database

    def create(self, state: AgentState) -> None:
        self.database.execute(
            """
            INSERT INTO agent_runs (
              run_id, conversation_id, user_id, property_code, user_goal, status,
              current_step, max_steps, plan_json, observations_json,
              pending_approval_json, artifacts_json, citations_json, tool_call_count,
              max_tool_calls, error_json, final_answer
            ) VALUES (
              %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            self._state_params(state),
        )

    def save(self, state: AgentState) -> None:
        affected = self.database.execute(
            """
            UPDATE agent_runs SET
              status = %s, current_step = %s, max_steps = %s, plan_json = %s,
              observations_json = %s, pending_approval_json = %s, artifacts_json = %s,
              citations_json = %s, tool_call_count = %s, max_tool_calls = %s,
              error_json = %s, final_answer = %s, version = version + 1
            WHERE run_id = %s AND user_id = %s AND property_code = %s
            """,
            (
                state["status"],
                state["current_step"],
                state["max_steps"],
                _json_dump(state["plan"]),
                _json_dump(state["observations"]),
                _json_dump(state["pending_approval"])
                if state["pending_approval"] is not None
                else None,
                _json_dump(state["artifacts"]),
                _json_dump(state["citations"]),
                state["tool_call_count"],
                state["max_tool_calls"],
                _json_dump(state["error"]) if state["error"] is not None else None,
                state["final_answer"],
                state["run_id"],
                state["user_id"],
                state["property_code"],
            ),
        )
        if affected != 1:
            raise LookupError("agent run was not found in the requested scope")

    def load(self, run_id: str, user_id: str, property_code: str) -> AgentState | None:
        row = self.database.fetch_one(
            """
            SELECT run_id, conversation_id, user_id, property_code, user_goal, status,
                   current_step, max_steps, plan_json, observations_json,
                   pending_approval_json, artifacts_json, citations_json,
                   tool_call_count, max_tool_calls, error_json, final_answer
            FROM agent_runs
            WHERE run_id = %s AND user_id = %s AND property_code = %s
            """,
            (run_id, user_id, property_code.lower()),
        )
        if row is None:
            return None
        return self._row_to_state(row)

    def start_step(
        self,
        state: AgentState,
        step_type: str,
        input_data: dict[str, Any] | None = None,
    ) -> str:
        step_id = str(uuid4())
        state["current_step"] += 1
        self.database.execute(
            """
            INSERT INTO agent_steps (
              step_id, run_id, step_number, step_type, status, input_json
            ) VALUES (%s, %s, %s, %s, 'running', %s)
            """,
            (
                step_id,
                state["run_id"],
                state["current_step"],
                step_type,
                _json_dump(input_data) if input_data is not None else None,
            ),
        )
        self.save(state)
        return step_id

    def finish_step(
        self,
        step_id: str,
        *,
        status: StepStatus,
        output_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        affected = self.database.execute(
            """
            UPDATE agent_steps SET status = %s, output_json = %s, error_json = %s,
                   completed_at = CURRENT_TIMESTAMP(6)
            WHERE step_id = %s
            """,
            (
                status,
                _json_dump(output_data) if output_data is not None else None,
                _json_dump(error) if error is not None else None,
                step_id,
            ),
        )
        if affected != 1:
            raise LookupError("agent step was not found")

    def list_steps(
        self,
        run_id: str,
        user_id: str,
        property_code: str,
    ) -> list[dict[str, Any]]:
        rows = self.database.fetch_all(
            """
            SELECT steps.step_id, steps.run_id, steps.step_number, steps.step_type,
                   steps.status, steps.input_json, steps.output_json, steps.error_json,
                   steps.started_at, steps.completed_at
            FROM agent_steps AS steps
            INNER JOIN agent_runs AS runs ON runs.run_id = steps.run_id
            WHERE steps.run_id = %s AND runs.user_id = %s AND runs.property_code = %s
            ORDER BY steps.step_number
            """,
            (run_id, user_id, property_code.lower()),
        )
        for row in rows:
            row["input"] = _json_load(row.pop("input_json", None), None)
            row["output"] = _json_load(row.pop("output_json", None), None)
            row["error"] = _json_load(row.pop("error_json", None), None)
        return rows

    @staticmethod
    def _state_params(state: AgentState) -> tuple[Any, ...]:
        return (
            state["run_id"],
            state["conversation_id"],
            state["user_id"],
            state["property_code"],
            state["user_goal"],
            state["status"],
            state["current_step"],
            state["max_steps"],
            _json_dump(state["plan"]),
            _json_dump(state["observations"]),
            _json_dump(state["pending_approval"])
            if state["pending_approval"] is not None
            else None,
            _json_dump(state["artifacts"]),
            _json_dump(state["citations"]),
            state["tool_call_count"],
            state["max_tool_calls"],
            _json_dump(state["error"]) if state["error"] is not None else None,
            state["final_answer"],
        )

    @staticmethod
    def _row_to_state(row: dict[str, Any]) -> AgentState:
        return AgentState(
            run_id=str(row["run_id"]),
            conversation_id=str(row["conversation_id"]),
            user_id=str(row["user_id"]),
            property_code=str(row["property_code"]),
            user_goal=str(row["user_goal"]),
            status=cast(RunStatus, row["status"]),
            current_step=int(row["current_step"]),
            max_steps=int(row["max_steps"]),
            plan=_json_load(row["plan_json"], []),
            observations=_json_load(row["observations_json"], []),
            pending_approval=_json_load(row["pending_approval_json"], None),
            artifacts=_json_load(row["artifacts_json"], []),
            citations=_json_load(row["citations_json"], []),
            tool_call_count=int(row["tool_call_count"]),
            max_tool_calls=int(row["max_tool_calls"]),
            error=_json_load(row["error_json"], None),
            final_answer=row["final_answer"],
        )
