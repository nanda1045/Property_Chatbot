"""Application-facing runtime for agent workflow execution."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from app.agents.state import (
    TERMINAL_RUN_STATUSES,
    AgentState,
    new_agent_state,
    transition_agent_state,
)
from app.core.config import Settings
from app.db.mysql import MySQLDatabase
from app.memory.run_store import AgentRunStore, StepStatus
from app.schemas import ChatResponse, UIComponent
from app.services.sql_approval import execute_approved_sql


class AgentWorkflow(Protocol):
    """Behavior required by the runtime's current synchronous chat workflow."""

    def answer(
        self,
        property_code: str,
        message: str,
        model: str,
        on_token: Callable[[str], None] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> ChatResponse: ...


WorkflowFactory = Callable[[Settings], AgentWorkflow]


class RunStore(Protocol):
    def create(self, state: AgentState) -> None: ...

    def save(self, state: AgentState) -> None: ...

    def checkpoint(self, state: AgentState, transition_name: str) -> str: ...

    def load_latest_checkpoint(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> AgentState | None: ...

    def claim_approval(self, run_id: str, user_id: str, property_code: str) -> bool: ...

    def start_step(
        self,
        state: AgentState,
        step_type: str,
        input_data: dict[str, Any] | None = None,
    ) -> str: ...

    def finish_step(
        self,
        step_id: str,
        *,
        status: StepStatus,
        output_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None: ...


RunStoreFactory = Callable[[Settings], RunStore]
SqlExecutor = Callable[[Settings, str, str], tuple[str, list[dict[str, Any]]]]


class AgentRunNotFoundError(LookupError):
    """Raised when a run does not exist inside the trusted request scope."""


class AgentRunConflictError(RuntimeError):
    """Raised when a run cannot accept the requested lifecycle action."""


class AgentRuntime:
    """Stable entry point between transport code and agent orchestration."""

    def __init__(
        self,
        settings: Settings,
        workflow_factory: WorkflowFactory | None = None,
        run_store_factory: RunStoreFactory | None = None,
        sql_executor: SqlExecutor = execute_approved_sql,
    ) -> None:
        self.settings = settings
        self._workflow_factory = workflow_factory or self._default_workflow_factory
        self._run_store_factory = run_store_factory or self._default_run_store_factory
        self._sql_executor = sql_executor

    @staticmethod
    def _default_workflow_factory(settings: Settings) -> AgentWorkflow:
        from app.agents.workflow import PropertyChatWorkflow

        return PropertyChatWorkflow(settings)

    @staticmethod
    def _default_run_store_factory(settings: Settings) -> RunStore:
        return AgentRunStore(MySQLDatabase(settings))

    @staticmethod
    def _capture_workflow_execution(
        state: AgentState,
        workflow: AgentWorkflow,
    ) -> None:
        """Copy sanitized bounded-loop state into the durable run snapshot."""
        state["tool_call_count"] = int(
            getattr(workflow, "tool_call_count", state["tool_call_count"])
        )
        execution_plan = list(getattr(workflow, "execution_plan", []) or [])
        execution_observations = list(
            getattr(workflow, "execution_observations", []) or []
        )
        planner_attempts = int(getattr(workflow, "planner_attempt_count", 0))
        sql_approval_count = int(getattr(workflow, "sql_approval_count", 0))
        if state["plan"] and execution_plan:
            state["plan"][0]["actions"] = execution_plan
        if execution_plan or execution_observations or planner_attempts or sql_approval_count:
            state["observations"].append(
                {
                    "type": "bounded_agent_loop",
                    "planner_attempts": planner_attempts,
                    "sql_approval_count": sql_approval_count,
                    "steps": len(execution_observations),
                    "observations": execution_observations,
                }
            )

    def answer(
        self,
        property_code: str,
        message: str,
        model: str,
        on_token: Callable[[str], None] | None = None,
        history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> ChatResponse:
        if not conversation_id:
            raise ValueError("conversation_id is required for durable agent execution")

        state = new_agent_state(
            conversation_id=conversation_id,
            user_id=user_id or self.settings.runtime_user_id,
            property_code=property_code,
            user_goal=message,
            max_steps=self.settings.agent_max_steps,
            max_tool_calls=self.settings.agent_max_tool_calls,
        )
        run_store = self._run_store_factory(self.settings)
        run_store.create(state)
        run_store.checkpoint(state, "run_created")
        transition_agent_state(state, "planning")
        state["plan"] = [
            {
                "step": 1,
                "type": "property_chat_workflow",
                "description": "Plan and execute the property-scoped request.",
                "status": "pending",
            }
        ]
        run_store.save(state)
        run_store.checkpoint(state, "plan_created")

        step_id: str | None = None
        try:
            workflow = self._workflow_factory(self.settings)
            transition_agent_state(state, "running")
            state["plan"][0]["status"] = "running"
            step_id = run_store.start_step(
                state,
                "property_chat_workflow",
                {"property_code": state["property_code"], "model": model},
            )
            run_store.checkpoint(state, "step_started")
            response = workflow.answer(
                property_code=property_code,
                message=message,
                model=model,
                on_token=on_token,
                history=history,
            )
            self._capture_workflow_execution(state, workflow)
            response.run_id = state["run_id"]
            if response.investigation is not None:
                response.investigation.run_id = state["run_id"]
                state["artifacts"].extend(
                    artifact.model_dump(mode="json")
                    for artifact in response.investigation.artifacts
                )
                response.tool_results["occupancy_investigation"] = (
                    response.investigation.model_dump(mode="json")
                )
            state["observations"].append(
                {
                    "step": state["current_step"],
                    "tool_result_keys": sorted(response.tool_results),
                    "component_types": [component.type for component in response.components],
                }
            )
            if response.investigation is not None:
                state["citations"] = [
                    citation.model_dump(mode="json")
                    for citation in response.investigation.citations
                ]
            else:
                state["citations"] = [source.model_dump() for source in response.sources]
            pending_component = next(
                (
                    component
                    for component in response.components
                    if component.type == "sql_approval"
                ),
                None,
            )
            if pending_component is not None:
                pending_component.data["run_id"] = state["run_id"]
                state["pending_approval"] = dict(pending_component.data)
                state["plan"][0]["status"] = "waiting_for_approval"
                transition_agent_state(state, "waiting_for_approval")
            else:
                transition_agent_state(state, "verifying")
                run_store.save(state)
                run_store.checkpoint(state, "verification_started")
                state["final_answer"] = response.answer_markdown
                state["plan"][0]["status"] = "completed"
                transition_agent_state(state, "completed")

            run_store.finish_step(
                step_id,
                status="succeeded",
                output_data={
                    "component_types": [
                        component.type for component in response.components
                    ],
                    "source_count": len(response.sources),
                    "tool_result_keys": sorted(response.tool_results),
                },
            )
            run_store.save(state)
            run_store.checkpoint(state, "step_completed")
            run_store.checkpoint(
                state,
                "approval_requested"
                if state["status"] == "waiting_for_approval"
                else "run_completed",
            )
            response.run_status = state["status"]
            return response
        except Exception as error:
            error_data = {"type": type(error).__name__, "message": str(error)}
            if state["status"] not in TERMINAL_RUN_STATUSES:
                if "workflow" in locals():
                    self._capture_workflow_execution(state, workflow)
                state["error"] = error_data
                state["plan"][0]["status"] = "failed"
                transition_agent_state(state, "failed")
                try:
                    if step_id is not None:
                        run_store.finish_step(step_id, status="failed", error=error_data)
                    run_store.save(state)
                    run_store.checkpoint(state, "step_failed")
                    run_store.checkpoint(state, "run_failed")
                except Exception:
                    pass
            raise

    def resolve_sql_approval(
        self,
        *,
        run_id: str,
        property_code: str,
        approved: bool,
        conversation_id: str,
        user_id: str | None = None,
    ) -> ChatResponse:
        """Resume a checkpointed SQL run using only the backend-stored SQL draft."""
        normalized_code = property_code.lower()
        trusted_user_id = user_id or self.settings.runtime_user_id
        run_store = self._run_store_factory(self.settings)
        state = run_store.load_latest_checkpoint(
            run_id,
            trusted_user_id,
            conversation_id,
            normalized_code,
        )
        if state is None:
            raise AgentRunNotFoundError("agent run was not found")
        if state["status"] != "waiting_for_approval":
            raise AgentRunConflictError(
                f"run is {state['status']}; expected waiting_for_approval"
            )

        pending = state["pending_approval"]
        if not pending or not isinstance(pending.get("sql"), str):
            raise AgentRunConflictError("run has no valid pending SQL approval")
        pending_property_code = str(pending.get("property_code") or normalized_code).lower()
        if pending_property_code != normalized_code:
            raise AgentRunConflictError("pending SQL approval has a different property scope")
        if not run_store.claim_approval(run_id, trusted_user_id, normalized_code):
            raise AgentRunConflictError("approval was already claimed or resolved")

        resolved_at = datetime.now(UTC).isoformat()
        transition_agent_state(state, "running")
        pending["status"] = "approved" if approved else "rejected"
        pending["resolved_at"] = resolved_at
        state["observations"].append(
            {
                "type": "sql_approval",
                "decision": "approved" if approved else "rejected",
                "timestamp": resolved_at,
            }
        )

        if not approved:
            state["plan"][0]["status"] = "cancelled"
            transition_agent_state(state, "cancelled")
            run_store.save(state)
            run_store.checkpoint(state, "approval_rejected")
            return ChatResponse(
                property_code=normalized_code,
                model=str(pending.get("model") or ""),
                conversation_id=state["conversation_id"],
                run_id=run_id,
                run_status=state["status"],
                answer_markdown="The SQL query was rejected and the agent run was cancelled.",
            )

        run_store.save(state)
        run_store.checkpoint(state, "approval_received")
        state["plan"][0]["status"] = "completed"
        state["plan"].append(
            {
                "step": len(state["plan"]) + 1,
                "type": "approved_sql_execution",
                "description": "Execute the approved property-scoped SQL and verify results.",
                "status": "running",
            }
        )
        step_id = run_store.start_step(
            state,
            "approved_sql_execution",
            {"property_code": normalized_code, "approval": "approved"},
        )
        state["tool_call_count"] += 1
        run_store.save(state)
        run_store.checkpoint(state, "sql_execution_started")

        try:
            validated_sql, rows = self._sql_executor(
                self.settings,
                pending["sql"],
                normalized_code,
            )
            state["artifacts"].append(
                {
                    "artifact_id": str(uuid4()),
                    "type": "approved_sql_result",
                    "sql": validated_sql,
                    "row_count": len(rows),
                    "rows": rows,
                }
            )
            state["observations"].append(
                {
                    "step": state["current_step"],
                    "type": "approved_sql_result",
                    "row_count": len(rows),
                }
            )
            state["pending_approval"] = None
            state["plan"][-1]["status"] = "completed"
            run_store.finish_step(
                step_id,
                status="succeeded",
                output_data={"row_count": len(rows)},
            )
            run_store.save(state)
            run_store.checkpoint(state, "sql_execution_completed")

            transition_agent_state(state, "verifying")
            run_store.save(state)
            run_store.checkpoint(state, "verification_started")
            answer = (
                "I ran the approved read-only query for the selected property. "
                f"It returned **{len(rows)}** row{'s' if len(rows) != 1 else ''}."
            )
            state["final_answer"] = answer
            transition_agent_state(state, "completed")
            run_store.save(state)
            run_store.checkpoint(state, "run_completed")

            return ChatResponse(
                property_code=normalized_code,
                model=str(pending.get("model") or ""),
                conversation_id=state["conversation_id"],
                run_id=run_id,
                run_status=state["status"],
                answer_markdown=answer,
                components=[
                    UIComponent(type="table", title="Approved SQL Results", data=rows)
                ],
                sources=[],
                tool_results={
                    "approved_sql": validated_sql,
                    "question": pending.get("question") or state["user_goal"],
                    "row_count": len(rows),
                },
            )
        except Exception as error:
            error_data = {"type": type(error).__name__, "message": str(error)}
            if state["status"] not in TERMINAL_RUN_STATUSES:
                state["error"] = error_data
                state["plan"][-1]["status"] = "failed"
                transition_agent_state(state, "failed")
                try:
                    run_store.finish_step(step_id, status="failed", error=error_data)
                    run_store.save(state)
                    run_store.checkpoint(state, "sql_execution_failed")
                except Exception:
                    pass
            raise
