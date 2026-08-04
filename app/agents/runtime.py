"""Application-facing runtime for agent workflow execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from app.agents.state import (
    TERMINAL_RUN_STATUSES,
    AgentState,
    new_agent_state,
    transition_agent_state,
)
from app.core.config import Settings
from app.db.mysql import MySQLDatabase
from app.memory.run_store import AgentRunStore, StepStatus
from app.schemas import ChatResponse


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


class AgentRuntime:
    """Stable entry point between transport code and agent orchestration."""

    def __init__(
        self,
        settings: Settings,
        workflow_factory: WorkflowFactory | None = None,
        run_store_factory: RunStoreFactory | None = None,
    ) -> None:
        self.settings = settings
        self._workflow_factory = workflow_factory or self._default_workflow_factory
        self._run_store_factory = run_store_factory or self._default_run_store_factory

    @staticmethod
    def _default_workflow_factory(settings: Settings) -> AgentWorkflow:
        from app.agents.workflow import PropertyChatWorkflow

        return PropertyChatWorkflow(settings)

    @staticmethod
    def _default_run_store_factory(settings: Settings) -> RunStore:
        return AgentRunStore(MySQLDatabase(settings))

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
        )
        run_store = self._run_store_factory(self.settings)
        run_store.create(state)
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
            response = workflow.answer(
                property_code=property_code,
                message=message,
                model=model,
                on_token=on_token,
                history=history,
            )
            response.run_id = state["run_id"]
            state["observations"].append(
                {
                    "step": state["current_step"],
                    "tool_result_keys": sorted(response.tool_results),
                    "component_types": [component.type for component in response.components],
                }
            )
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
            response.run_status = state["status"]
            return response
        except Exception as error:
            error_data = {"type": type(error).__name__, "message": str(error)}
            if state["status"] not in TERMINAL_RUN_STATUSES:
                state["error"] = error_data
                state["plan"][0]["status"] = "failed"
                transition_agent_state(state, "failed")
                try:
                    if step_id is not None:
                        run_store.finish_step(step_id, status="failed", error=error_data)
                    run_store.save(state)
                except Exception:
                    pass
            raise
