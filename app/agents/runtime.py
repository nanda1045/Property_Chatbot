"""Application-facing runtime for agent workflow execution."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from app.agents.cancellation import (
    AgentRunCancelledError,
    CancellationCheck,
    raise_if_cancelled,
)
from app.agents.state import (
    TERMINAL_RUN_STATUSES,
    AgentState,
    new_agent_state,
    transition_agent_state,
)
from app.core.auth import local_authenticated_user
from app.core.authorization import (
    AuthorizationContext,
    ToolPermission,
    authorize_permission,
    authorize_sql_approval,
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

    def record_event(
        self,
        state: AgentState,
        event_type: str,
        **kwargs: Any,
    ) -> str: ...

    def load_scoped(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> AgentState | None: ...

    def get_run_detail(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> dict[str, Any] | None: ...

    def list_steps_scoped(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> list[dict[str, Any]]: ...

    def list_events(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
        after_sequence: int = 0,
    ) -> list[dict[str, Any]]: ...

    def list_citations_scoped(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> list[dict[str, Any]]: ...

    def record_tool_invocation_event(
        self,
        state: AgentState,
        step_id: str,
        event_type: str,
        tool_name: str,
        attempt: int,
        duration_ms: int | None,
        error_type: str | None,
        payload: dict[str, Any],
    ) -> None: ...

    def claim_cancellation(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> bool: ...

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
        if (
            execution_plan
            or execution_observations
            or planner_attempts
            or sql_approval_count
        ):
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
        authorization_context: AuthorizationContext | None = None,
        run_id: str | None = None,
        cancellation_requested: CancellationCheck | None = None,
    ) -> ChatResponse:
        if not conversation_id:
            raise ValueError("conversation_id is required for durable agent execution")
        run_started = perf_counter()

        trusted_authorization = authorization_context or AuthorizationContext.from_settings(
            local_authenticated_user(self.settings).model_copy(
                update={"user_id": user_id or self.settings.local_auth_user_id}
            ),
            self.settings,
            property_code=property_code,
        )
        trusted_authorization = trusted_authorization.for_property(property_code)
        authorize_permission(trusted_authorization, ToolPermission.CHAT)

        state = new_agent_state(
            conversation_id=conversation_id,
            user_id=trusted_authorization.user.user_id,
            property_code=property_code,
            user_goal=message,
            max_steps=self.settings.agent_max_steps,
            max_tool_calls=self.settings.agent_max_tool_calls,
            run_id=run_id,
        )
        run_store = self._run_store_factory(self.settings)
        run_store.create(state)
        run_store.record_event(
            state,
            "run_created",
            duration_ms=0,
            payload={
                "max_steps": state["max_steps"],
                "max_tool_calls": state["max_tool_calls"],
            },
        )
        run_store.record_event(
            state,
            "AUTHENTICATED",
            duration_ms=0,
            payload={
                "user_id": trusted_authorization.user.user_id,
                "role": (
                    trusted_authorization.primary_role.value
                    if trusted_authorization.primary_role is not None
                    else None
                ),
                "property_code": state["property_code"],
                "outcome": "authenticated",
            },
        )
        run_store.record_event(
            state,
            "AUTHORIZATION_ALLOWED",
            duration_ms=0,
            payload={
                "user_id": trusted_authorization.user.user_id,
                "role": (
                    trusted_authorization.primary_role.value
                    if trusted_authorization.primary_role is not None
                    else None
                ),
                "property_code": state["property_code"],
                "permission": ToolPermission.CHAT.value,
                "outcome": "allowed",
            },
        )
        run_store.checkpoint(state, "run_created")
        transition_agent_state(state, "planning")
        run_store.record_event(state, "planning_started")
        state["plan"] = [
            {
                "step": 1,
                "type": "property_chat_workflow",
                "description": "Plan and execute the property-scoped request.",
                "status": "pending",
            }
        ]
        run_store.save(state)
        run_store.record_event(
            state,
            "plan_created",
            payload={
                "steps": [
                    {"step": item["step"], "type": item["type"]}
                    for item in state["plan"]
                ]
            },
        )
        run_store.checkpoint(state, "plan_created")

        step_id: str | None = None
        try:
            raise_if_cancelled(cancellation_requested)
            workflow = self._workflow_factory(self.settings)
            transition_agent_state(state, "running")
            state["plan"][0]["status"] = "running"
            step_id = run_store.start_step(
                state,
                "property_chat_workflow",
                {"property_code": state["property_code"], "model": model},
            )
            run_store.record_event(
                state,
                "step_started",
                step_id=step_id,
                payload={"step_type": "property_chat_workflow", "model": model},
            )
            run_store.checkpoint(state, "step_started")
            bind_run_context = getattr(workflow, "bind_run_context", None)
            if callable(bind_run_context):
                bind_run_context(
                    run_id=state["run_id"],
                    conversation_id=state["conversation_id"],
                    event_sink=lambda event: self._record_tool_event(
                        run_store,
                        state,
                        step_id,
                        event,
                    ),
                    cancellation_check=cancellation_requested,
                    authorization_context=trusted_authorization,
                )

            def publish_token(token: str) -> None:
                raise_if_cancelled(cancellation_requested)
                if on_token is not None:
                    on_token(token)

            response = workflow.answer(
                property_code=property_code,
                message=message,
                model=model,
                on_token=publish_token if on_token is not None else None,
                history=history,
            )
            raise_if_cancelled(cancellation_requested)
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
                state["citations"] = [
                    citation.model_dump(mode="json")
                    for citation in response.investigation.citations
                ]
            else:
                workflow_citations = list(
                    getattr(workflow, "citation_evidence", []) or []
                )
                state["citations"] = workflow_citations or [
                    source.model_dump(mode="json") for source in response.sources
                ]
            response.citation_ids = [
                str(citation["citation_id"])
                for citation in state["citations"]
                if citation.get("citation_id")
            ]
            structured_content = {
                "tool_results": {
                    key: value
                    for key, value in response.tool_results.items()
                    if key != "occupancy_investigation"
                },
                "components": [
                    component.model_dump(mode="json")
                    for component in response.components
                ],
                "sources": [
                    source.model_dump(mode="json") for source in response.sources
                ],
            }
            if any(structured_content.values()):
                state["artifacts"].append(
                    {
                        "artifact_id": str(uuid4()),
                        "type": "structured_output",
                        "name": "Property chat structured output",
                        "content": structured_content,
                    }
                )
            state["observations"].append(
                {
                    "step": state["current_step"],
                    "tool_result_keys": sorted(response.tool_results),
                    "component_types": [
                        component.type for component in response.components
                    ],
                }
            )
            pending_component = next(
                (
                    component
                    for component in response.components
                    if component.type == "sql_approval"
                ),
                None,
            )
            if pending_component is not None:
                current_role = (
                    trusted_authorization.primary_role.value
                    if trusted_authorization.primary_role is not None
                    else None
                )
                try:
                    authorize_permission(
                        trusted_authorization,
                        ToolPermission.CUSTOM_ANALYTICS,
                    )
                except PermissionError:
                    run_store.record_event(
                        state,
                        "AUTHORIZATION_DENIED",
                        step_id=step_id,
                        payload={
                            "user_id": trusted_authorization.user.user_id,
                            "role": current_role,
                            "property_code": state["property_code"],
                            "permission": ToolPermission.CUSTOM_ANALYTICS.value,
                            "outcome": "denied",
                        },
                    )
                    raise
                run_store.record_event(
                    state,
                    "AUTHORIZATION_ALLOWED",
                    step_id=step_id,
                    payload={
                        "user_id": trusted_authorization.user.user_id,
                        "role": current_role,
                        "property_code": state["property_code"],
                        "permission": ToolPermission.CUSTOM_ANALYTICS.value,
                        "outcome": "allowed",
                    },
                )
                pending_component.data["run_id"] = state["run_id"]
                pending_component.data.update(
                    {
                        "requested_action": pending_component.data.get("question")
                        or state["user_goal"],
                        "risk_level": "Privileged read-only SQL",
                        "required_permission": ToolPermission.SQL_APPROVE.value,
                        "required_role": "PropertyManager",
                        "current_role": current_role,
                        "property_code": state["property_code"],
                        "human_approval_required": True,
                    }
                )
                server_stored_proposal = dict(pending_component.data)
                state["artifacts"].append(
                    {
                        "artifact_id": str(uuid4()),
                        "type": "generated_sql",
                        "name": "Generated SQL approval proposal",
                        "content": server_stored_proposal,
                    }
                )
                run_store.record_event(
                    state,
                    "approval_requested",
                    step_id=step_id,
                    payload={
                        "approval_type": "sql",
                        "status": "authorization_check",
                        "property_code": state["property_code"],
                        "permission": ToolPermission.SQL_APPROVE.value,
                    },
                )
                try:
                    authorize_sql_approval(trusted_authorization)
                except PermissionError:
                    run_store.record_event(
                        state,
                        "SQL_APPROVAL_DENIED",
                        step_id=step_id,
                        payload={
                            "user_id": trusted_authorization.user.user_id,
                            "role": current_role,
                            "property_code": state["property_code"],
                            "permission": ToolPermission.SQL_APPROVE.value,
                            "outcome": "denied",
                            "authorization_phase": "approval_request",
                        },
                    )
                    pending_component.data.update(
                        {
                            "authorization": "denied",
                            "authorization_message": (
                                "This action requires PropertyManager permission."
                            ),
                            "status": "authorization_denied",
                            "executable": False,
                        }
                    )
                    pending_component.data.pop("sql", None)
                    pending_component.data.pop("parameters", None)
                    response.tool_results.pop("sql_draft", None)
                    response.answer_markdown += (
                        "\n\n**This action requires PropertyManager permission.** "
                        "The SQL draft passed safety validation, but it was not exposed "
                        "for approval or executed."
                    )
                    state["observations"].append(
                        {
                            "type": "authorization_decision",
                            "permission": ToolPermission.SQL_APPROVE.value,
                            "decision": "denied",
                            "role": current_role,
                        }
                    )
                    state["plan"][0]["status"] = "completed"
                    state["final_answer"] = response.answer_markdown
                    transition_agent_state(state, "completed")
                else:
                    run_store.record_event(
                        state,
                        "SQL_APPROVAL_AUTHORIZED",
                        step_id=step_id,
                        payload={
                            "user_id": trusted_authorization.user.user_id,
                            "role": current_role,
                            "property_code": state["property_code"],
                            "permission": ToolPermission.SQL_APPROVE.value,
                            "outcome": "allowed",
                            "authorization_phase": "approval_request",
                        },
                    )
                    pending_component.data.update(
                        {
                            "authorization": "allowed",
                            "authorization_message": "Human approval is required before execution.",
                            "status": "waiting_for_approval",
                            "executable": True,
                        }
                    )
                    state["pending_approval"] = dict(pending_component.data)
                    state["plan"][0]["status"] = "waiting_for_approval"
                    transition_agent_state(state, "waiting_for_approval")
            else:
                transition_agent_state(state, "verifying")
                run_store.save(state)
                run_store.record_event(
                    state,
                    "verification_started",
                    step_id=step_id,
                )
                run_store.checkpoint(state, "verification_started")
                raise_if_cancelled(cancellation_requested)
                if (
                    response.investigation is not None
                    and response.investigation.trace_summary.verification_status
                    == "failed"
                ):
                    raise RuntimeError("investigation evidence verification failed")
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
            raise_if_cancelled(cancellation_requested)
            run_store.save(state)
            run_store.checkpoint(state, "step_completed")
            run_store.checkpoint(
                state,
                "approval_requested"
                if state["status"] == "waiting_for_approval"
                else "run_completed",
            )
            if state["status"] == "completed":
                run_store.record_event(
                    state,
                    "run_completed",
                    step_id=step_id,
                    duration_ms=self._duration_ms(run_started),
                    payload={
                        "steps": state["current_step"],
                        "tool_calls": state["tool_call_count"],
                    },
                )
            response.run_status = state["status"]
            return response
        except AgentRunCancelledError:
            self._persist_interrupted_cancellation(run_store, state, step_id)
            raise
        except Exception as error:
            error_data = {"type": type(error).__name__, "message": str(error)}
            if state["status"] not in TERMINAL_RUN_STATUSES:
                verification_was_active = state["status"] == "verifying"
                if "workflow" in locals():
                    self._capture_workflow_execution(state, workflow)
                state["error"] = error_data
                state["plan"][0]["status"] = "failed"
                transition_agent_state(state, "failed")
                try:
                    if step_id is not None:
                        run_store.finish_step(
                            step_id, status="failed", error=error_data
                        )
                    run_store.save(state)
                    if verification_was_active:
                        run_store.record_event(
                            state,
                            "verification_failed",
                            step_id=step_id,
                            error_type=error_data["type"],
                            payload={"message": error_data["message"]},
                        )
                    run_store.record_event(
                        state,
                        "run_failed",
                        step_id=step_id,
                        duration_ms=self._duration_ms(run_started),
                        error_type=error_data["type"],
                        payload={"message": error_data["message"]},
                    )
                    run_store.checkpoint(state, "step_failed")
                    run_store.checkpoint(state, "run_failed")
                except Exception:
                    pass
            raise

    @staticmethod
    def _persist_interrupted_cancellation(
        run_store: RunStore,
        state: AgentState,
        step_id: str | None,
    ) -> None:
        """Make cooperative transport cancellation durable exactly once."""
        latest = run_store.load_scoped(
            state["run_id"],
            state["user_id"],
            state["conversation_id"],
            state["property_code"],
        )
        already_cancelled = latest is not None and latest["status"] == "cancelled"
        claimed = False
        if not already_cancelled:
            claimed = run_store.claim_cancellation(
                state["run_id"],
                state["user_id"],
                state["conversation_id"],
                state["property_code"],
            )
        if not already_cancelled and not claimed:
            return

        cancelled_state = latest if latest is not None else state
        for plan_step in cancelled_state["plan"]:
            if plan_step.get("status") in {
                "pending",
                "running",
                "waiting_for_approval",
            }:
                plan_step["status"] = "cancelled"
        if cancelled_state["status"] != "cancelled":
            transition_agent_state(cancelled_state, "cancelled")
        if step_id is not None:
            try:
                run_store.finish_step(step_id, status="cancelled")
            except LookupError:
                pass
        run_store.save(cancelled_state)
        if claimed:
            run_store.record_event(
                cancelled_state,
                "run_cancelled",
                step_id=step_id,
                payload={"reason": "client_disconnected"},
            )
            run_store.checkpoint(cancelled_state, "run_cancelled")

    def resolve_sql_approval(
        self,
        *,
        run_id: str,
        property_code: str,
        approved: bool,
        conversation_id: str,
        user_id: str | None = None,
        authorization_context: AuthorizationContext | None = None,
    ) -> ChatResponse:
        """Resume a checkpointed SQL run using only the backend-stored SQL draft."""
        resumed_at = perf_counter()
        normalized_code = property_code.lower()
        trusted_authorization = authorization_context or AuthorizationContext.from_settings(
            local_authenticated_user(self.settings).model_copy(
                update={"user_id": user_id or self.settings.local_auth_user_id}
            ),
            self.settings,
            property_code=normalized_code,
        )
        trusted_authorization = trusted_authorization.for_property(normalized_code)
        trusted_user_id = trusted_authorization.user.user_id
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

        try:
            authorize_sql_approval(trusted_authorization)
        except PermissionError:
            run_store.record_event(
                state,
                "SQL_APPROVAL_DENIED",
                payload={
                    "user_id": trusted_user_id,
                    "role": (
                        trusted_authorization.primary_role.value
                        if trusted_authorization.primary_role is not None
                        else None
                    ),
                    "property_code": normalized_code,
                    "permission": ToolPermission.SQL_APPROVE.value,
                    "outcome": "denied",
                    "authorization_phase": "approval_decision",
                },
            )
            raise
        run_store.record_event(
            state,
            "SQL_APPROVAL_AUTHORIZED",
            payload={
                "user_id": trusted_user_id,
                "role": (
                    trusted_authorization.primary_role.value
                    if trusted_authorization.primary_role is not None
                    else None
                ),
                "property_code": normalized_code,
                "permission": ToolPermission.SQL_APPROVE.value,
                "outcome": "allowed",
                "authorization_phase": "approval_decision",
            },
        )

        pending = state["pending_approval"]
        if not pending or not isinstance(pending.get("sql"), str):
            raise AgentRunConflictError("run has no valid pending SQL approval")
        pending_property_code = str(
            pending.get("property_code") or normalized_code
        ).lower()
        if pending_property_code != normalized_code:
            raise AgentRunConflictError(
                "pending SQL approval has a different property scope"
            )
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
        run_store.record_event(
            state,
            "approval_received",
            payload={
                "approval_type": "sql",
                "decision": "approved" if approved else "rejected",
                "user_id": trusted_user_id,
                "role": (
                    trusted_authorization.primary_role.value
                    if trusted_authorization.primary_role is not None
                    else None
                ),
                "property_code": normalized_code,
                "permission": ToolPermission.SQL_APPROVE.value,
                "outcome": "allowed",
            },
        )

        if not approved:
            state["plan"][0]["status"] = "cancelled"
            transition_agent_state(state, "cancelled")
            run_store.save(state)
            run_store.record_event(
                state,
                "run_cancelled",
                duration_ms=self._duration_ms(resumed_at),
                payload={"reason": "sql_approval_rejected"},
            )
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
        run_store.record_event(
            state,
            "step_started",
            step_id=step_id,
            payload={"step_type": "approved_sql_execution"},
        )
        state["tool_call_count"] += 1
        run_store.save(state)
        run_store.checkpoint(state, "sql_execution_started")

        try:
            sql_started = perf_counter()
            run_store.record_event(
                state,
                "tool_started",
                step_id=step_id,
                tool_name="execute_approved_sql",
                attempt=1,
                payload={"approval": "approved", "property_code": normalized_code},
            )
            try:
                validated_sql, rows = self._sql_executor(
                    self.settings,
                    pending["sql"],
                    normalized_code,
                )
            except Exception as sql_error:
                run_store.record_event(
                    state,
                    "tool_failed",
                    step_id=step_id,
                    tool_name="execute_approved_sql",
                    attempt=1,
                    duration_ms=self._duration_ms(sql_started),
                    error_type=type(sql_error).__name__,
                    payload={"message": str(sql_error)},
                )
                raise
            run_store.record_event(
                state,
                "tool_succeeded",
                step_id=step_id,
                tool_name="execute_approved_sql",
                attempt=1,
                duration_ms=self._duration_ms(sql_started),
                payload={"output_summary": {"type": "rows", "row_count": len(rows)}},
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
            run_store.record_event(
                state,
                "evidence_recorded",
                step_id=step_id,
                payload={
                    "evidence_type": "approved_sql_result",
                    "property_code": normalized_code,
                    "row_count": len(rows),
                },
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
            run_store.record_event(
                state,
                "verification_started",
                step_id=step_id,
            )
            run_store.checkpoint(state, "verification_started")
            answer = (
                "I ran the approved read-only query for the selected property. "
                f"It returned **{len(rows)}** row{'s' if len(rows) != 1 else ''}."
            )
            state["final_answer"] = answer
            run_store.record_event(
                state,
                "verification_succeeded",
                step_id=step_id,
                payload={
                    "evidence_type": "approved_sql_result",
                    "property_code": normalized_code,
                    "status": "verified",
                },
            )
            transition_agent_state(state, "completed")
            run_store.save(state)
            run_store.record_event(
                state,
                "run_completed",
                step_id=step_id,
                duration_ms=self._duration_ms(resumed_at),
                payload={
                    "steps": state["current_step"],
                    "tool_calls": state["tool_call_count"],
                },
            )
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
                verification_was_active = state["status"] == "verifying"
                state["error"] = error_data
                state["plan"][-1]["status"] = "failed"
                transition_agent_state(state, "failed")
                try:
                    run_store.finish_step(step_id, status="failed", error=error_data)
                    run_store.save(state)
                    if verification_was_active:
                        run_store.record_event(
                            state,
                            "verification_failed",
                            step_id=step_id,
                            error_type=error_data["type"],
                            payload={"message": error_data["message"]},
                        )
                    run_store.record_event(
                        state,
                        "run_failed",
                        step_id=step_id,
                        duration_ms=self._duration_ms(resumed_at),
                        error_type=error_data["type"],
                        payload={"message": error_data["message"]},
                    )
                    run_store.checkpoint(state, "sql_execution_failed")
                except Exception:
                    pass
            raise

    def get_run(
        self,
        *,
        run_id: str,
        property_code: str,
        conversation_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        store = self._run_store_factory(self.settings)
        detail = store.get_run_detail(
            run_id,
            user_id or self.settings.local_auth_user_id,
            conversation_id,
            property_code.lower(),
        )
        if detail is None:
            raise AgentRunNotFoundError("agent run was not found")
        return detail

    def list_run_steps(
        self,
        *,
        run_id: str,
        property_code: str,
        conversation_id: str,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        store = self._run_store_factory(self.settings)
        if (
            store.get_run_detail(
                run_id,
                user_id or self.settings.local_auth_user_id,
                conversation_id,
                property_code.lower(),
            )
            is None
        ):
            raise AgentRunNotFoundError("agent run was not found")
        return store.list_steps_scoped(
            run_id,
            user_id or self.settings.local_auth_user_id,
            conversation_id,
            property_code.lower(),
        )

    def list_run_events(
        self,
        *,
        run_id: str,
        property_code: str,
        conversation_id: str,
        user_id: str | None = None,
        after_sequence: int = 0,
    ) -> list[dict[str, Any]]:
        store = self._run_store_factory(self.settings)
        if (
            store.get_run_detail(
                run_id,
                user_id or self.settings.local_auth_user_id,
                conversation_id,
                property_code.lower(),
            )
            is None
        ):
            raise AgentRunNotFoundError("agent run was not found")
        return store.list_events(
            run_id,
            user_id or self.settings.local_auth_user_id,
            conversation_id,
            property_code.lower(),
            after_sequence,
        )

    def list_run_citations(
        self,
        *,
        run_id: str,
        property_code: str,
        conversation_id: str,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        store = self._run_store_factory(self.settings)
        trusted_user_id = user_id or self.settings.local_auth_user_id
        normalized_code = property_code.lower()
        if (
            store.get_run_detail(
                run_id,
                trusted_user_id,
                conversation_id,
                normalized_code,
            )
            is None
        ):
            raise AgentRunNotFoundError("agent run was not found")
        return store.list_citations_scoped(
            run_id,
            trusted_user_id,
            conversation_id,
            normalized_code,
        )

    def cancel_run(
        self,
        *,
        run_id: str,
        property_code: str,
        conversation_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        normalized_code = property_code.lower()
        trusted_user_id = user_id or self.settings.local_auth_user_id
        store = self._run_store_factory(self.settings)
        state = store.load_scoped(
            run_id,
            trusted_user_id,
            conversation_id,
            normalized_code,
        )
        if state is None:
            raise AgentRunNotFoundError("agent run was not found")
        if state["status"] in TERMINAL_RUN_STATUSES:
            raise AgentRunConflictError(f"run is already {state['status']}")
        if not store.claim_cancellation(
            run_id,
            trusted_user_id,
            conversation_id,
            normalized_code,
        ):
            raise AgentRunConflictError("run could not be cancelled")

        for plan_step in state["plan"]:
            if plan_step.get("status") in {
                "pending",
                "running",
                "waiting_for_approval",
            }:
                plan_step["status"] = "cancelled"
        if state["pending_approval"] is not None:
            state["pending_approval"]["status"] = "cancelled"
            state["pending_approval"]["resolved_at"] = datetime.now(UTC).isoformat()
        transition_agent_state(state, "cancelled")
        store.save(state)
        store.record_event(
            state,
            "run_cancelled",
            payload={"reason": "user_requested"},
        )
        store.checkpoint(state, "run_cancelled")
        detail = store.get_run_detail(
            run_id,
            trusted_user_id,
            conversation_id,
            normalized_code,
        )
        if detail is None:
            raise AgentRunNotFoundError("agent run was not found")
        return detail

    @staticmethod
    def _record_tool_event(
        run_store: RunStore,
        state: AgentState,
        step_id: str,
        event: dict[str, Any],
    ) -> None:
        payload = dict(event)
        event_type = str(payload.pop("event", ""))
        tool_name = payload.pop("tool_name", None)
        attempt = payload.pop("attempt", None)
        duration_ms = payload.pop("duration_ms", None)
        error_type = payload.pop("error_type", None)
        run_store.record_event(
            state,
            event_type,
            step_id=step_id,
            tool_name=str(tool_name) if tool_name is not None else None,
            attempt=int(attempt) if attempt is not None else None,
            duration_ms=int(duration_ms) if duration_ms is not None else None,
            error_type=str(error_type) if error_type is not None else None,
            payload=payload,
        )
        persist_invocation = getattr(run_store, "record_tool_invocation_event", None)
        if callable(persist_invocation) and tool_name is not None:
            persist_invocation(
                state,
                step_id,
                event_type,
                str(tool_name),
                int(attempt) if attempt is not None else 0,
                int(duration_ms) if duration_ms is not None else None,
                str(error_type) if error_type is not None else None,
                payload,
            )

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, round((perf_counter() - started) * 1000))
