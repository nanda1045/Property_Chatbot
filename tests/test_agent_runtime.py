from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any

from app.agents.policies import property_scope_conflict
from app.agents.runtime import AgentRunConflictError, AgentRuntime
from app.agents.state import AgentState
from app.core.config import Settings
from app.schemas import ChatResponse


class FakeWorkflow:
    def __init__(self, response: ChatResponse | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.run_context: dict[str, Any] | None = None

    def bind_run_context(self, **kwargs: Any) -> None:
        self.run_context = kwargs

    def answer(self, **kwargs: Any) -> ChatResponse:
        self.calls.append(kwargs)
        return self.response or ChatResponse(
            property_code=kwargs["property_code"].lower(),
            model=kwargs["model"],
            answer_markdown="delegated",
        )


class RecordingRunStore:
    def __init__(self) -> None:
        self.created: AgentState | None = None
        self.current: AgentState | None = None
        self.saved: list[AgentState] = []
        self.finished: list[dict[str, Any]] = []
        self.checkpoints: list[tuple[str, AgentState]] = []
        self.events: list[dict[str, Any]] = []

    def create(self, state: AgentState) -> None:
        self.created = deepcopy(state)
        self.current = deepcopy(state)

    def save(self, state: AgentState) -> None:
        snapshot = deepcopy(state)
        self.current = snapshot
        self.saved.append(snapshot)

    def checkpoint(self, state: AgentState, transition_name: str) -> str:
        self.checkpoints.append((transition_name, deepcopy(state)))
        return f"checkpoint-{len(self.checkpoints)}"

    def record_event(
        self,
        state: AgentState,
        event_type: str,
        **kwargs: Any,
    ) -> str:
        self.events.append(
            {
                "event_type": event_type,
                "state": deepcopy(state),
                **deepcopy(kwargs),
            }
        )
        return f"event-{len(self.events)}"

    def load_latest_checkpoint(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> AgentState | None:
        for _, state in reversed(self.checkpoints):
            if (
                state["run_id"] == run_id
                and state["user_id"] == user_id
                and state["conversation_id"] == conversation_id
                and state["property_code"] == property_code
            ):
                return deepcopy(state)
        return None

    def load_scoped(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> AgentState | None:
        state = self.current
        if (
            state is None
            or state["run_id"] != run_id
            or state["user_id"] != user_id
            or state["conversation_id"] != conversation_id
            or state["property_code"] != property_code
        ):
            return None
        return deepcopy(state)

    def get_run_detail(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> dict[str, Any] | None:
        state = self.load_scoped(run_id, user_id, conversation_id, property_code)
        if state is None:
            return None
        return {
            "run_id": state["run_id"],
            "conversation_id": state["conversation_id"],
            "property_code": state["property_code"],
            "user_goal": state["user_goal"],
            "status": state["status"],
            "current_step": state["current_step"],
            "max_steps": state["max_steps"],
            "plan": state["plan"],
            "pending_approval": state["pending_approval"],
            "tool_call_count": state["tool_call_count"],
            "max_tool_calls": state["max_tool_calls"],
            "error": state["error"],
            "final_answer": state["final_answer"],
            "created_at": "now",
            "updated_at": "now",
        }

    def list_steps_scoped(self, *args: Any) -> list[dict[str, Any]]:
        return []

    def list_events(self, *args: Any) -> list[dict[str, Any]]:
        return deepcopy(self.events)

    def claim_cancellation(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> bool:
        state = self.current
        if (
            state is None
            or state["run_id"] != run_id
            or state["user_id"] != user_id
            or state["conversation_id"] != conversation_id
            or state["property_code"] != property_code
            or state["status"] in {"completed", "failed", "cancelled"}
        ):
            return False
        state["status"] = "cancelled"
        return True

    def claim_approval(self, run_id: str, user_id: str, property_code: str) -> bool:
        state = self.current
        if (
            state is None
            or state["run_id"] != run_id
            or state["user_id"] != user_id
            or state["property_code"] != property_code
            or state["status"] != "waiting_for_approval"
        ):
            return False
        state["status"] = "running"
        return True

    def start_step(
        self,
        state: AgentState,
        step_type: str,
        input_data: dict[str, Any] | None = None,
    ) -> str:
        state["current_step"] += 1
        self.saved.append(deepcopy(state))
        return "step-1"

    def finish_step(self, step_id: str, **kwargs: Any) -> None:
        self.finished.append({"step_id": step_id, **kwargs})


class AgentRuntimeTests(unittest.TestCase):
    def test_runtime_delegates_transport_inputs_to_workflow(self) -> None:
        workflow = FakeWorkflow()
        workflow.tool_call_count = 2
        workflow.execution_plan = [
            {"key": "kpis", "tool_name": "get_latest_property_kpis", "arguments": {}}
        ]
        workflow.execution_observations = [
            {
                "step": 1,
                "action_key": "kpis",
                "tool_name": "get_latest_property_kpis",
                "status": "succeeded",
                "duration_ms": 4,
                "data": {"current": {}},
                "error": None,
            }
        ]
        workflow.planner_attempt_count = 1
        workflow.sql_approval_count = 0
        run_store = RecordingRunStore()
        settings = Settings(
            _env_file=None,
            agent_max_steps=6,
            agent_max_tool_calls=9,
        )
        runtime = AgentRuntime(
            settings,
            workflow_factory=lambda _: workflow,
            run_store_factory=lambda _: run_store,
        )
        history = [{"user": "Earlier", "assistant": "Answer"}]
        tokens: list[str] = []
        on_token = tokens.append

        response = runtime.answer(
            property_code="115R",
            message="Latest occupancy",
            model="mock:test",
            on_token=on_token,
            history=history,
            conversation_id="conversation-1",
            user_id="user-1",
        )

        self.assertEqual(response.answer_markdown, "delegated")
        self.assertEqual(len(workflow.calls), 1)
        self.assertEqual(workflow.calls[0]["property_code"], "115R")
        self.assertIs(workflow.calls[0]["history"], history)
        self.assertIs(workflow.calls[0]["on_token"], on_token)
        self.assertEqual(response.run_status, "completed")
        self.assertEqual(response.run_id, run_store.created["run_id"])
        self.assertEqual(run_store.saved[-1]["final_answer"], "delegated")
        self.assertEqual(run_store.saved[-1]["tool_call_count"], 2)
        self.assertEqual(run_store.created["max_steps"], 6)
        self.assertEqual(run_store.created["max_tool_calls"], 9)
        self.assertEqual(
            run_store.saved[-1]["plan"][0]["actions"][0]["tool_name"],
            "get_latest_property_kpis",
        )
        loop_observation = next(
            observation
            for observation in run_store.saved[-1]["observations"]
            if observation.get("type") == "bounded_agent_loop"
        )
        self.assertEqual(loop_observation["planner_attempts"], 1)
        self.assertEqual(loop_observation["steps"], 1)
        self.assertEqual(run_store.finished[0]["status"], "succeeded")
        self.assertEqual(
            [event["event_type"] for event in run_store.events],
            [
                "run_created",
                "planning_started",
                "plan_created",
                "step_started",
                "verification_started",
                "run_completed",
            ],
        )
        self.assertEqual(
            [name for name, _ in run_store.checkpoints],
            [
                "run_created",
                "plan_created",
                "step_started",
                "verification_started",
                "step_completed",
                "run_completed",
            ],
        )

    def test_sql_approval_is_persisted_as_waiting(self) -> None:
        response = ChatResponse(
            property_code="115r",
            model="mock:test",
            answer_markdown="Review this SQL.",
            components=[
                {
                    "type": "sql_approval",
                    "title": "Review SQL",
                    "data": {"sql": "SELECT 1", "status": "pending_approval"},
                }
            ],
        )
        workflow = FakeWorkflow(response)
        run_store = RecordingRunStore()
        runtime = AgentRuntime(
            Settings(_env_file=None),
            workflow_factory=lambda _: workflow,
            run_store_factory=lambda _: run_store,
        )

        result = runtime.answer(
            property_code="115r",
            message="Custom metric",
            model="mock:test",
            conversation_id="conversation-1",
            user_id="user-1",
        )

        self.assertEqual(result.run_status, "waiting_for_approval")
        self.assertEqual(result.components[0].data["run_id"], result.run_id)
        self.assertEqual(run_store.saved[-1]["pending_approval"]["sql"], "SELECT 1")
        self.assertEqual(
            {artifact["type"] for artifact in run_store.saved[-1]["artifacts"]},
            {"generated_sql", "structured_output"},
        )
        self.assertEqual(run_store.checkpoints[-1][0], "approval_requested")
        self.assertIn(
            "approval_requested",
            [event["event_type"] for event in run_store.events],
        )

    def test_investigation_report_is_scoped_to_run_and_persisted(self) -> None:
        response = ChatResponse(
            property_code="115r",
            model="mock:test",
            answer_markdown="Executive brief",
            investigation={
                "summary": "Occupancy evidence summary.",
                "findings": [],
                "citations": [
                    {
                        "citation_id": "citation-1",
                        "property_code": "115r",
                        "source_type": "structured_tool",
                        "source_name": "get_occupancy_trend",
                        "tool_invocation_id": "tool-1",
                        "content_hash": "hash",
                        "retrieved_at": "2026-08-05T00:00:00+00:00",
                        "evidence": {"rows": []},
                    }
                ],
                "artifacts": [
                    {
                        "artifact_id": "artifact-1",
                        "type": "executive_brief",
                        "name": "brief.md",
                        "content_type": "text/markdown",
                        "content": "Executive brief",
                    }
                ],
                "trace_summary": {
                    "steps": 1,
                    "tool_calls": 1,
                    "duration_ms": 5,
                    "tool_order": ["get_occupancy_trend"],
                    "stop_reason": "completed",
                    "verification_status": "passed",
                    "verification_checks": ["property_scope_valid:1"],
                },
            },
        )
        run_store = RecordingRunStore()
        runtime = AgentRuntime(
            Settings(_env_file=None),
            workflow_factory=lambda _: FakeWorkflow(response),
            run_store_factory=lambda _: run_store,
        )

        result = runtime.answer(
            property_code="115r",
            message="Investigate occupancy decline",
            model="mock:test",
            conversation_id="conversation-1",
            user_id="user-1",
        )

        self.assertEqual(result.investigation.run_id, result.run_id)
        self.assertEqual(run_store.saved[-1]["artifacts"][0]["artifact_id"], "artifact-1")
        self.assertEqual(run_store.saved[-1]["citations"][0]["citation_id"], "citation-1")
        self.assertEqual(
            result.tool_results["occupancy_investigation"]["run_id"],
            result.run_id,
        )
        self.assertEqual(result.citation_ids, ["citation-1"])

    def test_general_tool_evidence_is_linked_to_the_answer(self) -> None:
        class CitedWorkflow(FakeWorkflow):
            @property
            def citation_evidence(self) -> list[dict[str, Any]]:
                return [
                    {
                        "citation_id": "citation-structured",
                        "property_code": "115r",
                        "source_type": "structured_tool",
                        "source_name": "get_latest_property_kpis",
                        "tool_invocation_id": "tool-1",
                        "query_parameters": {},
                        "data_timestamp": "2025-03-01",
                        "content_hash": "a" * 64,
                        "retrieved_at": "2026-08-06T00:00:00+00:00",
                        "evidence": {"returned": {"occupancy": 91.0}},
                    }
                ]

        workflow = CitedWorkflow(
            ChatResponse(
                property_code="115r",
                model="mock:test",
                answer_markdown="Occupancy is 91%.",
            )
        )
        run_store = RecordingRunStore()
        result = AgentRuntime(
            Settings(_env_file=None),
            workflow_factory=lambda _: workflow,
            run_store_factory=lambda _: run_store,
        ).answer(
            property_code="115r",
            message="Latest occupancy",
            model="mock:test",
            conversation_id="conversation-1",
            user_id="user-1",
        )

        self.assertEqual(result.citation_ids, ["citation-structured"])
        self.assertEqual(
            run_store.saved[-1]["citations"][0]["tool_invocation_id"],
            "tool-1",
        )

    def test_workflow_failure_is_checkpointed(self) -> None:
        class FailingWorkflow(FakeWorkflow):
            def answer(self, **kwargs: Any) -> ChatResponse:
                raise TimeoutError("workflow timed out")

        run_store = RecordingRunStore()
        runtime = AgentRuntime(
            Settings(_env_file=None),
            workflow_factory=lambda _: FailingWorkflow(),
            run_store_factory=lambda _: run_store,
        )

        with self.assertRaisesRegex(TimeoutError, "workflow timed out"):
            runtime.answer(
                property_code="115r",
                message="Latest occupancy",
                model="mock:test",
                conversation_id="conversation-1",
                user_id="user-1",
            )

        self.assertEqual(run_store.saved[-1]["status"], "failed")
        self.assertEqual(run_store.events[-1]["event_type"], "run_failed")
        self.assertEqual(run_store.events[-1]["error_type"], "TimeoutError")
        self.assertEqual(
            [name for name, _ in run_store.checkpoints][-2:],
            ["step_failed", "run_failed"],
        )

    def test_approval_resumes_from_checkpoint_after_runtime_restart(self) -> None:
        response = ChatResponse(
            property_code="115r",
            model="mock:test",
            answer_markdown="Review this SQL.",
            components=[
                {
                    "type": "sql_approval",
                    "title": "Review SQL",
                    "data": {
                        "sql": "SELECT stored_sql",
                        "question": "Custom metric",
                        "model": "mock:test",
                        "status": "pending_approval",
                    },
                }
            ],
        )
        run_store = RecordingRunStore()
        settings = Settings(_env_file=None)
        first_runtime = AgentRuntime(
            settings,
            workflow_factory=lambda _: FakeWorkflow(response),
            run_store_factory=lambda _: run_store,
        )
        waiting = first_runtime.answer(
            property_code="115r",
            message="Custom metric",
            model="mock:test",
            conversation_id="conversation-1",
            user_id="user-1",
        )
        executor_calls: list[tuple[str, str]] = []

        def execute_sql(
            settings: Settings,
            sql: str,
            property_code: str,
        ) -> tuple[str, list[dict[str, Any]]]:
            executor_calls.append((sql, property_code))
            return "SELECT validated_sql", [{"unit_type": "A1", "unit_count": 3}]

        restarted_runtime = AgentRuntime(
            settings,
            workflow_factory=lambda _: FakeWorkflow(),
            run_store_factory=lambda _: run_store,
            sql_executor=execute_sql,
        )
        completed = restarted_runtime.resolve_sql_approval(
            run_id=str(waiting.run_id),
            property_code="115r",
            approved=True,
            conversation_id="conversation-1",
            user_id="user-1",
        )

        self.assertEqual(executor_calls, [("SELECT stored_sql", "115r")])
        self.assertEqual(completed.run_id, waiting.run_id)
        self.assertEqual(completed.run_status, "completed")
        self.assertEqual(completed.tool_results["row_count"], 1)
        self.assertEqual(run_store.saved[-1]["tool_call_count"], 1)
        checkpoint_names = [name for name, _ in run_store.checkpoints]
        self.assertIn("approval_received", checkpoint_names)
        self.assertIn("sql_execution_completed", checkpoint_names)
        self.assertEqual(checkpoint_names[-1], "run_completed")

        with self.assertRaisesRegex(AgentRunConflictError, "run is completed"):
            restarted_runtime.resolve_sql_approval(
                run_id=str(waiting.run_id),
                property_code="115r",
                approved=True,
                conversation_id="conversation-1",
                user_id="user-1",
            )
        self.assertEqual(len(executor_calls), 1)

    def test_tool_events_are_bound_to_the_current_run_and_step(self) -> None:
        class ToolEventWorkflow(FakeWorkflow):
            def answer(self, **kwargs: Any) -> ChatResponse:
                assert self.run_context is not None
                sink = self.run_context["event_sink"]
                sink(
                    {
                        "event": "tool_started",
                        "tool_name": "get_property_profile",
                        "attempt": 1,
                        "sanitized_arguments": {},
                    }
                )
                sink(
                    {
                        "event": "tool_succeeded",
                        "tool_name": "get_property_profile",
                        "attempt": 1,
                        "duration_ms": 4,
                        "output_summary": {"type": "object"},
                    }
                )
                return super().answer(**kwargs)

        workflow = ToolEventWorkflow()
        run_store = RecordingRunStore()
        result = AgentRuntime(
            Settings(_env_file=None),
            workflow_factory=lambda _: workflow,
            run_store_factory=lambda _: run_store,
        ).answer(
            property_code="115r",
            message="Latest occupancy",
            model="mock:test",
            conversation_id="conversation-1",
            user_id="user-1",
        )

        tool_events = [
            event for event in run_store.events if event["event_type"].startswith("tool_")
        ]
        self.assertEqual(
            [event["event_type"] for event in tool_events],
            ["tool_started", "tool_succeeded"],
        )
        self.assertTrue(all(event["step_id"] == "step-1" for event in tool_events))
        self.assertEqual(workflow.run_context["run_id"], result.run_id)

    def test_waiting_run_can_be_cancelled_and_checkpointed(self) -> None:
        response = ChatResponse(
            property_code="115r",
            model="mock:test",
            answer_markdown="Review this SQL.",
            components=[
                {
                    "type": "sql_approval",
                    "title": "Review SQL",
                    "data": {"sql": "SELECT 1", "status": "pending_approval"},
                }
            ],
        )
        run_store = RecordingRunStore()
        runtime = AgentRuntime(
            Settings(_env_file=None),
            workflow_factory=lambda _: FakeWorkflow(response),
            run_store_factory=lambda _: run_store,
        )
        waiting = runtime.answer(
            property_code="115r",
            message="Custom metric",
            model="mock:test",
            conversation_id="conversation-1",
            user_id="user-1",
        )

        detail = runtime.cancel_run(
            run_id=str(waiting.run_id),
            property_code="115r",
            conversation_id="conversation-1",
            user_id="user-1",
        )

        self.assertEqual(detail["status"], "cancelled")
        self.assertEqual(run_store.events[-1]["event_type"], "run_cancelled")
        self.assertEqual(run_store.checkpoints[-1][0], "run_cancelled")


class PropertyScopePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.active = {"property_code": "115r", "property_name": "Aker North"}
        self.properties = [
            self.active,
            {"property_code": "126a", "property_name": "Aker South"},
        ]

    def test_known_other_property_is_rejected(self) -> None:
        conflict = property_scope_conflict(
            "Show occupancy for Aker South",
            self.active,
            self.properties,
        )

        self.assertIsNotNone(conflict)
        self.assertEqual(conflict["property_code"], "126a")

    def test_active_property_is_allowed(self) -> None:
        conflict = property_scope_conflict(
            "Show occupancy for property Aker North",
            self.active,
            self.properties,
        )

        self.assertIsNone(conflict)

    def test_unknown_explicit_property_is_rejected(self) -> None:
        conflict = property_scope_conflict(
            "Show occupancy for property Mystery Place?",
            self.active,
            self.properties,
        )

        self.assertEqual(conflict, {"property_name": "Mystery Place"})


if __name__ == "__main__":
    unittest.main()
