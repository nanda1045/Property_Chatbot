from __future__ import annotations

import unittest
from copy import deepcopy
from typing import Any

from app.agents.policies import property_scope_conflict
from app.agents.runtime import AgentRuntime
from app.agents.state import AgentState
from app.core.config import Settings
from app.schemas import ChatResponse


class FakeWorkflow:
    def __init__(self, response: ChatResponse | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response

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
        self.saved: list[AgentState] = []
        self.finished: list[dict[str, Any]] = []

    def create(self, state: AgentState) -> None:
        self.created = deepcopy(state)

    def save(self, state: AgentState) -> None:
        self.saved.append(deepcopy(state))

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
        run_store = RecordingRunStore()
        settings = Settings(_env_file=None)
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
        self.assertEqual(run_store.finished[0]["status"], "succeeded")

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
