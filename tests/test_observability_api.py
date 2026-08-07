from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agents.runtime import AgentRunConflictError, AgentRunNotFoundError
from app.main import app

RUN_DETAIL = {
    "run_id": "run-1",
    "conversation_id": "conversation-1",
    "property_code": "115r",
    "user_goal": "Show occupancy",
    "status": "completed",
    "current_step": 1,
    "max_steps": 8,
    "plan": [],
    "pending_approval": None,
    "tool_call_count": 1,
    "max_tool_calls": 12,
    "error": None,
    "final_answer": "Completed.",
    "created_at": "2026-08-06T00:00:00",
    "updated_at": "2026-08-06T00:00:01",
}


class ObservabilityApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_run_trace_endpoints_pass_full_trusted_scope(self) -> None:
        calls: list[tuple[str, dict]] = []

        class FakeRuntime:
            def __init__(self, settings) -> None:
                pass

            def get_run(self, **kwargs):
                calls.append(("run", kwargs))
                return RUN_DETAIL

            def list_run_steps(self, **kwargs):
                calls.append(("steps", kwargs))
                return [
                    {
                        "step_id": "step-1",
                        "run_id": "run-1",
                        "step_number": 1,
                        "step_type": "property_chat_workflow",
                        "status": "succeeded",
                        "input": {"model": "mock:test"},
                        "output": {"source_count": 0},
                        "error": None,
                        "started_at": "2026-08-06T00:00:00",
                        "completed_at": "2026-08-06T00:00:01",
                        "duration_ms": 1000,
                    }
                ]

            def list_run_events(self, **kwargs):
                calls.append(("events", kwargs))
                return [
                    {
                        "event_id": "event-1",
                        "run_id": "run-1",
                        "event_type": "run_completed",
                        "conversation_id": "conversation-1",
                        "property_code": "115r",
                        "step_id": "step-1",
                        "tool_name": None,
                        "attempt": None,
                        "duration_ms": 1000,
                        "timestamp": "2026-08-06T00:00:01",
                        "error_type": None,
                        "payload": {"tool_calls": 1},
                    }
                ]

            def list_run_citations(self, **kwargs):
                calls.append(("citations", kwargs))
                return [
                    {
                        "citation_id": "citation-1",
                        "run_id": "run-1",
                        "property_code": "115r",
                        "source_type": "structured_tool",
                        "source_name": "get_occupancy_trend",
                        "tool_invocation_id": "tool-1",
                        "query_parameters": {"months": 12},
                        "data_timestamp": "2025-03-01",
                        "document_id": None,
                        "chunk_id": None,
                        "content_hash": "a" * 64,
                        "source_url": None,
                        "evidence": {"rows": []},
                        "retrieved_at": "2026-08-06T00:00:01",
                        "index_version": None,
                    }
                ]

        query = "property_code=115r&conversation_id=conversation-1"
        with patch("app.main.AgentRuntime", FakeRuntime):
            run_response = self.client.get(f"/api/agent-runs/run-1?{query}")
            step_response = self.client.get(f"/api/agent-runs/run-1/steps?{query}")
            event_response = self.client.get(f"/api/agent-runs/run-1/events?{query}")
            citation_response = self.client.get(
                f"/api/agent-runs/run-1/citations?{query}"
            )

        self.assertEqual(run_response.status_code, 200)
        self.assertEqual(step_response.json()[0]["duration_ms"], 1000)
        self.assertEqual(event_response.json()[0]["event_type"], "run_completed")
        self.assertEqual(
            citation_response.json()[0]["query_parameters"], {"months": 12}
        )
        for _, kwargs in calls:
            self.assertEqual(kwargs["run_id"], "run-1")
            self.assertEqual(kwargs["property_code"], "115r")
            self.assertEqual(kwargs["conversation_id"], "conversation-1")
            self.assertIn("user_id", kwargs)

    def test_run_endpoint_requires_conversation_and_property_scope(self) -> None:
        response = self.client.get("/api/agent-runs/run-1")

        self.assertEqual(response.status_code, 422)

    def test_out_of_scope_run_returns_not_found(self) -> None:
        class MissingRuntime:
            def __init__(self, settings) -> None:
                pass

            def get_run(self, **kwargs):
                raise AgentRunNotFoundError("agent run was not found")

        with patch("app.main.AgentRuntime", MissingRuntime):
            response = self.client.get(
                "/api/agent-runs/run-1?property_code=126a&conversation_id=conversation-1"
            )

        self.assertEqual(response.status_code, 404)

    def test_cancel_endpoint_returns_cancelled_detail(self) -> None:
        calls: list[dict] = []

        class FakeRuntime:
            def __init__(self, settings) -> None:
                pass

            def cancel_run(self, **kwargs):
                calls.append(kwargs)
                return {**RUN_DETAIL, "status": "cancelled", "final_answer": None}

        with patch("app.main.AgentRuntime", FakeRuntime):
            response = self.client.post(
                "/api/agent-runs/run-1/cancel",
                json={
                    "property_code": "115r",
                    "conversation_id": "conversation-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "cancelled")
        self.assertEqual(calls[0]["conversation_id"], "conversation-1")

    def test_terminal_run_cancel_returns_conflict(self) -> None:
        class ConflictRuntime:
            def __init__(self, settings) -> None:
                pass

            def cancel_run(self, **kwargs):
                raise AgentRunConflictError("run is already completed")

        with patch("app.main.AgentRuntime", ConflictRuntime):
            response = self.client.post(
                "/api/agent-runs/run-1/cancel",
                json={
                    "property_code": "115r",
                    "conversation_id": "conversation-1",
                },
            )

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
