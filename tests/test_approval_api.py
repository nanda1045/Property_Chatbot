from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.agents.runtime import AgentRunConflictError, AgentRunNotFoundError
from app.main import app
from app.schemas import ChatResponse


class ApprovalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_approval_endpoint_passes_only_run_scope_and_decision(self) -> None:
        calls: list[dict] = []

        class FakeRuntime:
            def __init__(self, settings) -> None:
                pass

            def resolve_sql_approval(self, **kwargs) -> ChatResponse:
                calls.append(kwargs)
                return ChatResponse(
                    property_code="115r",
                    model="mock:test",
                    conversation_id="conversation-1",
                    run_id="run-1",
                    run_status="completed",
                    answer_markdown="Completed.",
                )

        memory = MagicMock()
        with (
            patch("app.main.AgentRuntime", FakeRuntime),
            patch("app.main._conversation_memory", return_value=memory),
        ):
            response = self.client.post(
                "/api/agent-runs/run-1/approve",
                json={
                    "property_code": "115r",
                    "conversation_id": "conversation-1",
                    "approved": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_status"], "completed")
        self.assertEqual(calls[0]["run_id"], "run-1")
        self.assertNotIn("sql", calls[0])
        self.assertEqual(memory.add.call_args.kwargs["run_id"], "run-1")

    def test_approval_endpoint_rejects_client_supplied_sql(self) -> None:
        response = self.client.post(
            "/api/agent-runs/run-1/approve",
            json={
                "property_code": "115r",
                "conversation_id": "conversation-1",
                "approved": True,
                "sql": "SELECT client_tampering_is_rejected",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_legacy_endpoint_no_longer_accepts_client_supplied_sql(self) -> None:
        response = self.client.post(
            "/sql/execute",
            json={
                "property_code": "115r",
                "conversation_id": "conversation-1",
                "question": "Custom metric",
                "sql": "SELECT client_sql_is_never_executed",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_run_returns_404(self) -> None:
        class MissingRuntime:
            def __init__(self, settings) -> None:
                pass

            def resolve_sql_approval(self, **kwargs) -> ChatResponse:
                raise AgentRunNotFoundError("agent run was not found")

        with patch("app.main.AgentRuntime", MissingRuntime):
            response = self.client.post(
                "/api/agent-runs/missing/approve",
                json={"property_code": "115r", "conversation_id": "conversation-1"},
            )

        self.assertEqual(response.status_code, 404)

    def test_duplicate_or_resolved_approval_returns_409(self) -> None:
        class ConflictRuntime:
            def __init__(self, settings) -> None:
                pass

            def resolve_sql_approval(self, **kwargs) -> ChatResponse:
                raise AgentRunConflictError("approval was already claimed or resolved")

        with patch("app.main.AgentRuntime", ConflictRuntime):
            response = self.client.post(
                "/api/agent-runs/run-1/approve",
                json={"property_code": "115r", "conversation_id": "conversation-1"},
            )

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
