from __future__ import annotations

import json
import unittest
from typing import Any

from app.agents.state import new_agent_state, transition_agent_state
from app.memory.run_store import AgentRunStore


class MemoryDatabase:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, dict[str, Any]] = {}
        self.checkpoints: list[dict[str, Any]] = []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        normalized = " ".join(query.split())
        if normalized.startswith("UPDATE agent_runs SET status = 'running'"):
            run_id, user_id, property_code = params
            row = self.runs.get(str(run_id))
            if (
                not row
                or (row["user_id"], row["property_code"]) != (user_id, property_code)
                or row["status"] != "waiting_for_approval"
            ):
                return 0
            row["status"] = "running"
            return 1
        if normalized.startswith("INSERT INTO agent_runs"):
            keys = [
                "run_id",
                "conversation_id",
                "user_id",
                "property_code",
                "user_goal",
                "status",
                "current_step",
                "max_steps",
                "plan_json",
                "observations_json",
                "pending_approval_json",
                "artifacts_json",
                "citations_json",
                "tool_call_count",
                "max_tool_calls",
                "error_json",
                "final_answer",
            ]
            self.runs[str(params[0])] = dict(zip(keys, params, strict=True))
            return 1
        if normalized.startswith("INSERT INTO agent_checkpoints"):
            checkpoint_id, run_id, transition_name, state_json, _ = params
            sequence_number = 1 + max(
                (
                    checkpoint["sequence_number"]
                    for checkpoint in self.checkpoints
                    if checkpoint["run_id"] == run_id
                ),
                default=0,
            )
            self.checkpoints.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "run_id": run_id,
                    "sequence_number": sequence_number,
                    "transition_name": transition_name,
                    "state_json": state_json,
                }
            )
            return 1
        if normalized.startswith("UPDATE agent_runs SET"):
            run_id, user_id, property_code, requested_status = params[-4:]
            row = self.runs.get(str(run_id))
            if not row or (row["user_id"], row["property_code"]) != (
                user_id,
                property_code,
            ):
                return 0
            if row["status"] == "cancelled" and requested_status != "cancelled":
                return 0
            keys = [
                "status",
                "current_step",
                "max_steps",
                "plan_json",
                "observations_json",
                "pending_approval_json",
                "artifacts_json",
                "citations_json",
                "tool_call_count",
                "max_tool_calls",
                "error_json",
                "final_answer",
            ]
            row.update(dict(zip(keys, params[:12], strict=True)))
            return 1
        if normalized.startswith("INSERT INTO agent_steps"):
            step_id, run_id, step_number, step_type, input_json = params
            self.steps[str(step_id)] = {
                "step_id": step_id,
                "run_id": run_id,
                "step_number": step_number,
                "step_type": step_type,
                "status": "running",
                "input_json": input_json,
                "output_json": None,
                "error_json": None,
                "started_at": "now",
                "completed_at": None,
            }
            return 1
        if normalized.startswith("UPDATE agent_steps SET"):
            status, output_json, error_json, step_id = params
            row = self.steps.get(str(step_id))
            if not row:
                return 0
            row.update(
                {
                    "status": status,
                    "output_json": output_json,
                    "error_json": error_json,
                    "completed_at": "now",
                }
            )
            return 1
        raise AssertionError(f"Unexpected query: {normalized}")

    def fetch_one(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        normalized = " ".join(query.split())
        if "FROM agent_checkpoints AS checkpoints" in normalized:
            run_id, user_id, conversation_id, property_code = params
        else:
            run_id, user_id, property_code = params
            conversation_id = None
        row = self.runs.get(str(run_id))
        if not row or (row["user_id"], row["property_code"]) != (
            user_id,
            property_code,
        ):
            return None
        if "FROM agent_checkpoints AS checkpoints" in normalized:
            if row["conversation_id"] != conversation_id:
                return None
            matches = [
                checkpoint
                for checkpoint in self.checkpoints
                if checkpoint["run_id"] == run_id
            ]
            if not matches:
                return None
            latest = max(matches, key=lambda checkpoint: checkpoint["sequence_number"])
            return {"state_json": latest["state_json"]}
        if row:
            return dict(row)
        return None

    def fetch_all(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        run_id, user_id, property_code = params
        run = self.runs.get(str(run_id))
        if not run or (run["user_id"], run["property_code"]) != (
            user_id,
            property_code,
        ):
            return []
        rows = [row.copy() for row in self.steps.values() if row["run_id"] == run_id]
        return sorted(rows, key=lambda row: row["step_number"])


class AgentRunStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = MemoryDatabase()
        self.store = AgentRunStore(self.database)
        self.state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115r",
            user_goal="Investigate occupancy",
        )

    def test_run_can_be_loaded_by_a_new_store_instance(self) -> None:
        self.store.create(self.state)
        transition_agent_state(self.state, "planning")
        self.state["plan"] = [{"tool": "get_occupancy_trend"}]
        self.store.save(self.state)

        reloaded = AgentRunStore(self.database).load(
            self.state["run_id"], "user-1", "115r"
        )

        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded["status"], "planning")
        self.assertEqual(reloaded["plan"], [{"tool": "get_occupancy_trend"}])

    def test_run_cannot_be_loaded_outside_its_scope(self) -> None:
        self.store.create(self.state)

        self.assertIsNone(self.store.load(self.state["run_id"], "another-user", "115r"))
        self.assertIsNone(self.store.load(self.state["run_id"], "user-1", "126a"))

    def test_steps_are_persisted_in_order(self) -> None:
        self.store.create(self.state)
        transition_agent_state(self.state, "planning")
        step_id = self.store.start_step(self.state, "planning", {"model": "mock"})
        self.store.finish_step(
            step_id, status="succeeded", output_data={"route": "structured"}
        )

        steps = AgentRunStore(self.database).list_steps(
            self.state["run_id"], "user-1", "115r"
        )

        self.assertEqual(steps[0]["step_number"], 1)
        self.assertEqual(steps[0]["status"], "succeeded")
        self.assertEqual(steps[0]["output"], {"route": "structured"})

        self.assertEqual(
            AgentRunStore(self.database).list_steps(
                self.state["run_id"], "another-user", "115r"
            ),
            [],
        )

    def test_latest_checkpoint_survives_store_recreation_and_is_scoped(self) -> None:
        self.store.create(self.state)
        self.store.checkpoint(self.state, "run_created")
        transition_agent_state(self.state, "planning")
        self.store.save(self.state)
        self.store.checkpoint(self.state, "plan_created")

        restarted_store = AgentRunStore(self.database)
        checkpoint = restarted_store.load_latest_checkpoint(
            self.state["run_id"], "user-1", "conversation-1", "115r"
        )

        self.assertEqual(checkpoint["status"], "planning")
        self.assertIsNone(
            restarted_store.load_latest_checkpoint(
                self.state["run_id"], "another-user", "conversation-1", "115r"
            )
        )
        self.assertIsNone(
            restarted_store.load_latest_checkpoint(
                self.state["run_id"], "user-1", "another-conversation", "115r"
            )
        )

    def test_waiting_approval_can_only_be_claimed_once(self) -> None:
        self.store.create(self.state)
        transition_agent_state(self.state, "planning")
        transition_agent_state(self.state, "waiting_for_approval")
        self.store.save(self.state)

        self.assertTrue(
            self.store.claim_approval(self.state["run_id"], "user-1", "115r")
        )
        self.assertFalse(
            self.store.claim_approval(self.state["run_id"], "user-1", "115r")
        )

    def test_artifacts_and_evidence_are_normalized_with_run_scope(self) -> None:
        class RecordingDatabase:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[Any, ...]]] = []

            def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
                self.calls.append((" ".join(query.split()), params))
                return 1

            def fetch_one(
                self, query: str, params: tuple[Any, ...] = ()
            ) -> dict[str, Any] | None:
                if "FROM tool_invocations AS invocations" in query:
                    return {"invocation_id": "tool-1"}
                return None

            def fetch_all(
                self, query: str, params: tuple[Any, ...] = ()
            ) -> list[dict[str, Any]]:
                return []

        database = RecordingDatabase()
        state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115r",
            user_goal="Create a report",
        )
        state["artifacts"] = [
            {
                "artifact_id": "artifact-1",
                "type": "report",
                "name": "Occupancy report",
                "content": "report body",
            }
        ]
        state["citations"] = [
            {
                "citation_id": "citation-1",
                "property_code": "115r",
                "source_type": "retrieval",
                "source_name": "Property website",
                "tool_invocation_id": "tool-1",
                "document_id": "document-1",
                "chunk_id": "chunk-1",
                "source_url": "https://example.com/property",
                "retrieved_at": "2026-08-06T12:00:00+00:00",
                "query_parameters": {"query": "amenities"},
                "evidence": {
                    "id": "chunk-1",
                    "text": "evidence",
                    "metadata": {"property_code": "115r"},
                },
            }
        ]

        AgentRunStore(database).save(state)

        artifact_call = next(
            call for call in database.calls if "INSERT INTO agent_artifacts" in call[0]
        )
        citation_call = next(
            call
            for call in database.calls
            if "INSERT INTO citation_evidence" in call[0]
        )
        self.assertEqual(artifact_call[1][1], state["run_id"])
        self.assertEqual(citation_call[1][1], state["run_id"])
        self.assertEqual(citation_call[1][2], "115r")
        self.assertEqual(citation_call[1][5], "tool-1")
        self.assertEqual(
            citation_call[1][11].isoformat(),
            "2026-08-06T12:00:00",
        )

    def test_cross_property_citation_is_rejected(self) -> None:
        class RecordingDatabase:
            def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
                return 1

            def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> None:
                return None

            def fetch_all(
                self, query: str, params: tuple[Any, ...] = ()
            ) -> list[dict[str, Any]]:
                return []

        state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115r",
            user_goal="Create a report",
        )
        state["citations"] = [
            {
                "citation_id": "citation-1",
                "property_code": "176r",
                "source_type": "retrieval",
                "source_name": "Property website",
                "evidence": {"text": "wrong property"},
            }
        ]

        with self.assertRaisesRegex(ValueError, "another property"):
            AgentRunStore(RecordingDatabase()).save(state)

    def test_tool_events_are_normalized_for_citation_linkage(self) -> None:
        class RecordingDatabase:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[Any, ...]]] = []

            def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
                self.calls.append((" ".join(query.split()), params))
                return 1

            def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> None:
                return None

            def fetch_all(
                self, query: str, params: tuple[Any, ...] = ()
            ) -> list[dict[str, Any]]:
                return []

        database = RecordingDatabase()
        state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115r",
            user_goal="Show occupancy",
        )
        AgentRunStore(database).record_tool_invocation_event(
            state,
            "step-1",
            "tool_started",
            "get_occupancy_trend",
            1,
            None,
            None,
            {
                "invocation_id": "tool-1",
                "sanitized_arguments": {"months": 12},
            },
        )

        query, params = database.calls[0]
        self.assertIn("INSERT INTO tool_invocations", query)
        self.assertEqual(params[0], "tool-1")
        self.assertEqual(json.loads(params[5]), {"months": 12})
        self.assertEqual(params[-4:-1], ("user-1", "conversation-1", "115r"))

    def test_operational_event_is_scoped_and_private_reasoning_is_removed(self) -> None:
        class RecordingDatabase:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[Any, ...]]] = []

            def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
                self.calls.append((" ".join(query.split()), params))
                return 1

            def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> None:
                return None

            def fetch_all(
                self, query: str, params: tuple[Any, ...] = ()
            ) -> list[dict[str, Any]]:
                return []

        database = RecordingDatabase()
        store = AgentRunStore(database)
        state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115r",
            user_goal="Show occupancy",
        )

        store.record_event(
            state,
            "tool_started",
            step_id="step-1",
            tool_name="get_property_profile",
            attempt=1,
            payload={
                "sanitized_arguments": {"limit": 5},
                "reasoning": "must never be persisted",
                "access_token": "secret-access-token",
                "refreshToken": "secret-refresh-token",
                "raw_prompt": "private prompt",
                "nested": {
                    "chain_of_thought": "private",
                    "systemPrompt": "private",
                    "safe": "visible",
                },
            },
        )

        query, params = database.calls[0]
        payload = json.loads(params[7])
        self.assertIn("runs.user_id = %s", query)
        self.assertEqual(
            params[-6:],
            (
                state["run_id"],
                "user-1",
                "conversation-1",
                "115r",
                "step-1",
                "step-1",
            ),
        )
        self.assertNotIn("reasoning", payload)
        self.assertNotIn("access_token", payload)
        self.assertNotIn("refreshToken", payload)
        self.assertNotIn("raw_prompt", payload)
        self.assertNotIn("chain_of_thought", payload["nested"])
        self.assertNotIn("systemPrompt", payload["nested"])
        self.assertEqual(payload["nested"]["safe"], "visible")

        store.record_event(
            state,
            "AUTHORIZATION_DENIED",
            payload={
                "user_id": "user-1",
                "role": "Viewer",
                "property_code": "115r",
                "permission": "property.analytics.read",
                "outcome": "denied",
            },
        )

        with self.assertRaisesRegex(ValueError, "unsupported observability event"):
            store.record_event(state, "model_private_reasoning")

    def test_event_replay_uses_durable_sequence_cursor(self) -> None:
        class RecordingDatabase:
            def __init__(self) -> None:
                self.query = ""
                self.params: tuple[Any, ...] = ()

            def fetch_all(
                self, query: str, params: tuple[Any, ...] = ()
            ) -> list[dict[str, Any]]:
                self.query = " ".join(query.split())
                self.params = params
                return [
                    {
                        "sequence_id": 43,
                        "event_id": "event-43",
                        "run_id": "run-1",
                        "event_type": "run_completed",
                        "conversation_id": "conversation-1",
                        "property_code": "115r",
                        "step_id": None,
                        "tool_name": None,
                        "attempt": None,
                        "duration_ms": 12,
                        "timestamp": "2026-08-07T00:00:00",
                        "error_type": None,
                        "payload_json": '{"tool_calls": 1}',
                    }
                ]

        database = RecordingDatabase()
        events = AgentRunStore(database).list_events(
            "run-1",
            "user-1",
            "conversation-1",
            "115R",
            after_sequence=42,
        )

        self.assertIn("events.sequence_id > %s", database.query)
        self.assertIn("ORDER BY events.sequence_id", database.query)
        self.assertEqual(
            database.params,
            ("run-1", "user-1", "conversation-1", "115r", 42),
        )
        self.assertEqual(events[0]["sequence_id"], 43)
        self.assertEqual(events[0]["payload"], {"tool_calls": 1})


if __name__ == "__main__":
    unittest.main()
