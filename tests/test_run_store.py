from __future__ import annotations

import unittest
from typing import Any

from app.agents.state import new_agent_state, transition_agent_state
from app.memory.run_store import AgentRunStore


class MemoryDatabase:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.steps: dict[str, dict[str, Any]] = {}

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        normalized = " ".join(query.split())
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
        if normalized.startswith("UPDATE agent_runs SET"):
            run_id, user_id, property_code = params[-3:]
            row = self.runs.get(str(run_id))
            if not row or (row["user_id"], row["property_code"]) != (
                user_id,
                property_code,
            ):
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

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        run_id, user_id, property_code = params
        row = self.runs.get(str(run_id))
        if row and (row["user_id"], row["property_code"]) == (user_id, property_code):
            return dict(row)
        return None

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        run_id, user_id, property_code = params
        run = self.runs.get(str(run_id))
        if not run or (run["user_id"], run["property_code"]) != (user_id, property_code):
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

        self.assertIsNone(
            self.store.load(self.state["run_id"], "another-user", "115r")
        )
        self.assertIsNone(self.store.load(self.state["run_id"], "user-1", "126a"))

    def test_steps_are_persisted_in_order(self) -> None:
        self.store.create(self.state)
        transition_agent_state(self.state, "planning")
        step_id = self.store.start_step(self.state, "planning", {"model": "mock"})
        self.store.finish_step(step_id, status="succeeded", output_data={"route": "structured"})

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


if __name__ == "__main__":
    unittest.main()
