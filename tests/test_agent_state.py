from __future__ import annotations

import unittest

from app.agents.state import new_agent_state, transition_agent_state


class AgentStateTests(unittest.TestCase):
    def test_new_state_normalizes_trusted_scope_and_sets_budgets(self) -> None:
        state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115R",
            user_goal="Investigate occupancy",
        )

        self.assertEqual(state["property_code"], "115r")
        self.assertEqual(state["status"], "created")
        self.assertEqual(state["max_steps"], 8)
        self.assertEqual(state["max_tool_calls"], 12)

    def test_lifecycle_rejects_invalid_transition(self) -> None:
        state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115r",
            user_goal="Investigate occupancy",
        )

        with self.assertRaisesRegex(ValueError, "created -> completed"):
            transition_agent_state(state, "completed")

    def test_waiting_run_can_resume(self) -> None:
        state = new_agent_state(
            conversation_id="conversation-1",
            user_id="user-1",
            property_code="115r",
            user_goal="Prepare a custom query",
        )

        transition_agent_state(state, "planning")
        transition_agent_state(state, "waiting_for_approval")
        transition_agent_state(state, "running")

        self.assertEqual(state["status"], "running")


if __name__ == "__main__":
    unittest.main()
