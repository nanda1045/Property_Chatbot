from __future__ import annotations

import unittest

from app.agents.planner import ToolPlan, validate_tool_plan


class PlannerPolicyTests(unittest.TestCase):
    def test_validation_removes_untrusted_scope_and_unknown_tools(self) -> None:
        plan = ToolPlan(
            route="structured",
            structured_tools=[
                {
                    "name": "get_occupancy_trend",
                    "args": {"property_code": "other-property", "months": 99},
                },
                {"name": "unknown_tool", "args": {}},
            ],
        )

        validated = validate_tool_plan(plan, {"get_occupancy_trend"})

        self.assertEqual(len(validated.structured_tools), 1)
        self.assertEqual(validated.structured_tools[0].name, "get_occupancy_trend")
        self.assertEqual(validated.structured_tools[0].args, {"months": 36})

    def test_empty_structured_plan_becomes_clarification(self) -> None:
        plan = ToolPlan(
            route="structured",
            structured_tools=[{"name": "unknown_tool", "args": {}}],
        )

        validated = validate_tool_plan(plan, {"get_occupancy_trend"})

        self.assertEqual(validated.route, "clarification")
        self.assertIsNotNone(validated.clarification_question)


if __name__ == "__main__":
    unittest.main()
