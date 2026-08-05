from __future__ import annotations

import unittest

from app.agents.loop import (
    AgentAction,
    AgentLimitExceeded,
    BoundedAgentLoop,
    ExecutionLimits,
    LoopDecision,
    RepeatedActionError,
)


class BoundedAgentLoopTests(unittest.TestCase):
    def test_next_action_can_depend_on_previous_observation(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits())
        executed: list[str] = []

        def decide(observations):
            if not observations:
                return LoopDecision(
                    action=AgentAction(
                        key="trend",
                        tool_name="get_occupancy_trend",
                        arguments={"months": 12},
                    )
                )
            if observations[-1].tool_name == "get_occupancy_trend":
                if observations[-1].data["declined"]:
                    return LoopDecision(
                        action=AgentAction(
                            key="vacancies",
                            tool_name="get_vacant_units",
                            arguments={"limit": 20},
                        )
                    )
            return LoopDecision(complete=True, reason="evidence_complete")

        def execute(action: AgentAction):
            executed.append(action.tool_name)
            if action.tool_name == "get_occupancy_trend":
                return {"declined": True}
            return [{"unit": "101"}]

        result = loop.run(decide, execute)

        self.assertEqual(
            executed,
            ["get_occupancy_trend", "get_vacant_units"],
        )
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.reason, "evidence_complete")
        self.assertEqual(result.observations[1].data, [{"unit": "101"}])

    def test_step_limit_stops_continuation(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits(max_steps=2))

        def decide(observations):
            step = len(observations) + 1
            return LoopDecision(
                action=AgentAction(
                    key=f"step-{step}",
                    tool_name="lookup",
                    arguments={"page": step},
                )
            )

        with self.assertRaisesRegex(AgentLimitExceeded, "step limit"):
            loop.run(decide, lambda action: {"page": action.arguments["page"]})

        self.assertEqual(loop.step_count, 2)

    def test_tool_call_limit_is_independent_from_step_limit(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits(max_steps=5, max_tool_calls=1))

        def decide(observations):
            call = len(observations) + 1
            return LoopDecision(
                action=AgentAction(
                    key=f"call-{call}",
                    tool_name="lookup",
                    arguments={"call": call},
                )
            )

        with self.assertRaisesRegex(AgentLimitExceeded, "tool call limit"):
            loop.run(decide, lambda _action: {"ok": True})

        self.assertEqual(loop.step_count, 1)

    def test_repeated_identical_action_is_rejected(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits())
        repeated = AgentAction(key="same", tool_name="lookup", arguments={"value": 1})

        with self.assertRaisesRegex(RepeatedActionError, "Repeated identical"):
            loop.run(
                lambda _observations: LoopDecision(action=repeated),
                lambda _action: {"ok": True},
            )

        self.assertEqual(loop.step_count, 1)

    def test_failed_action_is_observed_before_error_propagates(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits())

        def fail(_action: AgentAction):
            raise ValueError("malformed observation")

        with self.assertRaisesRegex(ValueError, "malformed observation"):
            loop.run(
                lambda _observations: LoopDecision(
                    action=AgentAction(key="bad", tool_name="broken_tool")
                ),
                fail,
            )

        self.assertEqual(loop.observations[0].status, "failed")
        self.assertEqual(loop.observations[0].error["type"], "ValueError")

    def test_planner_retries_are_bounded(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits(max_planner_retries=2))
        responses = iter([None, None, {"route": "structured"}])

        plan = loop.plan_with_retries(lambda: next(responses))

        self.assertEqual(plan, {"route": "structured"})
        self.assertEqual(loop.planner_attempts, 3)

        exhausted = BoundedAgentLoop(ExecutionLimits(max_planner_retries=1))
        self.assertIsNone(exhausted.plan_with_retries(lambda: None))
        self.assertEqual(exhausted.planner_attempts, 2)

    def test_sql_approval_and_duration_limits_are_enforced(self) -> None:
        loop = BoundedAgentLoop(ExecutionLimits(max_sql_approvals=1))
        loop.request_sql_approval()
        with self.assertRaisesRegex(AgentLimitExceeded, "SQL approval limit"):
            loop.request_sql_approval()

        now = [0.0]
        timed = BoundedAgentLoop(
            ExecutionLimits(max_run_seconds=1),
            clock=lambda: now[0],
        )
        now[0] = 1.1
        with self.assertRaisesRegex(AgentLimitExceeded, "run duration limit"):
            timed.ensure_within_duration()


if __name__ == "__main__":
    unittest.main()
