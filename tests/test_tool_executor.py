from __future__ import annotations

import time
import unittest
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.tools.contracts import (
    ToolSpec,
    TransientToolError,
    TrustedToolContext,
)
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class NumberInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int


class NumberOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doubled: int


def build_executor(
    handler,
    *,
    max_attempts: int = 1,
    timeout_seconds: float = 1.0,
    idempotent: bool = True,
    max_tool_calls: int = 12,
    max_output_bytes: int = 250_000,
    trace: list[dict[str, Any]] | None = None,
    sleeps: list[float] | None = None,
) -> ToolExecutor:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="double",
            description="Double a number.",
            input_model=NumberInput,
            output_model=NumberOutput,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            idempotent=idempotent,
            required_scopes=("property_code",),
            max_output_bytes=max_output_bytes,
        ),
        handler,
    )
    return ToolExecutor(
        registry,
        max_tool_calls=max_tool_calls,
        trace_sink=(trace if trace is not None else []).append,
        sleep=(sleeps if sleeps is not None else []).append,
        random_value=lambda: 0.0,
    )


class ToolExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TrustedToolContext(property_code="115R", user_id="user-1")

    def test_trusted_scope_is_injected_and_model_scope_is_ignored(self) -> None:
        captured: dict[str, Any] = {}

        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            captured["input"] = tool_input.model_dump()
            captured["property_code"] = context.property_code
            return {"doubled": tool_input.value * 2}

        result = build_executor(handler).execute(
            "double",
            {
                "value": 4,
                "property_code": "attacker-property",
                "user_id": "attacker-user",
                "approval_status": "approved",
            },
            self.context,
        )

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.data, {"doubled": 8})
        self.assertEqual(captured["input"], {"value": 4})
        self.assertEqual(captured["property_code"], "115r")

    def test_input_validation_fails_before_handler_execution(self) -> None:
        calls = 0

        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            nonlocal calls
            calls += 1
            return {"doubled": 2}

        result = build_executor(handler).execute(
            "double",
            {"value": "not-an-integer"},
            self.context,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "input_validation")
        self.assertEqual(calls, 0)

    def test_malformed_output_is_rejected(self) -> None:
        result = build_executor(lambda _input, _context: {"wrong": 2}).execute(
            "double",
            {"value": 1},
            self.context,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "output_validation")

    def test_transient_failure_retries_with_backoff(self) -> None:
        attempts = 0
        sleeps: list[float] = []
        trace: list[dict[str, Any]] = []

        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise TransientToolError("database temporarily unavailable")
            return {"doubled": tool_input.value * 2}

        result = build_executor(
            handler,
            max_attempts=3,
            trace=trace,
            sleeps=sleeps,
        ).execute("double", {"value": 3}, self.context)

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.attempt, 3)
        self.assertEqual(attempts, 3)
        self.assertEqual(sleeps, [0.1, 0.2])
        self.assertEqual(
            [event["event"] for event in trace].count("tool_retried"),
            2,
        )

    def test_permanent_failure_does_not_retry(self) -> None:
        attempts = 0

        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            nonlocal attempts
            attempts += 1
            raise ValueError("permanent schema mismatch")

        result = build_executor(handler, max_attempts=3).execute(
            "double", {"value": 3}, self.context
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "permanent_failure")
        self.assertEqual(attempts, 1)

    def test_non_idempotent_tool_does_not_retry(self) -> None:
        attempts = 0

        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            nonlocal attempts
            attempts += 1
            raise TransientToolError("temporary failure")

        result = build_executor(handler, max_attempts=3, idempotent=False).execute(
            "double", {"value": 3}, self.context
        )

        self.assertEqual(result.status, "failed")
        self.assertTrue(result.error.retryable)
        self.assertEqual(attempts, 1)

    def test_timeout_returns_structured_error(self) -> None:
        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            time.sleep(0.05)
            return {"doubled": tool_input.value * 2}

        result = build_executor(handler, timeout_seconds=0.005).execute(
            "double", {"value": 2}, self.context
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "timeout")
        self.assertTrue(result.error.retryable)

    def test_budget_is_enforced_for_distinct_calls(self) -> None:
        executor = build_executor(
            lambda tool_input, _context: {"doubled": tool_input.value * 2},
            max_tool_calls=1,
        )

        first = executor.execute("double", {"value": 1}, self.context)
        second = executor.execute("double", {"value": 2}, self.context)

        self.assertEqual(first.status, "succeeded")
        self.assertEqual(second.error.type, "budget_exceeded")
        self.assertEqual(executor.budget.call_count, 1)

    def test_idempotent_duplicate_uses_cache_without_budget_charge(self) -> None:
        calls = 0

        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            nonlocal calls
            calls += 1
            return {"doubled": tool_input.value * 2}

        executor = build_executor(handler, max_tool_calls=1)
        first = executor.execute("double", {"value": 2}, self.context)
        second = executor.execute("double", {"value": 2}, self.context)

        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(calls, 1)
        self.assertEqual(executor.budget.call_count, 1)

    def test_new_execution_resets_budget_and_idempotency_scope(self) -> None:
        calls = 0

        def handler(tool_input: NumberInput, context: TrustedToolContext) -> dict:
            nonlocal calls
            calls += 1
            return {"doubled": tool_input.value * 2}

        executor = build_executor(handler, max_tool_calls=1)
        executor.execute("double", {"value": 2}, self.context)
        executor.reset_execution()
        result = executor.execute("double", {"value": 2}, self.context)

        self.assertEqual(result.status, "succeeded")
        self.assertFalse(result.cached)
        self.assertEqual(calls, 2)
        self.assertEqual(executor.budget.call_count, 1)

    def test_missing_scope_and_oversized_output_are_rejected(self) -> None:
        executor = build_executor(
            lambda _input, _context: {"doubled": 123456789},
            max_output_bytes=10,
        )

        missing_scope = executor.execute(
            "double",
            {"value": 1},
            TrustedToolContext(user_id="user-1"),
        )
        oversized = executor.execute("double", {"value": 1}, self.context)

        self.assertEqual(missing_scope.error.type, "scope_error")
        self.assertEqual(oversized.error.type, "output_too_large")

    def test_unknown_tool_returns_structured_error(self) -> None:
        executor = build_executor(lambda _input, _context: {"doubled": 2})

        result = executor.execute("missing", {}, self.context)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.type, "unknown_tool")


class ToolRegistryTests(unittest.TestCase):
    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec(
            name="double",
            description="Double a number.",
            input_model=NumberInput,
            output_model=NumberOutput,
        )
        registry.register(spec, lambda _input, _context: {"doubled": 2})

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(spec, lambda _input, _context: {"doubled": 2})


if __name__ == "__main__":
    unittest.main()
