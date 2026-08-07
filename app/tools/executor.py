"""Bounded, validated execution for every registered agent tool."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from copy import deepcopy
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from app.tools.contracts import (
    TRUSTED_ARGUMENT_NAMES,
    ToolError,
    ToolResult,
    TransientToolError,
    TrustedToolContext,
)
from app.tools.registry import RegisteredTool, ToolRegistry

logger = logging.getLogger(__name__)

# One bounded pool is shared across request-scoped executors. A timed-out call may
# finish in the background, but cannot create an unbounded number of worker threads.
_TOOL_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="aker-tool")

TraceSink = Callable[[dict[str, Any]], None]
Sleep = Callable[[float], None]
RandomValue = Callable[[], float]


class ToolBudget:
    def __init__(self, max_calls: int = 12) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.max_calls = max_calls
        self._call_count = 0
        self._lock = Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def consume(self) -> bool:
        with self._lock:
            if self._call_count >= self.max_calls:
                return False
            self._call_count += 1
            return True

    def reset(self) -> None:
        with self._lock:
            self._call_count = 0


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_tool_calls: int = 12,
        trace_sink: TraceSink | None = None,
        sleep: Sleep = time.sleep,
        random_value: RandomValue = random.random,
        pool: ThreadPoolExecutor = _TOOL_POOL,
    ) -> None:
        self.registry = registry
        self.budget = ToolBudget(max_tool_calls)
        self.trace_sink = trace_sink or self._log_trace
        self.sleep = sleep
        self.random_value = random_value
        self.pool = pool
        self._idempotency_cache: dict[str, ToolResult] = {}
        self._cache_lock = Lock()

    def reset_execution(self) -> None:
        """Start an independent run with a fresh budget and idempotency scope."""
        self.budget.reset()
        with self._cache_lock:
            self._idempotency_cache.clear()

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        context: TrustedToolContext,
        *,
        idempotency_key: str | None = None,
    ) -> ToolResult:
        invocation_id = str(uuid4())
        registered = self.registry.get(tool_name)
        if registered is None:
            return self._failure(
                invocation_id,
                tool_name,
                "unknown_tool",
                f"Unknown tool: {tool_name}",
            )

        scope_error = self._scope_error(registered, context)
        if scope_error:
            return self._failure(
                invocation_id,
                tool_name,
                "scope_error",
                scope_error,
            )

        untrusted_arguments = dict(arguments or {})
        supplied_trusted = sorted(TRUSTED_ARGUMENT_NAMES & untrusted_arguments.keys())
        for trusted_name in supplied_trusted:
            untrusted_arguments.pop(trusted_name, None)

        try:
            validated_input = registered.spec.input_model.model_validate(untrusted_arguments)
        except ValidationError as error:
            return self._failure(
                invocation_id,
                tool_name,
                "input_validation",
                "Tool arguments failed validation.",
                details={"errors": error.errors(include_url=False)},
            )

        cache_key = self._cache_key(
            registered,
            validated_input,
            context,
            idempotency_key,
        )
        if cache_key:
            with self._cache_lock:
                cached = self._idempotency_cache.get(cache_key)
            if cached is not None:
                cached_result = cached.model_copy(deep=True)
                cached_result.cached = True
                invocation_id = cached_result.invocation_id
                self._emit(
                    "tool_started",
                    invocation_id=invocation_id,
                    tool_name=tool_name,
                    attempt=cached_result.attempt,
                    sanitized_arguments=validated_input.model_dump(mode="json"),
                    property_code=context.property_code,
                    run_id=context.run_id,
                    cached=True,
                )
                self._emit(
                    "tool_succeeded",
                    invocation_id=invocation_id,
                    tool_name=tool_name,
                    attempt=cached_result.attempt,
                    duration_ms=0,
                    property_code=context.property_code,
                    run_id=context.run_id,
                    cached=True,
                    output_summary=self._output_summary(cached_result.data),
                )
                return cached_result

        if not self.budget.consume():
            return self._failure(
                invocation_id,
                tool_name,
                "budget_exceeded",
                f"Tool-call budget of {self.budget.max_calls} was exhausted.",
            )

        self._emit(
            "tool_started",
            invocation_id=invocation_id,
            tool_name=tool_name,
            attempt=1,
            sanitized_arguments=validated_input.model_dump(mode="json"),
            ignored_trusted_arguments=supplied_trusted,
            property_code=context.property_code,
            run_id=context.run_id,
        )

        result = self._execute_attempts(
            invocation_id,
            registered,
            validated_input,
            context,
        )
        if result.status == "succeeded" and cache_key:
            with self._cache_lock:
                self._idempotency_cache[cache_key] = result.model_copy(deep=True)
        return result

    def _execute_attempts(
        self,
        invocation_id: str,
        registered: RegisteredTool,
        validated_input: BaseModel,
        context: TrustedToolContext,
    ) -> ToolResult:
        spec = registered.spec
        total_started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, spec.max_attempts + 1):
            future: Future[Any] = self.pool.submit(
                registered.handler,
                validated_input,
                context,
            )
            try:
                raw_output = future.result(timeout=spec.timeout_seconds)
                validated_output = spec.output_model.model_validate(raw_output)
                data = validated_output.model_dump(mode="json")
                output_bytes = len(
                    json.dumps(data, ensure_ascii=True, default=str).encode("utf-8")
                )
                if output_bytes > spec.max_output_bytes:
                    return self._failure(
                        invocation_id,
                        spec.name,
                        "output_too_large",
                        (
                            f"Tool output was {output_bytes} bytes; maximum is "
                            f"{spec.max_output_bytes} bytes."
                        ),
                        attempt=attempt,
                        duration_ms=self._duration_ms(total_started),
                    )

                completed_at = datetime.now(UTC).isoformat()
                result = ToolResult(
                    invocation_id=invocation_id,
                    tool_name=spec.name,
                    status="succeeded",
                    attempt=attempt,
                    duration_ms=self._duration_ms(total_started),
                    query_parameters=validated_input.model_dump(mode="json"),
                    data_timestamp=self._data_timestamp(data),
                    completed_at=completed_at,
                    data=data,
                )
                self._emit(
                    "tool_succeeded",
                    invocation_id=invocation_id,
                    tool_name=spec.name,
                    attempt=attempt,
                    duration_ms=result.duration_ms,
                    property_code=context.property_code,
                    run_id=context.run_id,
                    data_timestamp=result.data_timestamp,
                    output_summary=self._output_summary(result.data),
                )
                return result
            except FutureTimeout:
                future.cancel()
                last_error = TimeoutError(
                    f"Tool exceeded its {spec.timeout_seconds:g}s timeout."
                )
            except ValidationError as error:
                return self._failure(
                    invocation_id,
                    spec.name,
                    "output_validation",
                    "Tool output failed validation.",
                    attempt=attempt,
                    duration_ms=self._duration_ms(total_started),
                    details={"errors": error.errors(include_url=False)},
                )
            except Exception as error:
                last_error = error

            retryable = self._is_transient(last_error)
            can_retry = retryable and spec.idempotent and attempt < spec.max_attempts
            if not can_retry:
                error_type = self._error_type(last_error, retryable)
                return self._failure(
                    invocation_id,
                    spec.name,
                    error_type,
                    str(last_error) or type(last_error).__name__,
                    attempt=attempt,
                    duration_ms=self._duration_ms(total_started),
                    retryable=retryable,
                )

            delay = min(2.0, 0.1 * (2 ** (attempt - 1))) + (0.05 * self.random_value())
            self._emit(
                "tool_retried",
                invocation_id=invocation_id,
                tool_name=spec.name,
                attempt=attempt,
                duration_ms=self._duration_ms(total_started),
                retry_in_seconds=delay,
                error_type=type(last_error).__name__,
                property_code=context.property_code,
                run_id=context.run_id,
            )
            self.sleep(delay)

        raise AssertionError("tool attempt loop exited unexpectedly")

    @staticmethod
    def _scope_error(
        registered: RegisteredTool,
        context: TrustedToolContext,
    ) -> str | None:
        for scope in registered.spec.required_scopes:
            if not getattr(context, scope, None):
                return f"Required trusted scope is missing: {scope}"
        return None

    @staticmethod
    def _cache_key(
        registered: RegisteredTool,
        validated_input: BaseModel,
        context: TrustedToolContext,
        idempotency_key: str | None,
    ) -> str | None:
        if not registered.spec.idempotent:
            return None
        payload = {
            "tool": registered.spec.name,
            "input": validated_input.model_dump(mode="json"),
            "property_code": context.property_code,
            "user_id": context.user_id,
            "tenant_id": context.tenant_id,
            "key": idempotency_key,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_transient(error: Exception | None) -> bool:
        if isinstance(error, TransientToolError | TimeoutError | ConnectionError):
            return True
        return getattr(error, "errno", None) in {1205, 1213, 2003, 2006, 2013, 2055}

    @staticmethod
    def _error_type(error: Exception | None, retryable: bool) -> str:
        if isinstance(error, TimeoutError):
            return "timeout"
        return "transient_failure" if retryable else "permanent_failure"

    def _failure(
        self,
        invocation_id: str,
        tool_name: str,
        error_type: Any,
        message: str,
        *,
        attempt: int = 0,
        duration_ms: int = 0,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> ToolResult:
        result = ToolResult(
            invocation_id=invocation_id,
            tool_name=tool_name,
            status="failed",
            attempt=attempt,
            duration_ms=duration_ms,
            completed_at=datetime.now(UTC).isoformat(),
            error=ToolError(
                type=error_type,
                message=message,
                retryable=retryable,
                details=details or {},
            ),
        )
        self._emit(
            "tool_failed",
            invocation_id=invocation_id,
            tool_name=tool_name,
            attempt=attempt,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=message,
            error_details=details or {},
        )
        return result

    def _emit(self, event_type: str, **payload: Any) -> None:
        self.trace_sink(
            {
                "event": event_type,
                "timestamp": datetime.now(UTC).isoformat(),
                **payload,
            }
        )

    @staticmethod
    def _log_trace(event: dict[str, Any]) -> None:
        logger.info("tool_event %s", json.dumps(event, sort_keys=True, default=str))

    @staticmethod
    def _duration_ms(started: float) -> int:
        return max(0, round((time.perf_counter() - started) * 1000))

    @staticmethod
    def _output_summary(data: Any) -> dict[str, Any]:
        if isinstance(data, list):
            return {"type": "list", "item_count": len(data)}
        if isinstance(data, dict):
            return {"type": "object", "keys": sorted(str(key) for key in data)[:30]}
        return {"type": type(data).__name__}

    @staticmethod
    def _data_timestamp(data: Any) -> str | None:
        """Return the newest source timestamp exposed by validated tool output."""
        candidates: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"report_month", "scraped_at", "data_timestamp", "updated_at"}:
                        if item is not None:
                            candidates.append(str(item))
                    elif isinstance(item, dict | list):
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(data)
        return max(candidates) if candidates else None

    def cached_results(self) -> dict[str, ToolResult]:
        with self._cache_lock:
            return deepcopy(self._idempotency_cache)
