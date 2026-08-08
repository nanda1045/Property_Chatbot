from __future__ import annotations

import asyncio
import json
import time
import unittest
from threading import Event
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.agents.cancellation import AgentRunCancelledError
from app.core.auth import local_authenticated_user
from app.core.config import Settings
from app.main import app, chat_stream
from app.schemas import ChatRequest, ChatResponse
from app.services.run_stream import (
    BoundedStreamExecutor,
    RunStreamBuffer,
    StreamExecutorSaturatedError,
    active_run_cancellations,
)

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
    "created_at": "2026-08-07T00:00:00",
    "updated_at": "2026-08-07T00:00:01",
}


def parse_sse(body: str) -> list[dict[str, str]]:
    parsed = []
    for block in body.strip().split("\n\n"):
        event: dict[str, str] = {}
        for line in block.splitlines():
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            event[key] = value
        if event:
            parsed.append(event)
    return parsed


class RunStreamBufferTests(unittest.TestCase):
    def test_queue_is_bounded_and_terminal_event_survives_token_pressure(self) -> None:
        buffer = RunStreamBuffer(max_size=2)

        self.assertTrue(buffer.publish("token", {"delta": "one"}))
        self.assertTrue(buffer.publish("token", {"delta": "two"}))
        self.assertFalse(buffer.publish("token", {"delta": "three"}))
        self.assertEqual(buffer.size, 2)
        self.assertTrue(buffer.publish("final", {"run_status": "completed"}))
        buffer.close()

        item = buffer.get(timeout=0.01)
        self.assertEqual(item.event, "final")
        self.assertIsNone(buffer.get(timeout=0.01))
        self.assertGreaterEqual(buffer.dropped_tokens, 2)

    def test_worker_pool_rejects_an_unbounded_pending_backlog(self) -> None:
        release_worker = Event()
        executor = BoundedStreamExecutor(max_workers=1, max_pending=0)
        first = executor.submit(lambda: release_worker.wait(timeout=1))
        try:
            with self.assertRaises(StreamExecutorSaturatedError):
                executor.submit(lambda: None)
        finally:
            release_worker.set()
            first.result(timeout=1)
            executor.shutdown()


class StreamingApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_chat_stream_announces_run_and_releases_worker_registration(self) -> None:
        calls: list[dict] = []

        class FakeRuntime:
            def __init__(self, settings) -> None:
                pass

            def answer(self, **kwargs) -> ChatResponse:
                calls.append(kwargs)
                kwargs["on_token"]("hello ")
                kwargs["on_token"]("world")
                return ChatResponse(
                    property_code="115r",
                    model="mock:test",
                    run_id=kwargs["run_id"],
                    run_status="completed",
                    answer_markdown="hello world",
                )

        memory = MagicMock()
        with (
            patch("app.main.AgentRuntime", FakeRuntime),
            patch("app.main._conversation_memory", return_value=memory),
        ):
            response = self.client.post(
                "/chat/stream",
                json={
                    "property_code": "115r",
                    "model": "mock:test",
                    "message": "Hello",
                    "conversation_id": "conversation-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-cache, no-transform")
        events = parse_sse(response.text)
        self.assertEqual(events[0]["event"], "status")
        start = json.loads(events[0]["data"])
        final = json.loads(next(item["data"] for item in events if item["event"] == "final"))
        self.assertEqual(start["run_id"], calls[0]["run_id"])
        self.assertEqual(final["run_id"], start["run_id"])
        self.assertIn(start["run_id"], start["reconnect_url"])
        self.assertFalse(active_run_cancellations.request(start["run_id"]))
        self.assertEqual(memory.add.call_args.kwargs["run_id"], start["run_id"])

    def test_chat_stream_does_not_expose_internal_failure_details(self) -> None:
        class FailingRuntime:
            def __init__(self, settings) -> None:
                pass

            def answer(self, **kwargs) -> ChatResponse:
                raise RuntimeError("private database and SQL details")

        with (
            patch("app.main.AgentRuntime", FailingRuntime),
            patch("app.main._conversation_memory", return_value=MagicMock()),
        ):
            response = self.client.post(
                "/chat/stream",
                json={
                    "property_code": "115r",
                    "model": "mock:test",
                    "message": "Hello",
                    "conversation_id": "conversation-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        error_event = next(
            item for item in parse_sse(response.text) if item["event"] == "error"
        )
        self.assertEqual(json.loads(error_event["data"])["detail"], "agent run failed")
        self.assertNotIn("private database", response.text)

    def test_run_event_stream_replays_only_missed_events(self) -> None:
        calls: list[dict] = []

        class FakeRuntime:
            def __init__(self, settings) -> None:
                pass

            def get_run(self, **kwargs):
                return RUN_DETAIL

            def list_run_events(self, **kwargs):
                calls.append(kwargs)
                return [
                    {
                        "sequence_id": 8,
                        "event_id": "event-8",
                        "run_id": "run-1",
                        "event_type": "run_completed",
                        "conversation_id": "conversation-1",
                        "property_code": "115r",
                        "timestamp": "2026-08-07T00:00:01",
                        "payload": {"tool_calls": 1},
                    }
                ]

        with patch("app.main.AgentRuntime", FakeRuntime):
            response = self.client.get(
                "/api/agent-runs/run-1/stream",
                params={
                    "property_code": "115r",
                    "conversation_id": "conversation-1",
                    "after_sequence": 3,
                    "follow": "false",
                },
                headers={"Last-Event-ID": "7"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls[0]["after_sequence"], 7)
        events = parse_sse(response.text)
        self.assertEqual(events[0]["id"], "8")
        self.assertEqual(events[0]["event"], "run_event")
        self.assertEqual(events[-1]["event"], "run_status")
        self.assertEqual(json.loads(events[-1]["data"])["status"], "completed")

    def test_run_event_stream_rejects_invalid_replay_cursor(self) -> None:
        response = self.client.get(
            "/api/agent-runs/run-1/stream",
            params={
                "property_code": "115r",
                "conversation_id": "conversation-1",
            },
            headers={"Last-Event-ID": "not-a-sequence"},
        )

        self.assertEqual(response.status_code, 400)

        negative_response = self.client.get(
            "/api/agent-runs/run-1/stream",
            params={
                "property_code": "115r",
                "conversation_id": "conversation-1",
            },
            headers={"Last-Event-ID": "-1"},
        )
        self.assertEqual(negative_response.status_code, 400)

    def test_client_disconnect_cancels_worker_and_unregisters_run(self) -> None:
        calls: dict[str, object] = {}

        class DisconnectedRequest:
            async def is_disconnected(self) -> bool:
                return True

        class FakeRuntime:
            def __init__(self, settings) -> None:
                pass

            def answer(self, **kwargs) -> ChatResponse:
                calls["run_id"] = kwargs["run_id"]
                while not kwargs["cancellation_requested"]():
                    time.sleep(0.001)
                raise AgentRunCancelledError("disconnected")

            def cancel_run(self, **kwargs):
                calls["cancelled"] = kwargs
                return {**RUN_DETAIL, "run_id": kwargs["run_id"], "status": "cancelled"}

        async def consume() -> list[str]:
            settings = Settings(
                _env_file=None,
                stream_poll_interval_seconds=0.01,
                stream_thread_join_seconds=1,
            )
            response = await chat_stream(
                ChatRequest(
                    property_code="115r",
                    model="mock:test",
                    message="Long response",
                    conversation_id="conversation-1",
                ),
                DisconnectedRequest(),
                settings,
                local_authenticated_user(settings),
                None,
            )
            return [chunk async for chunk in response.body_iterator]

        with (
            patch("app.main.AgentRuntime", FakeRuntime),
            patch("app.main._conversation_memory", return_value=MagicMock()),
        ):
            chunks = asyncio.run(consume())

        self.assertEqual(len(chunks), 1)
        self.assertIn("event: status", chunks[0])
        self.assertEqual(calls["cancelled"]["run_id"], calls["run_id"])
        self.assertFalse(active_run_cancellations.request(str(calls["run_id"])))


if __name__ == "__main__":
    unittest.main()
