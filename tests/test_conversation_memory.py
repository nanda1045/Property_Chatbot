from __future__ import annotations

import unittest
from typing import Any
from uuid import uuid4

from app.memory.conversation_store import ConversationMemoryStore, ConversationScope
from app.services.conversation_memory import ConversationMemory
from app.services.langchain_orchestrator import LangChainOrchestrator


class MemoryStore:
    def __init__(self) -> None:
        self.threads: dict[str, dict[str, Any]] = {}
        self.turns: list[dict[str, Any]] = []

    def ensure_thread(self, scope: ConversationScope) -> None:
        self.threads.setdefault(
            scope.thread_id,
            {
                "scope": scope,
                "thread_id": scope.thread_id,
                "summary_text": None,
                "summarized_through": 0,
            },
        )

    def add_turn(self, scope: ConversationScope, **values: Any) -> str:
        self.ensure_thread(scope)
        turn_id = str(uuid4())
        self.turns.append(
            {
                "scope": scope,
                "sequence_id": len(self.turns) + 1,
                "turn_id": turn_id,
                "run_id": values["run_id"],
                "user_message": values["user_message"],
                "assistant_answer": values["assistant_answer"],
                "tool_result_keys": values["tool_result_keys"],
                "component_types": values["component_types"],
                "created_at": "now",
            }
        )
        return turn_id

    def get_thread(self, scope: ConversationScope) -> dict[str, Any] | None:
        thread = self.threads.get(scope.thread_id)
        if thread is None or thread["scope"] != scope:
            return None
        return dict(thread)

    def list_turns(
        self,
        scope: ConversationScope,
        *,
        limit: int | None = None,
        after_sequence: int | None = None,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        rows = [turn for turn in self.turns if turn["scope"] == scope]
        if after_sequence is not None:
            rows = [turn for turn in rows if turn["sequence_id"] > after_sequence]
        rows = sorted(
            rows,
            key=lambda turn: turn["sequence_id"],
            reverse=newest_first,
        )
        if limit is not None:
            rows = rows[:limit]
        return [dict(row) for row in rows]

    def update_summary(
        self,
        scope: ConversationScope,
        *,
        summary: str,
        summarized_through: int,
        expected_through: int,
    ) -> bool:
        thread = self.threads.get(scope.thread_id)
        if (
            thread is None
            or thread["scope"] != scope
            or thread["summarized_through"] != expected_through
        ):
            return False
        thread["summary_text"] = summary
        thread["summarized_through"] = summarized_through
        return True

    def list_artifacts(
        self, scope: ConversationScope, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        return []

    def list_evidence(
        self, scope: ConversationScope, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        return []


class RecordingDatabase:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetched: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int:
        self.executed.append((" ".join(query.split()), params))
        return 1

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        self.fetched.append((" ".join(query.split()), params))
        return None

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        self.fetched.append((" ".join(query.split()), params))
        return []


class ConversationMemoryTests(unittest.TestCase):
    def test_turns_survive_service_recreation(self) -> None:
        store = MemoryStore()
        memory = ConversationMemory(store, max_turns=4)
        memory.add(
            user_id="user-1",
            conversation_id="conversation-1",
            property_code="115R",
            run_id="run-1",
            user_message="Show occupancy",
            assistant_answer="Occupancy is 93%.",
            tool_result_keys=["latest_kpis"],
            component_types=["kpi_group"],
        )

        history = ConversationMemory(store, max_turns=4).get(
            user_id="user-1",
            conversation_id="conversation-1",
            property_code="115r",
        )

        self.assertEqual(history[0]["user"], "Show occupancy")
        self.assertEqual(history[0]["run_id"], "run-1")
        self.assertEqual(history[0]["tool_result_keys"], ["latest_kpis"])

    def test_thread_memory_is_isolated_by_user_conversation_and_property(self) -> None:
        store = MemoryStore()
        memory = ConversationMemory(store)
        memory.add(
            user_id="user-1",
            conversation_id="conversation-1",
            property_code="115r",
            user_message="private question",
            assistant_answer="private answer",
        )

        mismatched_scopes = [
            ("user-2", "conversation-1", "115r"),
            ("user-1", "conversation-2", "115r"),
            ("user-1", "conversation-1", "126a"),
        ]
        for user_id, conversation_id, property_code in mismatched_scopes:
            self.assertEqual(
                memory.get(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    property_code=property_code,
                ),
                [],
            )

    def test_older_turns_are_compacted_and_recent_turns_remain_verbatim(self) -> None:
        store = MemoryStore()
        memory = ConversationMemory(store, max_turns=2, max_summary_chars=1000)
        for number in range(1, 5):
            memory.add(
                user_id="user-1",
                conversation_id="conversation-1",
                property_code="115r",
                run_id=f"run-{number}",
                user_message=f"question {number}",
                assistant_answer=f"answer {number}",
            )

        history = memory.get(
            user_id="user-1",
            conversation_id="conversation-1",
            property_code="115r",
        )

        self.assertEqual(history[0]["memory_type"], "summary")
        self.assertIn("question 1", history[0]["summary"])
        self.assertIn("question 2", history[0]["summary"])
        self.assertEqual([turn["user"] for turn in history[1:]], ["question 3", "question 4"])
        prompt_context = LangChainOrchestrator._history_context(history)
        self.assertIn("Earlier conversation summary", prompt_context)
        self.assertIn("User: question 4", prompt_context)

    def test_mysql_store_requires_matching_run_and_scopes_artifact_reads(self) -> None:
        database = RecordingDatabase()
        store = ConversationMemoryStore(database)
        scope = ConversationScope.create(
            user_id="user-1",
            conversation_id="conversation-1",
            property_code="115R",
        )

        store.add_turn(
            scope,
            run_id="run-1",
            user_message="question",
            assistant_answer="answer",
            tool_result_keys=[],
            component_types=[],
        )
        store.list_artifacts(scope, run_id="run-1")
        store.list_evidence(scope, run_id="run-1")

        turn_query, turn_params = database.executed[-1]
        self.assertIn("runs.user_id = threads.user_id", turn_query)
        self.assertEqual(turn_params[-2:], ("run-1", "run-1"))
        for query, params in database.fetched[-2:]:
            self.assertIn("runs.user_id = %s", query)
            self.assertIn("runs.conversation_id = %s", query)
            self.assertIn("runs.property_code = %s", query)
            self.assertEqual(params, ("user-1", "conversation-1", "115r", "run-1"))


if __name__ == "__main__":
    unittest.main()
