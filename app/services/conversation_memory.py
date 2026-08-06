from __future__ import annotations

from typing import Any

from app.memory.conversation_store import ConversationMemoryStore, ConversationScope


class ConversationMemory:
    """Durable recent-turn and rolling-summary memory for one trusted scope."""

    def __init__(
        self,
        store: ConversationMemoryStore,
        max_turns: int = 8,
        max_summary_chars: int = 4000,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if max_summary_chars < 256:
            raise ValueError("max_summary_chars must be at least 256")
        self.store = store
        self.max_turns = max_turns
        self.max_summary_chars = max_summary_chars

    def get(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        property_code: str,
    ) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        scope = ConversationScope.create(
            user_id=user_id,
            conversation_id=conversation_id,
            property_code=property_code,
        )
        thread = self.store.get_thread(scope)
        if thread is None:
            return []
        turns = self.store.list_turns(
            scope,
            limit=self.max_turns,
            newest_first=True,
        )
        history = [self._history_turn(turn) for turn in reversed(turns)]
        summary = str(thread.get("summary_text") or "").strip()
        if summary:
            history.insert(
                0,
                {
                    "memory_type": "summary",
                    "summary": summary,
                    "summarized_through": int(thread["summarized_through"]),
                },
            )
        return history

    def add(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        property_code: str,
        user_message: str,
        assistant_answer: str,
        run_id: str | None = None,
        tool_result_keys: list[str] | None = None,
        component_types: list[str] | None = None,
    ) -> None:
        if not conversation_id:
            return
        scope = ConversationScope.create(
            user_id=user_id,
            conversation_id=conversation_id,
            property_code=property_code,
        )
        self.store.add_turn(
            scope,
            run_id=run_id,
            user_message=user_message,
            assistant_answer=assistant_answer,
            tool_result_keys=tool_result_keys or [],
            component_types=component_types or [],
        )
        self._compact(scope)

    def artifacts(
        self,
        *,
        user_id: str,
        conversation_id: str,
        property_code: str,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = ConversationScope.create(
            user_id=user_id,
            conversation_id=conversation_id,
            property_code=property_code,
        )
        return self.store.list_artifacts(scope, run_id=run_id)

    def evidence(
        self,
        *,
        user_id: str,
        conversation_id: str,
        property_code: str,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        scope = ConversationScope.create(
            user_id=user_id,
            conversation_id=conversation_id,
            property_code=property_code,
        )
        return self.store.list_evidence(scope, run_id=run_id)

    def _compact(self, scope: ConversationScope) -> None:
        for _ in range(2):
            thread = self.store.get_thread(scope)
            if thread is None:
                return
            summarized_through = int(thread.get("summarized_through") or 0)
            unsummarized = self.store.list_turns(
                scope,
                after_sequence=summarized_through,
            )
            compactable = unsummarized[: -self.max_turns]
            if not compactable:
                return
            additions = [self._summary_line(turn) for turn in compactable]
            prior = str(thread.get("summary_text") or "").strip()
            summary = "\n".join(part for part in [prior, *additions] if part)
            if len(summary) > self.max_summary_chars:
                summary = "…" + summary[-(self.max_summary_chars - 1) :]
            latest_sequence = int(compactable[-1]["sequence_id"])
            if self.store.update_summary(
                scope,
                summary=summary,
                summarized_through=latest_sequence,
                expected_through=summarized_through,
            ):
                return

    @staticmethod
    def _history_turn(turn: dict[str, Any]) -> dict[str, Any]:
        return {
            "memory_type": "turn",
            "turn_id": str(turn["turn_id"]),
            "run_id": turn.get("run_id"),
            "user": str(turn["user_message"]),
            "assistant": str(turn["assistant_answer"]),
            "tool_result_keys": list(turn.get("tool_result_keys") or []),
            "component_types": list(turn.get("component_types") or []),
        }

    @staticmethod
    def _summary_line(turn: dict[str, Any]) -> str:
        user = " ".join(str(turn["user_message"]).split())[:240]
        assistant = " ".join(str(turn["assistant_answer"]).split())[:360]
        tools = ", ".join(str(value) for value in turn.get("tool_result_keys") or [])
        suffix = f" Tools: {tools}." if tools else ""
        return f"- User asked: {user} | Assistant answered: {assistant}.{suffix}"
