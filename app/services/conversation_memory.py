from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from typing import Any


class ConversationMemory:
    """Small in-process memory for recent chat turns.

    This keeps the prototype simple: no Redis, no database migration, and no extra
    infrastructure. Memory is scoped by conversation id and property code so a
    follow-up for one property cannot accidentally use another property's context.
    """

    def __init__(self, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self._lock = Lock()
        self._turns: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max_turns)
        )

    def get(self, conversation_id: str | None, property_code: str) -> list[dict[str, Any]]:
        if not conversation_id:
            return []
        key = self._key(conversation_id, property_code)
        with self._lock:
            return list(self._turns.get(key, []))

    def add(
        self,
        conversation_id: str | None,
        property_code: str,
        user_message: str,
        assistant_answer: str,
        tool_result_keys: list[str] | None = None,
        component_types: list[str] | None = None,
    ) -> None:
        if not conversation_id:
            return
        key = self._key(conversation_id, property_code)
        turn = {
            "user": user_message,
            "assistant": assistant_answer,
            "tool_result_keys": tool_result_keys or [],
            "component_types": component_types or [],
        }
        with self._lock:
            self._turns[key].append(turn)

    @staticmethod
    def _key(conversation_id: str, property_code: str) -> str:
        return f"{conversation_id}:{property_code.lower()}"
