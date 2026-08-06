"""MySQL persistence for property-scoped conversation memory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5


class ConversationDatabase(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int: ...

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None: ...

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ConversationScope:
    user_id: str
    conversation_id: str
    property_code: str

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> ConversationScope:
        values = (user_id.strip(), conversation_id.strip(), property_code.strip().lower())
        if not all(values):
            raise ValueError("user_id, conversation_id, and property_code are required")
        return cls(*values)

    @property
    def thread_id(self) -> str:
        identity = f"{self.user_id}\x1f{self.conversation_id}\x1f{self.property_code}"
        return str(uuid5(NAMESPACE_URL, identity))

    @property
    def params(self) -> tuple[str, str, str]:
        return self.user_id, self.conversation_id, self.property_code


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


class ConversationMemoryStore:
    """Store turns and summaries with scope predicates on every read and update."""

    def __init__(self, database: ConversationDatabase) -> None:
        self.database = database

    def ensure_thread(self, scope: ConversationScope) -> None:
        self.database.execute(
            """
            INSERT INTO conversation_threads (
              thread_id, user_id, conversation_id, property_code
            ) VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE updated_at = updated_at
            """,
            (scope.thread_id, *scope.params),
        )

    def add_turn(
        self,
        scope: ConversationScope,
        *,
        run_id: str | None,
        user_message: str,
        assistant_answer: str,
        tool_result_keys: list[str],
        component_types: list[str],
    ) -> str:
        self.ensure_thread(scope)
        turn_id = str(uuid4())
        affected = self.database.execute(
            """
            INSERT INTO conversation_turns (
              turn_id, thread_id, run_id, user_message, assistant_answer,
              tool_result_keys_json, component_types_json
            )
            SELECT %s, threads.thread_id, %s, %s, %s, %s, %s
            FROM conversation_threads AS threads
            WHERE threads.thread_id = %s AND threads.user_id = %s
              AND threads.conversation_id = %s AND threads.property_code = %s
              AND (
                %s IS NULL OR EXISTS (
                  SELECT 1 FROM agent_runs AS runs
                  WHERE runs.run_id = %s AND runs.user_id = threads.user_id
                    AND runs.conversation_id = threads.conversation_id
                    AND runs.property_code = threads.property_code
                )
              )
            """,
            (
                turn_id,
                run_id,
                user_message,
                assistant_answer,
                json.dumps(tool_result_keys, separators=(",", ":")),
                json.dumps(component_types, separators=(",", ":")),
                scope.thread_id,
                *scope.params,
                run_id,
                run_id,
            ),
        )
        if affected != 1:
            raise LookupError("conversation turn run was not found in the requested scope")
        return turn_id

    def get_thread(self, scope: ConversationScope) -> dict[str, Any] | None:
        return self.database.fetch_one(
            """
            SELECT thread_id, summary_text, summarized_through
            FROM conversation_threads
            WHERE thread_id = %s AND user_id = %s AND conversation_id = %s
              AND property_code = %s
            """,
            (scope.thread_id, *scope.params),
        )

    def list_turns(
        self,
        scope: ConversationScope,
        *,
        limit: int | None = None,
        after_sequence: int | None = None,
        newest_first: bool = False,
    ) -> list[dict[str, Any]]:
        filters = [
            "threads.thread_id = %s",
            "threads.user_id = %s",
            "threads.conversation_id = %s",
            "threads.property_code = %s",
        ]
        params: list[Any] = [scope.thread_id, *scope.params]
        if after_sequence is not None:
            filters.append("turns.sequence_id > %s")
            params.append(after_sequence)
        order = "DESC" if newest_first else "ASC"
        limit_sql = ""
        if limit is not None:
            limit_sql = " LIMIT %s"
            params.append(limit)
        rows = self.database.fetch_all(
            f"""
            SELECT turns.sequence_id, turns.turn_id, turns.run_id,
                   turns.user_message, turns.assistant_answer,
                   turns.tool_result_keys_json, turns.component_types_json,
                   turns.created_at
            FROM conversation_turns AS turns
            INNER JOIN conversation_threads AS threads
              ON threads.thread_id = turns.thread_id
            WHERE {" AND ".join(filters)}
            ORDER BY turns.sequence_id {order}{limit_sql}
            """,
            tuple(params),
        )
        for row in rows:
            row["tool_result_keys"] = _json_load(row.pop("tool_result_keys_json", None), [])
            row["component_types"] = _json_load(row.pop("component_types_json", None), [])
        return rows

    def update_summary(
        self,
        scope: ConversationScope,
        *,
        summary: str,
        summarized_through: int,
        expected_through: int,
    ) -> bool:
        affected = self.database.execute(
            """
            UPDATE conversation_threads
            SET summary_text = %s, summarized_through = %s
            WHERE thread_id = %s AND user_id = %s AND conversation_id = %s
              AND property_code = %s AND summarized_through = %s
            """,
            (
                summary,
                summarized_through,
                scope.thread_id,
                *scope.params,
                expected_through,
            ),
        )
        return affected == 1

    def list_artifacts(
        self,
        scope: ConversationScope,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = [
            "runs.user_id = %s",
            "runs.conversation_id = %s",
            "runs.property_code = %s",
        ]
        params: list[Any] = list(scope.params)
        if run_id is not None:
            filters.append("runs.run_id = %s")
            params.append(run_id)
        rows = self.database.fetch_all(
            f"""
            SELECT artifacts.artifact_id, artifacts.run_id,
                   artifacts.artifact_type, artifacts.name,
                   artifacts.content_json, artifacts.storage_uri,
                   artifacts.content_hash, artifacts.created_at
            FROM agent_artifacts AS artifacts
            INNER JOIN agent_runs AS runs ON runs.run_id = artifacts.run_id
            WHERE {" AND ".join(filters)}
            ORDER BY artifacts.created_at, artifacts.artifact_id
            """,
            tuple(params),
        )
        for row in rows:
            row["content"] = _json_load(row.pop("content_json", None), None)
        return rows

    def list_evidence(
        self,
        scope: ConversationScope,
        *,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = [
            "runs.user_id = %s",
            "runs.conversation_id = %s",
            "runs.property_code = %s",
            "evidence.property_code = runs.property_code",
        ]
        params: list[Any] = list(scope.params)
        if run_id is not None:
            filters.append("runs.run_id = %s")
            params.append(run_id)
        rows = self.database.fetch_all(
            f"""
            SELECT evidence.citation_id, evidence.run_id, evidence.source_type,
                   evidence.source_name, evidence.document_id, evidence.chunk_id,
                   evidence.content_hash, evidence.source_url,
                   evidence.evidence_json, evidence.retrieved_at
            FROM citation_evidence AS evidence
            INNER JOIN agent_runs AS runs ON runs.run_id = evidence.run_id
            WHERE {" AND ".join(filters)}
            ORDER BY evidence.retrieved_at, evidence.citation_id
            """,
            tuple(params),
        )
        for row in rows:
            row["evidence"] = _json_load(row.pop("evidence_json", None), None)
        return rows
