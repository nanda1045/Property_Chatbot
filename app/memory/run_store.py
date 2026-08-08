"""MySQL-backed persistence for agent runs and execution steps."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal, Protocol, cast
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.agents.state import AgentState, RunStatus

StepStatus = Literal["succeeded", "failed", "cancelled"]

OBSERVABILITY_EVENT_TYPES = frozenset(
    {
        "run_created",
        "planning_started",
        "plan_created",
        "step_started",
        "tool_started",
        "tool_succeeded",
        "tool_failed",
        "tool_retried",
        "approval_requested",
        "approval_received",
        "verification_started",
        "verification_failed",
        "run_completed",
        "run_failed",
        "run_cancelled",
        "AUTHENTICATED",
        "AUTHORIZATION_ALLOWED",
        "AUTHORIZATION_DENIED",
        "SQL_APPROVAL_AUTHORIZED",
        "SQL_APPROVAL_DENIED",
    }
)

PRIVATE_PAYLOAD_KEYS = frozenset(
    {
        "chainofthought",
        "hiddenreasoning",
        "privatereasoning",
        "reasoning",
        "systemprompt",
        "thoughts",
        "accesstoken",
        "authorizationheader",
        "bearertoken",
        "prompt",
        "rawprompt",
        "refreshtoken",
        "token",
    }
)


class RunDatabase(Protocol):
    def execute(self, query: str, params: tuple[Any, ...] = ()) -> int: ...

    def fetch_one(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None: ...

    def fetch_all(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]: ...


def _json_dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _json_load(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


class AgentRunStore:
    """Persist and reload run state within trusted user/property scope."""

    def __init__(self, database: RunDatabase) -> None:
        self.database = database

    def create(self, state: AgentState) -> None:
        self.database.execute(
            """
            INSERT INTO agent_runs (
              run_id, conversation_id, user_id, property_code, user_goal, status,
              current_step, max_steps, plan_json, observations_json,
              pending_approval_json, artifacts_json, citations_json, tool_call_count,
              max_tool_calls, error_json, final_answer
            ) VALUES (
              %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            self._state_params(state),
        )

    def save(self, state: AgentState) -> None:
        affected = self.database.execute(
            """
            UPDATE agent_runs SET
              status = %s, current_step = %s, max_steps = %s, plan_json = %s,
              observations_json = %s, pending_approval_json = %s, artifacts_json = %s,
              citations_json = %s, tool_call_count = %s, max_tool_calls = %s,
              error_json = %s, final_answer = %s, version = version + 1
            WHERE run_id = %s AND user_id = %s AND property_code = %s
              AND (status <> 'cancelled' OR %s = 'cancelled')
            """,
            (
                state["status"],
                state["current_step"],
                state["max_steps"],
                _json_dump(state["plan"]),
                _json_dump(state["observations"]),
                _json_dump(state["pending_approval"])
                if state["pending_approval"] is not None
                else None,
                _json_dump(state["artifacts"]),
                _json_dump(state["citations"]),
                state["tool_call_count"],
                state["max_tool_calls"],
                _json_dump(state["error"]) if state["error"] is not None else None,
                state["final_answer"],
                state["run_id"],
                state["user_id"],
                state["property_code"],
                state["status"],
            ),
        )
        if affected != 1:
            raise LookupError("agent run was not found in the requested scope")
        self._persist_artifacts(state)
        self._persist_citations(state)

    def load(self, run_id: str, user_id: str, property_code: str) -> AgentState | None:
        row = self.database.fetch_one(
            """
            SELECT run_id, conversation_id, user_id, property_code, user_goal, status,
                   current_step, max_steps, plan_json, observations_json,
                   pending_approval_json, artifacts_json, citations_json,
                   tool_call_count, max_tool_calls, error_json, final_answer
            FROM agent_runs
            WHERE run_id = %s AND user_id = %s AND property_code = %s
            """,
            (run_id, user_id, property_code.lower()),
        )
        if row is None:
            return None
        return self._row_to_state(row)

    def load_scoped(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> AgentState | None:
        row = self.database.fetch_one(
            """
            SELECT run_id, conversation_id, user_id, property_code, user_goal, status,
                   current_step, max_steps, plan_json, observations_json,
                   pending_approval_json, artifacts_json, citations_json,
                   tool_call_count, max_tool_calls, error_json, final_answer
            FROM agent_runs
            WHERE run_id = %s AND user_id = %s AND conversation_id = %s
              AND property_code = %s
            """,
            (run_id, user_id, conversation_id, property_code.lower()),
        )
        if row is None:
            return None
        return self._row_to_state(row)

    def get_run_detail(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> dict[str, Any] | None:
        row = self.database.fetch_one(
            """
            SELECT run_id, conversation_id, property_code, user_goal, status,
                   current_step, max_steps, plan_json, pending_approval_json,
                   tool_call_count, max_tool_calls, error_json, final_answer,
                   created_at, updated_at
            FROM agent_runs
            WHERE run_id = %s AND user_id = %s AND conversation_id = %s
              AND property_code = %s
            """,
            (run_id, user_id, conversation_id, property_code.lower()),
        )
        if row is None:
            return None
        row["plan"] = _json_load(row.pop("plan_json", None), [])
        row["pending_approval"] = _json_load(
            row.pop("pending_approval_json", None), None
        )
        row["error"] = _json_load(row.pop("error_json", None), None)
        return row

    def checkpoint(self, state: AgentState, transition_name: str) -> str:
        """Append an immutable state snapshot after a meaningful transition."""
        checkpoint_id = str(uuid4())
        self.database.execute(
            """
            INSERT INTO agent_checkpoints (
              checkpoint_id, run_id, sequence_number, transition_name, state_json
            )
            SELECT %s, %s, COALESCE(MAX(sequence_number), 0) + 1, %s, %s
            FROM agent_checkpoints WHERE run_id = %s
            """,
            (
                checkpoint_id,
                state["run_id"],
                transition_name,
                _json_dump(state),
                state["run_id"],
            ),
        )
        return checkpoint_id

    def load_latest_checkpoint(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> AgentState | None:
        """Reload the latest checkpoint inside the trusted run scope."""
        row = self.database.fetch_one(
            """
            SELECT checkpoints.state_json
            FROM agent_checkpoints AS checkpoints
            INNER JOIN agent_runs AS runs ON runs.run_id = checkpoints.run_id
            WHERE checkpoints.run_id = %s
              AND runs.user_id = %s
              AND runs.conversation_id = %s
              AND runs.property_code = %s
            ORDER BY checkpoints.sequence_number DESC
            LIMIT 1
            """,
            (run_id, user_id, conversation_id, property_code.lower()),
        )
        if row is None:
            return None
        return cast(AgentState, _json_load(row["state_json"], None))

    def claim_approval(self, run_id: str, user_id: str, property_code: str) -> bool:
        """Atomically claim a waiting approval so duplicate clicks cannot execute twice."""
        affected = self.database.execute(
            """
            UPDATE agent_runs SET status = 'running', version = version + 1
            WHERE run_id = %s AND user_id = %s AND property_code = %s
              AND status = 'waiting_for_approval'
            """,
            (run_id, user_id, property_code.lower()),
        )
        return affected == 1

    def claim_cancellation(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> bool:
        """Atomically prevent a nonterminal run from doing more durable work."""
        affected = self.database.execute(
            """
            UPDATE agent_runs SET status = 'cancelled', version = version + 1
            WHERE run_id = %s AND user_id = %s AND conversation_id = %s
              AND property_code = %s
              AND status NOT IN ('completed', 'failed', 'cancelled')
            """,
            (run_id, user_id, conversation_id, property_code.lower()),
        )
        return affected == 1

    def start_step(
        self,
        state: AgentState,
        step_type: str,
        input_data: dict[str, Any] | None = None,
    ) -> str:
        step_id = str(uuid4())
        state["current_step"] += 1
        self.database.execute(
            """
            INSERT INTO agent_steps (
              step_id, run_id, step_number, step_type, status, input_json
            ) VALUES (%s, %s, %s, %s, 'running', %s)
            """,
            (
                step_id,
                state["run_id"],
                state["current_step"],
                step_type,
                _json_dump(input_data) if input_data is not None else None,
            ),
        )
        self.save(state)
        return step_id

    def finish_step(
        self,
        step_id: str,
        *,
        status: StepStatus,
        output_data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        affected = self.database.execute(
            """
            UPDATE agent_steps SET status = %s, output_json = %s, error_json = %s,
                   completed_at = CURRENT_TIMESTAMP(6)
            WHERE step_id = %s
            """,
            (
                status,
                _json_dump(output_data) if output_data is not None else None,
                _json_dump(error) if error is not None else None,
                step_id,
            ),
        )
        if affected != 1:
            raise LookupError("agent step was not found")

    def list_steps(
        self,
        run_id: str,
        user_id: str,
        property_code: str,
    ) -> list[dict[str, Any]]:
        rows = self.database.fetch_all(
            """
            SELECT steps.step_id, steps.run_id, steps.step_number, steps.step_type,
                   steps.status, steps.input_json, steps.output_json, steps.error_json,
                   steps.started_at, steps.completed_at
            FROM agent_steps AS steps
            INNER JOIN agent_runs AS runs ON runs.run_id = steps.run_id
            WHERE steps.run_id = %s AND runs.user_id = %s AND runs.property_code = %s
            ORDER BY steps.step_number
            """,
            (run_id, user_id, property_code.lower()),
        )
        for row in rows:
            row["input"] = _json_load(row.pop("input_json", None), None)
            row["output"] = _json_load(row.pop("output_json", None), None)
            row["error"] = _json_load(row.pop("error_json", None), None)
        return rows

    def list_steps_scoped(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> list[dict[str, Any]]:
        rows = self.database.fetch_all(
            """
            SELECT steps.step_id, steps.run_id, steps.step_number, steps.step_type,
                   steps.status, steps.input_json, steps.output_json, steps.error_json,
                   steps.started_at, steps.completed_at,
                   TIMESTAMPDIFF(
                     MICROSECOND, steps.started_at,
                     COALESCE(steps.completed_at, CURRENT_TIMESTAMP(6))
                   ) DIV 1000 AS duration_ms
            FROM agent_steps AS steps
            INNER JOIN agent_runs AS runs ON runs.run_id = steps.run_id
            WHERE steps.run_id = %s AND runs.user_id = %s
              AND runs.conversation_id = %s AND runs.property_code = %s
            ORDER BY steps.step_number
            """,
            (run_id, user_id, conversation_id, property_code.lower()),
        )
        for row in rows:
            row["input"] = _json_load(row.pop("input_json", None), None)
            row["output"] = _json_load(row.pop("output_json", None), None)
            row["error"] = _json_load(row.pop("error_json", None), None)
            if row.get("duration_ms") is not None:
                row["duration_ms"] = int(row["duration_ms"])
        return rows

    def record_event(
        self,
        state: AgentState,
        event_type: str,
        *,
        step_id: str | None = None,
        tool_name: str | None = None,
        attempt: int | None = None,
        duration_ms: int | None = None,
        error_type: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        if event_type not in OBSERVABILITY_EVENT_TYPES:
            raise ValueError(f"unsupported observability event: {event_type}")
        event_id = str(uuid4())
        sanitized_payload = self._sanitize_operational_payload(payload or {})
        affected = self.database.execute(
            """
            INSERT INTO agent_events (
              event_id, run_id, event_type, conversation_id, property_code,
              step_id, tool_name, attempt, duration_ms, error_type, payload_json
            )
            SELECT %s, runs.run_id, %s, runs.conversation_id, runs.property_code,
                   %s, %s, %s, %s, %s, %s
            FROM agent_runs AS runs
            WHERE runs.run_id = %s AND runs.user_id = %s
              AND runs.conversation_id = %s AND runs.property_code = %s
              AND (
                %s IS NULL OR EXISTS (
                  SELECT 1 FROM agent_steps AS scoped_steps
                  WHERE scoped_steps.step_id = %s
                    AND scoped_steps.run_id = runs.run_id
                )
              )
            """,
            (
                event_id,
                event_type,
                step_id,
                tool_name,
                attempt,
                duration_ms,
                error_type,
                _json_dump(sanitized_payload),
                state["run_id"],
                state["user_id"],
                state["conversation_id"],
                state["property_code"],
                step_id,
                step_id,
            ),
        )
        if affected != 1:
            raise LookupError("agent run event scope was not found")
        return event_id

    def list_events(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
        after_sequence: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self.database.fetch_all(
            """
            SELECT events.sequence_id, events.event_id, events.run_id, events.event_type,
                   events.conversation_id, events.property_code, events.step_id,
                   events.tool_name, events.attempt, events.duration_ms,
                   events.event_timestamp AS timestamp, events.error_type,
                   events.payload_json
            FROM agent_events AS events
            INNER JOIN agent_runs AS runs ON runs.run_id = events.run_id
            WHERE events.run_id = %s AND runs.user_id = %s
              AND runs.conversation_id = %s AND runs.property_code = %s
              AND events.conversation_id = runs.conversation_id
              AND events.property_code = runs.property_code
              AND events.sequence_id > %s
            ORDER BY events.sequence_id
            """,
            (
                run_id,
                user_id,
                conversation_id,
                property_code.lower(),
                after_sequence,
            ),
        )
        for row in rows:
            row["payload"] = _json_load(row.pop("payload_json", None), {})
            if row.get("duration_ms") is not None:
                row["duration_ms"] = int(row["duration_ms"])
        return rows

    def record_tool_invocation_event(
        self,
        state: AgentState,
        step_id: str,
        event_type: str,
        tool_name: str,
        attempt: int,
        duration_ms: int | None,
        error_type: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Upsert the durable tool record represented by an operational event."""
        invocation_id = str(payload.get("invocation_id") or "")
        if not invocation_id:
            return
        status = {
            "tool_started": "running",
            "tool_retried": "running",
            "tool_succeeded": "succeeded",
            "tool_failed": "failed",
        }.get(event_type)
        if status is None:
            return
        sanitized_input = payload.get("sanitized_arguments")
        output = payload.get("output_summary")
        error = None
        if status == "failed":
            error = {
                "type": error_type,
                "message": payload.get("error_message"),
                "details": payload.get("error_details") or {},
            }
        affected = self.database.execute(
            """
            INSERT INTO tool_invocations (
              invocation_id, run_id, step_id, tool_name, status, attempt,
              sanitized_input_json, output_json, citation_refs_json, error_json,
              duration_ms, completed_at
            )
            SELECT %s, runs.run_id, %s, %s, %s, %s, %s, %s, '[]', %s, %s,
                   CASE WHEN %s IN ('succeeded', 'failed')
                        THEN CURRENT_TIMESTAMP(6) ELSE NULL END
            FROM agent_runs AS runs
            WHERE runs.run_id = %s AND runs.user_id = %s
              AND runs.conversation_id = %s AND runs.property_code = %s
              AND EXISTS (
                SELECT 1 FROM agent_steps AS steps
                WHERE steps.step_id = %s AND steps.run_id = runs.run_id
              )
            ON DUPLICATE KEY UPDATE
              status = VALUES(status), attempt = VALUES(attempt),
              sanitized_input_json = COALESCE(
                VALUES(sanitized_input_json), sanitized_input_json
              ),
              output_json = COALESCE(VALUES(output_json), output_json),
              error_json = COALESCE(
                VALUES(error_json), tool_invocations.error_json
              ),
              duration_ms = COALESCE(VALUES(duration_ms), duration_ms),
              completed_at = COALESCE(VALUES(completed_at), completed_at)
            """,
            (
                invocation_id,
                step_id,
                tool_name,
                status,
                attempt,
                _json_dump(sanitized_input) if sanitized_input is not None else None,
                _json_dump(output) if output is not None else None,
                _json_dump(error) if error is not None else None,
                duration_ms,
                status,
                state["run_id"],
                state["user_id"],
                state["conversation_id"],
                state["property_code"],
                step_id,
            ),
        )
        if affected not in {1, 2}:
            raise LookupError("tool invocation scope was not found")

    def list_citations_scoped(
        self,
        run_id: str,
        user_id: str,
        conversation_id: str,
        property_code: str,
    ) -> list[dict[str, Any]]:
        rows = self.database.fetch_all(
            """
            SELECT evidence.citation_id, evidence.run_id, evidence.property_code,
                   evidence.source_type, evidence.source_name,
                   evidence.tool_invocation_id, evidence.document_id,
                   evidence.chunk_id, evidence.content_hash, evidence.source_url,
                   evidence.evidence_json, evidence.retrieved_at,
                   evidence.index_version
            FROM citation_evidence AS evidence
            INNER JOIN agent_runs AS runs ON runs.run_id = evidence.run_id
            WHERE evidence.run_id = %s AND runs.user_id = %s
              AND runs.conversation_id = %s AND runs.property_code = %s
              AND evidence.property_code = runs.property_code
            ORDER BY evidence.retrieved_at, evidence.citation_id
            """,
            (run_id, user_id, conversation_id, property_code.lower()),
        )
        for row in rows:
            stored = _json_load(row.pop("evidence_json", None), {})
            row["query_parameters"] = dict(stored.get("query_parameters") or {})
            row["data_timestamp"] = stored.get("data_timestamp")
            row["evidence"] = dict(stored.get("evidence") or stored)
        return rows

    @classmethod
    def _sanitize_operational_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._sanitize_operational_payload(item)
                for key, item in value.items()
                if not cls._is_private_payload_key(str(key))
            }
        if isinstance(value, list):
            return [cls._sanitize_operational_payload(item) for item in value[:50]]
        if isinstance(value, tuple):
            return [cls._sanitize_operational_payload(item) for item in value[:50]]
        if isinstance(value, str) and len(value) > 2000:
            return value[:1997] + "..."
        return value

    @staticmethod
    def _is_private_payload_key(key: str) -> bool:
        normalized = "".join(
            character for character in key.lower() if character.isalnum()
        )
        return normalized in PRIVATE_PAYLOAD_KEYS

    def _persist_artifacts(self, state: AgentState) -> None:
        """Normalize append-only run artifacts for scoped retrieval and audit."""
        for artifact in state["artifacts"]:
            artifact_id = str(artifact.get("artifact_id") or uuid4())
            artifact["artifact_id"] = artifact_id
            artifact_type = str(artifact.get("type") or "structured_output")
            name = str(artifact.get("name") or artifact_type.replace("_", " ").title())
            content_json = _json_dump(artifact)
            content_hash = sha256(content_json.encode("utf-8")).hexdigest()
            self.database.execute(
                """
                INSERT INTO agent_artifacts (
                  artifact_id, run_id, artifact_type, name, content_json,
                  storage_uri, content_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  artifact_type = VALUES(artifact_type), name = VALUES(name),
                  content_json = VALUES(content_json),
                  storage_uri = VALUES(storage_uri),
                  content_hash = VALUES(content_hash)
                """,
                (
                    artifact_id,
                    state["run_id"],
                    artifact_type,
                    name,
                    content_json,
                    artifact.get("storage_uri"),
                    content_hash,
                ),
            )

    def _persist_citations(self, state: AgentState) -> None:
        """Normalize evidence while enforcing the run's property scope."""
        citation_ids = [
            str(citation.get("citation_id"))
            for citation in state["citations"]
            if citation.get("citation_id")
        ]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("duplicate citation IDs were returned")
        for citation in state["citations"]:
            citation_property = str(
                citation.get("property_code") or state["property_code"]
            ).lower()
            if citation_property != state["property_code"]:
                raise ValueError("citation belongs to another property")
            canonical = _json_dump(citation)
            citation_id = str(
                citation.get("citation_id")
                or uuid5(NAMESPACE_URL, f"{state['run_id']}:{canonical}")
            )
            source_type = str(citation.get("source_type") or "retrieval")
            evidence = dict(citation.get("evidence") or {})
            if source_type == "retrieval":
                if not citation.get("chunk_id"):
                    raise ValueError("retrieval citation is missing a chunk ID")
                if not citation.get("document_id"):
                    raise ValueError("retrieval citation is missing a document ID")
                evidence_chunk = str(evidence.get("id") or "")
                if evidence_chunk and evidence_chunk != str(citation["chunk_id"]):
                    raise ValueError(
                        "retrieval citation does not reference its returned chunk"
                    )
                metadata = dict(evidence.get("metadata") or {})
                evidence_property = str(metadata.get("property_code") or "").lower()
                if evidence_property and evidence_property != state["property_code"]:
                    raise ValueError("citation evidence belongs to another property")
            elif source_type != "structured_tool":
                raise ValueError(f"unsupported citation source type: {source_type}")
            source_name = str(
                citation.get("source_name")
                or citation.get("title")
                or citation.get("tool")
                or "property evidence"
            )
            content_hash = str(
                citation.get("content_hash")
                or sha256(canonical.encode("utf-8")).hexdigest()
            )
            if evidence and citation.get("content_hash"):
                evidence_json = json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    default=str,
                )
                expected_hash = sha256(evidence_json.encode("utf-8")).hexdigest()
                if content_hash != expected_hash:
                    raise ValueError("citation content hash does not match evidence")
            tool_invocation_id = citation.get("tool_invocation_id")
            if source_type == "structured_tool" and tool_invocation_id is None:
                raise ValueError("structured citation is missing its tool invocation")
            if tool_invocation_id is not None:
                invocation = self.database.fetch_one(
                    """
                    SELECT invocations.invocation_id
                    FROM tool_invocations AS invocations
                    INNER JOIN agent_runs AS runs ON runs.run_id = invocations.run_id
                    WHERE invocations.invocation_id = %s
                      AND invocations.run_id = %s AND runs.user_id = %s
                      AND runs.conversation_id = %s AND runs.property_code = %s
                    """,
                    (
                        tool_invocation_id,
                        state["run_id"],
                        state["user_id"],
                        state["conversation_id"],
                        state["property_code"],
                    ),
                )
                if invocation is None:
                    raise ValueError(
                        "citation references an unsupported tool invocation"
                    )
            retrieved_at = self._citation_timestamp(citation.get("retrieved_at"))
            self.database.execute(
                """
                INSERT INTO citation_evidence (
                  citation_id, run_id, property_code, source_type, source_name,
                  tool_invocation_id, document_id, chunk_id, content_hash,
                  source_url, evidence_json, retrieved_at, index_version
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  source_type = VALUES(source_type),
                  source_name = VALUES(source_name),
                  document_id = VALUES(document_id),
                  chunk_id = VALUES(chunk_id),
                  content_hash = VALUES(content_hash),
                  source_url = VALUES(source_url),
                  evidence_json = VALUES(evidence_json),
                  retrieved_at = VALUES(retrieved_at),
                  index_version = VALUES(index_version)
                """,
                (
                    citation_id,
                    state["run_id"],
                    state["property_code"],
                    source_type,
                    source_name,
                    tool_invocation_id,
                    citation.get("document_id"),
                    citation.get("chunk_id"),
                    content_hash,
                    citation.get("source_url"),
                    canonical,
                    retrieved_at,
                    citation.get("index_version"),
                ),
            )

    @staticmethod
    def _citation_timestamp(value: Any) -> datetime:
        if not value:
            return datetime.now(UTC).replace(tzinfo=None)
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC).replace(tzinfo=None)
        return parsed

    @staticmethod
    def _state_params(state: AgentState) -> tuple[Any, ...]:
        return (
            state["run_id"],
            state["conversation_id"],
            state["user_id"],
            state["property_code"],
            state["user_goal"],
            state["status"],
            state["current_step"],
            state["max_steps"],
            _json_dump(state["plan"]),
            _json_dump(state["observations"]),
            _json_dump(state["pending_approval"])
            if state["pending_approval"] is not None
            else None,
            _json_dump(state["artifacts"]),
            _json_dump(state["citations"]),
            state["tool_call_count"],
            state["max_tool_calls"],
            _json_dump(state["error"]) if state["error"] is not None else None,
            state["final_answer"],
        )

    @staticmethod
    def _row_to_state(row: dict[str, Any]) -> AgentState:
        return AgentState(
            run_id=str(row["run_id"]),
            conversation_id=str(row["conversation_id"]),
            user_id=str(row["user_id"]),
            property_code=str(row["property_code"]),
            user_goal=str(row["user_goal"]),
            status=cast(RunStatus, row["status"]),
            current_step=int(row["current_step"]),
            max_steps=int(row["max_steps"]),
            plan=_json_load(row["plan_json"], []),
            observations=_json_load(row["observations_json"], []),
            pending_approval=_json_load(row["pending_approval_json"], None),
            artifacts=_json_load(row["artifacts_json"], []),
            citations=_json_load(row["citations_json"], []),
            tool_call_count=int(row["tool_call_count"]),
            max_tool_calls=int(row["max_tool_calls"]),
            error=_json_load(row["error_json"], None),
            final_answer=row["final_answer"],
        )
