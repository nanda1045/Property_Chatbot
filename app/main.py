from __future__ import annotations

import asyncio
import json
import queue
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agents.cancellation import AgentRunCancelledError
from app.agents.runtime import AgentRunConflictError, AgentRunNotFoundError, AgentRuntime
from app.core.config import Settings, get_settings
from app.db.mysql import MySQLDatabase
from app.memory.conversation_store import ConversationMemoryStore
from app.schemas import (
    AgentApprovalRequest,
    AgentRunCitation,
    AgentRunDetail,
    AgentRunEvent,
    AgentRunScopeRequest,
    AgentRunStep,
    ChatRequest,
    ChatResponse,
    SqlApprovalRequest,
    UIComponent,
)
from app.services.conversation_memory import ConversationMemory
from app.services.rent_roll_repository import RentRollRepository
from app.services.run_stream import (
    BoundedStreamExecutor,
    RunStreamBuffer,
    StreamExecutorSaturatedError,
    active_run_cancellations,
)
from app.services.sql_approval import execute_approved_sql

SettingsDep = Annotated[Settings, Depends(get_settings)]

_STREAM_WORKERS = BoundedStreamExecutor(max_workers=8, max_pending=8)
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


app = FastAPI(
    title="Aker Property Assistant",
    version="0.1.0",
    description="Property-scoped AI assistant prototype.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(settings: SettingsDep) -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "default_property_code": settings.default_property_code,
    }


@app.get("/models")
def models(settings: SettingsDep) -> dict[str, object]:
    return {
        "models": [
            {
                "id": "openai:gpt-4.1-mini",
                "label": "OpenAI GPT-4.1 Mini",
                "provider": "openai",
            },
            {
                "id": "anthropic:claude-haiku-4-5-20251001",
                "label": "Claude Haiku 4.5",
                "provider": "anthropic",
            },
            {
                "id": "anthropic:claude-sonnet-4-6",
                "label": "Claude Sonnet 4.6",
                "provider": "anthropic",
            },
        ],
        "default": f"{settings.default_llm_provider}:{settings.default_llm_model}",
    }


@app.get("/properties")
def properties(settings: SettingsDep) -> dict[str, list[dict]]:
    repository = RentRollRepository(MySQLDatabase(settings))
    return {"properties": repository.list_properties()}


@app.post("/chat")
def chat(request: ChatRequest, settings: SettingsDep) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    memory = _conversation_memory(settings)
    history = memory.get(
        user_id=settings.runtime_user_id,
        conversation_id=conversation_id,
        property_code=request.property_code,
    )
    runtime = AgentRuntime(settings)
    response = runtime.answer(
        property_code=request.property_code,
        message=request.message,
        model=request.model,
        history=history,
        conversation_id=conversation_id,
        user_id=settings.runtime_user_id,
    )
    response.conversation_id = conversation_id
    memory.add(
        user_id=settings.runtime_user_id,
        conversation_id=conversation_id,
        property_code=response.property_code,
        run_id=response.run_id,
        user_message=request.message,
        assistant_answer=response.answer_markdown,
        tool_result_keys=sorted(response.tool_results),
        component_types=[component.type for component in response.components],
    )
    return response


@app.post("/sql/execute")
def execute_sql(request: SqlApprovalRequest, settings: SettingsDep) -> ChatResponse:
    if request.run_id:
        if not request.conversation_id:
            raise HTTPException(
                status_code=422,
                detail="conversation_id is required when resuming an agent run",
            )
        response = _resolve_agent_approval(
            run_id=request.run_id,
            property_code=request.property_code,
            conversation_id=request.conversation_id,
            approved=True,
            settings=settings,
        )
        _record_approval_response(response, request.question, settings)
        return response

    normalized_code = request.property_code.lower()
    validated_sql, rows = execute_approved_sql(settings, request.sql, normalized_code)
    response = ChatResponse(
        property_code=normalized_code,
        model=request.model,
        conversation_id=request.conversation_id,
        answer_markdown=(
            "I ran the approved read-only query for the selected property. "
            f"It returned **{len(rows)}** row{'s' if len(rows) != 1 else ''}."
        ),
        components=[
            UIComponent(
                type="table",
                title="Approved SQL Results",
                data=rows,
            )
        ],
        sources=[],
        tool_results={
            "approved_sql": validated_sql,
            "question": request.question,
            "row_count": len(rows),
        },
    )
    _conversation_memory(settings).add(
        user_id=settings.runtime_user_id,
        conversation_id=request.conversation_id,
        property_code=normalized_code,
        user_message=f"Approved SQL for: {request.question}",
        assistant_answer=response.answer_markdown,
        tool_result_keys=sorted(response.tool_results),
        component_types=[component.type for component in response.components],
    )
    return response


@app.post("/api/agent-runs/{run_id}/approve")
def approve_agent_run(
    run_id: str,
    request: AgentApprovalRequest,
    settings: SettingsDep,
) -> ChatResponse:
    response = _resolve_agent_approval(
        run_id=run_id,
        property_code=request.property_code,
        conversation_id=request.conversation_id,
        approved=request.approved,
        settings=settings,
    )
    question = str(response.tool_results.get("question") or "custom SQL request")
    _record_approval_response(response, question, settings)
    return response


@app.get("/api/agent-runs/{run_id}")
def get_agent_run(
    run_id: str,
    property_code: str,
    conversation_id: str,
    settings: SettingsDep,
) -> AgentRunDetail:
    try:
        detail = AgentRuntime(settings).get_run(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=settings.runtime_user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return AgentRunDetail.model_validate(detail)


@app.get("/api/agent-runs/{run_id}/steps")
def get_agent_run_steps(
    run_id: str,
    property_code: str,
    conversation_id: str,
    settings: SettingsDep,
) -> list[AgentRunStep]:
    try:
        steps = AgentRuntime(settings).list_run_steps(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=settings.runtime_user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [AgentRunStep.model_validate(step) for step in steps]


@app.get("/api/agent-runs/{run_id}/events")
def get_agent_run_events(
    run_id: str,
    property_code: str,
    conversation_id: str,
    settings: SettingsDep,
    after_sequence: int = Query(default=0, ge=0),
) -> list[AgentRunEvent]:
    try:
        events = AgentRuntime(settings).list_run_events(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=settings.runtime_user_id,
            after_sequence=after_sequence,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [AgentRunEvent.model_validate(event) for event in events]


@app.get("/api/agent-runs/{run_id}/stream")
async def stream_agent_run_events(
    run_id: str,
    request: Request,
    property_code: str,
    conversation_id: str,
    settings: SettingsDep,
    after_sequence: int = Query(default=0, ge=0),
    follow: bool = True,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Replay durable run events and optionally follow until a terminal status."""
    cursor = after_sequence
    if last_event_id:
        try:
            parsed_event_id = int(last_event_id)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail="Last-Event-ID must be an event sequence number",
            ) from error
        if parsed_event_id < 0:
            raise HTTPException(
                status_code=400,
                detail="Last-Event-ID must be a non-negative event sequence number",
            )
        cursor = max(cursor, parsed_event_id)
    runtime = AgentRuntime(settings)
    try:
        runtime.get_run(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=settings.runtime_user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    async def durable_event_stream():
        nonlocal cursor
        last_heartbeat = asyncio.get_running_loop().time()
        while True:
            if await request.is_disconnected():
                break
            events = runtime.list_run_events(
                run_id=run_id,
                property_code=property_code,
                conversation_id=conversation_id,
                user_id=settings.runtime_user_id,
                after_sequence=cursor,
            )
            for event in events:
                sequence_id = int(event.get("sequence_id") or cursor)
                cursor = max(cursor, sequence_id)
                yield _encode_sse("run_event", event, event_id=str(sequence_id))

            detail = runtime.get_run(
                run_id=run_id,
                property_code=property_code,
                conversation_id=conversation_id,
                user_id=settings.runtime_user_id,
            )
            if detail["status"] in _TERMINAL_RUN_STATUSES or not follow:
                yield _encode_sse(
                    "run_status",
                    {
                        "run_id": run_id,
                        "conversation_id": conversation_id,
                        "property_code": property_code.lower(),
                        "status": detail["status"],
                        "final_answer": detail.get("final_answer"),
                    },
                    event_id=str(cursor) if cursor else None,
                )
                break

            now = asyncio.get_running_loop().time()
            if now - last_heartbeat >= settings.stream_heartbeat_seconds:
                yield ": keep-alive\n\n"
                last_heartbeat = now
            await asyncio.sleep(settings.stream_poll_interval_seconds)

    return StreamingResponse(
        durable_event_stream(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.get("/api/agent-runs/{run_id}/citations")
def get_agent_run_citations(
    run_id: str,
    property_code: str,
    conversation_id: str,
    settings: SettingsDep,
) -> list[AgentRunCitation]:
    try:
        citations = AgentRuntime(settings).list_run_citations(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=settings.runtime_user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [AgentRunCitation.model_validate(citation) for citation in citations]


@app.post("/api/agent-runs/{run_id}/cancel")
def cancel_agent_run(
    run_id: str,
    request: AgentRunScopeRequest,
    settings: SettingsDep,
) -> AgentRunDetail:
    try:
        detail = AgentRuntime(settings).cancel_run(
            run_id=run_id,
            property_code=request.property_code,
            conversation_id=request.conversation_id,
            user_id=settings.runtime_user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AgentRunConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    active_run_cancellations.request(run_id)
    return AgentRunDetail.model_validate(detail)


def _resolve_agent_approval(
    *,
    run_id: str,
    property_code: str,
    conversation_id: str,
    approved: bool,
    settings: Settings,
) -> ChatResponse:
    try:
        return AgentRuntime(settings).resolve_sql_approval(
            run_id=run_id,
            property_code=property_code,
            approved=approved,
            conversation_id=conversation_id,
            user_id=settings.runtime_user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AgentRunConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _record_approval_response(
    response: ChatResponse,
    question: str,
    settings: Settings,
) -> None:
    _conversation_memory(settings).add(
        user_id=settings.runtime_user_id,
        conversation_id=response.conversation_id,
        property_code=response.property_code,
        run_id=response.run_id,
        user_message=f"SQL approval decision for: {question}",
        assistant_answer=response.answer_markdown,
        tool_result_keys=sorted(response.tool_results),
        component_types=[component.type for component in response.components],
    )


@app.post("/chat/stream")
async def chat_stream(
    chat_request: ChatRequest,
    http_request: Request,
    settings: SettingsDep,
) -> StreamingResponse:
    conversation_id = chat_request.conversation_id or str(uuid4())
    run_id = str(uuid4())
    buffer = RunStreamBuffer(settings.stream_queue_max_size)
    runtime = AgentRuntime(settings)
    active_run_cancellations.register(run_id, buffer.cancelled)

    def run_chat() -> None:
        try:
            memory = _conversation_memory(settings)
            history = memory.get(
                user_id=settings.runtime_user_id,
                conversation_id=conversation_id,
                property_code=chat_request.property_code,
            )
            response = runtime.answer(
                property_code=chat_request.property_code,
                message=chat_request.message,
                model=chat_request.model,
                on_token=buffer.publish_token,
                history=history,
                conversation_id=conversation_id,
                user_id=settings.runtime_user_id,
                run_id=run_id,
                cancellation_requested=buffer.cancelled.is_set,
            )
            response.conversation_id = conversation_id
            memory.add(
                user_id=settings.runtime_user_id,
                conversation_id=conversation_id,
                property_code=response.property_code,
                run_id=response.run_id,
                user_message=chat_request.message,
                assistant_answer=response.answer_markdown,
                tool_result_keys=sorted(response.tool_results),
                component_types=[component.type for component in response.components],
            )
            buffer.publish("final", response.model_dump(mode="json"))
        except AgentRunCancelledError:
            pass
        except Exception as error:
            buffer.publish("error", {"detail": str(error), "run_id": run_id})
        finally:
            buffer.close()

    try:
        worker = _STREAM_WORKERS.submit(run_chat)
    except StreamExecutorSaturatedError as error:
        active_run_cancellations.unregister(run_id, buffer.cancelled)
        raise HTTPException(
            status_code=503,
            detail="stream capacity is currently exhausted; retry shortly",
        ) from error

    def release_worker_registration(_future) -> None:
        active_run_cancellations.unregister(run_id, buffer.cancelled)

    worker.add_done_callback(release_worker_registration)

    async def event_stream():
        stream_finished = False
        yield _encode_sse(
            "status",
            {
                "message": "started",
                "run_id": run_id,
                "conversation_id": conversation_id,
                "property_code": chat_request.property_code.lower(),
                "reconnect_url": f"/api/agent-runs/{run_id}/stream",
            },
        )
        try:
            while True:
                if await http_request.is_disconnected():
                    break
                try:
                    item = buffer.get(timeout=0)
                except queue.Empty:
                    await asyncio.sleep(settings.stream_poll_interval_seconds)
                    continue
                if item is None:
                    stream_finished = True
                    break
                yield _encode_sse(item.event, item.payload)
        finally:
            if not stream_finished and not worker.done():
                buffer.request_cancellation()
                try:
                    runtime.cancel_run(
                        run_id=run_id,
                        property_code=chat_request.property_code,
                        conversation_id=conversation_id,
                        user_id=settings.runtime_user_id,
                    )
                except (AgentRunNotFoundError, AgentRunConflictError):
                    pass
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.wrap_future(worker)),
                    timeout=settings.stream_thread_join_seconds,
                )
            except TimeoutError:
                buffer.request_cancellation()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


def _encode_sse(
    event: str,
    payload: dict,
    *,
    event_id: str | None = None,
) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(payload, ensure_ascii=True, default=str)}")
    return "\n".join(lines) + "\n\n"


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _conversation_memory(settings: Settings) -> ConversationMemory:
    return ConversationMemory(ConversationMemoryStore(MySQLDatabase(settings)))


def run() -> None:
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
