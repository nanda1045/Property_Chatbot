from __future__ import annotations

import json
import queue
import threading
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
from app.services.sql_approval import execute_approved_sql

SettingsDep = Annotated[Settings, Depends(get_settings)]


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
) -> list[AgentRunEvent]:
    try:
        events = AgentRuntime(settings).list_run_events(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=settings.runtime_user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [AgentRunEvent.model_validate(event) for event in events]


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
def chat_stream(request: ChatRequest, settings: SettingsDep) -> StreamingResponse:
    def encode_event(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"

    def event_stream():
        events: queue.Queue[tuple[str, dict] | None] = queue.Queue()

        def publish_token(token: str) -> None:
            events.put(("token", {"delta": token}))

        def run_chat() -> None:
            try:
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
                    on_token=publish_token,
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
                events.put(("final", response.model_dump()))
            except Exception as error:
                events.put(("error", {"detail": str(error)}))
            finally:
                events.put(None)

        thread = threading.Thread(target=run_chat, daemon=True)
        thread.start()

        yield encode_event("status", {"message": "started"})
        while True:
            item = events.get()
            if item is None:
                break
            event, payload = item
            yield encode_event(event, payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _conversation_memory(settings: Settings) -> ConversationMemory:
    return ConversationMemory(ConversationMemoryStore(MySQLDatabase(settings)))


def run() -> None:
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
