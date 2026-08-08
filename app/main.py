from __future__ import annotations

import asyncio
import json
import logging
import queue
from typing import Annotated
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.agents.cancellation import AgentRunCancelledError
from app.agents.runtime import AgentRunConflictError, AgentRunNotFoundError, AgentRuntime
from app.core.auth import (
    LOCAL_DEMO_IDENTITY_COOKIE,
    LOCAL_DEMO_IDENTITY_TTL_SECONDS,
    AuthenticatedUser,
    AuthenticatedUserDep,
    Role,
    issue_local_demo_identity_token,
    local_demo_authenticated_user,
)
from app.core.authorization import (
    AuthorizationContext,
    AuthorizationDeniedError,
    ToolPermission,
    authorize_permission,
)
from app.core.config import Settings, get_settings
from app.core.errors import install_exception_handlers
from app.core.http import RequestContextMiddleware
from app.core.logging import configure_logging
from app.core.rate_limit import AllowAllRateLimiter, enforce_rate_limit
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
    DemoIdentityRequest,
    SqlApprovalRequest,
)
from app.services.conversation_memory import ConversationMemory
from app.services.rent_roll_repository import RentRollRepository
from app.services.run_stream import (
    BoundedStreamExecutor,
    RunStreamBuffer,
    StreamExecutorSaturatedError,
    active_run_cancellations,
)

SettingsDep = Annotated[Settings, Depends(get_settings)]
RateLimitDep = Annotated[None, Depends(enforce_rate_limit)]

_STREAM_WORKERS = BoundedStreamExecutor(max_workers=8, max_pending=8)
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
_BOOT_SETTINGS = get_settings()
_LOGGER = configure_logging(_BOOT_SETTINGS.log_level)
_AUDIT_LOGGER = logging.getLogger("aker.audit")


app = FastAPI(
    title="Aker Property Assistant",
    version="0.1.0",
    description="Property-scoped AI assistant prototype.",
)
app.state.rate_limiter = AllowAllRateLimiter()
install_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_BOOT_SETTINGS.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestContextMiddleware)


@app.get("/health")
def health(settings: SettingsDep) -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "default_property_code": settings.default_property_code,
    }


@app.get("/ready")
def readiness(settings: SettingsDep) -> dict[str, object]:
    """Report whether required backend dependencies can serve requests."""
    try:
        database_ready = MySQLDatabase(settings).ping()
    except Exception as error:
        _LOGGER.warning(
            "readiness_check_failed",
            extra={
                "event": "readiness_check_failed",
                "error_type": type(error).__name__,
            },
        )
        raise HTTPException(status_code=503, detail="database is unavailable") from error
    if not database_ready:
        raise HTTPException(status_code=503, detail="database readiness check failed")
    return {
        "status": "ready",
        "checks": {"database": "ok"},
        "env": settings.app_env,
    }


def _identity_payload(user: AuthenticatedUser) -> dict[str, object]:
    return {
        "user_id": user.user_id,
        "display_name": user.display_name,
        "email": user.email,
        "tenant_id": user.tenant_id,
        "roles": [role.value for role in user.roles],
        "role": user.primary_role.value if user.primary_role is not None else None,
    }


@app.get("/auth/me")
def authenticated_identity(user: AuthenticatedUserDep) -> dict[str, object]:
    return _identity_payload(user)


@app.post("/auth/demo-identity")
def select_demo_identity(
    request: DemoIdentityRequest,
    response: Response,
    settings: SettingsDep,
    _rate_limit: RateLimitDep,
) -> dict[str, object]:
    """Select a predefined backend-owned identity in local demo mode only."""
    if settings.auth_mode != "local":
        raise HTTPException(status_code=404, detail="Local demo identity switching is unavailable")
    role = Role(request.role)
    response.set_cookie(
        key=LOCAL_DEMO_IDENTITY_COOKIE,
        value=issue_local_demo_identity_token(role),
        max_age=LOCAL_DEMO_IDENTITY_TTL_SECONDS,
        httponly=True,
        secure=False,
        samesite="strict",
        path="/",
    )
    return _identity_payload(local_demo_authenticated_user(role))


@app.get("/models")
def models(settings: SettingsDep, _user: AuthenticatedUserDep) -> dict[str, object]:
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
def properties(
    settings: SettingsDep,
    user: AuthenticatedUserDep,
) -> dict[str, list[dict]]:
    repository = RentRollRepository(MySQLDatabase(settings))
    authorization = _require_permission(
        user,
        settings,
        None,
        ToolPermission.PROPERTY_BASIC_READ,
    )
    return {
        "properties": [
            item
            for item in repository.list_properties()
            if authorization.can_access_property(str(item.get("property_code") or ""))
        ]
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
    settings: SettingsDep,
    user: AuthenticatedUserDep,
    _rate_limit: RateLimitDep,
) -> ChatResponse:
    authorization = _require_permission(
        user,
        settings,
        request.property_code,
        ToolPermission.CHAT,
    )
    conversation_id = request.conversation_id or str(uuid4())
    memory = _conversation_memory(settings)
    history = memory.get(
        user_id=user.user_id,
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
        user_id=user.user_id,
        authorization_context=authorization,
    )
    response.conversation_id = conversation_id
    memory.add(
        user_id=user.user_id,
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
def execute_sql(
    request: SqlApprovalRequest,
    settings: SettingsDep,
    user: AuthenticatedUserDep,
    _rate_limit: RateLimitDep,
) -> ChatResponse:
    if not request.run_id or not request.conversation_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Direct SQL submission is disabled; approve a server-stored agent run draft."
            ),
        )
    authorization = AuthorizationContext.from_settings(
        user,
        settings,
        property_code=request.property_code,
    )
    response = _resolve_agent_approval(
        run_id=request.run_id,
        property_code=request.property_code,
        conversation_id=request.conversation_id,
        approved=True,
        settings=settings,
        authorization_context=authorization,
    )
    _record_approval_response(response, request.question or "custom SQL request", settings, user)
    return response


@app.post("/api/agent-runs/{run_id}/approve")
def approve_agent_run(
    run_id: str,
    request: AgentApprovalRequest,
    settings: SettingsDep,
    user: AuthenticatedUserDep,
    _rate_limit: RateLimitDep,
) -> ChatResponse:
    authorization = AuthorizationContext.from_settings(
        user,
        settings,
        property_code=request.property_code,
    )
    response = _resolve_agent_approval(
        run_id=run_id,
        property_code=request.property_code,
        conversation_id=request.conversation_id,
        approved=request.approved,
        settings=settings,
        authorization_context=authorization,
    )
    question = str(response.tool_results.get("question") or "custom SQL request")
    _record_approval_response(response, question, settings, user)
    return response


@app.get("/api/agent-runs/{run_id}")
def get_agent_run(
    run_id: str,
    property_code: str,
    conversation_id: str,
    settings: SettingsDep,
    user: AuthenticatedUserDep,
) -> AgentRunDetail:
    _require_permission(user, settings, property_code, ToolPermission.RUN_READ)
    try:
        detail = AgentRuntime(settings).get_run(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=user.user_id,
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
    user: AuthenticatedUserDep,
) -> list[AgentRunStep]:
    _require_permission(user, settings, property_code, ToolPermission.RUN_READ)
    try:
        steps = AgentRuntime(settings).list_run_steps(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=user.user_id,
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
    user: AuthenticatedUserDep,
    after_sequence: int = Query(default=0, ge=0),
) -> list[AgentRunEvent]:
    _require_permission(user, settings, property_code, ToolPermission.RUN_READ)
    try:
        events = AgentRuntime(settings).list_run_events(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=user.user_id,
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
    user: AuthenticatedUserDep,
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
    _require_permission(user, settings, property_code, ToolPermission.RUN_READ)
    try:
        runtime.get_run(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=user.user_id,
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
                user_id=user.user_id,
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
                user_id=user.user_id,
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
    user: AuthenticatedUserDep,
) -> list[AgentRunCitation]:
    _require_permission(user, settings, property_code, ToolPermission.RUN_READ)
    try:
        citations = AgentRuntime(settings).list_run_citations(
            run_id=run_id,
            property_code=property_code,
            conversation_id=conversation_id,
            user_id=user.user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [AgentRunCitation.model_validate(citation) for citation in citations]


@app.post("/api/agent-runs/{run_id}/cancel")
def cancel_agent_run(
    run_id: str,
    request: AgentRunScopeRequest,
    settings: SettingsDep,
    user: AuthenticatedUserDep,
    _rate_limit: RateLimitDep,
) -> AgentRunDetail:
    _require_permission(user, settings, request.property_code, ToolPermission.RUN_CANCEL)
    try:
        detail = AgentRuntime(settings).cancel_run(
            run_id=run_id,
            property_code=request.property_code,
            conversation_id=request.conversation_id,
            user_id=user.user_id,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AgentRunConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except AuthorizationDeniedError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    active_run_cancellations.request(run_id)
    return AgentRunDetail.model_validate(detail)


def _resolve_agent_approval(
    *,
    run_id: str,
    property_code: str,
    conversation_id: str,
    approved: bool,
    settings: Settings,
    authorization_context: AuthorizationContext,
) -> ChatResponse:
    try:
        response = AgentRuntime(settings).resolve_sql_approval(
            run_id=run_id,
            property_code=property_code,
            approved=approved,
            conversation_id=conversation_id,
            user_id=authorization_context.user.user_id,
            authorization_context=authorization_context,
        )
    except AgentRunNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AgentRunConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except AuthorizationDeniedError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    _AUDIT_LOGGER.info(
        "sql_approval_decision",
        extra={
            "event": "sql_approval_decision",
            "run_id": run_id,
            "conversation_id": conversation_id,
            "property_code": property_code.lower(),
            "decision": "approved" if approved else "rejected",
            "user_id": authorization_context.user.user_id,
            "role": (
                authorization_context.primary_role.value
                if authorization_context.primary_role is not None
                else None
            ),
        },
    )
    return response


def _record_approval_response(
    response: ChatResponse,
    question: str,
    settings: Settings,
    user: AuthenticatedUser,
) -> None:
    _conversation_memory(settings).add(
        user_id=user.user_id,
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
    user: AuthenticatedUserDep,
    _rate_limit: RateLimitDep,
) -> StreamingResponse:
    authorization = _require_permission(
        user,
        settings,
        chat_request.property_code,
        ToolPermission.CHAT,
    )
    conversation_id = chat_request.conversation_id or str(uuid4())
    run_id = str(uuid4())
    buffer = RunStreamBuffer(settings.stream_queue_max_size)
    runtime = AgentRuntime(settings)
    active_run_cancellations.register(run_id, buffer.cancelled)

    def run_chat() -> None:
        try:
            memory = _conversation_memory(settings)
            history = memory.get(
                user_id=user.user_id,
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
                user_id=user.user_id,
                authorization_context=authorization,
                run_id=run_id,
                cancellation_requested=buffer.cancelled.is_set,
            )
            response.conversation_id = conversation_id
            memory.add(
                user_id=user.user_id,
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
            _LOGGER.error(
                "stream_run_failed",
                extra={
                    "event": "stream_run_failed",
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "property_code": chat_request.property_code.lower(),
                    "error_type": type(error).__name__,
                },
            )
            buffer.publish("error", {"detail": "agent run failed", "run_id": run_id})
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
                        user_id=user.user_id,
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


def _require_permission(
    user: AuthenticatedUser,
    settings: Settings,
    property_code: str | None,
    permission: ToolPermission,
) -> AuthorizationContext:
    authorization = AuthorizationContext.from_settings(
        user,
        settings,
        property_code=property_code,
    )
    try:
        authorize_permission(
            authorization,
            permission,
            require_property=property_code is not None,
        )
    except AuthorizationDeniedError as error:
        _AUDIT_LOGGER.info(
            "authorization_denied",
            extra={
                "event": "AUTHORIZATION_DENIED",
                "user_id": user.user_id,
                "role": user.primary_role.value if user.primary_role is not None else None,
                "property_code": property_code.lower() if property_code else None,
                "permission": permission.value,
                "outcome": "denied",
            },
        )
        raise HTTPException(status_code=403, detail=str(error)) from error
    return authorization


def _conversation_memory(settings: Settings) -> ConversationMemory:
    return ConversationMemory(ConversationMemoryStore(MySQLDatabase(settings)))


def run() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
        log_level=settings.log_level.lower(),
    )
