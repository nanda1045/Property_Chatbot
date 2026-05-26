from __future__ import annotations

import json
import queue
import threading
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.db.mysql import MySQLDatabase
from app.schemas import ChatRequest, ChatResponse
from app.services.langchain_orchestrator import LangChainOrchestrator
from app.services.rent_roll_repository import RentRollRepository

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
                "id": "mock:mock-property-assistant",
                "label": "Mock Assistant",
                "provider": "mock",
            },
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
    orchestrator = LangChainOrchestrator(settings)
    return orchestrator.answer(
        property_code=request.property_code,
        message=request.message,
        model=request.model,
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
                orchestrator = LangChainOrchestrator(settings)
                response = orchestrator.answer(
                    property_code=request.property_code,
                    message=request.message,
                    model=request.model,
                    on_token=publish_token,
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


def run() -> None:
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
