from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    property_code: str = Field(description="Active property code, for example 115r.")
    message: str
    model: str = "mock:mock-property-assistant"


class UIComponent(BaseModel):
    type: str
    title: str
    data: Any
    description: str | None = None


class Source(BaseModel):
    property_code: str
    title: str | None = None
    source_url: str | None = None
    page_type: str | None = None
    tool: str | None = None


class ChatResponse(BaseModel):
    property_code: str
    model: str
    answer_markdown: str
    components: list[UIComponent] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
