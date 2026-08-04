from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    property_code: str = Field(description="Active property code, for example 115r.")
    message: str
    model: str = "anthropic:claude-haiku-4-5-20251001"
    conversation_id: str | None = Field(
        default=None,
        description="Client-generated conversation/session id for lightweight memory.",
    )


class SqlApprovalRequest(BaseModel):
    run_id: str | None = Field(
        default=None,
        description="Durable run to resume. Legacy requests may omit this field.",
    )
    property_code: str = Field(description="Active property code, for example 115r.")
    model: str = "anthropic:claude-haiku-4-5-20251001"
    sql: str = Field(description="Backend-validated read-only SQL proposed for approval.")
    question: str = Field(description="Original user question that produced the SQL proposal.")
    conversation_id: str | None = None


class AgentApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_code: str = Field(description="Backend-selected active property code.")
    conversation_id: str
    approved: bool = True


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
    conversation_id: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    answer_markdown: str
    components: list[UIComponent] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
