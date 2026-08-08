from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    property_code: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Active property code, for example 115r.",
    )
    message: str = Field(min_length=1, max_length=8000)
    model: str = Field(default="anthropic:claude-haiku-4-5-20251001", max_length=128)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Client-generated id for durable property-scoped conversation memory.",
    )


class DemoIdentityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["Viewer", "Analyst", "PropertyManager"]


class SqlApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    run_id: str = Field(description="Durable run containing the server-stored SQL draft.")
    property_code: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Active property code, for example 115r.",
    )
    question: str | None = Field(default=None, min_length=1, max_length=8000)
    conversation_id: str = Field(min_length=1, max_length=128)


class AgentApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_code: str = Field(description="Backend-selected active property code.")
    conversation_id: str = Field(min_length=1, max_length=128)
    approved: bool = True


class AgentRunScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_code: str = Field(min_length=1, max_length=32)
    conversation_id: str = Field(min_length=1, max_length=128)


class AgentRunDetail(BaseModel):
    run_id: str
    conversation_id: str
    property_code: str
    user_goal: str
    status: str
    current_step: int
    max_steps: int
    plan: list[dict[str, Any]] = Field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    tool_call_count: int
    max_tool_calls: int
    error: dict[str, Any] | None = None
    final_answer: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class AgentRunStep(BaseModel):
    step_id: str
    run_id: str
    step_number: int
    step_type: str
    status: str
    input: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None


class AgentRunEvent(BaseModel):
    sequence_id: int | None = None
    event_id: str
    run_id: str
    event_type: str
    conversation_id: str
    property_code: str
    step_id: str | None = None
    tool_name: str | None = None
    attempt: int | None = None
    duration_ms: int | None = None
    timestamp: str
    error_type: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRunCitation(BaseModel):
    citation_id: str
    run_id: str
    property_code: str
    source_type: Literal["structured_tool", "retrieval"]
    source_name: str
    tool_invocation_id: str | None = None
    query_parameters: dict[str, Any] = Field(default_factory=dict)
    data_timestamp: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    content_hash: str
    source_url: str | None = None
    evidence: dict[str, Any]
    retrieved_at: str
    index_version: str | None = None


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


class InvestigationMetric(BaseModel):
    label: str
    value: float | int | str
    unit: str | None = None
    citation_id: str


class InvestigationFinding(BaseModel):
    finding_id: str
    title: str
    narrative: str
    metrics: list[InvestigationMetric] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)


class InvestigationCitation(BaseModel):
    citation_id: str
    property_code: str
    source_type: Literal["structured_tool", "retrieval"]
    source_name: str
    tool_invocation_id: str
    query_parameters: dict[str, Any] = Field(default_factory=dict)
    data_timestamp: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    source_url: str | None = None
    content_hash: str
    retrieved_at: str
    index_version: str | None = None
    evidence: dict[str, Any]


class InvestigationArtifact(BaseModel):
    artifact_id: str
    type: str
    name: str
    content_type: str
    content: str


class InvestigationTraceSummary(BaseModel):
    steps: int
    tool_calls: int
    duration_ms: int
    tool_order: list[str]
    stop_reason: str
    verification_status: Literal["passed", "failed"]
    verification_checks: list[str] = Field(default_factory=list)


class OccupancyInvestigationReport(BaseModel):
    run_id: str | None = None
    status: Literal["completed"] = "completed"
    summary: str
    findings: list[InvestigationFinding]
    citations: list[InvestigationCitation]
    artifacts: list[InvestigationArtifact]
    trace_summary: InvestigationTraceSummary


class ChatResponse(BaseModel):
    property_code: str
    model: str
    conversation_id: str | None = None
    run_id: str | None = None
    run_status: str | None = None
    answer_markdown: str
    components: list[UIComponent] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    tool_results: dict[str, Any] = Field(default_factory=dict)
    investigation: OccupancyInvestigationReport | None = None
