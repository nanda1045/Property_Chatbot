from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    property_code: str = Field(description="Active property code, for example 115r.")
    message: str
    model: str = "anthropic:claude-haiku-4-5-20251001"
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Client-generated id for durable property-scoped conversation memory.",
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
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class AgentApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_code: str = Field(description="Backend-selected active property code.")
    conversation_id: str = Field(min_length=1, max_length=128)
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
    document_id: str | None = None
    chunk_id: str | None = None
    source_url: str | None = None
    content_hash: str
    retrieved_at: str
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
    tool_results: dict[str, Any] = Field(default_factory=dict)
    investigation: OccupancyInvestigationReport | None = None
