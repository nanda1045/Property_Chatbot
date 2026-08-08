"""Shared typed contracts for agent tools and their execution results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.auth import Role
from app.core.authorization import ToolPermission

RiskLevel = Literal["read", "compute", "write"]
ToolStatus = Literal["succeeded", "failed"]
ToolErrorType = Literal[
    "unknown_tool",
    "authorization_denied",
    "scope_error",
    "input_validation",
    "budget_exceeded",
    "timeout",
    "transient_failure",
    "permanent_failure",
    "output_validation",
    "output_too_large",
]

TRUSTED_ARGUMENT_NAMES = frozenset(
    {
        "property_code",
        "user_id",
        "tenant_id",
        "role",
        "roles",
        "permission",
        "permissions",
        "allowed_property_codes",
        "database_connection",
        "approval_status",
    }
)


class TrustedToolContext(BaseModel):
    """Backend-owned values that must never be accepted from model arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    property_code: str | None = None
    user_id: str
    tenant_id: str | None = None
    roles: tuple[Role, ...]
    allowed_property_codes: tuple[str, ...]
    run_id: str | None = None

    @field_validator("property_code")
    @classmethod
    def normalize_property_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("property_code cannot be empty")
        return normalized

    @field_validator("allowed_property_codes")
    @classmethod
    def normalize_allowed_properties(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({item.strip().lower() for item in value if item.strip()}))
        if not normalized:
            raise ValueError("allowed_property_codes cannot be empty")
        return normalized


class ToolError(BaseModel):
    type: ToolErrorType
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    invocation_id: str
    tool_name: str
    status: ToolStatus
    attempt: int
    duration_ms: int
    query_parameters: dict[str, Any] = Field(default_factory=dict)
    data_timestamp: str | None = None
    completed_at: str | None = None
    data: Any = None
    citation_refs: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    cached: bool = False


class ToolHandler(Protocol):
    def __call__(
        self,
        tool_input: BaseModel,
        context: TrustedToolContext,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    risk_level: RiskLevel = "read"
    required_permission: ToolPermission = ToolPermission.PROPERTY_BASIC_READ
    timeout_seconds: float = 10.0
    max_attempts: int = 1
    idempotent: bool = True
    required_scopes: tuple[str, ...] = ("property_code",)
    max_output_bytes: int = 250_000

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name is required")
        if self.risk_level not in {"read", "compute", "write"}:
            raise ValueError(f"invalid risk_level: {self.risk_level}")
        if not issubclass(self.input_model, BaseModel):
            raise TypeError("input_model must be a Pydantic model")
        if not issubclass(self.output_model, BaseModel):
            raise TypeError("output_model must be a Pydantic model")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        unknown_scopes = set(self.required_scopes) - set(TrustedToolContext.model_fields)
        if unknown_scopes:
            raise ValueError(f"unknown trusted scopes: {sorted(unknown_scopes)}")


class TransientToolError(RuntimeError):
    """Signal that an idempotent tool call may be retried."""
