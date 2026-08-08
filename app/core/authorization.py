"""Deterministic identity, property, tool, and approval authorization policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from app.core.auth import AuthenticatedUser, Role
from app.core.config import Settings


class ToolPermission(StrEnum):
    CHAT = "chat"
    PROPERTY_BASIC_READ = "property.basic.read"
    KPI_READ = "property.kpi.read"
    RETRIEVAL_READ = "property.retrieval.read"
    ANALYTICS_READ = "property.analytics.read"
    CUSTOM_ANALYTICS = "property.analytics.custom"
    RUN_READ = "run.read"
    RUN_CANCEL = "run.cancel"
    SQL_APPROVE = "sql.approve"


ROLE_PERMISSIONS: dict[Role, frozenset[ToolPermission]] = {
    Role.VIEWER: frozenset(
        {
            ToolPermission.CHAT,
            ToolPermission.PROPERTY_BASIC_READ,
            ToolPermission.KPI_READ,
            ToolPermission.RETRIEVAL_READ,
            ToolPermission.RUN_READ,
            ToolPermission.RUN_CANCEL,
        }
    ),
    Role.ANALYST: frozenset(
        {
            ToolPermission.CHAT,
            ToolPermission.PROPERTY_BASIC_READ,
            ToolPermission.KPI_READ,
            ToolPermission.RETRIEVAL_READ,
            ToolPermission.ANALYTICS_READ,
            ToolPermission.CUSTOM_ANALYTICS,
            ToolPermission.RUN_READ,
            ToolPermission.RUN_CANCEL,
        }
    ),
    Role.PROPERTY_MANAGER: frozenset(ToolPermission),
}


class ToolPolicyMetadata(Protocol):
    name: str
    required_permission: ToolPermission


class AuthorizationDeniedError(PermissionError):
    def __init__(
        self,
        message: str,
        *,
        permission: ToolPermission,
        property_code: str | None = None,
        tool_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.permission = permission
        self.property_code = property_code
        self.tool_name = tool_name


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    user: AuthenticatedUser
    allowed_property_codes: tuple[str, ...]
    property_code: str | None = None

    @classmethod
    def from_settings(
        cls,
        user: AuthenticatedUser,
        settings: Settings,
        *,
        property_code: str | None = None,
    ) -> AuthorizationContext:
        if settings.auth_mode == "local":
            allowed = set(settings.local_auth_allowed_properties)
        else:
            allowed: set[str] = set()
            for key in (user.user_id, "*", *(f"role:{role.value}" for role in user.roles)):
                allowed.update(settings.auth_property_access.get(key, []))
        return cls(
            user=user,
            allowed_property_codes=tuple(sorted(allowed)),
            property_code=property_code.strip().lower() if property_code else None,
        )

    def for_property(self, property_code: str) -> AuthorizationContext:
        return AuthorizationContext(
            user=self.user,
            allowed_property_codes=self.allowed_property_codes,
            property_code=property_code.strip().lower(),
        )

    @property
    def primary_role(self) -> Role | None:
        return self.user.primary_role

    def can_access_property(self, property_code: str) -> bool:
        normalized = property_code.strip().lower()
        return "*" in self.allowed_property_codes or normalized in self.allowed_property_codes


def authorize_property(context: AuthorizationContext) -> None:
    property_code = context.property_code
    if not property_code or not context.can_access_property(property_code):
        raise AuthorizationDeniedError(
            "You do not have access to the requested property.",
            permission=ToolPermission.PROPERTY_BASIC_READ,
            property_code=property_code,
        )


def authorize_permission(
    context: AuthorizationContext,
    permission: ToolPermission,
    *,
    require_property: bool = True,
) -> None:
    if require_property:
        authorize_property(context)
    granted = any(permission in ROLE_PERMISSIONS[role] for role in context.user.roles)
    if not granted:
        raise AuthorizationDeniedError(
            "Your assigned role does not permit this action.",
            permission=permission,
            property_code=context.property_code,
        )


def authorize_tool(
    context: AuthorizationContext,
    tool: ToolPolicyMetadata,
    *,
    require_property: bool = True,
) -> None:
    try:
        authorize_permission(
            context,
            tool.required_permission,
            require_property=require_property,
        )
    except AuthorizationDeniedError as error:
        error.tool_name = tool.name
        raise


def authorize_sql_approval(context: AuthorizationContext) -> None:
    authorize_permission(context, ToolPermission.SQL_APPROVE)
