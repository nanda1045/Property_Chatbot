"""Microsoft Entra authentication with a deterministic local-development mode."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Any
from urllib.request import urlopen

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from pydantic import BaseModel, ConfigDict, field_validator

from app.core.config import Settings, get_settings


class Role(StrEnum):
    VIEWER = "Viewer"
    ANALYST = "Analyst"
    PROPERTY_MANAGER = "PropertyManager"


ROLE_ORDER = {
    Role.VIEWER: 0,
    Role.ANALYST: 1,
    Role.PROPERTY_MANAGER: 2,
}


class AuthenticatedUser(BaseModel):
    """Identity claims trusted only after local construction or JWT validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str
    display_name: str
    email: str | None = None
    tenant_id: str
    roles: tuple[Role, ...]

    @field_validator("user_id", "display_name", "tenant_id")
    @classmethod
    def reject_blank_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("trusted identity fields cannot be blank")
        return normalized

    @property
    def primary_role(self) -> Role | None:
        return max(self.roles, key=ROLE_ORDER.__getitem__) if self.roles else None


class AuthenticationError(ValueError):
    """Raised when a bearer token cannot establish a trusted identity."""


MetadataFetcher = Callable[[str], dict[str, Any]]


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=5) as response:  # noqa: S310 - URL is validated config
        return dict(json.load(response))


@dataclass
class EntraTokenValidator:
    tenant_id: str
    audience: str
    authority_host: str = "https://login.microsoftonline.com"
    required_scope: str = "access_as_user"
    jwks_cache_seconds: int = 3600
    metadata_fetcher: MetadataFetcher = _fetch_json

    def __post_init__(self) -> None:
        self.authority_host = self.authority_host.rstrip("/")
        self.metadata_url = (
            f"{self.authority_host}/{self.tenant_id}/v2.0/"
            ".well-known/openid-configuration"
        )
        self._metadata: dict[str, Any] | None = None
        self._jwks_client: PyJWKClient | None = None

    def validate(self, token: str) -> AuthenticatedUser:
        try:
            metadata = self._load_metadata()
            signing_key = self._jwk_client(metadata).get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=str(metadata["issuer"]),
                options={
                    "require": ["aud", "exp", "iss", "oid", "tid"],
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_exp": True,
                    "verify_iss": True,
                },
            )
        except Exception as error:
            raise AuthenticationError("bearer token validation failed") from error

        token_tenant = str(claims.get("tid") or "")
        if token_tenant.lower() != self.tenant_id.lower():
            raise AuthenticationError("bearer token tenant is not allowed")
        if str(claims.get("ver") or "2.0") != "2.0":
            raise AuthenticationError("only Microsoft identity platform v2 tokens are allowed")
        scopes = set(str(claims.get("scp") or "").split())
        if self.required_scope not in scopes:
            raise AuthenticationError("bearer token is missing the delegated API scope")

        raw_roles = claims.get("roles") or []
        if isinstance(raw_roles, str):
            raw_roles = [raw_roles]
        roles = tuple(
            sorted(
                {Role(value) for value in raw_roles if value in Role._value2member_map_},
                key=ROLE_ORDER.__getitem__,
            )
        )
        email = claims.get("preferred_username") or claims.get("email") or claims.get("upn")
        return AuthenticatedUser(
            user_id=str(claims["oid"]),
            display_name=str(claims.get("name") or email or claims["oid"]),
            email=str(email) if email else None,
            tenant_id=token_tenant,
            roles=roles,
        )

    def _load_metadata(self) -> dict[str, Any]:
        if self._metadata is None:
            metadata = self.metadata_fetcher(self.metadata_url)
            if not metadata.get("issuer") or not metadata.get("jwks_uri"):
                raise AuthenticationError("OpenID metadata is incomplete")
            self._metadata = metadata
        return self._metadata

    def _jwk_client(self, metadata: dict[str, Any]) -> PyJWKClient:
        if self._jwks_client is None:
            self._jwks_client = PyJWKClient(
                str(metadata["jwks_uri"]),
                cache_jwk_set=True,
                lifespan=self.jwks_cache_seconds,
            )
        return self._jwks_client


def local_authenticated_user(settings: Settings) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=settings.local_auth_user_id,
        display_name=settings.local_auth_display_name,
        email=settings.local_auth_email or None,
        tenant_id="local-development",
        roles=(Role(settings.local_auth_role),),
    )


@lru_cache(maxsize=16)
def _cached_entra_validator(
    tenant_id: str,
    audience: str,
    authority_host: str,
    required_scope: str,
    jwks_cache_seconds: int,
) -> EntraTokenValidator:
    return EntraTokenValidator(
        tenant_id=tenant_id,
        audience=audience,
        authority_host=authority_host,
        required_scope=required_scope,
        jwks_cache_seconds=jwks_cache_seconds,
    )


def token_validator(settings: Settings) -> EntraTokenValidator:
    if not settings.entra_tenant_id or not settings.entra_api_audience:
        raise AuthenticationError("Entra authentication is not configured")
    return _cached_entra_validator(
        settings.entra_tenant_id,
        settings.entra_api_audience,
        settings.entra_authority_host,
        settings.entra_required_scope,
        settings.entra_jwks_cache_seconds,
    )


_BEARER = HTTPBearer(auto_error=False)


def get_authenticated_user(
    settings: Annotated[Settings, Depends(get_settings)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_BEARER)],
) -> AuthenticatedUser:
    if settings.auth_mode == "local":
        return local_authenticated_user(settings)
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return token_validator(settings).validate(credentials.credentials)
    except AuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]
