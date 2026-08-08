"""Replaceable rate-limit contract; local development allows all requests."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Protocol

from fastapi import HTTPException, Request


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int | None = None


class RateLimiter(Protocol):
    def check(
        self,
        *,
        client_id: str,
        route: str,
    ) -> RateLimitDecision | Awaitable[RateLimitDecision]: ...


class AllowAllRateLimiter:
    """Default local hook. Replace `app.state.rate_limiter` at deployment time."""

    def check(self, *, client_id: str, route: str) -> RateLimitDecision:
        return RateLimitDecision(allowed=True)


async def enforce_rate_limit(request: Request) -> None:
    limiter: RateLimiter = getattr(request.app.state, "rate_limiter", AllowAllRateLimiter())
    client_id = request.client.host if request.client else "unknown"
    decision = limiter.check(client_id=client_id, route=request.url.path)
    if isawaitable(decision):
        decision = await decision
    if decision.allowed:
        return
    retry_after = max(1, decision.retry_after_seconds or 1)
    raise HTTPException(
        status_code=429,
        detail="request rate limit exceeded",
        headers={"Retry-After": str(retry_after)},
    )
