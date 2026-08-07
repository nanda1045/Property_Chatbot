"""Shared cooperative cancellation primitives for synchronous agent execution."""

from __future__ import annotations

from collections.abc import Callable

CancellationCheck = Callable[[], bool]


class AgentRunCancelledError(RuntimeError):
    """Raised when the transport or user cancels an active agent run."""


def raise_if_cancelled(check: CancellationCheck | None) -> None:
    if check is not None and check():
        raise AgentRunCancelledError("agent run cancellation was requested")
