"""Thread-safe registry for typed agent tool specifications and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from app.tools.contracts import ToolHandler, ToolSpec


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._lock = RLock()

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        with self._lock:
            if spec.name in self._tools:
                raise ValueError(f"tool is already registered: {spec.name}")
            self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def get(self, name: str) -> RegisteredTool | None:
        with self._lock:
            return self._tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        registered = self.get(name)
        if registered is None:
            raise KeyError(f"unknown tool: {name}")
        return registered

    def specs(self) -> list[ToolSpec]:
        with self._lock:
            return [registered.spec for registered in self._tools.values()]

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._tools)
