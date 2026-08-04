"""Property chat workflow boundary.

The subclass intentionally preserves the current orchestration behavior. New durable
workflow steps can move behind this boundary without changing API or evaluation callers.
"""

from app.services.langchain_orchestrator import LangChainOrchestrator


class PropertyChatWorkflow(LangChainOrchestrator):
    """Execute the existing property-scoped chat workflow."""
