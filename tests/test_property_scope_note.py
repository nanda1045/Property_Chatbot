from __future__ import annotations

from app.services.langchain_orchestrator import LangChainOrchestrator


def test_scope_note_detects_different_mentioned_property() -> None:
    orchestrator = object.__new__(LangChainOrchestrator)
    orchestrator._call_tool = lambda name, **kwargs: [
        {"property_code": "115r", "property_name": "Canfield Park"},
        {"property_code": "176r", "property_name": "The Alexander"},
    ]

    note = orchestrator._property_scope_note(
        "For The Alexander, does it have a pool?",
        {"property_code": "115r", "property_name": "Canfield Park"},
    )

    assert note is not None
    assert "You mentioned **The Alexander**" in note
    assert "active property is **Canfield Park (`115r`)**" in note


def test_scope_note_ignores_active_property_mentions() -> None:
    orchestrator = object.__new__(LangChainOrchestrator)
    orchestrator._call_tool = lambda name, **kwargs: [
        {"property_code": "115r", "property_name": "Canfield Park"},
        {"property_code": "176r", "property_name": "The Alexander"},
    ]

    note = orchestrator._property_scope_note(
        "For Canfield Park, does it have a pool?",
        {"property_code": "115r", "property_name": "Canfield Park"},
    )

    assert note is None
