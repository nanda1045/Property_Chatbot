"""Deterministic policies enforced before property-scoped tool execution."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def normalize_property_phrase(text: str) -> str:
    """Normalize property names and codes for conservative comparisons."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def explicit_property_name_from_message(message: str) -> str | None:
    """Extract a property explicitly named with `for/about property ...`."""
    patterns = [
        r"\bfor\s+(?:the\s+)?property\s+(.+?)(?:,|\?|$)",
        r"\babout\s+(?:the\s+)?property\s+(.+?)(?:,|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if not match:
            continue
        name = match.group(1).strip(" .:-")
        if name:
            return name
    return None


def mentioned_other_property(
    message: str,
    active_profile: dict[str, Any],
    available_properties: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return a known non-active property explicitly mentioned by the user."""
    message_text = message.lower()
    normalized_message = normalize_property_phrase(message)
    active_code = str(active_profile["property_code"]).lower()
    active_name = normalize_property_phrase(str(active_profile["property_name"]))

    matches: list[dict[str, Any]] = []
    for property_profile in available_properties:
        candidate_code = str(property_profile.get("property_code") or "").lower()
        candidate_name = str(property_profile.get("property_name") or "").strip()
        normalized_name = normalize_property_phrase(candidate_name)
        if not candidate_code or not candidate_name:
            continue
        if candidate_code == active_code or normalized_name == active_name:
            continue

        code_matches = bool(re.search(rf"\b{re.escape(candidate_code)}\b", message_text))
        name_matches = (
            len(normalized_name) >= 4
            and f" {normalized_name} " in f" {normalized_message} "
        )
        if code_matches or name_matches:
            matches.append(property_profile)

    if not matches:
        return None
    matches.sort(key=lambda item: len(str(item.get("property_name") or "")), reverse=True)
    return matches[0]


def property_scope_conflict(
    message: str,
    active_profile: dict[str, Any],
    available_properties: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Reject requests that explicitly target a property other than the active one."""
    mentioned = mentioned_other_property(message, active_profile, available_properties)
    if mentioned:
        return mentioned

    explicit_name = explicit_property_name_from_message(message)
    if not explicit_name:
        return None

    normalized_name = normalize_property_phrase(explicit_name)
    active_code = str(active_profile["property_code"]).lower()
    active_name = normalize_property_phrase(str(active_profile["property_name"]))
    if normalized_name in {"active", "selected", "this"}:
        return None
    if normalized_name == active_code or normalized_name == active_name:
        return None
    return {"property_name": explicit_name.strip()}


def property_scope_conflict_answer(
    active_profile: dict[str, Any],
    mentioned_property: str,
) -> str:
    """Create the standard user-facing response for a scope conflict."""
    active_name = active_profile["property_name"]
    active_code = active_profile["property_code"]
    return (
        f"### {active_name} (`{active_code}`)\n\n"
        f"I can't answer that because the selected property is **{active_name} "
        f"(`{active_code}`)**, but your question asks about **{mentioned_property}**.\n\n"
        "Please select the correct property code first, then ask the question again."
    )
