"""Adaptive occupancy-decline investigation and evidence verification."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.agents.loop import AgentAction, AgentLoopResult, AgentObservation, LoopDecision
from app.schemas import (
    InvestigationArtifact,
    InvestigationCitation,
    InvestigationFinding,
    InvestigationMetric,
    InvestigationTraceSummary,
    OccupancyInvestigationReport,
)
from app.tools.contracts import ToolResult

OCCUPANCY_TERMS = re.compile(r"\boccup(?:ancy|ied)\b", re.IGNORECASE)
DECLINE_TERMS = re.compile(
    r"\b(?:declin(?:e|ed|ing)|drop(?:ped|ping)?|decreas(?:e|ed|ing)|fell|falling)\b",
    re.IGNORECASE,
)
INVESTIGATION_TERMS = re.compile(
    r"\b(?:investigat(?:e|ion)|why|reason|driver|executive\s+(?:brief|summary)|analy[sz]e)\b",
    re.IGNORECASE,
)


class EvidenceVerificationError(RuntimeError):
    """Raised when a report claim is not supported by its cited evidence."""


def is_occupancy_investigation_request(message: str) -> bool:
    """Identify explicit requests for an occupancy-decline investigation."""
    return bool(
        OCCUPANCY_TERMS.search(message)
        and DECLINE_TERMS.search(message)
        and INVESTIGATION_TERMS.search(message)
    )


def analyze_occupancy_trend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the largest consecutive decline and concurrent financial movement."""
    usable = [
        row
        for row in sorted(rows, key=lambda item: str(item.get("report_month") or ""))
        if row.get("unit_occupancy_pct") is not None
    ]
    if not usable:
        return {"rows": [], "has_decline": False, "reason": "no_occupancy_rows"}

    start = usable[0]
    end = usable[-1]
    changes: list[dict[str, Any]] = []
    for before, after in zip(usable, usable[1:], strict=False):
        delta = round(
            float(after["unit_occupancy_pct"])
            - float(before["unit_occupancy_pct"]),
            2,
        )
        changes.append(
            {
                "from_month": before["report_month"],
                "to_month": after["report_month"],
                "from_occupancy_pct": float(before["unit_occupancy_pct"]),
                "to_occupancy_pct": float(after["unit_occupancy_pct"]),
                "change_percentage_points": delta,
                "before": before,
                "after": after,
            }
        )

    largest_decline = min(
        changes,
        key=lambda item: item["change_percentage_points"],
        default=None,
    )
    has_decline = bool(
        largest_decline
        and float(largest_decline["change_percentage_points"]) < 0
    )
    if has_decline and largest_decline:
        for field in ("market_rent", "lease_charges"):
            before_value = largest_decline["before"].get(field)
            after_value = largest_decline["after"].get(field)
            if before_value is not None and after_value is not None:
                largest_decline[f"{field}_change"] = round(
                    float(after_value) - float(before_value),
                    2,
                )
    return {
        "rows": usable,
        "start_month": start["report_month"],
        "end_month": end["report_month"],
        "start_occupancy_pct": float(start["unit_occupancy_pct"]),
        "end_occupancy_pct": float(end["unit_occupancy_pct"]),
        "overall_change_percentage_points": round(
            float(end["unit_occupancy_pct"])
            - float(start["unit_occupancy_pct"]),
            2,
        ),
        "changes": changes,
        "has_decline": has_decline,
        "largest_decline": largest_decline if has_decline else None,
    }


class OccupancyInvestigationPolicy:
    """Select each next evidence action from observations already received."""

    def __init__(self, months: int = 12) -> None:
        self.months = months
        self.analysis: dict[str, Any] | None = None

    def decide(self, observations: tuple[AgentObservation, ...]) -> LoopDecision:
        if not observations:
            return LoopDecision(
                action=AgentAction(
                    key="occupancy_trend",
                    tool_name="get_occupancy_trend",
                    arguments={"months": self.months},
                )
            )

        last = observations[-1]
        if last.tool_name == "get_occupancy_trend":
            self.analysis = analyze_occupancy_trend(last.data or [])
            if not self.analysis.get("has_decline"):
                return LoopDecision(
                    complete=True,
                    reason="no_occupancy_decline_found",
                )
            decline = self.analysis["largest_decline"]
            return LoopDecision(
                action=AgentAction(
                    key="vacancies_at_decline",
                    tool_name="get_vacant_units",
                    arguments={
                        "limit": 50,
                        "report_month": decline["to_month"],
                    },
                )
            )

        if last.tool_name == "get_vacant_units":
            if last.data:
                return LoopDecision(
                    action=AgentAction(
                        key="rent_mix_at_decline",
                        tool_name="get_rent_by_unit_type",
                        arguments={
                            "report_month": self.analysis["largest_decline"]["to_month"]
                        },
                    )
                )
            return self._website_action()

        if last.tool_name == "get_rent_by_unit_type":
            return self._website_action()

        if last.tool_name == "search_property_content":
            return LoopDecision(
                complete=True,
                reason="occupancy_evidence_complete",
            )

        return LoopDecision(complete=True, reason="no_additional_evidence_action")

    @staticmethod
    def _website_action() -> LoopDecision:
        return LoopDecision(
            action=AgentAction(
                key="public_leasing_context",
                tool_name="search_property_content",
                arguments={
                    "query": (
                        "leasing amenities apartment features floorplans parking "
                        "resident convenience"
                    ),
                    "page_type": "amenities",
                    "n_results": 3,
                },
            )
        )


def build_occupancy_investigation_report(
    *,
    property_code: str,
    property_name: str,
    observations: tuple[AgentObservation, ...],
    invocations: dict[str, ToolResult],
    loop_result: AgentLoopResult,
    total_tool_calls: int,
) -> tuple[OccupancyInvestigationReport, str]:
    """Build and verify a deterministic executive brief from collected evidence."""
    by_tool = {observation.tool_name: observation.data for observation in observations}
    trend = list(by_tool.get("get_occupancy_trend") or [])
    vacancies = list(by_tool.get("get_vacant_units") or [])
    rent_by_type = list(by_tool.get("get_rent_by_unit_type") or [])
    retrieval = list(by_tool.get("search_property_content") or [])
    analysis = analyze_occupancy_trend(trend)
    retrieved_at = datetime.now(UTC).isoformat()

    citations: list[InvestigationCitation] = []
    citation_by_tool: dict[str, InvestigationCitation] = {}

    structured_evidence = {
        "get_occupancy_trend": {
            "rows": trend,
            "derived": analysis,
        },
        "get_vacant_units": {"rows": vacancies, "returned_count": len(vacancies)},
        "get_rent_by_unit_type": {"rows": rent_by_type},
    }
    for tool_name, evidence in structured_evidence.items():
        invocation = _invocation_for_tool(invocations, tool_name)
        if invocation is None:
            continue
        citation = _citation(
            property_code=property_code,
            source_type="structured_tool",
            source_name=tool_name,
            tool_invocation_id=invocation.invocation_id,
            evidence=evidence,
            retrieved_at=retrieved_at,
        )
        citations.append(citation)
        citation_by_tool[tool_name] = citation

    retrieval_invocation = _invocation_for_tool(
        invocations,
        "search_property_content",
    )
    if retrieval_invocation:
        for result in retrieval[:3]:
            metadata = dict(result.get("metadata") or {})
            evidence = {
                "id": result.get("id"),
                "content": result.get("content"),
                "metadata": metadata,
            }
            citations.append(
                _citation(
                    property_code=property_code,
                    source_type="retrieval",
                    source_name="search_property_content",
                    tool_invocation_id=retrieval_invocation.invocation_id,
                    document_id=str(metadata.get("document_id") or "") or None,
                    chunk_id=str(result.get("id") or "") or None,
                    source_url=str(metadata.get("source_url") or "") or None,
                    evidence=evidence,
                    retrieved_at=retrieved_at,
                )
            )

    trend_citation = citation_by_tool.get("get_occupancy_trend")
    if trend_citation is None:
        raise EvidenceVerificationError("occupancy trend evidence is missing")

    findings: list[InvestigationFinding] = []
    if not analysis.get("rows"):
        summary = (
            f"No occupancy history was available for {property_name} (`{property_code}`), "
            "so the requested decline investigation could not establish a trend."
        )
        findings.append(
            InvestigationFinding(
                finding_id="occupancy_history",
                title="Occupancy history unavailable",
                narrative="The structured occupancy tool returned no usable monthly rows.",
                citation_ids=[trend_citation.citation_id],
            )
        )
    elif not analysis.get("has_decline"):
        summary = (
            f"The available {len(analysis['rows'])}-month series for {property_name} "
            "does not contain a month-over-month occupancy decline. The investigation "
            "stopped before requesting downstream vacancy or website evidence."
        )
        findings.append(
            InvestigationFinding(
                finding_id="occupancy_movement",
                title="No decline found in available history",
                narrative=(
                    f"Occupancy moved from {analysis['start_occupancy_pct']:.2f}% in "
                    f"{analysis['start_month']} to {analysis['end_occupancy_pct']:.2f}% "
                    f"in {analysis['end_month']}."
                ),
                metrics=[
                    _metric(
                        "Starting occupancy",
                        analysis["start_occupancy_pct"],
                        "%",
                        trend_citation,
                    ),
                    _metric(
                        "Ending occupancy",
                        analysis["end_occupancy_pct"],
                        "%",
                        trend_citation,
                    ),
                    _metric(
                        "Overall change",
                        analysis["overall_change_percentage_points"],
                        "percentage points",
                        trend_citation,
                    ),
                ],
                citation_ids=[trend_citation.citation_id],
            )
        )
    else:
        decline = analysis["largest_decline"]
        overall_change = analysis["overall_change_percentage_points"]
        summary = (
            f"For {property_name} (`{property_code}`), occupancy moved from "
            f"{analysis['start_occupancy_pct']:.2f}% to "
            f"{analysis['end_occupancy_pct']:.2f}% across the available period "
            f"({overall_change:+.2f} percentage points). The largest monthly decline "
            f"was {decline['change_percentage_points']:.2f} points from "
            f"{decline['from_month']} to {decline['to_month']}. The evidence below "
            "describes concurrent rent-roll and leasing conditions; it does not prove "
            "that any one condition caused the decline."
        )
        findings.extend(
            _decline_findings(
                analysis=analysis,
                vacancies=vacancies,
                rent_by_type=rent_by_type,
                retrieval=retrieval,
                citations=citations,
                citation_by_tool=citation_by_tool,
            )
        )

    markdown = _report_markdown(property_name, property_code, summary, findings, citations)
    artifact = InvestigationArtifact(
        artifact_id=str(uuid4()),
        type="executive_brief",
        name="occupancy-decline-executive-brief.md",
        content_type="text/markdown",
        content=markdown,
    )
    report = OccupancyInvestigationReport(
        summary=summary,
        findings=findings,
        citations=citations,
        artifacts=[artifact],
        trace_summary=InvestigationTraceSummary(
            steps=loop_result.steps,
            tool_calls=total_tool_calls,
            duration_ms=loop_result.duration_ms,
            tool_order=[action.tool_name for action in loop_result.actions],
            stop_reason=loop_result.reason or "completed",
            verification_status="passed",
            verification_checks=[],
        ),
    )
    checks = verify_occupancy_investigation_report(report, property_code)
    report.trace_summary.verification_checks = checks
    return report, markdown


def verify_occupancy_investigation_report(
    report: OccupancyInvestigationReport,
    property_code: str,
) -> list[str]:
    """Reject unsupported metrics, missing chunks, and cross-property citations."""
    normalized_code = property_code.lower()
    citation_index = {citation.citation_id: citation for citation in report.citations}
    if len(citation_index) != len(report.citations):
        raise EvidenceVerificationError("duplicate citation IDs were returned")

    for citation in report.citations:
        if citation.property_code.lower() != normalized_code:
            raise EvidenceVerificationError("citation belongs to another property")
        if citation.source_type == "retrieval" and not citation.chunk_id:
            raise EvidenceVerificationError("retrieval citation is missing a chunk ID")
        if citation.source_type == "retrieval":
            metadata = dict(citation.evidence.get("metadata") or {})
            evidence_property = str(metadata.get("property_code") or "").lower()
            if evidence_property != normalized_code:
                raise EvidenceVerificationError(
                    "retrieval evidence belongs to another property"
                )

    metric_count = 0
    for finding in report.findings:
        for citation_id in finding.citation_ids:
            if citation_id not in citation_index:
                raise EvidenceVerificationError(
                    f"finding references missing evidence: {citation_id}"
                )
        for metric in finding.metrics:
            metric_count += 1
            citation = citation_index.get(metric.citation_id)
            if citation is None:
                raise EvidenceVerificationError(
                    f"metric references missing evidence: {metric.citation_id}"
                )
            if not _evidence_contains_value(citation.evidence, metric.value):
                raise EvidenceVerificationError(
                    f"unsupported numerical metric: {metric.label}={metric.value}"
                )

    return [
        f"property_scope_valid:{len(report.citations)}",
        f"citation_references_valid:{len(citation_index)}",
        f"numerical_metrics_grounded:{metric_count}",
    ]


def _decline_findings(
    *,
    analysis: dict[str, Any],
    vacancies: list[dict[str, Any]],
    rent_by_type: list[dict[str, Any]],
    retrieval: list[dict[str, Any]],
    citations: list[InvestigationCitation],
    citation_by_tool: dict[str, InvestigationCitation],
) -> list[InvestigationFinding]:
    decline = analysis["largest_decline"]
    trend_citation = citation_by_tool["get_occupancy_trend"]
    findings = [
        InvestigationFinding(
            finding_id="occupancy_movement",
            title="Largest occupancy decline",
            narrative=(
                f"The sharpest monthly movement occurred from {decline['from_month']} "
                f"to {decline['to_month']}, when occupancy moved from "
                f"{decline['from_occupancy_pct']:.2f}% to "
                f"{decline['to_occupancy_pct']:.2f}%."
            ),
            metrics=[
                _metric(
                    "Starting occupancy",
                    analysis["start_occupancy_pct"],
                    "%",
                    trend_citation,
                ),
                _metric(
                    "Ending occupancy",
                    analysis["end_occupancy_pct"],
                    "%",
                    trend_citation,
                ),
                _metric(
                    "Largest monthly decline",
                    decline["change_percentage_points"],
                    "percentage points",
                    trend_citation,
                ),
            ],
            citation_ids=[trend_citation.citation_id],
        )
    ]

    financial_metrics: list[InvestigationMetric] = []
    for label, field in (
        ("Market rent change", "market_rent"),
        ("Lease charges change", "lease_charges"),
    ):
        change = decline.get(f"{field}_change")
        if change is None:
            continue
        financial_metrics.append(_metric(label, change, "USD", trend_citation))
    findings.append(
        InvestigationFinding(
            finding_id="concurrent_financial_movement",
            title="Concurrent rent-roll movement",
            narrative=(
                "Market rent and lease-charge changes are reported for the same two "
                "months as the largest occupancy decline. They are correlations in the "
                "available rent roll, not verified causal drivers."
            ),
            metrics=financial_metrics,
            citation_ids=[trend_citation.citation_id],
        )
    )

    vacancy_citation = citation_by_tool.get("get_vacant_units")
    if vacancy_citation:
        unit_types: dict[str, int] = {}
        for row in vacancies:
            unit_type = str(row.get("unit_type") or "Unknown")
            unit_types[unit_type] = unit_types.get(unit_type, 0) + 1
        leading_types = sorted(unit_types.items(), key=lambda item: (-item[1], item[0]))[:3]
        mix_text = ", ".join(f"{name}: {count}" for name, count in leading_types)
        narrative = (
            f"The period-specific vacancy query returned {len(vacancies)} vacant units."
        )
        if mix_text:
            narrative += f" The leading returned unit types were {mix_text}."
        if rent_by_type:
            narrative += (
                f" Unit-type rent evidence covered {len(rent_by_type)} unit types for "
                "the same report month."
            )
        citation_ids = [vacancy_citation.citation_id]
        rent_citation = citation_by_tool.get("get_rent_by_unit_type")
        if rent_citation:
            citation_ids.append(rent_citation.citation_id)
        findings.append(
            InvestigationFinding(
                finding_id="vacancy_exposure",
                title="Vacancy and unit-type exposure",
                narrative=narrative,
                metrics=[
                    _metric(
                        "Returned vacant units",
                        len(vacancies),
                        "units",
                        vacancy_citation,
                    )
                ],
                citation_ids=citation_ids,
            )
        )

    retrieval_citations = [
        citation for citation in citations if citation.source_type == "retrieval"
    ]
    if retrieval and retrieval_citations:
        findings.append(
            InvestigationFinding(
                finding_id="public_leasing_context",
                title="Public leasing context",
                narrative=(
                    "The property website evidence supplies current advertised amenities "
                    "and leasing context. It should be treated as positioning evidence, "
                    "not as proof of the occupancy decline's cause."
                ),
                citation_ids=[citation.citation_id for citation in retrieval_citations],
            )
        )
    return findings


def _citation(
    *,
    property_code: str,
    source_type: str,
    source_name: str,
    tool_invocation_id: str,
    evidence: dict[str, Any],
    retrieved_at: str,
    document_id: str | None = None,
    chunk_id: str | None = None,
    source_url: str | None = None,
) -> InvestigationCitation:
    serialized = json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=str)
    return InvestigationCitation(
        citation_id=f"citation_{uuid4()}",
        property_code=property_code.lower(),
        source_type=source_type,
        source_name=source_name,
        tool_invocation_id=tool_invocation_id,
        document_id=document_id,
        chunk_id=chunk_id,
        source_url=source_url,
        content_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        retrieved_at=retrieved_at,
        evidence=evidence,
    )


def _metric(
    label: str,
    value: float | int,
    unit: str,
    citation: InvestigationCitation,
) -> InvestigationMetric:
    return InvestigationMetric(
        label=label,
        value=value,
        unit=unit,
        citation_id=citation.citation_id,
    )


def _invocation_for_tool(
    invocations: dict[str, ToolResult],
    tool_name: str,
) -> ToolResult | None:
    return next(
        (result for result in invocations.values() if result.tool_name == tool_name),
        None,
    )


def _evidence_contains_value(evidence: Any, expected: float | int | str) -> bool:
    if isinstance(evidence, dict):
        return any(_evidence_contains_value(value, expected) for value in evidence.values())
    if isinstance(evidence, list):
        return any(_evidence_contains_value(value, expected) for value in evidence)
    if isinstance(expected, int | float) and isinstance(evidence, int | float):
        return abs(float(evidence) - float(expected)) <= 0.011
    return evidence == expected


def _report_markdown(
    property_name: str,
    property_code: str,
    summary: str,
    findings: list[InvestigationFinding],
    citations: list[InvestigationCitation],
) -> str:
    lines = [
        f"### Occupancy Investigation: {property_name} (`{property_code}`)",
        "",
        "#### Executive Brief",
        "",
        summary,
        "",
        "#### Findings",
    ]
    for finding in findings:
        references = ", ".join(f"`{item}`" for item in finding.citation_ids)
        lines.extend(["", f"- **{finding.title}:** {finding.narrative}"])
        if references:
            lines.append(f"  Evidence: {references}")

    source_urls = sorted(
        {citation.source_url for citation in citations if citation.source_url}
    )
    if source_urls:
        lines.extend(["", "#### Public Sources", ""])
        lines.extend(f"- [{url}]({url})" for url in source_urls)
    return "\n".join(lines)
