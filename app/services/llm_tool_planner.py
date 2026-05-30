from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.services.intent_router import RETRIEVAL_INTENTS, STRUCTURED_INTENTS, IntentRouter

RouteType = Literal[
    "structured",
    "retrieval",
    "hybrid",
    "unsupported",
    "clarification",
    "sql_approval",
]


class StructuredToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    query: str
    page_type: str | None = None
    n_results: int = 5


class ToolPlan(BaseModel):
    route: RouteType
    structured_tools: list[StructuredToolCall] = Field(default_factory=list)
    retrieval_queries: list[RetrievalQuery] = Field(default_factory=list)
    unsupported_reason: str | None = None
    clarification_question: str | None = None
    sql_request: str | None = None
    confidence: float = 0.0


ALLOWED_PAGE_TYPES = {
    "amenities",
    "floorplans",
    "neighborhood",
    "contact",
    "gallery",
    "fee-guide",
    "residents",
    "faqs",
}


UNSUPPORTED_FACT_TERMS = {
    "crime",
    "crime rate",
    "public safety",
    "safety",
    "police",
    "neighborhood safety",
    "violent crime",
    "property crime",
    "robbery",
    "assault",
    "homicide",
    "gun violence",
    "school rating",
    "school ratings",
    "google rating",
    "google ratings",
    "google review",
    "google reviews",
    "resident review",
    "resident reviews",
    "review",
    "reviews",
    "rating",
    "ratings",
    "testimonial",
    "testimonials",
    "maintenance response time",
    "maintenance SLA",
    "resident satisfaction",
    "satisfaction score",
    "demographics",
    "income level",
    "median income",
    "cap rate",
    "noi",
    "net operating income",
    "market comps",
    "walk score",
    "transit score",
}


PII_OR_SENSITIVE_TERMS = {
    "resident name",
    "resident names",
    "tenant name",
    "tenant names",
    "names",
    "name",
    "email",
    "emails",
    "phone",
    "phone number",
    "phone numbers",
    "ssn",
    "social security",
    "date of birth",
    "dob",
}


UNSAFE_SQL_TERMS = {
    "delete",
    "update",
    "insert",
    "drop",
    "alter",
    "truncate",
    "create",
    "replace",
    "merge",
    "grant",
    "revoke",
    "execute",
    "run this sql",
    "raw sql",
    "drop table",
}


AMBIGUOUS_EXACT_REQUESTS = {
    "compare",
    "compare this property",
    "analyze",
    "analyze this",
    "show",
    "show me",
    "details",
    "more details",
}


AMBIGUOUS_DOMAIN_REQUESTS = {
    "charge",
    "charges",
    "fee",
    "fees",
    "lease",
    "rent",
    "balance",
    "balances",
    "vacancy",
    "vacant",
    "occupancy",
}


CUSTOM_SQL_PATTERNS = [
    # Counts / grouping
    r"\bcount\b.*\b(unit|units|occupied|vacant)\b",
    r"\bcount occupied and vacant\b",
    r"\bgroup(?:ed)? by\b",

    # Market rent totals / grouped rent
    r"\btotal market rent by\b",

    # Top / highest / lowest individual market rent rows
    r"\btop\s+\d+\s+market rents?\b",
    r"\btop\s+\d+\s+rents?\b",
    r"\bhighest\s+\d+\s+market rents?\b",
    r"\bhighest\s+\d+\s+rents?\b",
    r"\blowest\s+\d+\s+market rents?\b",
    r"\blowest\s+\d+\s+rents?\b",
    r"\btop market rents?\b",
    r"\bhighest market rents?\b",
    r"\blowest market rents?\b",

    # Balance queries not covered by top_balances safe tool
    r"\baverage balance by\b",
    r"\bavg balance by\b",
    r"\bmedian balance by\b",
    r"\bbalance by\b",

    # Lease charge custom groupings
    r"\blease charges? by\b",
    r"\baverage lease charges? by\b",
    r"\bavg lease charges? by\b",
    r"\bmedian lease charges? by\b",

    # Vacancy / occupancy grouped counts
    r"\bvacant units? by\b",
    r"\boccupied units? by\b",
    r"\bvacancy count by\b",
    r"\boccupancy count by\b",
]


def _contains_any(text: str, terms: set[str] | tuple[str, ...] | list[str]) -> bool:
    return any(term in text for term in terms)


def _looks_like_pii_or_sensitive_request(message: str) -> bool:
    lowered = message.lower()
    return any(term in lowered for term in PII_OR_SENSITIVE_TERMS)


def _looks_like_unsupported_fact(message: str) -> bool:
    lowered = message.lower()
    return _contains_any(lowered, UNSUPPORTED_FACT_TERMS)


def _looks_like_unsafe_or_cross_property_sql(message: str) -> bool:
    lowered = message.lower()

    if _looks_like_pii_or_sensitive_request(lowered):
        return True

    if _contains_any(lowered, UNSAFE_SQL_TERMS):
        return True

    if re.search(r"\b(?:query|use|switch to|for)\s+\d{3}[a-z]?\b", lowered):
        return True

    return any(
        phrase in lowered
        for phrase in [
            "all properties",
            "across properties",
            "across all properties",
            "every property",
            "ignore selected property",
            "ignore the selected property",
            "other property",
            "another property",
            "not the selected property",
        ]
    )


def _is_predefined_unit_count_by_type(message: str) -> bool:
    lowered = message.lower()

    if "occupied" in lowered or "vacant" in lowered or "vacancy" in lowered:
        return False

    return (
        "unit type" in lowered
        and any(term in lowered for term in ["how many", "count", "units"])
        and not any(term in lowered for term in ["top", "highest", "lowest", "median"])
    )


def _is_predefined_average_market_rent_by_type(message: str) -> bool:
    lowered = message.lower()
    return (
        "unit type" in lowered
        and any(
            term in lowered
            for term in [
                "average market rent",
                "avg market rent",
                "average rent",
                "avg rent",
            ]
        )
        and not any(term in lowered for term in ["top", "highest", "lowest", "median"])
    )


def _is_predefined_rent_lease_comparison(message: str) -> bool:
    lowered = message.lower()
    return (
        any(
            term in lowered
            for term in ["compare", "comparison", " vs ", "versus", "against"]
        )
        and ("rent" in lowered or "market rent" in lowered)
        and ("lease charge" in lowered or "lease charges" in lowered)
    )


def _looks_like_custom_structured_sql(message: str) -> bool:
    lowered = message.lower()

    if (
        _looks_like_unsupported_fact(lowered)
        or _looks_like_unsafe_or_cross_property_sql(lowered)
        or _looks_like_pii_or_sensitive_request(lowered)
    ):
        return False

    # These are handled directly by predefined safe tools.
    if (
        _is_predefined_unit_count_by_type(lowered)
        or _is_predefined_average_market_rent_by_type(lowered)
        or _is_predefined_rent_lease_comparison(lowered)
    ):
        return False

    if not any(
        term in lowered
        for term in [
            "balance",
            "charge",
            "charges",
            "lease",
            "market rent",
            "market rents",
            "rent",
            "rents",
            "unit",
            "units",
            "vacant",
            "occupied",
        ]
    ):
        return False

    return any(re.search(pattern, lowered) for pattern in CUSTOM_SQL_PATTERNS)


def _sql_request_from_message(message: str) -> str:
    return (
        "Draft a property-scoped SELECT query to answer this custom structured "
        f"rent-roll question: {message.strip()}"
    )


def _looks_ambiguous(message: str) -> bool:
    lowered = " ".join(message.lower().strip().split())
    if not lowered:
        return True

    if lowered in AMBIGUOUS_EXACT_REQUESTS:
        return True

    tokens = re.findall(r"[a-z0-9'-]+", lowered)

    if tokens and all(token in AMBIGUOUS_DOMAIN_REQUESTS for token in tokens):
        return True

    return len(tokens) <= 2 and lowered in {"compare", "analyze", "details", "show"}


def _clarification_question_for_ambiguous(message: str) -> str:
    lowered = message.lower().strip()

    if any(term in lowered for term in ["charge", "charges", "fee", "fees", "lease"]):
        return (
            "What charge view would you like? You can ask for lease charge total, "
            "biggest charge categories, charge breakdown, or rent vs lease charges comparison."
        )

    if any(term in lowered for term in ["rent", "occupancy", "vacancy", "vacant"]):
        return (
            "Which metric view would you like? You can ask for latest KPIs, occupancy trend, "
            "vacant units, or rent by unit type."
        )

    if any(term in lowered for term in ["balance", "balances"]):
        return (
            "What balance view would you like? Currently I can show top balances "
            "or highest balance units."
        )

    return "What would you like to compare or analyze for this property?"


def _infer_page_type(message: str) -> str | None:
    lowered = message.lower()

    if "floorplan" in lowered or "floor plan" in lowered or "bedroom" in lowered:
        return "floorplans"

    if "neighborhood" in lowered or "nearby" in lowered:
        return "neighborhood"

    if "contact" in lowered or "phone" in lowered or "email" in lowered:
        return "contact"

    if "gallery" in lowered or "photos" in lowered or "images" in lowered:
        return "gallery"

    if "fee" in lowered or "fees" in lowered:
        return "fee-guide"

    if "resident" in lowered:
        return "residents"

    if "faq" in lowered or "question" in lowered:
        return "faqs"

    if any(
        term in lowered
        for term in [
            "amenit",
            "ev",
            "charging",
            "parking",
            "pet",
            "bike",
            "storage",
            "pool",
            "fitness",
            "gym",
            "washer",
            "dryer",
            "feature",
            "features",
        ]
    ):
        return "amenities"

    return None


def _structured_tools_from_intent(intent: str | None) -> list[StructuredToolCall]:
    if intent in {"latest_kpis", "executive_summary"}:
        return [StructuredToolCall(name="get_latest_property_kpis")]

    if intent == "occupancy_trend":
        return [StructuredToolCall(name="get_occupancy_trend", args={"months": 12})]

    if intent == "charge_breakdown":
        return [StructuredToolCall(name="get_charge_breakdown", args={"limit": 10})]

    if intent == "top_balances":
        return [StructuredToolCall(name="get_top_balances", args={"limit": 10})]

    if intent == "vacant_units":
        return [StructuredToolCall(name="get_vacant_units", args={"limit": 20})]

    if intent == "rent_by_unit_type":
        return [StructuredToolCall(name="get_rent_by_unit_type")]

    if intent == "rent_lease_comparison":
        return [StructuredToolCall(name="get_latest_property_kpis")]

    return []


def _heuristic_structured_tools(message: str) -> list[StructuredToolCall]:
    lowered = message.lower()

    # Strict known intents first.
    if "executive summary" in lowered or "quick executive summary" in lowered:
        return [StructuredToolCall(name="get_latest_property_kpis")]

    if _is_predefined_rent_lease_comparison(lowered):
        return [StructuredToolCall(name="get_latest_property_kpis")]

    if _is_predefined_unit_count_by_type(lowered):
        return [StructuredToolCall(name="get_rent_by_unit_type")]

    if _is_predefined_average_market_rent_by_type(lowered):
        return [StructuredToolCall(name="get_rent_by_unit_type")]

    # Custom SQL-style questions should not fall into broad KPI/rent heuristics.
    if _looks_like_custom_structured_sql(message):
        return []

    if any(
        phrase in lowered
        for phrase in [
            "occupancy trend",
            "occupancy over time",
            "monthly occupancy",
            "occupancy history",
            "occupancy move",
            "occupancy changed",
            "occupancy change",
        ]
    ):
        return [StructuredToolCall(name="get_occupancy_trend", args={"months": 12})]

    if any(
        phrase in lowered
        for phrase in [
            "charge breakdown",
            "largest charge",
            "biggest charge",
            "income bucket",
            "income buckets",
            "fee breakdown",
        ]
    ):
        return [StructuredToolCall(name="get_charge_breakdown", args={"limit": 10})]

    if any(term in lowered for term in ["top balance", "top balances", "highest balance"]):
        return [StructuredToolCall(name="get_top_balances", args={"limit": 10})]

    if (
        "vacant" in lowered
        and any(term in lowered for term in ["unit", "units", "list", "show", "open"])
    ):
        return [StructuredToolCall(name="get_vacant_units", args={"limit": 20})]

    if (
        "unit type" in lowered
        or "rent by" in lowered
        or "average market rent" in lowered
        or "average rent" in lowered
    ):
        return [StructuredToolCall(name="get_rent_by_unit_type")]

    if any(
        term in lowered
        for term in [
            "latest occupancy",
            "current occupancy",
            "latest kpi",
            "market rent",
            "lease charge",
            "vacant count",
            "executive summary",
            "leadership overview",
            "quick summary",
            "summary",
        ]
    ):
        return [StructuredToolCall(name="get_latest_property_kpis")]

    return []


def _looks_retrieval_related(message: str) -> bool:
    lowered = message.lower()
    return any(
        term in lowered
        for term in [
            "amenity",
            "amenities",
            "feature",
            "features",
            "ev",
            "charging",
            "parking",
            "pet",
            "bike",
            "storage",
            "floorplan",
            "floor plan",
            "neighborhood",
            "contact",
            "website",
            "pool",
            "fitness",
            "gym",
            "washer",
            "dryer",
            "nearby",
        ]
    )


def _build_retrieval_query(message: str) -> RetrievalQuery:
    lowered = message.lower()
    page_type = _infer_page_type(message)

    if "ev" in lowered or "electric" in lowered:
        query = "EV charging electric vehicle parking amenities"
        page_type = page_type or "amenities"
    elif "parking" in lowered:
        query = "parking garage resident parking amenities"
        page_type = page_type or "amenities"
    elif "pet" in lowered:
        query = "pet policy pet friendly amenities fees"
        page_type = page_type or "amenities"
    elif "bike" in lowered or "storage" in lowered:
        query = "bike storage bicycle storage amenities"
        page_type = page_type or "amenities"
    elif "floor" in lowered or "bedroom" in lowered:
        query = "floorplans bedroom apartment layouts"
        page_type = "floorplans"
    else:
        query = message

    return RetrievalQuery(
        query=query,
        page_type=page_type,
        n_results=10 if page_type == "floorplans" else 5,
    )


def validate_tool_plan(
    plan: ToolPlan,
    structured_allowlist: set[str],
) -> ToolPlan:
    cleaned_tools: list[StructuredToolCall] = []

    for tool_call in plan.structured_tools:
        if tool_call.name not in structured_allowlist:
            continue

        args = dict(tool_call.args or {})
        args.pop("property_code", None)

        if "limit" in args:
            try:
                args["limit"] = max(1, min(50, int(args["limit"])))
            except (TypeError, ValueError):
                args.pop("limit", None)

        if "months" in args:
            try:
                args["months"] = max(1, min(36, int(args["months"])))
            except (TypeError, ValueError):
                args.pop("months", None)

        cleaned_tools.append(StructuredToolCall(name=tool_call.name, args=args))

    cleaned_queries: list[RetrievalQuery] = []
    for query in plan.retrieval_queries:
        clean_query = (query.query or "").strip()
        if not clean_query:
            continue

        try:
            n_results = int(query.n_results or 5)
        except (TypeError, ValueError):
            n_results = 5

        n_results = max(1, min(10, n_results))
        page_type = query.page_type if query.page_type in ALLOWED_PAGE_TYPES else None

        cleaned_queries.append(
            RetrievalQuery(query=clean_query, page_type=page_type, n_results=n_results)
        )

    route = plan.route
    unsupported_reason = plan.unsupported_reason
    clarification_question = plan.clarification_question
    sql_request = plan.sql_request

    if route == "unsupported":
        cleaned_tools = []
        cleaned_queries = []
        sql_request = None
        unsupported_reason = unsupported_reason or (
            "I don't have that data for this property. The available sources are "
            "rent-roll metrics and scraped property website content, and this question "
            "is outside those sources."
        )

    if route == "clarification":
        cleaned_tools = []
        cleaned_queries = []
        sql_request = None
        clarification_question = clarification_question or (
            "What would you like to look at for this property?"
        )

    if route == "sql_approval":
        cleaned_tools = []
        cleaned_queries = []
        if not sql_request:
            route = "clarification"
            clarification_question = (
                "What custom structured metric would you like me to prepare for review?"
            )

    if route == "structured" and not cleaned_tools:
        route = "clarification"
        clarification_question = clarification_question or (
            "Which structured metric would you like: occupancy, rent, lease charges, "
            "vacancy, balances, or charge breakdown?"
        )

    if route == "retrieval" and not cleaned_queries:
        route = "clarification"
        clarification_question = clarification_question or (
            "Which website detail would you like: amenities, floorplans, parking, "
            "pet policy, neighborhood, or contact information?"
        )

    if route == "hybrid":
        if cleaned_tools and not cleaned_queries:
            route = "structured"
        elif cleaned_queries and not cleaned_tools:
            route = "retrieval"
        elif not cleaned_tools and not cleaned_queries:
            route = "clarification"
            clarification_question = clarification_question or (
                "What would you like to compare or analyze?"
            )

    return ToolPlan(
        route=route,
        structured_tools=cleaned_tools,
        retrieval_queries=cleaned_queries,
        unsupported_reason=unsupported_reason,
        clarification_question=clarification_question,
        sql_request=sql_request,
        confidence=max(0.0, min(1.0, float(plan.confidence or 0.0))),
    )


class LLMToolPlanner:
    def __init__(
        self,
        structured_allowlist: set[str],
        intent_router: IntentRouter,
    ) -> None:
        self.structured_allowlist = structured_allowlist
        self.intent_router = intent_router

    def _predefined_tool_plan(self, message: str) -> ToolPlan | None:
        """Return strict predefined tool plans only when a safe tool clearly supports it."""
        intent_route = self.intent_router.route(message)
        intent = intent_route.intent

        # Explicit predefined tool cases.
        if _is_predefined_unit_count_by_type(message):
            return ToolPlan(
                route="structured",
                structured_tools=[StructuredToolCall(name="get_rent_by_unit_type")],
                retrieval_queries=[],
                confidence=max(intent_route.confidence, 0.9),
            )

        if _is_predefined_average_market_rent_by_type(message):
            return ToolPlan(
                route="structured",
                structured_tools=[StructuredToolCall(name="get_rent_by_unit_type")],
                retrieval_queries=[],
                confidence=max(intent_route.confidence, 0.9),
            )

        if _is_predefined_rent_lease_comparison(message):
            return ToolPlan(
                route="structured",
                structured_tools=[StructuredToolCall(name="get_latest_property_kpis")],
                retrieval_queries=[],
                confidence=max(intent_route.confidence, 0.9),
            )

        # If it looks like custom SQL, never allow broad semantic intents to swallow it.
        if _looks_like_custom_structured_sql(message):
            return None

        structured_tools = _structured_tools_from_intent(intent)
        if not structured_tools:
            structured_tools = _heuristic_structured_tools(message)

        if not structured_tools:
            return None

        has_retrieval = intent in RETRIEVAL_INTENTS or _looks_retrieval_related(message)

        if has_retrieval:
            return ToolPlan(
                route="hybrid",
                structured_tools=structured_tools,
                retrieval_queries=[_build_retrieval_query(message)],
                confidence=max(intent_route.confidence, 0.8),
            )

        return ToolPlan(
            route="structured",
            structured_tools=structured_tools,
            retrieval_queries=[],
            confidence=max(intent_route.confidence, 0.8),
        )

    def plan_with_llm(
        self,
        *,
        property_code: str,
        property_name: str,
        message: str,
        structured_tool_descriptions: str,
        retrieval_tool_description: str,
        data_sources_description: str,
        chat_model: Callable[[list[dict[str, str]]], str],
    ) -> ToolPlan | None:
        if _looks_ambiguous(message):
            return ToolPlan(
                route="clarification",
                structured_tools=[],
                retrieval_queries=[],
                unsupported_reason=None,
                clarification_question=_clarification_question_for_ambiguous(message),
                sql_request=None,
                confidence=0.95,
            )

        if _looks_like_unsafe_or_cross_property_sql(message):
            return ToolPlan(
                route="unsupported",
                structured_tools=[],
                retrieval_queries=[],
                unsupported_reason=(
                    "I can't prepare that query because it is unsafe, requests sensitive "
                    "resident-level information, or is outside the selected property scope."
                ),
                clarification_question=None,
                sql_request=None,
                confidence=0.95,
            )

        if _looks_like_unsupported_fact(message):
            return ToolPlan(
                route="unsupported",
                structured_tools=[],
                retrieval_queries=[],
                unsupported_reason=(
                    "I don't have that data for this property. The available sources are "
                    "rent-roll metrics and scraped property website content, and this "
                    "question is outside those sources."
                ),
                clarification_question=None,
                sql_request=None,
                confidence=0.95,
            )

        predefined_plan = self._predefined_tool_plan(message)
        if predefined_plan:
            return validate_tool_plan(predefined_plan, self.structured_allowlist)

        if _looks_like_custom_structured_sql(message):
            return ToolPlan(
                route="sql_approval",
                structured_tools=[],
                retrieval_queries=[],
                unsupported_reason=None,
                clarification_question=None,
                sql_request=_sql_request_from_message(message),
                confidence=0.9,
            )

        system_prompt = (
            "You are a tool planner for a property-specific AI assistant.\n"
            "Return only valid JSON. Do not answer the user directly.\n\n"
            "Your job is to choose a route and propose safe tool calls.\n\n"
            "Allowed routes:\n"
            "- structured: use MySQL rent-roll tools only\n"
            "- retrieval: use scraped property website retrieval only\n"
            "- hybrid: use both structured tools and website retrieval\n"
            "- sql_approval: custom structured rent-roll question that needs SQL review\n"
            "- unsupported: requested data is outside current data sources\n"
            "- clarification: user request is too vague\n\n"
            "Strict rules:\n"
            "- Do NOT output SQL.\n"
            "- Do NOT include property_code in tool args.\n"
            "- Do NOT choose or override the active property.\n"
            "- Backend will inject property_code server-side.\n"
            "- Use only the provided structured tool names.\n"
            "- Prefer predefined structured tools when they can answer the question.\n"
            "- Use sql_approval only when no predefined tool can answer a custom structured "
            "rent-roll question.\n"
            "- Use unsupported for crime rate, school ratings, Google reviews, resident "
            "reviews, ratings, testimonials, demographics, cap rate, NOI, market comps, "
            "resident satisfaction, maintenance response time, or PII requests.\n"
            "- Use retrieval only for likely public website content: amenities, floorplans, "
            "parking, EV charging, pet policy, neighborhood, contact, fee guide, FAQs.\n"
            "- Use structured only for rent-roll metrics: occupancy, vacancy, market rent, "
            "lease charges, balances, charge breakdown, rent by unit type.\n"
            "- Use hybrid when the user asks for both analytics and website facts.\n"
            "- Use clarification for vague requests like 'compare this property' or "
            "'analyze this'.\n\n"
            f"Active property context only: {property_name} ({property_code})\n\n"
            f"Available structured tools:\n{structured_tool_descriptions}\n\n"
            f"Retrieval tool:\n{retrieval_tool_description}\n\n"
            f"Available data sources:\n{data_sources_description}\n\n"
            "Return JSON with exactly these keys:\n"
            "{\n"
            '  "route": "structured|retrieval|hybrid|sql_approval|unsupported|clarification",\n'
            '  "structured_tools": [{"name": "tool_name", "args": {}}],\n'
            '  "retrieval_queries": ['
            '{"query": "search query", "page_type": null, "n_results": 5}'
            "],\n"
            '  "unsupported_reason": null,\n'
            '  "clarification_question": null,\n'
            '  "sql_request": null,\n'
            '  "confidence": 0.0\n'
            "}\n\n"
            "Examples:\n"
            'User: "What is the latest occupancy?"\n'
            '{"route":"structured","structured_tools":[{"name":"get_latest_property_kpis","args":{}}],'
            '"retrieval_queries":[],"unsupported_reason":null,"clarification_question":null,'
            '"sql_request":null,"confidence":0.95}\n\n'
            'User: "Does this property have EV charging?"\n'
            '{"route":"retrieval","structured_tools":[],"retrieval_queries":['
            '{"query":"EV charging electric vehicle parking amenities",'
            '"page_type":"amenities","n_results":5}],'
            '"unsupported_reason":null,"clarification_question":null,"sql_request":null,"confidence":0.9}\n\n'
            'User: "What is the latest occupancy and does it have parking?"\n'
            '{"route":"hybrid","structured_tools":[{"name":"get_latest_property_kpis","args":{}}],'
            '"retrieval_queries":[{"query":"parking garage resident parking amenities",'
            '"page_type":"amenities","n_results":5}],'
            '"unsupported_reason":null,"clarification_question":null,"sql_request":null,"confidence":0.9}\n\n'
            'User: "Give me the top 10 market rents for this property."\n'
            '{"route":"sql_approval","structured_tools":[],"retrieval_queries":[],'
            '"unsupported_reason":null,"clarification_question":null,'
            '"sql_request":"Draft a property-scoped SELECT query to answer this '
            "custom structured rent-roll question: Give me the top 10 market "
            'rents for this property.",'
            '"confidence":0.9}\n\n'
            'User: "What is the crime rate around this property?"\n'
            '{"route":"unsupported","structured_tools":[],"retrieval_queries":[],'
            '"unsupported_reason":"I don’t have crime-rate or public-safety data '
            "in the current dataset. The available sources are rent-roll metrics "
            'and scraped property website content.",'
            '"clarification_question":null,"sql_request":null,"confidence":0.95}\n\n'
            'User: "Compare this property"\n'
            '{"route":"clarification","structured_tools":[],"retrieval_queries":[],'
            '"unsupported_reason":null,"clarification_question":"What would you '
            "like to compare — occupancy, rent, lease charges, amenities, "
            'or floorplans?",'
            '"sql_request":null,"confidence":0.85}'
        )

        try:
            payload = chat_model(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ]
            )
            content = str(payload).strip()
            if not content.startswith("{"):
                match = re.search(r"\{.*\}", content, re.DOTALL)
                content = match.group(0) if match else content

            raw = json.loads(content)
            plan = ToolPlan(**raw)
        except Exception:
            return None

        return validate_tool_plan(plan, self.structured_allowlist)

    def deterministic_plan(self, message: str) -> ToolPlan:
        lowered = message.lower().strip()

        if _looks_ambiguous(lowered):
            return ToolPlan(
                route="clarification",
                structured_tools=[],
                retrieval_queries=[],
                clarification_question=_clarification_question_for_ambiguous(message),
                confidence=0.95,
            )

        if _looks_like_unsafe_or_cross_property_sql(lowered):
            return ToolPlan(
                route="unsupported",
                structured_tools=[],
                retrieval_queries=[],
                unsupported_reason=(
                    "I can't prepare that query because it is unsafe, requests sensitive "
                    "resident-level information, or is outside the selected property scope."
                ),
                sql_request=None,
                confidence=0.95,
            )

        if _looks_like_unsupported_fact(lowered):
            return ToolPlan(
                route="unsupported",
                structured_tools=[],
                retrieval_queries=[],
                unsupported_reason=(
                    "I don't have that data for this property. The available sources are "
                    "rent-roll metrics and scraped property website content, and this question "
                    "is outside those sources."
                ),
                sql_request=None,
                confidence=0.95,
            )

        predefined_plan = self._predefined_tool_plan(message)
        if predefined_plan:
            return validate_tool_plan(predefined_plan, self.structured_allowlist)

        if _looks_like_custom_structured_sql(lowered):
            return ToolPlan(
                route="sql_approval",
                structured_tools=[],
                retrieval_queries=[],
                sql_request=_sql_request_from_message(message),
                confidence=0.9,
            )

        intent_route = self.intent_router.route(message)
        intent = intent_route.intent
        has_retrieval = intent in RETRIEVAL_INTENTS or _looks_retrieval_related(message)

        if intent in STRUCTURED_INTENTS:
            return ToolPlan(
                route="structured",
                structured_tools=[StructuredToolCall(name="get_latest_property_kpis")],
                retrieval_queries=[],
                confidence=max(intent_route.confidence, 0.75),
            )

        if has_retrieval:
            return ToolPlan(
                route="retrieval",
                structured_tools=[],
                retrieval_queries=[_build_retrieval_query(message)],
                confidence=max(intent_route.confidence, 0.75),
            )

        return ToolPlan(
            route="clarification",
            structured_tools=[],
            retrieval_queries=[],
            clarification_question="What would you like to look at for this property?",
            confidence=0.5,
        )
