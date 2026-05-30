from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse, urlunparse

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.schemas import ChatResponse, Source, UIComponent
from app.services.intent_router import get_intent_router
from app.services.langchain_tools import build_langchain_tools
from app.services.llm_tool_planner import (
    UNSUPPORTED_FACT_TERMS,
    LLMToolPlanner,
    StructuredToolCall,
    ToolPlan,
    validate_tool_plan,
)
from app.services.sql_approval import draft_sql_for_approval, validate_drafted_sql

REVIEW_RE = re.compile(r"\b(?:reviews?|ratings?|testimonials?)\b", re.IGNORECASE)
EVIDENCE_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
MIN_RETRIEVAL_CONFIDENCE = "medium"
SUPPORTED_STRUCTURED_CAPABILITIES: dict[str, dict[str, Any]] = {
    "latest_kpis": {
        "tool": "get_latest_property_kpis",
        "supports": (
            "latest occupancy",
            "latest market rent",
            "latest lease charges",
            "latest vacant unit count",
            "current KPI snapshot",
        ),
        "complete_dataset": True,
    },
    "occupancy_trend": {
        "tool": "get_occupancy_trend",
        "supports": ("occupancy trend", "monthly occupancy", "occupancy over time"),
        "complete_dataset": True,
    },
    "charge_breakdown": {
        "tool": "get_charge_breakdown",
        "supports": ("charge breakdown", "largest charge categories"),
        "complete_dataset": True,
    },
    "top_balances": {
        "tool": "get_top_balances",
        "supports": ("top balances", "highest balance units"),
        "does_not_support": (
            "average balance",
            "median balance",
            "balance by bedroom category",
        ),
        "complete_dataset": False,
    },
    "vacant_units": {
        "tool": "get_vacant_units",
        "supports": ("vacant unit list", "vacant units and unit types"),
        "complete_dataset": True,
    },
    "rent_by_unit_type": {
        "tool": "get_rent_by_unit_type",
        "supports": (
            "average market rent by unit type",
            "average market rent by bedroom category",
        ),
        "does_not_support": ("median rent", "lease charges by unit type"),
        "complete_dataset": True,
    },
    "rent_lease_comparison": {
        "tool": "get_latest_property_kpis",
        "supports": ("market rent vs lease charges", "rent and lease charge comparison"),
        "complete_dataset": True,
    },
}
UNSUPPORTED_STRUCTURED_AGGREGATE_RULES = (
    {
        "label": "balance aggregates by category",
        "metric_terms": ("balance", "balances", "delinquency", "delinquent"),
        "aggregate_terms": ("average", "avg", "mean", "median", "rate", "ratio", " by "),
        "grouping_terms": (
            "bedroom",
            "category",
            "categories",
            "layout",
            "floorplan",
            "floor plan",
            "unit type",
            "resident status",
        ),
        "allowed_terms": ("top", "highest", "largest", "biggest"),
    },
    {
        "label": "vacancy aggregates by category",
        "metric_terms": ("vacancy", "vacant"),
        "aggregate_terms": ("average", "avg", "rate", "ratio", " by ", "per ", "group"),
        "grouping_terms": (
            "bedroom",
            "category",
            "categories",
            "layout",
            "floorplan",
            "floor plan",
            "unit type",
        ),
        "allowed_terms": (),
    },
    {
        "label": "charge aggregates by category",
        "metric_terms": ("lease charge", "lease charges", "charge", "charges", "fee", "fees"),
        "aggregate_terms": ("median", "rate", "ratio", " by ", "per ", "group"),
        "grouping_terms": (
            "bedroom",
            "category",
            "categories",
            "layout",
            "floorplan",
            "floor plan",
            "unit type",
        ),
        "allowed_terms": (),
    },
    {
        "label": "unsupported structured aggregate",
        "metric_terms": ("market rent", "rent", "occupancy"),
        "aggregate_terms": ("median", "rate", "ratio"),
        "grouping_terms": (),
        "allowed_terms": (),
    },
    {
        "label": "top market rents by unit",
        "metric_terms": ("market rent", "market rents", "rent", "rents"),
        "aggregate_terms": ("top", "highest", "largest"),
        "grouping_terms": (),
        "allowed_terms": (),
    },
)
STRUCTURED_TOOL_ALLOWLIST = {
    "get_property_profile",
    "get_latest_property_kpis",
    "get_occupancy_trend",
    "get_charge_breakdown",
    "get_top_balances",
    "get_vacant_units",
    "get_rent_by_unit_type",
    "get_report_periods",
}
ANSWER_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{1,}", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(20\d{2})\b")
ANSWER_STOPWORDS = {
    "about",
    "active",
    "and",
    "answer",
    "cite",
    "data",
    "does",
    "do",
    "for",
    "from",
    "have",
    "only",
    "or",
    "property",
    "selected",
    "source",
    "sources",
    "they",
    "them",
    "the",
    "this",
    "use",
    "using",
    "website",
    "with",
}
WEBSITE_BOILERPLATE_EXACT = {
    "available",
    "community fee guide",
    "contact",
    "contact us",
    "e-brochure",
    "email us",
    "fee guide",
    "follow us",
    "got it!",
    "legal",
    "map",
    "our address",
    "page overview",
    "schedule a tour",
    "stay connected with us",
    "virtual tours",
    "welcome home",
}
WEBSITE_BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^image:",
        r"^@",
        r"additional fees?",
        r"all dimensions are approximate",
        r"artist.?s rendering",
        r"base rent",
        r"by using this website",
        r"contact a representative",
        r"cookie policy",
        r"fee list",
        r"floorplans? are artist",
        r"monthly leasing prices",
        r"move-?in|move-?out",
        r"not all features are available",
        r"optional services",
        r"prices and availability",
        r"required monthly fees",
        r"subject to change",
        r"there.?s room for you",
        r"variable or usage-based",
    ]
]


class LangChainOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tools = {tool.name: tool for tool in build_langchain_tools(settings)}
        self.intent_router = get_intent_router(settings)

    def answer(
        self,
        property_code: str,
        message: str,
        model: str,
        on_token: Callable[[str], None] | None = None,
    ) -> ChatResponse:
        normalized_code = property_code.lower()
        tool_results: dict[str, Any] = {}
        components: list[UIComponent] = []
        sources: list[Source] = []

        profile = self._call_tool("get_property_profile", property_code=normalized_code)
        if profile is None:
            return ChatResponse(
                property_code=normalized_code,
                model=model,
                answer_markdown=f"I could not find property code `{normalized_code}`.",
            )

        tool_results["property_profile"] = profile
        scope_note = self._property_scope_note(message, profile)

        intent_route = self.intent_router.route(message)
        intent = intent_route.intent
        provider, _, model_name = model.partition(":")

        location_only = (
            self._wants_location_answer(message.lower()) or intent == "location"
        ) and not self._wants_location_website_context(message.lower())

        if location_only:
            answer_markdown = (
                f"### {profile['property_name']} (`{normalized_code}`)\n\n"
                f"{self._location_answer(profile, attempted_website_lookup=False)}"
            )
            return ChatResponse(
                property_code=normalized_code,
                model=model,
                answer_markdown=self._with_scope_note(answer_markdown, scope_note),
                components=[],
                sources=[],
                tool_results=tool_results,
            )

        planner = LLMToolPlanner(STRUCTURED_TOOL_ALLOWLIST, self.intent_router)

        structured_tool_desc = "\n".join(
            f"- {name}: {self.tools[name].description or 'No description'}"
            for name in sorted(STRUCTURED_TOOL_ALLOWLIST)
            if name in self.tools
        )

        retrieval_desc = (
            self.tools["search_property_content"].description
            if "search_property_content" in self.tools
            else "Search scraped website content for property-specific evidence."
        )

        data_sources_desc = (
            "Available data sources are structured MySQL rent-roll data and scraped "
            "public property website content. MySQL supports occupancy, vacancy, market "
            "rent, lease charges, balances, charge breakdown, rent by unit type, and "
            "report periods. Website retrieval supports public page content such as "
            "amenities, floorplans, parking, EV charging, pet policy, neighborhood, "
            "contact, fee guide, residents, FAQs, and gallery text. External data such "
            "as crime rate, school ratings, Google reviews, resident satisfaction, "
            "demographics, NOI, cap rate, and market comps is not available unless "
            "explicitly scraped or loaded."
        )

        plan: ToolPlan | None = None
        chat_invoke: Callable[[list[dict[str, str]]], str] | None = None

        if provider != "mock" and model_name:
            def chat_invoke(messages: list[dict[str, str]]) -> str:
                response = self._chat_model(provider, model_name).invoke(
                    [
                        SystemMessage(content=messages[0]["content"]),
                        HumanMessage(content=messages[1]["content"]),
                    ]
                )
                return str(response.content)

            plan = planner.plan_with_llm(
                property_code=normalized_code,
                property_name=profile.get("property_name") or normalized_code,
                message=message,
                structured_tool_descriptions=structured_tool_desc,
                retrieval_tool_description=retrieval_desc,
                data_sources_description=data_sources_desc,
                chat_model=chat_invoke,
            )

        if not plan:
            plan = planner.deterministic_plan(message)

        plan = validate_tool_plan(plan, STRUCTURED_TOOL_ALLOWLIST)

        if plan.route == "clarification":
            return ChatResponse(
                property_code=normalized_code,
                model=model,
                answer_markdown=self._with_scope_note(
                    plan.clarification_question
                    or self._clarification_answer(profile, normalized_code, message),
                    scope_note,
                ),
                components=[],
                sources=[],
                tool_results=tool_results,
            )

        if plan.route == "unsupported":
            return ChatResponse(
                property_code=normalized_code,
                model=model,
                answer_markdown=self._with_scope_note(
                    plan.unsupported_reason
                    or self._unsupported_fact_answer(profile, normalized_code, message),
                    scope_note,
                ),
                components=[],
                sources=[],
                tool_results=tool_results,
            )

        if plan.route == "sql_approval":
            if not chat_invoke or not plan.sql_request:
                return ChatResponse(
                    property_code=normalized_code,
                    model=model,
                    answer_markdown=self._with_scope_note(
                        self._sql_approval_unavailable_answer(profile, normalized_code),
                        scope_note,
                    ),
                    components=[],
                    sources=[],
                    tool_results=tool_results,
                )

            drafted = draft_sql_for_approval(
                message=message,
                property_code=normalized_code,
                property_name=profile.get("property_name") or normalized_code,
                sql_request=plan.sql_request,
                chat_model=chat_invoke,
            )
            if not drafted:
                return ChatResponse(
                    property_code=normalized_code,
                    model=model,
                    answer_markdown=self._with_scope_note(
                        self._sql_approval_unavailable_answer(profile, normalized_code),
                        scope_note,
                    ),
                    components=[],
                    sources=[],
                    tool_results=tool_results,
                )

            tool_results["sql_draft"] = drafted.model_dump()

            if drafted.answerable is False:
                unavailable_reason = (
                    drafted.unavailable_reason
                    or "This question is not answerable from the current MySQL schema."
                )
                return ChatResponse(
                    property_code=normalized_code,
                    model=model,
                    answer_markdown=self._with_scope_note(
                        (
                            f"### {profile['property_name']} (`{normalized_code}`)\n\n"
                            f"{unavailable_reason}"
                        ),
                        scope_note,
                    ),
                    components=[],
                    sources=[],
                    tool_results=tool_results,
                )

            is_valid, validation_message = validate_drafted_sql(drafted.sql)
            if not is_valid:
                tool_results["sql_draft_error"] = validation_message
                return ChatResponse(
                    property_code=normalized_code,
                    model=model,
                    answer_markdown=self._with_scope_note(
                        self._sql_approval_unavailable_answer(profile, normalized_code),
                        scope_note,
                    ),
                    components=[],
                    sources=[],
                    tool_results=tool_results,
                )

            component = UIComponent(
                type="sql_approval",
                title="Review SQL before execution",
                description=drafted.explanation,
                data={
                    "question": message,
                    "property_code": normalized_code,
                    "model": model,
                    "sql": drafted.sql,
                    "explanation": drafted.explanation,
                    "parameters": {"property_code": normalized_code},
                    "safety_notes": drafted.safety_notes,
                    "status": "pending_approval",
                    "executable": True,
                },
            )
            tool_results["sql_draft"] = component.data
            return ChatResponse(
                property_code=normalized_code,
                model=model,
                answer_markdown=(
                    f"### {profile['property_name']} (`{normalized_code}`)\n\n"
                    "This question needs a custom rent-roll query. I drafted a "
                    "read-only SQL query for review, but it has not been run."
                ),
                components=[component],
                sources=[],
                tool_results=tool_results,
            )

        wants_structured = plan.route in {"structured", "hybrid"}
        wants_retrieval = plan.route in {"retrieval", "hybrid"}

        requested_years = self._requested_years(message)
        if wants_structured and requested_years:
            report_periods = self._call_tool("get_report_periods", property_code=normalized_code)
            tool_results["report_periods"] = report_periods

            unavailable_years = [
                year for year in requested_years if year not in report_periods.get("years", [])
            ]

            if unavailable_years:
                return ChatResponse(
                    property_code=normalized_code,
                    model=model,
                    answer_markdown=self._with_scope_note(
                        self._unavailable_year_answer(
                            profile=profile,
                            property_code=normalized_code,
                            requested_years=unavailable_years,
                            report_periods=report_periods,
                        ),
                        scope_note,
                    ),
                    components=[],
                    sources=[],
                    tool_results=tool_results,
                )

        if wants_structured:
            unsupported_metric = self._unsupported_structured_aggregate(message, intent)
            if unsupported_metric:
                return ChatResponse(
                    property_code=normalized_code,
                    model=model,
                    answer_markdown=self._with_scope_note(
                        self._unsupported_structured_metric_answer(
                            profile=profile,
                            property_code=normalized_code,
                            unsupported_metric=unsupported_metric,
                        ),
                        scope_note,
                    ),
                    components=[],
                    sources=[],
                    tool_results=tool_results,
                )

            self._collect_structured_from_plan(
                property_code=normalized_code,
                message=message,
                tool_results=tool_results,
                components=components,
                structured_tools=plan.structured_tools,
                intent=intent,
            )

        if wants_retrieval:
            retrieval_results: list[dict[str, Any]] = []

            for query in plan.retrieval_queries:
                results = self._call_tool(
                    "search_property_content",
                    property_code=normalized_code,
                    query=query.query,
                    page_type=query.page_type,
                    n_results=query.n_results,
                )
                retrieval_results.extend(results)

            retrieval_results = self._annotate_retrieval_evidence(message, retrieval_results)

            is_review_query = self._wants_reviews(message.lower())

            if is_review_query:
                review_results = self._filter_review_results(retrieval_results)
                if not review_results:
                    return ChatResponse(
                        property_code=normalized_code,
                        model=model,
                        answer_markdown=self._with_scope_note(
                            self._unsupported_fact_answer(profile, normalized_code, message),
                            scope_note,
                        ),
                        components=[],
                        sources=[],
                        tool_results={"property_profile": profile},
                    )

                tool_results["property_content"] = review_results
                sources.extend(self._sources_from_retrieval(review_results))

            else:
                tool_results["property_content"] = retrieval_results
                sources.extend(self._sources_from_retrieval(retrieval_results))

                if (
                    not wants_structured
                    and self._requires_grounded_retrieval_answer(message, intent)
                    and not self._has_confident_evidence(retrieval_results)
                ):
                    tool_results["property_content"] = []
                    return ChatResponse(
                        property_code=normalized_code,
                        model=model,
                        answer_markdown=self._with_scope_note(
                            self._no_matching_website_evidence_answer(
                                profile,
                                normalized_code,
                            ),
                            scope_note,
                        ),
                        components=[],
                        sources=[],
                        tool_results=tool_results,
                    )

        answer_markdown = self._generate_answer(
            model=model,
            property_code=normalized_code,
            message=message,
            tool_results=tool_results,
            components=components,
            intent=intent,
            on_token=on_token,
        )

        if scope_note:
            answer_markdown = self._remove_scope_boilerplate(answer_markdown)

        if any(component.type == "table" for component in components):
            answer_markdown = self._remove_markdown_tables(answer_markdown)

        return ChatResponse(
            property_code=normalized_code,
            model=model,
            answer_markdown=answer_markdown,
            components=components,
            sources=sources,
            tool_results=tool_results,
        )

    def _collect_structured_from_plan(
        self,
        property_code: str,
        message: str,
        tool_results: dict[str, Any],
        components: list[UIComponent],
        structured_tools: list[StructuredToolCall],
        intent: str | None = None,
    ) -> None:
        text = message.lower()

        for tool_call in structured_tools:
            name = tool_call.name
            args = dict(getattr(tool_call, "args", {}) or {})
            args.pop("property_code", None)

            if name == "get_property_profile":
                tool_results["property_profile"] = self._call_tool(
                    "get_property_profile",
                    property_code=property_code,
                )
                continue

            if name == "get_latest_property_kpis":
                latest_kpis = tool_results.get("latest_kpis")
                if latest_kpis is None:
                    latest_kpis = self._call_tool(
                        "get_latest_property_kpis",
                        property_code=property_code,
                    )
                    tool_results["latest_kpis"] = latest_kpis

                if self._wants_kpi_cards(text, intent):
                    self._add_kpi_components(latest_kpis, components)

                if self._wants_rent_lease_comparison(text, intent):
                    comparison = self._rent_lease_comparison(latest_kpis)
                    if comparison:
                        tool_results["rent_lease_comparison"] = comparison
                        components.append(
                            UIComponent(
                                type="comparison_view",
                                title="Market Rent vs Lease Charges",
                                data=comparison["items"],
                            )
                        )

                continue

            if name == "get_occupancy_trend":
                months = int(args.get("months") or 12)
                months = max(1, min(36, months))

                trend = self._call_tool(
                    "get_occupancy_trend",
                    property_code=property_code,
                    months=months,
                )
                tool_results["occupancy_trend"] = trend
                components.append(
                    UIComponent(
                        type="line_chart",
                        title="Occupancy Trend",
                        data=[
                            {
                                "label": row["report_month"],
                                "value": row["unit_occupancy_pct"],
                                "unit": "%",
                            }
                            for row in trend
                        ],
                    )
                )
                continue

            if name == "get_charge_breakdown":
                limit = int(args.get("limit") or 10)
                limit = max(1, min(50, limit))

                charges = self._call_tool(
                    "get_charge_breakdown",
                    property_code=property_code,
                    limit=limit,
                )
                tool_results["charge_breakdown"] = charges
                components.append(
                    UIComponent(
                        type="bar_chart",
                        title="Charge Breakdown",
                        data=[
                            {
                                "label": row["charge_code"],
                                "value": row["amount"],
                                "unit": "USD",
                            }
                            for row in charges
                        ],
                    )
                )
                continue

            if name == "get_top_balances":
                limit = int(args.get("limit") or 10)
                limit = max(1, min(50, limit))

                balances = self._call_tool(
                    "get_top_balances",
                    property_code=property_code,
                    limit=limit,
                )
                tool_results["top_balances"] = balances
                components.append(
                    UIComponent(
                        type="table",
                        title="Top Balances",
                        data=balances,
                    )
                )
                continue

            if name == "get_vacant_units":
                limit = int(args.get("limit") or 20)
                limit = max(1, min(50, limit))

                vacant_units = self._call_tool(
                    "get_vacant_units",
                    property_code=property_code,
                    limit=limit,
                )
                vacant_units = self._with_bedroom_categories(vacant_units)
                tool_results["vacant_units"] = vacant_units
                components.append(
                    UIComponent(
                        type="table",
                        title="Vacant Units",
                        data=vacant_units,
                    )
                )
                continue

            if name == "get_rent_by_unit_type":
                rent_by_type = self._call_tool(
                    "get_rent_by_unit_type",
                    property_code=property_code,
                )
                tool_results["rent_by_unit_type"] = rent_by_type

                rent_by_bedroom = self._rent_by_bedroom_category(rent_by_type)
                if rent_by_bedroom:
                    tool_results["rent_by_bedroom_category"] = rent_by_bedroom
                    components.append(
                        UIComponent(
                            type="bar_chart",
                            title="Average Market Rent by Bedroom Category",
                            description=(
                                "Each bar shows the average monthly market rent in USD, "
                                "grouped from recognizable rent-roll unit type codes."
                            ),
                            data=[
                                {
                                    "label": row["bedroom_category"],
                                    "value": row["avg_market_rent"],
                                    "unit": "USD",
                                    "unit_count": row["unit_count"],
                                }
                                for row in rent_by_bedroom
                            ],
                        )
                    )

                components.append(
                    UIComponent(
                        type="bar_chart",
                        title="Average Market Rent by Floorplan Code",
                        description=(
                            "Each bar shows the average monthly market rent in USD for a "
                            "specific rent-roll unit type or floorplan code."
                        ),
                        data=[
                            {
                                "label": row["unit_type"],
                                "value": row["avg_market_rent"],
                                "unit": "USD",
                                "unit_count": row["unit_count"],
                            }
                            for row in rent_by_type
                        ],
                    )
                )
                continue

            if name == "get_report_periods":
                if "report_periods" not in tool_results:
                    tool_results["report_periods"] = self._call_tool(
                        "get_report_periods",
                        property_code=property_code,
                    )
                continue

    def _generate_answer(
        self,
        model: str,
        property_code: str,
        message: str,
        tool_results: dict[str, Any],
        components: list[UIComponent],
        intent: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        provider, _, model_name = model.partition(":")
        if (
            provider == "mock"
            or not model_name
            or self._should_use_deterministic_answer(tool_results, components)
        ):
            return self._mock_answer(
                property_code,
                message,
                tool_results,
                components,
                intent=intent,
            )

        system_prompt = (
            "You are a property-specific AI assistant for rent-roll and property "
            "website questions.\n\n"
            "Core rules:\n"
            "- Answer only for the active property_code supplied by the backend.\n"
            "- Never switch property scope based on a property name in the user message.\n"
            "- Use only supplied tool results and retrieved evidence; do not invent data.\n"
            "- If the supplied data does not answer the question, say it is not available.\n"
            "- Do not generate, rewrite, or claim to execute SQL. If a requested metric "
            "would require a new SQL query, say that metric is not available yet unless "
            "the backend has supplied approved results for it.\n"
            "- If the user mentions a different property, the backend already handles "
            "scope internally. Do not repeat the mismatch or ask for another property code.\n\n"
            "Answer style:\n"
            "- Return concise Markdown.\n"
            "- For yes/no questions, answer yes or no first.\n"
            "- When using website context, cite the source URL in Markdown.\n"
            "- If UI components are provided, summarize the main takeaway only. Do not "
            "recreate the same table, chart, KPI cards, or comparison view in Markdown.\n"
            "- Do not mention UI components, frontend, JSON, rendering, or where a "
            "table/chart appears.\n"
            "- Never output Markdown tables or pipe-delimited table rows.\n\n"
            "Evidence rules:\n"
            "- For website facts, only make claims supported by retrieved chunks whose "
            "evidence.confidence is medium or high.\n"
            "- If website evidence is low-confidence or missing, say you do not see "
            "matching website evidence in the selected property sample.\n"
            "- Do not dump raw website chunks, navigation labels, fee-guide boilerplate, "
            "or disclaimer text; summarize the user-facing facts."
        )
        prompt_tool_results = self._tool_results_for_prompt(message, tool_results, components)
        prompt_components = self._components_for_prompt(components)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"Active property_code: {property_code}\n"
                        f"Routed intent: {intent or 'unspecified'}\n"
                        f"User question: {message}\n\n"
                        f"Tool results JSON:\n"
                        f"{json.dumps(prompt_tool_results, ensure_ascii=True)}\n\n"
                        f"UI components JSON:\n"
                        f"{json.dumps(prompt_components, ensure_ascii=True)}"
                    )
                ),
            ]
        )
        chat_model = self._chat_model(provider, model_name)
        should_stream = on_token and not any(component.type == "table" for component in components)
        if should_stream:
            chunks: list[str] = []
            for chunk in (prompt | chat_model).stream({}):
                content = str(chunk.content)
                if not content:
                    continue
                chunks.append(content)
                on_token(content)
            return "".join(chunks)
        response = (prompt | chat_model).invoke({})
        return str(response.content)

    @staticmethod
    def _should_use_deterministic_answer(
        tool_results: dict[str, Any],
        components: list[UIComponent],
    ) -> bool:
        if components:
            return True
        deterministic_keys = {
            "latest_kpis",
            "rent_lease_comparison",
            "occupancy_trend",
            "charge_breakdown",
            "top_balances",
            "vacant_units",
            "rent_by_unit_type",
            "rent_by_bedroom_category",
            "property_content",
        }
        return any(tool_results.get(key) for key in deterministic_keys)

    def _tool_results_for_prompt(
        self,
        message: str,
        tool_results: dict[str, Any],
        components: list[UIComponent],
    ) -> dict[str, Any]:
        prompt_results = dict(tool_results)
        table_titles = {
            component.title
            for component in components
            if component.type == "table"
        }
        if "Top Balances" in table_titles and tool_results.get("top_balances"):
            rows = tool_results["top_balances"]
            prompt_results["top_balances"] = {
                "ui_component": "Top Balances table",
                "row_count": len(rows),
                "summary": self._top_balances_summary(rows),
                "note": (
                    "Do not recreate the table in Markdown. Do not mention the UI, "
                    "frontend, component, or where the table appears."
                ),
            }
        if "Vacant Units" in table_titles and tool_results.get("vacant_units"):
            rows = tool_results["vacant_units"]
            prompt_results["vacant_units"] = {
                "ui_component": "Vacant Units table",
                "row_count": len(rows),
                "summary": self._vacant_units_summary(rows),
                "note": (
                    "Do not recreate the table in Markdown. Do not mention the UI, "
                    "frontend, component, or where the table appears."
                ),
            }
        retrieval_results = tool_results.get("property_content") or []
        if retrieval_results:
            text = message.lower()
            if self._wants_floorplan_answer(text):
                prompt_results["website_answer_hint"] = self._floorplan_answer(retrieval_results)
            elif self._wants_amenity_list(text):
                prompt_results["website_answer_hint"] = self._amenity_list_answer(
                    retrieval_results,
                    section_filter=self._amenity_section_filter(text),
                )
            prompt_results["property_content"] = [
                {
                    **result,
                    "content": "\n".join(
                        self._clean_website_lines(
                            result["content"],
                            section=result["metadata"].get("section_heading"),
                        )
                    ),
                }
                for result in retrieval_results
            ]
        return prompt_results

    def _chat_model(self, provider: str, model_name: str):
        if provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI models.")
            return ChatOpenAI(
                model=model_name,
                api_key=self.settings.openai_api_key,
                max_tokens=900,
                temperature=0.1,
            )
        if provider == "anthropic":
            if not self.settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models.")
            return ChatAnthropic(
                model=model_name,
                api_key=self.settings.anthropic_api_key,
                max_tokens=900,
                temperature=0.1,
            )
        raise ValueError(f"Unsupported model provider: {provider}")

    @staticmethod
    def _components_for_prompt(components: list[UIComponent]) -> list[dict[str, Any]]:
        prompt_components: list[dict[str, Any]] = []
        for component in components:
            data = component.data or []
            summary: dict[str, Any] = {
                "type": component.type,
                "title": component.title,
            }
            if component.description:
                summary["description"] = component.description
            if component.type == "table":
                summary["row_count"] = len(data)
                summary["columns"] = list(data[0].keys()) if data else []
            elif component.type in {"bar_chart", "line_chart", "comparison_view"}:
                summary["data_point_count"] = len(data)
            elif component.type == "kpi_card":
                summary["metric_count"] = len(data)
            prompt_components.append(summary)
        return prompt_components

    @staticmethod
    def _is_unsupported_fact_question(message: str) -> bool:
        lowered = message.lower()
        return any(term in lowered for term in UNSUPPORTED_FACT_TERMS)

    @staticmethod
    def _unsupported_fact_answer(profile: dict, property_code: str, message: str) -> str:
        return (
            f"### {profile['property_name']} (`{property_code}`)\n\n"
            "I don't have that data for this property. The available sources are "
            "rent-roll metrics and the scraped property website content, and this question "
            "is outside those sources."
        )

    @staticmethod
    def _sql_approval_unavailable_answer(profile: dict, property_code: str) -> str:
        return (
            f"### {profile['property_name']} (`{property_code}`)\n\n"
            "I can't prepare a safe custom query for that request. I can still help with "
            "supported views like latest occupancy, occupancy trend, charge breakdown, "
            "top balances, vacant units, average market rent by unit type, and market "
            "rent vs lease charges."
        )

    def _mock_answer(
        self,
        property_code: str,
        message: str,
        tool_results: dict[str, Any],
        components: list[UIComponent],
        intent: str | None = None,
    ) -> str:
        profile = tool_results["property_profile"]
        lines = [f"### {profile['property_name']} (`{property_code}`)"]
        text = message.lower()

        current = tool_results.get("latest_kpis", {}).get("current")
        vacant = tool_results.get("latest_kpis", {}).get("vacant")
        if current and self._wants_executive_summary(text, intent):
            lines.append(self._executive_summary(current, vacant))
        elif current and self._should_include_snapshot_summary(text, tool_results, intent):
            lines.append(self._snapshot_summary(current, vacant))

        if tool_results.get("rent_lease_comparison"):
            lines.append(
                self._rent_lease_comparison_summary(tool_results["rent_lease_comparison"])
            )

        if tool_results.get("occupancy_trend"):
            lines.append(self._occupancy_trend_summary(tool_results["occupancy_trend"]))

        if tool_results.get("charge_breakdown"):
            lines.append(self._charge_breakdown_summary(tool_results["charge_breakdown"]))

        if tool_results.get("top_balances"):
            lines.append(self._top_balances_summary(tool_results["top_balances"]))

        if tool_results.get("vacant_units"):
            lines.append(self._vacant_units_summary(tool_results["vacant_units"]))

        if tool_results.get("rent_by_unit_type"):
            lines.append(self._rent_by_unit_type_summary(tool_results["rent_by_unit_type"]))

        if self._wants_location_answer(text):
            lines.append(self._location_answer(profile, "property_content" in tool_results))
            if not self._wants_location_website_context(text):
                return "\n\n".join(lines)

        retrieval_results = tool_results.get("property_content") or []
        if retrieval_results:
            if self._wants_amenity_list(text, intent):
                lines.append(
                    self._amenity_list_answer(
                        retrieval_results,
                        section_filter=self._amenity_section_filter(text),
                    )
                )
                return "\n\n".join(lines)
            if self._wants_floorplan_answer(text, intent):
                lines.append(self._floorplan_answer(retrieval_results))
                return "\n\n".join(lines)

            matched_terms: set[str] = set()
            matched_evidence: list[dict[str, str]] = []
            fallback_evidence: list[dict[str, str]] = []
            for result in retrieval_results[:3]:
                metadata = result["metadata"]
                matched_line_details = self._matching_content_line_details(
                    message,
                    result["content"],
                )
                if matched_line_details:
                    strong_matches = [
                        detail for detail in matched_line_details if detail["score"] >= 2
                    ]
                    if strong_matches:
                        matched_line_details = strong_matches
                matched_lines = [detail["line"] for detail in matched_line_details]
                snippet = (
                    "; ".join(matched_lines)
                    if matched_lines
                    else self._content_preview(result["content"])
                )
                if not snippet:
                    continue
                section = metadata.get("section_heading") or metadata.get("page_type")
                item = {
                    "section": self._display_section_label(section),
                    "snippet": snippet,
                }
                if matched_lines:
                    for detail in matched_line_details:
                        matched_terms.update(detail["terms"])
                    matched_evidence.append(item)
                else:
                    fallback_evidence.append(item)

            evidence = matched_evidence or fallback_evidence
            if matched_evidence:
                lines.append(self._natural_retrieval_fact_answer(message, evidence))
            elif self._is_yes_no_question(message):
                lines.append(
                    "I don't see matching website evidence for that in the selected "
                    "property sample."
                )
                return "\n\n".join(lines)
            else:
                lines.append(self._website_evidence_answer(evidence))

        if len(lines) == 1:
            lines.append(f"I found the property, but no matching data for: {message}")
        return "\n\n".join(lines)

    def _unavailable_year_answer(
        self,
        profile: dict,
        property_code: str,
        requested_years: list[int],
        report_periods: dict,
    ) -> str:
        requested = ", ".join(str(year) for year in requested_years)
        available_years = report_periods.get("years") or []
        min_month = report_periods.get("min_report_month")
        max_month = report_periods.get("max_report_month")

        if min_month and max_month:
            available = f"**{min_month}** through **{max_month}**"
        elif available_years:
            available = ", ".join(f"**{year}**" for year in available_years)
        else:
            available = "no loaded rent-roll periods"

        return (
            f"### {profile['property_name']} (`{property_code}`)\n\n"
            f"I don't have rent-roll data for **{requested}** for this property, "
            "so I can't provide that year-specific breakdown.\n\n"
            f"I only have rent-roll data for {available}."
        )

    @staticmethod
    def _no_matching_website_evidence_answer(profile: dict, property_code: str) -> str:
        return (
            f"### {profile['property_name']} (`{property_code}`)\n\n"
            "I don't see matching website evidence for that in the selected "
            "property sample."
        )

    def _clarification_answer(self, profile: dict, property_code: str, message: str) -> str:
        text = message.strip().lower()
        if any(term in text for term in ["charge", "charges", "fee", "fees", "lease"]):
            return (
                f"### {profile['property_name']} (`{property_code}`)\n\n"
                "What charge view would you like?\n\n"
                "- Lease charge total for the latest month\n"
                "- Biggest charge categories\n"
                "- Charge breakdown chart\n"
                "- Rent vs lease charges comparison"
            )
        return (
            f"### {profile['property_name']} (`{property_code}`)\n\n"
            "What would you like to look at for this property?\n\n"
            "- Rent-roll KPIs: occupancy, market rent, lease charges, vacancies\n"
            "- Financial detail: charge breakdowns, top balances, rent by unit type\n"
            "- Website details: amenities, EV charging, bike storage, floorplans, sources"
        )

    def _property_scope_note(self, message: str, active_profile: dict) -> str | None:
        mentioned = self._mentioned_other_property(message, active_profile)
        if not mentioned:
            return None

        active_name = active_profile["property_name"]
        active_code = active_profile["property_code"]
        mentioned_name = mentioned["property_name"]
        return (
            f"> **Scope note:** You mentioned **{mentioned_name}**, but the active "
            f"property is **{active_name} (`{active_code}`)**. I’ll answer using only "
            f"{active_name} data."
        )

    def _mentioned_other_property(
        self,
        message: str,
        active_profile: dict,
    ) -> dict | None:
        message_text = message.lower()
        normalized_message = self._normalize_property_phrase(message)
        active_code = str(active_profile["property_code"]).lower()
        active_name = self._normalize_property_phrase(active_profile["property_name"])
        properties = self._call_tool("list_properties")

        matches = []
        for property_profile in properties:
            candidate_code = str(property_profile.get("property_code") or "").lower()
            candidate_name = str(property_profile.get("property_name") or "").strip()
            normalized_name = self._normalize_property_phrase(candidate_name)
            if not candidate_code or not candidate_name:
                continue
            if candidate_code == active_code or normalized_name == active_name:
                continue

            code_matches = bool(
                re.search(rf"\b{re.escape(candidate_code)}\b", message_text)
            )
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

    @staticmethod
    def _normalize_property_phrase(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @staticmethod
    def _with_scope_note(answer_markdown: str, scope_note: str | None) -> str:
        return answer_markdown

    @staticmethod
    def _remove_scope_boilerplate(answer_markdown: str) -> str:
        filtered_lines = []
        skip_phrases = [
            "i notice there's a mismatch",
            "there's a mismatch",
            "you asked about",
            "if you meant to ask about",
            "please provide its property code",
            "active property code",
        ]
        for line in answer_markdown.splitlines():
            lowered = line.strip().lower()
            if any(phrase in lowered for phrase in skip_phrases):
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines).strip()

    @staticmethod
    def _remove_markdown_tables(answer_markdown: str) -> str:
        filtered_lines: list[str] = []
        in_table = False
        for line in answer_markdown.splitlines():
            stripped = line.strip()
            pipe_count = stripped.count("|")
            looks_like_separator = bool(re.fullmatch(r"[\s|:=-]+", stripped))
            looks_like_table = pipe_count >= 2 or looks_like_separator
            if looks_like_table:
                in_table = True
                continue
            if in_table and not stripped:
                in_table = False
                continue
            table_row_pattern = r"^[-+*]?\s*(unit|a\d+|b\d+|\+\s*\d+\s+more)\b"
            if in_table and re.match(table_row_pattern, stripped, re.IGNORECASE):
                continue
            in_table = False
            filtered_lines.append(line)

        compacted = "\n".join(filtered_lines)
        compacted = re.sub(r"\n{3,}", "\n\n", compacted)
        return compacted.strip()

    @staticmethod
    def _display_section_label(section: str | None) -> str:
        if not section:
            return "Website Evidence"
        if section.islower() or "_" in section:
            return section.replace("_", " ").title()
        return section

    @staticmethod
    def _location_answer(profile: dict, attempted_website_lookup: bool) -> str:
        answer = f"The property address is **{profile.get('address', 'not available')}**."
        source_site = profile.get("source_site")
        if attempted_website_lookup:
            answer += " I don't have a clean neighborhood summary in the scraped website sample."
        if source_site:
            answer += f"\n\nSource: [{profile['property_name']} website]({source_site})"
        return answer

    def _amenity_list_answer(
        self,
        retrieval_results: list[dict],
        section_filter: str | None = None,
    ) -> str:
        amenities_by_section: dict[str, list[str]] = {}
        source_title = None
        source_url = None

        for result in retrieval_results:
            metadata = result["metadata"]
            if metadata.get("page_type") != "amenities":
                continue
            section = metadata.get("section_heading") or "Amenities"
            if section_filter and section != section_filter:
                continue
            items = self._amenity_items_from_content(result["content"], section)
            if not items:
                continue
            amenities_by_section.setdefault(section, [])
            for item in items:
                if item not in amenities_by_section[section]:
                    amenities_by_section[section].append(item)
            source_title = source_title or metadata.get("title")
            source_url = source_url or metadata.get("source_url")

        if not amenities_by_section:
            return "I found the amenities page, but could not extract a clean amenity list."

        lines = ["The property website lists these amenities:"]
        preferred_sections = ["Community Features", "Apartment Features"]
        ordered_sections = [
            *[section for section in preferred_sections if section in amenities_by_section],
            *[
                section
                for section in amenities_by_section
                if section not in preferred_sections
            ],
        ]
        for section in ordered_sections:
            item_limit = 18 if section == "Community Features" else 12
            items = amenities_by_section[section][:item_limit]
            lines.append(f"\n**{section}**")
            lines.extend(f"- {item}" for item in items)
            if len(amenities_by_section[section]) > len(items):
                lines.append(f"- Plus {len(amenities_by_section[section]) - len(items)} more")

        return "\n".join(lines)

    def _amenity_items_from_content(self, content: str, section: str) -> list[str]:
        return [
            line
            for line in self._clean_website_lines(content, section=section)
            if line.lower() != section.lower()
        ]

    def _floorplan_answer(self, retrieval_results: list[dict]) -> str:
        floorplan_results = [
            result
            for result in retrieval_results
            if result["metadata"].get("page_type") == "floorplans"
        ] or retrieval_results
        floorplan_types: list[str] = []
        floorplan_details: list[tuple[str, str]] = []
        source_title = None
        source_url = None

        for result in floorplan_results:
            metadata = result["metadata"]
            main_floorplans_url = self._main_floorplans_url(metadata.get("source_url"))
            source_url = source_url or main_floorplans_url
            if self._is_floorplans_listing_url(metadata.get("source_url")):
                source_title = source_title or metadata.get("title")
            elif not source_title:
                source_title = f"{metadata.get('property_name', 'Property')} Floorplans"
            detail = self._floorplan_detail_from_result(result)
            if detail and detail not in floorplan_details:
                floorplan_details.append(detail)
            for label in self._floorplan_types_from_text(metadata.get("title") or ""):
                if label not in floorplan_types:
                    floorplan_types.append(label)
            for line in self._clean_website_lines(
                result["content"],
                section=metadata.get("section_heading"),
            ):
                for label in self._floorplan_types_from_text(line):
                    if label not in floorplan_types:
                        floorplan_types.append(label)

        if floorplan_types:
            lines = ["The website advertises these floorplan categories:"]
            lines.extend(f"- {label}" for label in floorplan_types)
        else:
            lines = [
                "I found the floorplans page, but it did not expose a clean list of "
                "floorplan categories or plan names."
            ]

        if floorplan_details:
            lines.append("\nI also found these individual floorplan pages:")
            for name, detail_text in floorplan_details[:10]:
                suffix = f": {detail_text}" if detail_text else ""
                lines.append(f"- **{name}**{suffix}")
            if len(floorplan_details) > 10:
                lines.append(f"- Plus {len(floorplan_details) - 10} more")
        elif floorplan_types:
            lines.append(
                "\nI don't see individual floorplan names in the available page text."
            )

        return "\n".join(lines)

    def _floorplan_detail_from_result(self, result: dict) -> tuple[str, str] | None:
        metadata = result["metadata"]
        source_url = metadata.get("source_url") or ""
        path = urlparse(source_url).path.rstrip("/")
        if not re.search(r"/floorplans/[^/]+$", path):
            return None

        section = metadata.get("section_heading") or ""
        title = metadata.get("title") or ""
        name = section if section and section.lower() != "page overview" else title.split("|")[0]
        name = name.strip()
        if not name or len(name) > 40:
            return None
        if re.search(r"follow|instagram|overview|virtual tour", name, re.IGNORECASE):
            return None
        if "brochure" in name.lower():
            return None

        detail_lines = []
        for line in self._clean_website_lines(result["content"], section=section):
            if line.lower() == name.lower():
                continue
            if re.search(r"\b\d+\s+beds?\b|\b\d+\s+baths?\b|sq\.\s*ft\.|/mo|den", line, re.I):
                detail_lines.append(line)
            if len(detail_lines) >= 4:
                break

        if not detail_lines:
            return None
        return name, ", ".join(detail_lines)

    @staticmethod
    def _main_floorplans_url(source_url: str | None) -> str | None:
        if not source_url:
            return None
        parsed = urlparse(source_url)
        path_parts = [part for part in parsed.path.split("/") if part]
        if "floorplans" not in path_parts:
            return source_url
        floorplan_index = path_parts.index("floorplans")
        main_path = "/" + "/".join(path_parts[: floorplan_index + 1]) + "/"
        return urlunparse((parsed.scheme, parsed.netloc, main_path, "", "", ""))

    @staticmethod
    def _is_floorplans_listing_url(source_url: str | None) -> bool:
        if not source_url:
            return False
        parsed = urlparse(source_url)
        return parsed.path.rstrip("/").endswith("/floorplans")

    @staticmethod
    def _floorplan_types_from_text(text: str) -> list[str]:
        labels: list[str] = []
        for match in re.finditer(
            r"((?:studio|\b[1-5]\b)(?:\s*,\s*(?:or\s+)?(?:studio|\b[1-5]\b))*"
            r"\s*(?:,\s*)?(?:or\s+)?(?:studio|\b[1-5]\b))\s+"
            r"(?:bed(?:room)?s?)\b",
            text,
            re.IGNORECASE,
        ):
            for token in re.findall(r"studio|[1-5]", match.group(1), re.IGNORECASE):
                label = "Studio" if token.lower() == "studio" else f"{int(token)}-bedroom"
                if label not in labels:
                    labels.append(label)
        for match in re.finditer(r"\b([1-5])\s+(?:bed(?:room)?s?)\b", text, re.IGNORECASE):
            label = f"{int(match.group(1))}-bedroom"
            if label not in labels:
                labels.append(label)
        return labels

    @staticmethod
    def _should_include_snapshot_summary(
        text: str,
        tool_results: dict[str, Any],
        intent: str | None = None,
    ) -> bool:
        if LangChainOrchestrator._wants_occupancy_trend(text, intent):
            return False
        if LangChainOrchestrator._wants_rent_lease_comparison(text, intent):
            return False
        explicit_kpi_request = (
            "latest occupancy" in text
            or "latest kpi" in text
            or "snapshot" in text
            or ("occupancy" in text and "trend" not in text)
        )
        if intent in {
            "charge_breakdown",
            "executive_summary",
            "rent_by_unit_type",
            "top_balances",
            "vacant_units",
        }:
            return intent == "executive_summary" or explicit_kpi_request
        if explicit_kpi_request:
            return True
        if "market rent" in text and "unit type" not in text and "average" not in text:
            return True
        if "lease charge" in text and "breakdown" not in text and "categor" not in text:
            return True
        if "vacant" in text and "vacant_units" not in tool_results:
            return True
        detail_keys = [
            "occupancy_trend",
            "charge_breakdown",
            "top_balances",
            "vacant_units",
            "rent_by_unit_type",
            "property_content",
        ]
        return not any(tool_results.get(key) for key in detail_keys)

    @staticmethod
    def _wants_charge_breakdown(text: str, intent: str | None = None) -> bool:
        if intent == "charge_breakdown":
            return True
        if "lease charge" in text and not any(
            term in text for term in ["breakdown", "categor", "biggest", "largest", "top"]
        ):
            return False
        return any(
            term in text
            for term in ["charge breakdown", "charge categor", "biggest charge", "largest charge"]
        ) or ("fee" in text or "revenue" in text or "breakdown" in text)

    @staticmethod
    def _wants_vacant_unit_detail(text: str, intent: str | None = None) -> bool:
        if intent == "vacant_units":
            return True
        if "vacant unit count" in text or "vacant count" in text:
            return False
        if "vacant" in text and "unit" in text and any(
            term in text
            for term in [
                "all",
                "detail",
                "give me",
                "list",
                "only",
                "show",
                "their",
                "those",
                "type",
                "which",
            ]
        ):
            return True
        return any(
            phrase in text
            for phrase in [
                "which units are vacant",
                "which vacant units",
                "show vacant units",
                "list vacant units",
                "vacant unit detail",
                "vacant units include",
            ]
        )

    @staticmethod
    def _wants_occupancy_trend(text: str, intent: str | None = None) -> bool:
        if intent == "occupancy_trend":
            return True
        if any(
            phrase in text
            for phrase in [
                "occupancy trend",
                "trend over time",
                "over time",
                "monthly occupancy",
                "occupancy history",
            ]
        ):
            return True
        return "occupancy" in text and any(
            term in text
            for term in ["available months", "across months", "changed", "change", "month-to-month"]
        )

    @staticmethod
    def _wants_rent_lease_comparison(text: str, intent: str | None = None) -> bool:
        if intent == "rent_lease_comparison":
            return True
        has_comparison_intent = any(
            term in text
            for term in ["against", "compare", "comparison", " vs ", " versus "]
        )
        return (
            has_comparison_intent
            and ("rent" in text or "market rent" in text)
            and ("lease charge" in text or "lease charges" in text)
        )

    @staticmethod
    def _wants_executive_summary(text: str, intent: str | None = None) -> bool:
        return (
            intent == "executive_summary"
            or "executive summary" in text
            or "quick summary" in text
        )

    @staticmethod
    def _wants_rent_by_unit_type(text: str, intent: str | None = None) -> bool:
        if LangChainOrchestrator._wants_vacant_unit_detail(text, intent):
            return False
        return (
            intent == "rent_by_unit_type"
            or "unit type" in text
            or "rent by" in text
            or "average rent" in text
        )

    @staticmethod
    def _unsupported_structured_aggregate(
        message: str,
        intent: str | None = None,
    ) -> str | None:
        text = message.lower()

        for rule in UNSUPPORTED_STRUCTURED_AGGREGATE_RULES:
            if LangChainOrchestrator._contains_any(text, rule["allowed_terms"]):
                continue
            has_metric = LangChainOrchestrator._contains_any(text, rule["metric_terms"])
            has_aggregate = LangChainOrchestrator._contains_any(text, rule["aggregate_terms"])
            grouping_terms = rule["grouping_terms"]
            has_grouping = (
                not grouping_terms
                or LangChainOrchestrator._contains_any(text, grouping_terms)
            )
            if has_metric and has_aggregate and has_grouping:
                return str(rule["label"])

        if LangChainOrchestrator._supported_structured_capability(text, intent):
            return None

        return None

    @staticmethod
    def _supported_structured_capability(text: str, intent: str | None = None) -> str | None:
        if LangChainOrchestrator._wants_executive_summary(text, intent):
            return "latest_kpis"
        if LangChainOrchestrator._wants_kpi_cards(text, intent):
            return "latest_kpis"
        if LangChainOrchestrator._wants_rent_by_unit_type(text, intent):
            if (
                ("market rent" in text or "average rent" in text or "avg rent" in text)
                and "lease charge" not in text
                and "lease charges" not in text
            ):
                return "rent_by_unit_type"
            return None
        if LangChainOrchestrator._wants_charge_breakdown(text, intent):
            return "charge_breakdown"
        if LangChainOrchestrator._wants_occupancy_trend(text, intent):
            return "occupancy_trend"
        if LangChainOrchestrator._wants_rent_lease_comparison(text, intent):
            return "rent_lease_comparison"
        if LangChainOrchestrator._wants_vacant_unit_detail(text, intent):
            return "vacant_units"
        if any(
            phrase in text
            for phrase in [
                "latest occupancy",
                "latest kpi",
                "latest market rent",
                "latest lease charge",
                "latest vacant",
                "current occupancy",
                "current market rent",
                "current lease charge",
                "current vacant",
            ]
        ):
            return "latest_kpis"
        if any(term in text for term in ["top balance", "top balances", "highest balance"]):
            return "top_balances"

        return None

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _wants_amenity_list(text: str, intent: str | None = None) -> bool:
        return (
            intent == "amenity_list"
            or (
                ("amenit" in text or "feature" in text)
                and any(term in text for term in ["what", "list", "listed", "show"])
                and not any(
                    term in text for term in ["ev", "charging", "bike", "parking", "pet"]
                )
            )
        )

    @staticmethod
    def _amenity_section_filter(text: str) -> str | None:
        if "apartment feature" in text:
            return "Apartment Features"
        if "community feature" in text:
            return "Community Features"
        return None

    @staticmethod
    def _wants_floorplan_answer(text: str, intent: str | None = None) -> bool:
        return (
            intent == "floorplans"
            or (
                ("floorplan" in text or "floor plan" in text or "bedroom" in text)
                and any(
                    term in text
                    for term in ["advertised", "available", "list", "show", "what"]
                )
            )
        )

    @staticmethod
    def _wants_kpi_cards(text: str, intent: str | None = None) -> bool:
        if LangChainOrchestrator._wants_occupancy_trend(text, intent):
            return False
        if LangChainOrchestrator._wants_rent_lease_comparison(text, intent):
            return False
        if LangChainOrchestrator._wants_vacant_unit_detail(text, intent):
            return False
        explicit_kpi_request = (
            "latest occupancy" in text
            or "latest kpi" in text
            or ("occupancy" in text and "trend" not in text)
        )
        if explicit_kpi_request:
            return True
        if intent in {"charge_breakdown", "rent_by_unit_type", "top_balances"}:
            return False
        if intent in {"latest_kpis", "executive_summary"}:
            return True
        if LangChainOrchestrator._wants_rent_by_unit_type(text, intent):
            return False
        if "balance" in text:
            return False
        if any(term in text for term in ["breakdown", "categor", "biggest charge"]):
            return False
        return any(
            term in text
            for term in [
                "latest",
                "occupancy",
                "market rent",
                "lease charge",
                "vacant",
                "kpi",
                "summary",
            ]
        )

    @staticmethod
    def _requested_years(message: str) -> list[int]:
        return list(dict.fromkeys(int(match.group(1)) for match in YEAR_RE.finditer(message)))

    def _snapshot_summary(self, current: dict, vacant: dict | None) -> str:
        summary = (
            f"As of **{current['report_month']}**, occupancy is "
            f"**{self._format_percent(current['unit_occupancy_pct'])}** across "
            f"**{current['unit_count']} units**. Market rent is "
            f"**{self._format_money(current['market_rent'])}** and lease charges are "
            f"**{self._format_money(current['lease_charges'])}**."
        )
        if vacant:
            summary += f" The latest summary shows **{vacant['unit_count']} vacant units**."
        return summary

    def _executive_summary(self, current: dict, vacant: dict | None) -> str:
        market_rent = float(current["market_rent"])
        lease_charges = float(current["lease_charges"])
        difference = lease_charges - market_rent
        percent_difference = (difference / market_rent * 100) if market_rent else 0.0
        vacancy_text = (
            f"with **{vacant['unit_count']} vacant units**"
            if vacant
            else "with vacancy detail unavailable"
        )
        return (
            f"Quick executive summary: as of **{current['report_month']}**, this property "
            f"is highly occupied at **{self._format_percent(current['unit_occupancy_pct'])}** "
            f"across **{current['unit_count']} units**, {vacancy_text}. Market rent is "
            f"**{self._format_money(market_rent)}** and lease charges are "
            f"**{self._format_money(lease_charges)}**, putting lease charges "
            f"**{self._format_money(abs(difference))}** "
            f"{'above' if difference >= 0 else 'below'} market rent "
            f"(**{abs(percent_difference):.1f}%** difference). Overall, the latest "
            f"snapshot suggests a stabilized property with strong occupancy and meaningful "
            f"monthly lease-charge volume."
        )

    def _rent_lease_comparison(self, latest_kpis: dict) -> dict | None:
        current = latest_kpis.get("current") if latest_kpis else None
        if not current:
            return None
        market_rent = float(current["market_rent"])
        lease_charges = float(current["lease_charges"])
        difference = lease_charges - market_rent
        percent_difference = (difference / market_rent * 100) if market_rent else 0.0
        return {
            "report_month": current["report_month"],
            "market_rent": market_rent,
            "lease_charges": lease_charges,
            "difference": difference,
            "percent_difference": percent_difference,
            "items": [
                {
                    "metric": "Market Rent",
                    "value": market_rent,
                    "unit": "USD",
                    "report_month": current["report_month"],
                },
                {
                    "metric": "Lease Charges",
                    "value": lease_charges,
                    "unit": "USD",
                    "report_month": current["report_month"],
                },
                {
                    "metric": "Difference",
                    "value": difference,
                    "unit": "USD",
                    "report_month": current["report_month"],
                },
            ],
        }

    def _rent_lease_comparison_summary(self, comparison: dict) -> str:
        difference = comparison["difference"]
        direction = "higher than" if difference >= 0 else "lower than"
        return (
            f"For **{comparison['report_month']}**, market rent is "
            f"**{self._format_money(comparison['market_rent'])}** and lease charges are "
            f"**{self._format_money(comparison['lease_charges'])}**. Lease charges are "
            f"**{self._format_money(abs(difference))}** {direction} market rent "
            f"(**{abs(comparison['percent_difference']):.1f}%** difference)."
        )

    @staticmethod
    def _unsupported_structured_metric_answer(
        profile: dict,
        property_code: str,
        unsupported_metric: str,
    ) -> str:
        supported_examples = [
            capability["supports"][0]
            for capability in SUPPORTED_STRUCTURED_CAPABILITIES.values()
        ]
        supported_text = ", ".join(supported_examples[:-1])
        supported_text = f"{supported_text}, and {supported_examples[-1]}"
        return (
            f"### {profile['property_name']} (`{property_code}`)\n\n"
            f"I can't calculate **{unsupported_metric}** reliably from the available "
            "analytics yet.\n\n"
            f"I can help with supported views like {supported_text}."
        )

    def _occupancy_trend_summary(self, rows: list[dict]) -> str:
        if not rows:
            return "I could not find occupancy trend data for this property."

        first = rows[0]
        last = rows[-1]
        lowest = min(rows, key=lambda row: row["unit_occupancy_pct"])
        highest = max(rows, key=lambda row: row["unit_occupancy_pct"])
        change = last["unit_occupancy_pct"] - first["unit_occupancy_pct"]
        direction = "up" if change > 0 else "down" if change < 0 else "flat"

        return (
            f"Occupancy stayed high over the **{len(rows)}-month trend window**, moving "
            f"from **{self._format_percent(first['unit_occupancy_pct'])}** in "
            f"**{first['report_month']}** to "
            f"**{self._format_percent(last['unit_occupancy_pct'])}** "
            f"in **{last['report_month']}** ({direction} "
            f"**{abs(change):.2f} percentage points**). The low point was "
            f"**{self._format_percent(lowest['unit_occupancy_pct'])}** in "
            f"**{lowest['report_month']}**, and the high point was "
            f"**{self._format_percent(highest['unit_occupancy_pct'])}** in "
            f"**{highest['report_month']}**."
        )

    def _charge_breakdown_summary(self, rows: list[dict]) -> str:
        if not rows:
            return "I could not find charge breakdown data for this property."

        report_month = rows[0].get("report_month")
        top_rows = rows[:3]
        top_text = ", ".join(
            f"**{row['charge_code']}** ({self._format_money(row['amount'])})"
            for row in top_rows
        )
        total = sum(row["amount"] for row in rows)
        return (
            f"For **{report_month}**, the largest charge categories are {top_text}. "
            f"The displayed top categories sum to **{self._format_money(total)}**."
        )

    def _top_balances_summary(self, rows: list[dict]) -> str:
        if not rows:
            return "I could not find balance data for this property."

        report_month = rows[0].get("report_month")
        top_rows = rows[:3]
        top_text = ", ".join(
            f"**unit {row['unit']}** at {self._format_money(row['balance'])}"
            for row in top_rows
        )
        return f"As of **{report_month}**, the highest resident balances are {top_text}."

    def _vacant_units_summary(self, rows: list[dict]) -> str:
        if not rows:
            return "I could not find vacant-unit detail for this property."

        report_month = rows[0].get("report_month")
        units = ", ".join(
            self._vacant_unit_label(row) for row in rows[:8]
        )
        extra = "" if len(rows) <= 8 else f", plus {len(rows) - 8} more"
        return (
            f"As of **{report_month}**, there are **{len(rows)} vacant units**. "
            f"They include {units}{extra}."
        )

    @staticmethod
    def _vacant_unit_label(row: dict) -> str:
        category = row.get("bedroom_category")
        unit_type = row.get("unit_type")
        if category and unit_type:
            return f"**{row['unit']}** ({category}, {unit_type})"
        if category:
            return f"**{row['unit']}** ({category})"
        return f"**{row['unit']}** ({unit_type})"

    def _with_bedroom_categories(self, rows: list[dict]) -> list[dict]:
        enriched_rows = []
        for row in rows:
            enriched = dict(row)
            category = self._bedroom_category_from_unit_type(
                str(enriched.get("unit_type", ""))
            )
            if category:
                enriched["bedroom_category"] = category
            enriched_rows.append(enriched)
        return enriched_rows

    def _rent_by_unit_type_summary(self, rows: list[dict]) -> str:
        if not rows:
            return "I could not find unit-type rent data for this property."

        bedroom_rows = self._rent_by_bedroom_category(rows)
        bedroom_sentence = ""
        if bedroom_rows:
            bedroom_text = ", ".join(
                f"**{row['bedroom_category']}** at "
                f"{self._format_money(row['avg_market_rent'])}"
                for row in bedroom_rows
            )
            bedroom_sentence = (
                f"By bedroom category, the average market rents are {bedroom_text}. "
            )

        sorted_rows = sorted(rows, key=lambda row: row["avg_market_rent"], reverse=True)
        highest = sorted_rows[0]
        lowest = sorted_rows[-1]
        top_rows = sorted_rows[:3]
        top_text = ", ".join(
            f"**{row['unit_type']}** at {self._format_money(row['avg_market_rent'])}"
            for row in top_rows
        )
        return (
            f"{bedroom_sentence}Across detailed floorplan codes, average market rent "
            f"ranges from "
            f"**{self._format_money(lowest['avg_market_rent'])}** "
            f"(**{lowest['unit_type']}**) to "
            f"**{self._format_money(highest['avg_market_rent'])}** "
            f"(**{highest['unit_type']}**). The top average rents are {top_text}."
        )

    def _rent_by_bedroom_category(self, rows: list[dict]) -> list[dict]:
        buckets: dict[str, dict[str, Any]] = {}
        for row in rows:
            category = self._bedroom_category_from_unit_type(str(row.get("unit_type", "")))
            if not category:
                continue

            unit_count = int(row.get("unit_count") or 0)
            avg_market_rent = float(row.get("avg_market_rent") or 0)
            bucket = buckets.setdefault(
                category,
                {
                    "bedroom_category": category,
                    "unit_count": 0,
                    "weighted_rent_total": 0.0,
                },
            )
            bucket["unit_count"] += unit_count
            bucket["weighted_rent_total"] += avg_market_rent * unit_count

        category_order = {"Studio": 0, "1-bedroom": 1, "2-bedroom": 2, "3-bedroom": 3}
        results = []
        for bucket in buckets.values():
            unit_count = bucket["unit_count"]
            if unit_count <= 0:
                continue
            results.append(
                {
                    "bedroom_category": bucket["bedroom_category"],
                    "unit_count": unit_count,
                    "avg_market_rent": round(bucket["weighted_rent_total"] / unit_count, 2),
                }
            )
        return sorted(
            results,
            key=lambda row: (
                category_order.get(row["bedroom_category"], 99),
                row["bedroom_category"],
            ),
        )

    @staticmethod
    def _bedroom_category_from_unit_type(unit_type: str) -> str | None:
        normalized = unit_type.upper().strip()
        match = re.search(r"(?:^|MX)([SABC])\d", normalized)
        if not match:
            match = re.search(r"\b([SABC])\d", normalized)
        if not match:
            return None
        return {
            "S": "Studio",
            "A": "1-bedroom",
            "B": "2-bedroom",
            "C": "3-bedroom",
        }[match.group(1)]

    def _matching_content_lines(
        self,
        message: str,
        content: str,
        limit: int = 4,
    ) -> list[str]:
        return [
            detail["line"]
            for detail in self._matching_content_line_details(
                message=message,
                content=content,
                limit=limit,
            )
        ]

    def _matching_content_line_details(
        self,
        message: str,
        content: str,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        terms = self._answer_terms(message)
        if not terms:
            return []

        scored_lines = []
        for index, cleaned in enumerate(self._clean_website_lines(content)):
            normalized = cleaned.lower()
            line_tokens = {
                match.group(0).lower() for match in ANSWER_TOKEN_RE.finditer(normalized)
            }
            matched_terms = [term for term in terms if term in line_tokens]
            score = len(matched_terms)
            if score:
                scored_lines.append((score, index, cleaned, matched_terms))

        scored_lines.sort(key=lambda item: (-item[0], item[1]))
        return [
            {"line": line, "terms": matched_terms, "score": score}
            for _, _, line, matched_terms in scored_lines[:limit]
        ]

    @staticmethod
    def _answer_terms(message: str) -> list[str]:
        tokens = []
        for match in ANSWER_TOKEN_RE.finditer(message):
            token = match.group(0).lower()
            candidates = [token]
            if "-" in token:
                candidates.extend(part for part in token.split("-") if part)
            tokens.extend(
                candidate
                for candidate in candidates
                if candidate not in ANSWER_STOPWORDS
            )
        if "ev" in tokens and "charging" not in tokens:
            tokens.append("charging")
        if "bike" in tokens and "storage" not in tokens:
            tokens.append("storage")
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _yes_no_retrieval_answer(matched_terms: set[str], evidence: list[dict[str, str]]) -> str:
        amenities = []
        if "bike" in matched_terms or "storage" in matched_terms:
            amenities.append("bike storage")
        if "ev" in matched_terms or "charging" in matched_terms:
            amenities.append("EV charging")
        if "pet" in matched_terms:
            amenities.append("pet-friendly features")

        details = [
            f"- **{item['section']}**: {item['snippet']}"
            for item in evidence[:3]
        ]
        if amenities:
            intro = f"Yes, this property has {LangChainOrchestrator._join_phrase(amenities)}."
        else:
            intro = "Yes. The selected property's website lists a matching feature."
        return intro + "\n\n" + "\n".join(details)

    def _natural_retrieval_fact_answer(
        self,
        message: str,
        evidence: list[dict[str, str]],
    ) -> str:
        if not evidence:
            return "I don't see matching website evidence for that in the selected property sample."

        text = message.lower()
        first = evidence[0]
        snippet = self._clean_snippet_spacing(first.get("snippet", ""))
        section = first.get("section", "Website Evidence")

        if "parking" in text:
            return (
                "Yes, the property website mentions parking. "
                f"Specifically, it lists **{snippet}** under **{section}**."
            )

        if "ev" in text or "charging" in text or "electric" in text:
            return (
                "Yes, the property website mentions EV/electric-vehicle related amenities. "
                f"Specifically, it lists **{snippet}** under **{section}**."
            )

        if "bike" in text or "storage" in text:
            return (
                "Yes, the property website mentions bike/storage related amenities. "
                f"Specifically, it lists **{snippet}** under **{section}**."
            )

        if "pet" in text:
            return (
                "Yes, the property website mentions pet-related information. "
                f"Specifically, it lists **{snippet}** under **{section}**."
            )

        return (
            "The property website has matching evidence: "
            f"**{snippet}** under **{section}**."
        )

    @staticmethod
    def _clean_snippet_spacing(text: str) -> str:
        text = re.sub(r"\bC ar\b", "Car", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _website_evidence_answer(evidence: list[dict[str, str]]) -> str:
        if not evidence:
            return "I don't see matching website evidence for that in the selected property sample."
        details = [
            f"- **{item['section']}**: {item['snippet']}"
            for item in evidence[:3]
        ]
        return (
            "Here is what I found on the selected property's website:\n\n"
            + "\n".join(details)
        )

    @staticmethod
    def _join_phrase(items: list[str]) -> str:
        if len(items) <= 1:
            return "".join(items)
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _content_preview(self, content: str, max_length: int = 260) -> str:
        preview = " ".join(self._clean_website_lines(content))
        if len(preview) <= max_length:
            return preview
        return preview[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _format_money(value: float | int | None) -> str:
        if value is None:
            return "not available"
        return f"${value:,.0f}" if abs(value) >= 100 else f"${value:,.2f}"

    @staticmethod
    def _format_percent(value: float | int | None) -> str:
        if value is None:
            return "not available"
        number = float(value)
        if number.is_integer():
            return f"{number:.1f}%"
        return f"{number:.2f}".rstrip("0").rstrip(".") + "%"

    @staticmethod
    def _looks_like_address(text: str) -> bool:
        lowered = text.lower()
        if "sq. ft" in lowered or re.search(r"\b(?:beds?|baths?)\b", lowered):
            return False
        return bool(
            re.search(r"\b\d{2,6}\s+[A-Za-z]", text)
            or re.search(r"\b[A-Z]{2}\s+\d{5}\b", text)
            or re.search(r"\b[A-Za-z .'-]+,\s*[A-Z]{2}\s*\d{5}\b", text)
        )

    @classmethod
    def _clean_website_lines(cls, content: str, section: str | None = None) -> list[str]:
        skipped = {section.lower()} if section else set()
        cleaned_lines: list[str] = []
        for raw_line in content.splitlines():
            line = re.sub(r"\s+", " ", raw_line.strip())
            normalized = line.lower()
            if len(line) < 3 or normalized in skipped:
                continue
            if cls._is_boilerplate_line(line) or cls._looks_like_address(line):
                continue
            if line not in cleaned_lines:
                cleaned_lines.append(line)
        return cleaned_lines

    @staticmethod
    def _is_boilerplate_line(text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in WEBSITE_BOILERPLATE_EXACT or any(
            pattern.search(text) for pattern in WEBSITE_BOILERPLATE_PATTERNS
        )

    @staticmethod
    def _is_yes_no_question(message: str) -> bool:
        first_token = next(ANSWER_TOKEN_RE.finditer(message), None)
        return bool(
            first_token
            and first_token.group(0).lower()
            in {"are", "can", "could", "did", "do", "does", "has", "have", "is"}
        )

    @classmethod
    def _looks_like_property_fact_question(cls, message: str) -> bool:
        text = message.lower()
        tokens = [match.group(0).lower() for match in ANSWER_TOKEN_RE.finditer(text)]
        if not tokens:
            return False
        if cls._is_yes_no_question(message):
            return True
        if tokens[0] in {"what", "which", "where"} and any(
            term in text
            for term in [
                "available",
                "include",
                "included",
                "listed",
                "mention",
                "mentioned",
                "offer",
                "offered",
                "provide",
                "provided",
            ]
        ):
            return True
        return text.rstrip().endswith("?") and any(
            term in text
            for term in ["amenity", "feature", "website", "available", "included", "listed"]
        )

    def _call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        raw = self.tools[tool_name].invoke(kwargs)
        return json.loads(raw)

    @staticmethod
    def _wants_reviews(text: str) -> bool:
        return bool(REVIEW_RE.search(text))

    @staticmethod
    def _filter_review_results(results: list[dict]) -> list[dict]:
        """Keep only chunks that actually look like review/rating/testimonial evidence.

        A generic amenities/contact chunk can be returned by semantic retrieval for a
        reviews query. This filter prevents those weak fallback chunks from being
        treated as evidence and shown as sources.
        """
        filtered: list[dict] = []
        for result in results or []:
            metadata = result.get("metadata") or {}
            haystack = " ".join(
                [
                    str(result.get("content") or ""),
                    str(metadata.get("page_type") or ""),
                    str(metadata.get("section_heading") or ""),
                    str(metadata.get("title") or ""),
                    str(metadata.get("source_url") or ""),
                ]
            ).lower()
            if REVIEW_RE.search(haystack):
                filtered.append(result)
        return filtered

    def _filter_matching_evidence_results(
        self,
        message: str,
        results: list[dict],
        require_complete_match: bool = False,
        min_confidence: str | None = None,
    ) -> list[dict]:
        filtered = []
        required_terms = set(self._answer_terms(message))
        for result in results or []:
            evidence = result.get("evidence") or {}
            line_details = evidence.get("line_matches") or []
            if not line_details:
                continue
            if min_confidence and not self._passes_evidence_confidence(
                evidence,
                min_confidence,
            ):
                continue
            if require_complete_match and len(required_terms) > 1:
                matched_terms = {
                    term for term in evidence.get("matched_terms", [])
                }
                if not required_terms.issubset(matched_terms):
                    continue
            filtered.append(result)
        return filtered

    @staticmethod
    def _passes_evidence_confidence(evidence: dict, minimum: str) -> bool:
        actual = str(evidence.get("confidence") or "low")
        return EVIDENCE_CONFIDENCE_RANK.get(actual, 0) >= EVIDENCE_CONFIDENCE_RANK[minimum]

    def _has_confident_evidence(
        self,
        results: list[dict],
        minimum: str = MIN_RETRIEVAL_CONFIDENCE,
    ) -> bool:
        return any(
            self._passes_evidence_confidence(result.get("evidence") or {}, minimum)
            for result in results or []
        )

    def _requires_grounded_retrieval_answer(
        self,
        message: str,
        intent: str | None,
    ) -> bool:
        text = message.lower()
        if intent in {"amenity_list", "floorplans", "gallery", "location"}:
            return False
        if self._wants_amenity_list(text, intent) or self._wants_floorplan_answer(text, intent):
            return False
        return self._is_yes_no_question(message) or self._looks_like_property_fact_question(message)

    def _annotate_retrieval_evidence(
        self,
        message: str,
        results: list[dict],
    ) -> list[dict]:
        annotated = []
        required_terms = set(self._answer_terms(message))
        for result in results or []:
            line_details = self._matching_content_line_details(
                message,
                result.get("content") or "",
            )
            matched_terms = {
                term for detail in line_details for term in detail.get("terms", [])
            }
            confidence = self._evidence_confidence(
                required_terms=required_terms,
                matched_terms=matched_terms,
                line_details=line_details,
                result=result,
            )
            enriched = dict(result)
            enriched["evidence"] = {
                "confidence": confidence,
                "matched_terms": sorted(matched_terms),
                "required_terms": sorted(required_terms),
                "match_count": len(matched_terms),
                "line_matches": line_details,
            }
            annotated.append(enriched)
        return annotated

    @staticmethod
    def _evidence_confidence(
        required_terms: set[str],
        matched_terms: set[str],
        line_details: list[dict[str, Any]],
        result: dict,
    ) -> str:
        if not line_details:
            return "low"
        if required_terms and required_terms.issubset(matched_terms):
            if result.get("vector_rank") is not None and result.get("keyword_rank") is not None:
                return "high"
            return "medium"
        if len(matched_terms) >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _wants_location_answer(text: str) -> bool:
        return any(term in text for term in ["address", "located", "location", "where is"])

    @staticmethod
    def _wants_location_website_context(text: str) -> bool:
        return any(term in text for term in ["website", "neighborhood", "nearby", "context"])

    @staticmethod
    def _sources_from_retrieval(results: list[dict]) -> list[Source]:
        sources = []
        seen = set()
        for result in results:
            metadata = result["metadata"]
            source_url = metadata.get("source_url")
            title = metadata.get("title")
            if metadata.get("page_type") == "floorplans":
                source_url = LangChainOrchestrator._main_floorplans_url(source_url)
                if not LangChainOrchestrator._is_floorplans_listing_url(
                    metadata.get("source_url")
                ):
                    title = f"{metadata.get('property_name', 'Property')} Floorplans"

            key = (source_url, metadata.get("page_type"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                Source(
                    property_code=metadata["property_code"],
                    title=title,
                    source_url=source_url,
                    page_type=metadata.get("page_type"),
                    tool="search_property_content",
                )
            )
        return sources

    @staticmethod
    def _add_kpi_components(latest_kpis: dict, components: list[UIComponent]) -> None:
        current = latest_kpis.get("current") if latest_kpis else None
        vacant = latest_kpis.get("vacant") if latest_kpis else None
        if current:
            components.extend(
                [
                    UIComponent(
                        type="kpi_card",
                        title="Occupancy",
                        data={
                            "value": current["unit_occupancy_pct"],
                            "unit": "%",
                            "report_month": current["report_month"],
                        },
                    ),
                    UIComponent(
                        type="kpi_card",
                        title="Lease Charges",
                        data={
                            "value": current["lease_charges"],
                            "unit": "USD",
                            "report_month": current["report_month"],
                        },
                    ),
                ]
            )
        if vacant:
            components.append(
                UIComponent(
                    type="kpi_card",
                    title="Vacant Units",
                    data={
                        "value": vacant["unit_count"],
                        "report_month": vacant["report_month"],
                    },
                )
            )
