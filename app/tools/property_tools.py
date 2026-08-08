"""Typed specifications and handlers for property-scoped tools."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel

from app.core.authorization import ToolPermission
from app.core.config import Settings
from app.db.mysql import MySQLDatabase
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid_store import HybridPropertyRetriever
from app.services.rent_roll_repository import RentRollRepository
from app.tools.contracts import ToolSpec, TrustedToolContext
from app.tools.registry import ToolRegistry


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyInput(StrictInput):
    pass


class TrendInput(StrictInput):
    months: int = Field(default=12, ge=1, le=36)


class LimitInput(StrictInput):
    limit: int = Field(default=10, ge=1, le=50)


class VacantUnitsInput(LimitInput):
    report_month: date | None = None


class ReportMonthInput(StrictInput):
    report_month: date | None = None


class SearchContentInput(StrictInput):
    query: str = Field(min_length=1, max_length=1000)
    page_type: str | None = Field(default=None, max_length=64)
    n_results: int = Field(default=5, ge=1, le=10)


class PropertyProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    property_code: str
    property_name: str
    address: str | None = None
    source_site: str | None = None


class PropertyProfileOutput(RootModel[PropertyProfile | None]):
    pass


class PropertyCatalogOutput(RootModel[list[PropertyProfile]]):
    pass


class ReportPeriodsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_code: str
    min_report_month: str | None
    max_report_month: str | None
    months: list[str]
    years: list[int]


class KpiRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    property_code: str
    report_month: str


class LatestKpisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current: KpiRow | None
    vacant: KpiRow | None


class OccupancyTrendRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    report_month: str
    unit_occupancy_pct: float | None = None


class OccupancyTrendOutput(RootModel[list[OccupancyTrendRow]]):
    pass


class ChargeBreakdownRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    charge_code: str
    amount: float | None = None
    report_month: str


class ChargeBreakdownOutput(RootModel[list[ChargeBreakdownRow]]):
    pass


class BalanceRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    unit: str
    balance: float | None = None
    report_month: str


class TopBalancesOutput(RootModel[list[BalanceRow]]):
    pass


class VacantUnitRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    unit: str
    unit_type: str | None = None
    report_month: str


class VacantUnitsOutput(RootModel[list[VacantUnitRow]]):
    pass


class RentByUnitTypeRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    unit_type: str | None
    unit_count: int
    avg_market_rent: float | None = None
    report_month: str


class RentByUnitTypeOutput(RootModel[list[RentByUnitTypeRow]]):
    pass


class RetrievalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    distance: float | None = None
    score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    metadata: dict[str, Any]


class SearchContentOutput(RootModel[list[RetrievalItem]]):
    pass


TOOL_DESCRIPTIONS = {
    "get_property_profile": (
        "Fetch profile metadata for exactly one active property. Use this first to verify "
        "the selected property exists and retrieve its display fields."
    ),
    "list_properties": (
        "Fetch the property catalog only for detecting possible property-scope mismatches."
    ),
    "get_report_periods": (
        "Fetch available rent-roll report months and years for the active property."
    ),
    "get_latest_property_kpis": (
        "Fetch the latest property-level occupancy, unit, rent, charge, and vacancy KPIs."
    ),
    "get_occupancy_trend": "Fetch monthly occupancy history for the active property.",
    "get_charge_breakdown": "Fetch latest charge totals grouped by charge code.",
    "get_top_balances": "Fetch units with the highest latest resident balances.",
    "get_vacant_units": "Fetch the latest vacant-unit list for the active property.",
    "get_rent_by_unit_type": (
        "Fetch latest average market rent grouped by rent-roll unit type."
    ),
    "search_property_content": (
        "Search scraped website content using property-scoped hybrid retrieval."
    ),
}


def _property_code(context: TrustedToolContext) -> str:
    if context.property_code is None:
        raise ValueError("property_code scope is required")
    return context.property_code


def build_property_tool_registry(
    settings: Settings,
    *,
    repository: RentRollRepository | None = None,
    property_retriever: HybridPropertyRetriever | None = None,
) -> ToolRegistry:
    """Build the central registry with local database and retrieval handlers."""
    repository = repository or RentRollRepository(MySQLDatabase(settings))
    property_retriever = property_retriever or HybridPropertyRetriever(
        chroma_path=settings.chroma_path,
        chroma_collection=settings.chroma_collection,
        bm25_path=settings.bm25_path,
        embedder=build_embedder(settings),
    )
    registry = ToolRegistry()

    def register(
        name: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        handler,
        *,
        timeout_seconds: float = 5.0,
        max_attempts: int = 3,
        required_scopes: tuple[str, ...] = ("property_code",),
        required_permission: ToolPermission = ToolPermission.PROPERTY_BASIC_READ,
    ) -> None:
        registry.register(
            ToolSpec(
                name=name,
                description=TOOL_DESCRIPTIONS[name],
                input_model=input_model,
                output_model=output_model,
                risk_level="read",
                required_permission=required_permission,
                timeout_seconds=timeout_seconds,
                max_attempts=max_attempts,
                idempotent=True,
                required_scopes=required_scopes,
            ),
            handler,
        )

    register(
        "get_property_profile",
        EmptyInput,
        PropertyProfileOutput,
        lambda _input, context: repository.get_property_profile(_property_code(context)),
    )
    register(
        "list_properties",
        EmptyInput,
        PropertyCatalogOutput,
        lambda _input, context: [
            item
            for item in repository.list_properties()
            if "*" in context.allowed_property_codes
            or str(item.get("property_code") or "").lower()
            in context.allowed_property_codes
        ],
        required_scopes=(),
    )
    register(
        "get_report_periods",
        EmptyInput,
        ReportPeriodsOutput,
        lambda _input, context: repository.get_report_periods(_property_code(context)),
    )
    register(
        "get_latest_property_kpis",
        EmptyInput,
        LatestKpisOutput,
        lambda _input, context: repository.get_latest_kpis(_property_code(context)),
        required_permission=ToolPermission.KPI_READ,
    )
    register(
        "get_occupancy_trend",
        TrendInput,
        OccupancyTrendOutput,
        lambda tool_input, context: repository.get_occupancy_trend(
            _property_code(context),
            months=tool_input.months,
        ),
        required_permission=ToolPermission.ANALYTICS_READ,
    )
    register(
        "get_charge_breakdown",
        LimitInput,
        ChargeBreakdownOutput,
        lambda tool_input, context: repository.get_charge_breakdown(
            _property_code(context),
            limit=tool_input.limit,
        ),
        required_permission=ToolPermission.ANALYTICS_READ,
    )
    register(
        "get_top_balances",
        LimitInput,
        TopBalancesOutput,
        lambda tool_input, context: repository.get_top_balances(
            _property_code(context),
            limit=tool_input.limit,
        ),
        required_permission=ToolPermission.ANALYTICS_READ,
    )
    register(
        "get_vacant_units",
        VacantUnitsInput,
        VacantUnitsOutput,
        lambda tool_input, context: repository.get_vacant_units(
            _property_code(context),
            limit=tool_input.limit,
            report_month=(
                tool_input.report_month.isoformat() if tool_input.report_month else None
            ),
        ),
        required_permission=ToolPermission.ANALYTICS_READ,
    )
    register(
        "get_rent_by_unit_type",
        ReportMonthInput,
        RentByUnitTypeOutput,
        lambda tool_input, context: repository.get_rent_by_unit_type(
            _property_code(context),
            report_month=(
                tool_input.report_month.isoformat() if tool_input.report_month else None
            ),
        ),
        required_permission=ToolPermission.ANALYTICS_READ,
    )

    def search_content(tool_input: SearchContentInput, context: TrustedToolContext) -> list[dict]:
        results = property_retriever.search(
            query=tool_input.query,
            property_code=_property_code(context),
            page_type=tool_input.page_type,
            n_results=tool_input.n_results,
        )
        return [
            {
                "id": result.id,
                "content": result.content,
                "distance": result.distance,
                "score": result.score,
                "vector_rank": result.vector_rank,
                "keyword_rank": result.keyword_rank,
                "metadata": result.metadata,
            }
            for result in results
        ]

    register(
        "search_property_content",
        SearchContentInput,
        SearchContentOutput,
        search_content,
        timeout_seconds=15.0,
        max_attempts=2,
        required_permission=ToolPermission.RETRIEVAL_READ,
    )
    return registry
