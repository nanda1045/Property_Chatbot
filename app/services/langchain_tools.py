from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.db.mysql import MySQLDatabase
from app.retrieval.embeddings import build_embedder
from app.retrieval.hybrid_store import HybridPropertyRetriever
from app.services.rent_roll_repository import RentRollRepository


class PropertyCodeInput(BaseModel):
    property_code: str = Field(description="Active property code, for example 115r.")


class TrendInput(PropertyCodeInput):
    months: int = Field(default=12, ge=1, le=24)


class LimitInput(PropertyCodeInput):
    limit: int = Field(default=10, ge=1, le=50)


class SearchContentInput(PropertyCodeInput):
    query: str = Field(description="User search query scoped to the active property.")
    page_type: str | None = Field(default=None, description="Optional page type filter.")
    n_results: int = Field(default=5, ge=1, le=10)


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def build_langchain_tools(settings: Settings) -> list[BaseTool]:
    repository = RentRollRepository(MySQLDatabase(settings))
    property_retriever = HybridPropertyRetriever(
        chroma_path=settings.chroma_path,
        chroma_collection=settings.chroma_collection,
        bm25_path=settings.bm25_path,
        embedder=build_embedder(settings),
    )

    @tool(args_schema=PropertyCodeInput)
    def get_property_profile(property_code: str) -> str:
        """Fetch property metadata for the active property code."""
        return to_json(repository.get_property_profile(property_code))

    @tool
    def list_properties() -> str:
        """Fetch the property catalog for scope-mismatch detection."""
        return to_json(repository.list_properties())

    @tool(args_schema=PropertyCodeInput)
    def get_report_periods(property_code: str) -> str:
        """Fetch available structured rent-roll report months and years."""
        return to_json(repository.get_report_periods(property_code))

    @tool(args_schema=PropertyCodeInput)
    def get_latest_property_kpis(property_code: str) -> str:
        """Fetch latest rent-roll KPI summaries for the active property code."""
        return to_json(repository.get_latest_kpis(property_code))

    @tool(args_schema=TrendInput)
    def get_occupancy_trend(property_code: str, months: int = 12) -> str:
        """Fetch month-over-month occupancy KPIs for the active property code."""
        return to_json(repository.get_occupancy_trend(property_code, months=months))

    @tool(args_schema=LimitInput)
    def get_charge_breakdown(property_code: str, limit: int = 10) -> str:
        """Fetch latest charge-code totals for the active property code."""
        return to_json(repository.get_charge_breakdown(property_code, limit=limit))

    @tool(args_schema=LimitInput)
    def get_top_balances(property_code: str, limit: int = 10) -> str:
        """Fetch units with the highest latest balances for the active property code."""
        return to_json(repository.get_top_balances(property_code, limit=limit))

    @tool(args_schema=LimitInput)
    def get_vacant_units(property_code: str, limit: int = 20) -> str:
        """Fetch latest vacant units for the active property code."""
        return to_json(repository.get_vacant_units(property_code, limit=limit))

    @tool(args_schema=PropertyCodeInput)
    def get_rent_by_unit_type(property_code: str) -> str:
        """Fetch latest market-rent statistics grouped by unit type for the active property."""
        return to_json(repository.get_rent_by_unit_type(property_code))

    @tool(args_schema=SearchContentInput)
    def search_property_content(
        property_code: str,
        query: str,
        page_type: str | None = None,
        n_results: int = 5,
    ) -> str:
        """Search scraped website chunks using hybrid retrieval scoped to the property code."""
        results = property_retriever.search(
            query=query,
            property_code=property_code,
            page_type=page_type,
            n_results=n_results,
        )
        return to_json(
            [
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
        )

    return [
        get_property_profile,
        list_properties,
        get_report_periods,
        get_latest_property_kpis,
        get_occupancy_trend,
        get_charge_breakdown,
        get_top_balances,
        get_vacant_units,
        get_rent_by_unit_type,
        search_property_content,
    ]
