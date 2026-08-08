from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from app.core.auth import local_authenticated_user
from app.core.authorization import AuthorizationContext
from app.core.config import Settings
from app.tools.contracts import TrustedToolContext
from app.tools.executor import ToolExecutor
from app.tools.property_tools import build_property_tool_registry


class PropertyCodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    property_code: str = Field(
        description=(
            "The active property code selected by the user, for example 115r. "
            "This value must always come from the active UI selection and must "
            "never be inferred from the user's natural-language question."
        )
    )


class TrendInput(PropertyCodeInput):
    months: int = Field(default=12, ge=1, le=36)


class LimitInput(PropertyCodeInput):
    limit: int = Field(default=10, ge=1, le=50)


class SearchContentInput(PropertyCodeInput):
    query: str = Field(
        description=(
            "Natural-language website search query. The search is always scoped "
            "to property_code metadata before returning results."
        )
    )
    page_type: str | None = Field(
        default=None,
        description=(
            "Optional website page type filter such as amenities, floorplans, "
            "gallery, home, or fees. Leave null when the page type is uncertain."
        ),
    )
    n_results: int = Field(default=5, ge=1, le=10)


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def build_langchain_tools(
    settings: Settings,
    executor: ToolExecutor | None = None,
) -> list[BaseTool]:
    """Expose compatibility wrappers backed by the controlled tool executor."""
    executor = executor or ToolExecutor(build_property_tool_registry(settings))
    authorization = AuthorizationContext.from_settings(
        local_authenticated_user(settings),
        settings,
    )

    def execute_json(
        name: str,
        arguments: dict[str, Any],
        property_code: str | None = None,
    ) -> str:
        result = executor.execute(
            name,
            arguments,
            TrustedToolContext(
                property_code=property_code,
                user_id=authorization.user.user_id,
                tenant_id=authorization.user.tenant_id,
                roles=authorization.user.roles,
                allowed_property_codes=authorization.allowed_property_codes,
            ),
        )
        if result.status == "failed":
            error = result.error
            raise RuntimeError(
                f"{name} failed ({error.type if error else 'unknown'}): "
                f"{error.message if error else 'Unknown tool failure.'}"
            )
        return to_json(result.data)

    @tool(args_schema=PropertyCodeInput)
    def get_property_profile(property_code: str) -> str:
        """Fetch profile metadata for exactly one active property.

        Use this first to verify the selected property exists and to retrieve
        display fields such as property name, address, and website URL. This
        tool is not for cross-property search or comparison.
        """
        return execute_json("get_property_profile", {}, property_code)

    @tool
    def list_properties() -> str:
        """Fetch the property catalog for detecting possible scope mismatches.

        Use this only to recognize when a user mentions a property name that is
        different from the active property. Do not switch the active property
        based on this result.
        """
        return execute_json("list_properties", {})

    @tool(args_schema=PropertyCodeInput)
    def get_report_periods(property_code: str) -> str:
        """Fetch available rent-roll report months and years for one property.

        Use this before answering date-specific structured questions. If the
        requested year or month is missing, do not substitute another period.
        """
        return execute_json("get_report_periods", {}, property_code)

    @tool(args_schema=PropertyCodeInput)
    def get_latest_property_kpis(property_code: str) -> str:
        """Fetch the latest property-level rent-roll KPI snapshot.

        Returns high-level metrics such as occupancy, unit count, market rent,
        lease charges, and vacant unit count for the latest available report
        month. Use this for latest/current KPI questions only.
        """
        return execute_json("get_latest_property_kpis", {}, property_code)

    @tool(args_schema=TrendInput)
    def get_occupancy_trend(property_code: str, months: int = 12) -> str:
        """Fetch monthly occupancy history for the active property.

        Use this for occupancy trend or month-over-month occupancy questions.
        The result is property-scoped and should not be used for unrelated
        rent, balance, or charge trends.
        """
        return execute_json("get_occupancy_trend", {"months": months}, property_code)

    @tool(args_schema=LimitInput)
    def get_charge_breakdown(property_code: str, limit: int = 10) -> str:
        """Fetch latest charge totals grouped by charge code.

        Use this for questions about largest charge categories or lease charge
        breakdowns. This is not a unit-level charge-detail tool and should not
        be used to calculate unsupported grouped aggregates.
        """
        return execute_json("get_charge_breakdown", {"limit": limit}, property_code)

    @tool(args_schema=LimitInput)
    def get_top_balances(property_code: str, limit: int = 10) -> str:
        """Fetch units with the highest latest resident balances.

        This returns only the top N balance rows for the latest report month.
        Use it for "top balances" or "highest balance units." Do not use it
        to calculate full-property averages, medians, rates, or grouped balance
        aggregates because it is a partial ranked result.
        """
        return execute_json("get_top_balances", {"limit": limit}, property_code)

    @tool(args_schema=LimitInput)
    def get_vacant_units(property_code: str, limit: int = 20) -> str:
        """Fetch the latest vacant-unit list for the active property.

        Use this for questions asking which units are vacant, their unit types,
        bedroom categories, square footage, or market rent. Do not use it to
        calculate unsupported vacancy rates by category.
        """
        return execute_json("get_vacant_units", {"limit": limit}, property_code)

    @tool(args_schema=PropertyCodeInput)
    def get_rent_by_unit_type(property_code: str) -> str:
        """Fetch latest average market rent grouped by rent-roll unit type.

        Use this for average market rent by unit type, floorplan code, or
        recognizable bedroom category. Do not use it for top individual unit
        rents, median rent, lease charges by unit type, or arbitrary SQL-style
        aggregates.
        """
        return execute_json("get_rent_by_unit_type", {}, property_code)

    @tool(args_schema=SearchContentInput)
    def search_property_content(
        property_code: str,
        query: str,
        page_type: str | None = None,
        n_results: int = 5,
    ) -> str:
        """Search scraped website content for property-specific evidence.

        Uses hybrid retrieval over Chroma vector search and BM25 keyword search,
        always filtered by active property_code. Use this for website facts such
        as amenities, apartment features, EV charging, bike storage, floorplans,
        location, and fees. If evidence is weak or missing, say the selected
        property sample does not contain matching website evidence.
        """
        return execute_json(
            "search_property_content",
            {"query": query, "page_type": page_type, "n_results": n_results},
            property_code,
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
