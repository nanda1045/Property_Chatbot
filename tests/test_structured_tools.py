from __future__ import annotations

import pytest

from app.services.rent_roll_repository import RentRollRepository


def test_property_profile_is_loaded(rent_roll_repository: RentRollRepository) -> None:
    profile = rent_roll_repository.get_property_profile("115r")

    assert profile is not None
    assert profile["property_code"] == "115r"
    assert profile["property_name"] == "Canfield Park"
    assert "Canfield Avenue" in profile["address"]


def test_latest_kpis_match_known_canfield_snapshot(
    rent_roll_repository: RentRollRepository,
) -> None:
    kpis = rent_roll_repository.get_latest_kpis("115r")

    assert kpis["current"]["report_month"] == "2025-12-01"
    assert kpis["current"]["unit_count"] == 300
    assert kpis["current"]["unit_occupancy_pct"] == pytest.approx(96.66)
    assert kpis["vacant"]["unit_count"] == 10


def test_charge_breakdown_is_scoped_and_contains_rent(
    rent_roll_repository: RentRollRepository,
) -> None:
    charges = rent_roll_repository.get_charge_breakdown("115r")

    assert charges
    assert {row["report_month"] for row in charges} == {"2025-12-01"}
    assert "RENT" in {row["charge_code"] for row in charges}


def test_top_balances_are_sorted_descending(
    rent_roll_repository: RentRollRepository,
) -> None:
    rows = rent_roll_repository.get_top_balances("115r", limit=10)
    balances = [row["balance"] for row in rows]

    assert rows
    assert balances == sorted(balances, reverse=True)


def test_occupancy_trend_returns_months_in_ascending_order(
    rent_roll_repository: RentRollRepository,
) -> None:
    rows = rent_roll_repository.get_occupancy_trend("115r", months=6)
    months = [row["report_month"] for row in rows]

    assert len(rows) == 6
    assert months == sorted(months)
