from __future__ import annotations

import json

from app.services.sql_approval import draft_sql_for_approval, validate_drafted_sql


def test_valid_top_market_rents_sql_passes_validation() -> None:
    sql = """
    SELECT u.unit, u.unit_type, u.market_rent
    FROM rent_roll_units u
    WHERE u.property_code = :property_code
      AND u.report_id = (
        SELECT MAX(r.id)
        FROM rent_roll_reports r
        WHERE r.property_code = :property_code
      )
    ORDER BY u.market_rent DESC
    LIMIT 10
    """

    ok, reason = validate_drafted_sql(sql)

    assert ok, reason


def test_valid_units_by_unit_type_sql_passes_validation() -> None:
    sql = """
    SELECT u.unit_type, COUNT(*) AS unit_count
    FROM rent_roll_units u
    WHERE u.property_code = :property_code
      AND u.report_id = (
        SELECT MAX(r.id)
        FROM rent_roll_reports r
        WHERE r.property_code = :property_code
      )
    GROUP BY u.unit_type
    ORDER BY unit_count DESC
    """

    ok, reason = validate_drafted_sql(sql)

    assert ok, reason


def test_sql_without_property_placeholder_fails() -> None:
    sql = """
    SELECT unit, market_rent
    FROM rent_roll_units
    WHERE property_code = '115r'
    LIMIT 10
    """

    ok, reason = validate_drafted_sql(sql)

    assert not ok
    assert "placeholder" in reason.lower() or "hardcode" in reason.lower()


def test_sql_without_property_code_fails() -> None:
    sql = """
    SELECT unit, market_rent
    FROM rent_roll_units
    ORDER BY market_rent DESC
    LIMIT 10
    """

    ok, reason = validate_drafted_sql(sql)

    assert not ok
    assert "property_code" in reason.lower()


def test_sql_with_semicolon_fails() -> None:
    ok, reason = validate_drafted_sql(
        "SELECT unit FROM rent_roll_units WHERE property_code = :property_code LIMIT 10;"
    )

    assert not ok
    assert "semicolon" in reason.lower()


def test_sql_with_delete_fails() -> None:
    ok, reason = validate_drafted_sql(
        "DELETE FROM rent_roll_units WHERE property_code = :property_code"
    )

    assert not ok
    assert "select" in reason.lower() or "forbidden" in reason.lower()


def test_sql_with_union_fails() -> None:
    sql = """
    SELECT unit FROM rent_roll_units WHERE property_code = :property_code
    UNION
    SELECT unit FROM rent_roll_units WHERE property_code = :property_code
    LIMIT 10
    """

    ok, reason = validate_drafted_sql(sql)

    assert not ok
    assert "union" in reason.lower()


def test_sql_with_unknown_table_fails() -> None:
    sql = """
    SELECT *
    FROM resident_reviews
    WHERE property_code = :property_code
    LIMIT 10
    """

    ok, reason = validate_drafted_sql(sql)

    assert not ok
    assert "unsupported table" in reason.lower()


def test_sql_with_pii_column_fails() -> None:
    sql = """
    SELECT resident_name, balance
    FROM rent_roll_units
    WHERE property_code = :property_code
    ORDER BY balance DESC
    LIMIT 10
    """

    ok, reason = validate_drafted_sql(sql)

    assert not ok
    assert "pii" in reason.lower() or "sensitive" in reason.lower()


def test_drafter_answerable_false_is_supported() -> None:
    def fake_chat_model(messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "answerable": False,
                "unavailable_reason": "Crime-rate data is not present in the current MySQL schema.",
                "sql": None,
                "explanation": "The requested metric requires an external public safety dataset.",
                "parameters": {},
                "safety_notes": [],
            }
        )

    draft = draft_sql_for_approval(
        message="What is the crime rate around this property?",
        property_code="115r",
        property_name="Canfield Park",
        sql_request="Draft SQL for crime rate.",
        chat_model=fake_chat_model,
    )

    assert draft is not None
    assert draft.answerable is False
    assert draft.sql is None
    assert "crime" in (draft.unavailable_reason or "").lower()