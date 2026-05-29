from app.services.sql_approval import validate_drafted_sql, validate_read_only_sql


def test_valid_single_table_query_can_use_unqualified_property_filter() -> None:
    result = validate_read_only_sql(
        """
        SELECT unit, market_rent
        FROM rent_roll_units
        WHERE property_code = '115r'
        ORDER BY market_rent DESC
        LIMIT 10
        """,
        "115r",
    )

    assert result.ok


def test_valid_join_requires_every_alias_to_be_property_scoped() -> None:
    result = validate_read_only_sql(
        """
        SELECT u.unit, r.report_month
        FROM rent_roll_units u
        JOIN rent_roll_reports r ON r.id = u.report_id
        WHERE u.property_code = '115r'
          AND r.property_code = '115r'
        LIMIT 10
        """,
        "115r",
    )

    assert result.ok


def test_join_with_unscoped_alias_is_rejected() -> None:
    result = validate_read_only_sql(
        """
        SELECT u.unit, r.report_month
        FROM rent_roll_units u
        JOIN rent_roll_reports r ON r.id = u.report_id
        WHERE u.property_code = '115r'
        LIMIT 10
        """,
        "115r",
    )

    assert not result.ok
    assert "rent_roll_reports AS r" in (result.error or "")


def test_string_literal_cannot_satisfy_property_filter() -> None:
    result = validate_read_only_sql(
        """
        SELECT 'property_code = ''115r''' AS fake_scope
        FROM rent_roll_units
        LIMIT 10
        """,
        "115r",
    )

    assert not result.ok
    assert "active property_code" in (result.error or "")


def test_string_literal_cannot_satisfy_table_reference() -> None:
    result = validate_read_only_sql(
        """
        SELECT 'FROM rent_roll_units WHERE property_code = ''115r''' AS fake_query
        LIMIT 10
        """,
        "115r",
    )

    assert not result.ok
    assert "read from at least one allowed table" in (result.error or "")


def test_unsupported_table_is_rejected() -> None:
    result = validate_read_only_sql(
        "SELECT unit FROM users WHERE property_code = '115r'",
        "115r",
    )

    assert not result.ok
    assert "unsupported table" in (result.error or "")


def test_write_statement_is_rejected() -> None:
    result = validate_read_only_sql(
        "DELETE FROM rent_roll_units WHERE property_code = '115r'",
        "115r",
    )

    assert not result.ok
    assert "SELECT" in (result.error or "")


def test_valid_drafted_sql_uses_property_placeholder() -> None:
    ok, reason = validate_drafted_sql(
        """
        SELECT u.unit_type, COUNT(*) AS unit_count
        FROM rent_roll_units u
        JOIN rent_roll_reports r ON r.id = u.report_id
        WHERE u.property_code = :property_code
          AND r.property_code = :property_code
          AND r.report_month = (
            SELECT MAX(r2.report_month)
            FROM rent_roll_reports r2
            WHERE r2.property_code = :property_code
          )
        GROUP BY u.unit_type
        LIMIT 50
        """
    )

    assert ok, reason


def test_drafted_delete_is_rejected() -> None:
    ok, reason = validate_drafted_sql(
        "DELETE FROM rent_roll_units WHERE property_code = :property_code"
    )

    assert not ok
    assert "SELECT" in reason


def test_drafted_sql_without_property_code_is_rejected() -> None:
    ok, reason = validate_drafted_sql("SELECT unit, market_rent FROM rent_roll_units LIMIT 10")

    assert not ok
    assert "property_code" in reason


def test_drafted_sql_without_property_placeholder_is_rejected() -> None:
    ok, reason = validate_drafted_sql(
        "SELECT unit, market_rent FROM rent_roll_units WHERE property_code = '115r' LIMIT 10"
    )

    assert not ok
    assert ":property_code" in reason or "hardcode" in reason


def test_drafted_sql_with_semicolon_is_rejected() -> None:
    ok, reason = validate_drafted_sql(
        "SELECT unit FROM rent_roll_units WHERE property_code = :property_code;"
    )

    assert not ok
    assert "Semicolons" in reason


def test_drafted_sql_with_union_is_rejected() -> None:
    ok, reason = validate_drafted_sql(
        """
        SELECT unit FROM rent_roll_units WHERE property_code = :property_code
        UNION
        SELECT unit FROM rent_roll_units WHERE property_code = :property_code
        """
    )

    assert not ok
    assert "UNION" in reason


def test_drafted_sql_with_comments_is_rejected() -> None:
    ok, reason = validate_drafted_sql(
        """
        SELECT unit FROM rent_roll_units
        WHERE property_code = :property_code -- scoped
        """
    )

    assert not ok
    assert "comments" in reason
