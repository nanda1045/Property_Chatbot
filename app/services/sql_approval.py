from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from app.core.config import Settings
from app.db.mysql import MySQLDatabase


ALLOWED_TABLES = {
    "properties",
    "rent_roll_reports",
    "rent_roll_units",
    "lease_charges",
    "rent_roll_summary_groups",
    "rent_roll_charge_summary",
}
FORBIDDEN_SQL_RE = re.compile(
    r"\b("
    r"alter|call|create|delete|drop|grant|insert|load|lock|merge|replace|revoke|set|truncate|"
    r"unlock|update|use"
    r")\b",
    re.IGNORECASE,
)
TABLE_REF_RE = re.compile(r"\b(?:from|join)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?", re.IGNORECASE)
TABLE_ALIAS_RE = re.compile(
    r"\b(?:from|join)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?"
    r"(?:\s+(?:as\s+)?`?([a-zA-Z_][a-zA-Z0-9_]*)`?)?",
    re.IGNORECASE,
)
LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)
SQL_KEYWORDS = {
    "cross",
    "full",
    "group",
    "having",
    "inner",
    "join",
    "left",
    "limit",
    "natural",
    "on",
    "order",
    "right",
    "union",
    "where",
}


@dataclass(frozen=True)
class SqlValidationResult:
    sql: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.sql is not None and self.error is None


class DraftedSql(BaseModel):
    sql: str
    explanation: str
    parameters: dict[str, Any]
    safety_notes: list[str]


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def _mask_comments_and_string_literals(sql: str) -> str:
    masked: list[str] = []
    index = 0
    quote: str | None = None

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if quote is None and char == "-" and next_char == "-":
            masked.extend("  ")
            index += 2
            while index < len(sql) and sql[index] != "\n":
                masked.append(" ")
                index += 1
            continue

        if quote is None and char == "/" and next_char == "*":
            masked.extend("  ")
            index += 2
            while index < len(sql):
                if sql[index] == "*" and index + 1 < len(sql) and sql[index + 1] == "/":
                    masked.extend("  ")
                    index += 2
                    break
                masked.append(" ")
                index += 1
            continue

        if quote is None and char in {"'", '"'}:
            quote = char
            masked.append(char)
            index += 1
            continue

        if quote is not None:
            masked.append(char if char == quote else " ")
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    masked.append(" ")
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        masked.append(char)
        index += 1

    return "".join(masked)


def _table_scopes(sql: str) -> list[tuple[str, str]]:
    scopes: list[tuple[str, str]] = []
    for match in TABLE_ALIAS_RE.finditer(sql):
        table = match.group(1).lower()
        alias = (match.group(2) or table).lower()
        if alias in SQL_KEYWORDS:
            alias = table
        scopes.append((table, alias))
    return scopes


def _has_property_filter(
    sql: str,
    masked_sql: str,
    scope_name: str,
    property_code: str,
    allow_unqualified: bool,
) -> bool:
    escaped_scope = re.escape(scope_name)
    escaped_code = re.escape(property_code.lower())
    qualifier = rf"\b`?{escaped_scope}`?\s*\.\s*" if not allow_unqualified else (
        rf"(?:\b`?{escaped_scope}`?\s*\.\s*)?"
    )
    pattern = re.compile(
        rf"{qualifier}`?property_code`?\s*=\s*"
        rf"(?P<quote>['\"]){escaped_code}(?P=quote)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql.lower()):
        masked_match = masked_sql[match.start() : match.end()].lower()
        if "property_code" in masked_match:
            return True
    return False


def _all_table_scopes_are_filtered(
    sql: str,
    masked_sql: str,
    property_code: str,
) -> SqlValidationResult:
    table_scopes = _table_scopes(masked_sql)
    if not table_scopes:
        return SqlValidationResult(None, "The SQL must read from at least one allowed table.")

    disallowed = sorted({table for table, _alias in table_scopes if table not in ALLOWED_TABLES})
    if disallowed:
        return SqlValidationResult(
            None,
            f"The SQL references unsupported table(s): {', '.join(disallowed)}.",
        )

    if len(table_scopes) == 1:
        table, alias = table_scopes[0]
        if _has_property_filter(
            sql, masked_sql, alias, property_code, allow_unqualified=True
        ) or _has_property_filter(
            sql, masked_sql, table, property_code, allow_unqualified=True
        ):
            return SqlValidationResult(sql)
        return SqlValidationResult(None, "The SQL must filter the table by active property_code.")

    unscoped: list[str] = []
    for table, alias in table_scopes:
        if _has_property_filter(
            sql, masked_sql, alias, property_code, allow_unqualified=False
        ) or _has_property_filter(
            sql, masked_sql, table, property_code, allow_unqualified=False
        ):
            continue
        unscoped.append(f"{table} AS {alias}" if alias != table else table)

    if unscoped:
        return SqlValidationResult(
            None,
            "Every referenced table must be explicitly filtered by active property_code. "
            f"Missing filter for: {', '.join(unscoped)}.",
        )
    return SqlValidationResult(sql)


def validate_read_only_sql(sql: str, property_code: str, max_rows: int = 100) -> SqlValidationResult:
    candidate = sql.strip()
    if not candidate:
        return SqlValidationResult(None, "The proposed SQL was empty.")

    if ";" in candidate:
        return SqlValidationResult(None, "Semicolons and multiple statements are not allowed.")

    if not re.match(r"^select\b", candidate, re.IGNORECASE):
        return SqlValidationResult(None, "Only read-only SELECT queries can be approved.")

    masked_sql = _mask_comments_and_string_literals(candidate)
    if FORBIDDEN_SQL_RE.search(masked_sql):
        return SqlValidationResult(None, "The SQL contains a forbidden operation.")

    scope_result = _all_table_scopes_are_filtered(candidate, masked_sql, property_code)
    if not scope_result.ok:
        return scope_result

    limit_match = LIMIT_RE.search(candidate)
    if limit_match:
        if int(limit_match.group(1)) > max_rows:
            candidate = LIMIT_RE.sub(f"LIMIT {max_rows}", candidate)
    else:
        candidate = f"{candidate}\nLIMIT {max_rows}"

    return SqlValidationResult(candidate)


def validate_drafted_sql(sql: str) -> tuple[bool, str]:
    candidate = sql.strip()
    if not candidate:
        return False, "The drafted SQL was empty."

    lowered = candidate.lower()
    if ";" in candidate:
        return False, "Semicolons and multiple statements are not allowed."
    if "--" in candidate or "/*" in candidate or "*/" in candidate:
        return False, "SQL comments are not allowed."
    if not re.match(r"^(select|with)\b", candidate, re.IGNORECASE):
        return False, "Only read-only SELECT queries can be drafted."

    masked_sql = _mask_comments_and_string_literals(candidate)
    if FORBIDDEN_SQL_RE.search(masked_sql):
        return False, "The SQL contains a forbidden operation."
    if re.search(r"\bor\s+1\s*=\s*1\b", masked_sql, re.IGNORECASE):
        return False, "Always-true OR predicates are not allowed."
    if re.search(r"\bunion\b", masked_sql, re.IGNORECASE):
        return False, "UNION queries are not allowed for approval drafts."
    if re.search(r"\bresident_(?:name|id)\b", masked_sql, re.IGNORECASE):
        return False, "Sensitive resident identifiers cannot be selected."
    if "property_code" not in masked_sql.lower():
        return False, "The SQL must filter by property_code."
    if ":property_code" not in candidate:
        return False, "The SQL must use the :property_code placeholder."
    if re.search(
        r"(?:\b`?[a-zA-Z_][a-zA-Z0-9_]*`?\s*\.\s*)?`?property_code`?\s*=\s*"
        r"(?P<quote>['\"])[^'\"]+(?P=quote)",
        candidate,
        re.IGNORECASE,
    ):
        return False, "The SQL must not hardcode a property code."

    table_scopes = _table_scopes(masked_sql)
    if not table_scopes:
        return False, "The SQL must read from at least one approved table."
    disallowed = sorted({table for table, _alias in table_scopes if table not in ALLOWED_TABLES})
    if disallowed:
        return False, f"The SQL references unsupported table(s): {', '.join(disallowed)}."

    for table, alias in table_scopes:
        alias_pattern = re.compile(
            rf"\b`?{re.escape(alias)}`?\s*\.\s*`?property_code`?\s*=\s*:property_code\b",
            re.IGNORECASE,
        )
        table_pattern = re.compile(
            rf"\b`?{re.escape(table)}`?\s*\.\s*`?property_code`?\s*=\s*:property_code\b",
            re.IGNORECASE,
        )
        unqualified_pattern = re.compile(
            r"\b`?property_code`?\s*=\s*:property_code\b",
            re.IGNORECASE,
        )
        if len(table_scopes) == 1 and unqualified_pattern.search(masked_sql):
            continue
        if alias_pattern.search(masked_sql) or table_pattern.search(masked_sql):
            continue
        return False, f"Missing property_code = :property_code filter for {table}."

    return True, "ok"


def draft_sql_for_approval(
    message: str,
    property_code: str,
    property_name: str,
    sql_request: str,
    chat_model: Callable[[list[dict[str, str]]], str],
) -> DraftedSql | None:
    system_prompt = (
        "You draft SQL for a property-scoped rent-roll assistant. Return only valid JSON.\n\n"
        "The backend will show your SQL to the user for manual approval. It will not be "
        "executed automatically.\n\n"
        "Rules:\n"
        "- Generate one SELECT-only MySQL query.\n"
        "- Use the parameter placeholder :property_code. Do not hardcode the property code.\n"
        "- Always filter every table by property_code = :property_code.\n"
        "- Use the latest report month with a MAX(report_month) subquery when the user asks "
        "for latest/current unit-level or report-level data.\n"
        "- Return no more than 50 rows unless the result is aggregate-only.\n"
        "- Use only known tables and columns from this schema.\n"
        "- Do not include semicolons, comments, CTEs, UNION, write operations, temporary "
        "tables, stored procedures, or resident_name/resident_id.\n\n"
        "Schema:\n"
        "- properties(property_code, property_name, address, source_site)\n"
        "- rent_roll_reports(id, property_code, report_month, as_of_date, source_filename)\n"
        "- rent_roll_units(id, report_id, property_code, resident_group, unit, unit_type, "
        "sqft, resident_status, market_rent, resident_deposit, other_deposit, "
        "move_in_date, lease_expiration_date, move_out_date, balance)\n"
        "- lease_charges(id, rent_roll_unit_id, report_id, property_code, charge_code, amount)\n"
        "- rent_roll_summary_groups(id, report_id, property_code, group_name, square_footage, "
        "market_rent, lease_charges, security_deposit, other_deposits, unit_count, "
        "unit_occupancy_pct, sqft_occupied_pct, balance)\n"
        "- rent_roll_charge_summary(id, report_id, property_code, charge_code, amount)\n\n"
        "Return JSON exactly like:\n"
        "{\n"
        '  "sql": "SELECT ... WHERE alias.property_code = :property_code LIMIT 50",\n'
        '  "explanation": "Short user-facing explanation.",\n'
        '  "parameters": {"property_code": "active property code placeholder"},\n'
        '  "safety_notes": ["Read-only SELECT", "Scoped to active property"]\n'
        "}"
    )
    user_prompt = (
        f"Active property: {property_name} ({property_code})\n"
        f"User message: {message}\n"
        f"SQL request: {sql_request}\n\n"
        "Draft the safest approval-ready SQL."
    )
    try:
        payload = chat_model(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = str(payload).strip()
        if not content.startswith("{"):
            match = re.search(r"\{.*\}", content, re.DOTALL)
            content = match.group(0) if match else content
        drafted = DraftedSql.model_validate_json(content)
    except Exception:
        return None

    return DraftedSql(
        sql=drafted.sql.strip(),
        explanation=drafted.explanation.strip(),
        parameters={"property_code": property_code},
        safety_notes=drafted.safety_notes
        or ["Read-only SELECT", "Scoped to the active property with :property_code"],
    )


def execute_approved_sql(settings: Settings, sql: str, property_code: str) -> tuple[str, list[dict]]:
    validation = validate_read_only_sql(sql, property_code)
    if not validation.ok or not validation.sql:
        raise ValueError(validation.error or "The SQL could not be validated.")
    return validation.sql, MySQLDatabase(settings).fetch_sql(validation.sql)
