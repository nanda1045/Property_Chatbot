from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.db.mysql import MySQLDatabase

ALLOWED_SQL_TABLES = {
    "properties",
    "rent_roll_reports",
    "rent_roll_units",
    "lease_charges",
    "rent_roll_summary_groups",
    "rent_roll_charge_summary",
}

ALLOWED_SQL_COLUMNS: dict[str, set[str]] = {
    "properties": {
        "property_code",
        "property_name",
        "address",
        "source_site",
    },
    "rent_roll_reports": {
        "id",
        "property_code",
        "report_month",
        "as_of_date",
        "source_filename",
    },
    "rent_roll_units": {
        "id",
        "report_id",
        "property_code",
        "resident_group",
        "unit",
        "unit_type",
        "sqft",
        "resident_status",
        "market_rent",
        "resident_deposit",
        "other_deposit",
        "move_in_date",
        "lease_expiration_date",
        "move_out_date",
        "balance",
    },
    "lease_charges": {
        "id",
        "rent_roll_unit_id",
        "report_id",
        "property_code",
        "charge_code",
        "amount",
    },
    "rent_roll_summary_groups": {
        "id",
        "report_id",
        "property_code",
        "group_name",
        "square_footage",
        "market_rent",
        "lease_charges",
        "security_deposit",
        "other_deposits",
        "unit_count",
        "unit_occupancy_pct",
        "sqft_occupied_pct",
        "balance",
    },
    "rent_roll_charge_summary": {
        "id",
        "report_id",
        "property_code",
        "charge_code",
        "amount",
    },
}

SQL_SCHEMA_DESCRIPTION = """
Available MySQL schema:

properties:
- property_code
- property_name
- address
- source_site

rent_roll_reports:
- id
- property_code
- report_month
- as_of_date
- source_filename

rent_roll_units:
- id
- report_id
- property_code
- resident_group
- unit
- unit_type
- sqft
- resident_status
- market_rent
- resident_deposit
- other_deposit
- move_in_date
- lease_expiration_date
- move_out_date
- balance

lease_charges:
- id
- rent_roll_unit_id
- report_id
- property_code
- charge_code
- amount

rent_roll_summary_groups:
- id
- report_id
- property_code
- group_name
- square_footage
- market_rent
- lease_charges
- security_deposit
- other_deposits
- unit_count
- unit_occupancy_pct
- sqft_occupied_pct
- balance

rent_roll_charge_summary:
- id
- report_id
- property_code
- charge_code
- amount

Unavailable data:
- crime rate
- school ratings
- Google reviews
- resident reviews
- resident satisfaction
- maintenance tickets
- demographics
- NOI
- cap rate
- market comps
- external neighborhood statistics
- resident names, emails, phone numbers, SSNs, or other PII
"""

FORBIDDEN_SQL_RE = re.compile(
    r"\b("
    r"alter|call|create|delete|drop|grant|insert|load|lock|merge|replace|revoke|"
    r"set|truncate|unlock|update|use"
    r")\b",
    re.IGNORECASE,
)
TABLE_ALIAS_RE = re.compile(
    r"\b(?:from|join)\s+`?([a-zA-Z_][a-zA-Z0-9_]*)`?"
    r"(?:\s+(?:as\s+)?`?([a-zA-Z_][a-zA-Z0-9_]*)`?)?",
    re.IGNORECASE,
)
QUALIFIED_IDENTIFIER_RE = re.compile(
    r"\b`?([a-zA-Z_][a-zA-Z0-9_]*)`?\s*\.\s*`?([a-zA-Z_][a-zA-Z0-9_]*)`?\b"
)
LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)

SQL_KEYWORDS = {
    "and",
    "as",
    "asc",
    "avg",
    "between",
    "by",
    "case",
    "cast",
    "coalesce",
    "count",
    "cross",
    "date",
    "desc",
    "distinct",
    "else",
    "end",
    "from",
    "group",
    "having",
    "ifnull",
    "in",
    "inner",
    "is",
    "join",
    "left",
    "limit",
    "max",
    "min",
    "natural",
    "not",
    "null",
    "on",
    "or",
    "order",
    "right",
    "round",
    "select",
    "sum",
    "then",
    "where",
    "when",
    "with",
}
SQL_FUNCTIONS = {"avg", "count", "sum", "max", "min", "round", "coalesce", "ifnull", "date"}
PII_TERMS = {
    "resident_name",
    "tenant_name",
    "name",
    "email",
    "phone",
    "ssn",
    "social_security",
    "dob",
}


@dataclass(frozen=True)
class SqlValidationResult:
    sql: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.sql is not None and self.error is None


class DraftedSql(BaseModel):
    answerable: bool
    unavailable_reason: str | None = None
    sql: str | None = None
    explanation: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    safety_notes: list[str] = Field(default_factory=list)


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


def _contains_pii_identifier(sql: str) -> bool:
    lowered = sql.lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in PII_TERMS)


def _validate_tables(masked_sql: str) -> tuple[bool, str]:
    scopes = _table_scopes(masked_sql)
    if not scopes:
        return False, "The SQL must read from at least one approved table."

    disallowed = sorted({table for table, _alias in scopes if table not in ALLOWED_SQL_TABLES})
    if disallowed:
        return False, f"The SQL references unsupported table(s): {', '.join(disallowed)}."

    return True, "ok"


def _validate_qualified_columns(masked_sql: str) -> tuple[bool, str]:
    scopes = _table_scopes(masked_sql)
    alias_to_table = {alias: table for table, alias in scopes}
    alias_to_table.update({table: table for table, _alias in scopes})

    for match in QUALIFIED_IDENTIFIER_RE.finditer(masked_sql):
        qualifier = match.group(1).lower()
        column = match.group(2).lower()

        table = alias_to_table.get(qualifier)
        if not table:
            continue

        if column not in ALLOWED_SQL_COLUMNS.get(table, set()):
            return False, f"Column `{column}` is not allowed on table `{table}`."

    return True, "ok"


def _has_property_placeholder_filter(masked_sql: str) -> bool:
    return bool(
        re.search(
            r"(?:\b`?[a-zA-Z_][a-zA-Z0-9_]*`?\s*\.\s*)?"
            r"`?property_code`?\s*=\s*:property_code\b",
            masked_sql,
            re.IGNORECASE,
        )
    )


def _has_hardcoded_property_code(sql: str) -> bool:
    return bool(
        re.search(
            r"(?:\b`?[a-zA-Z_][a-zA-Z0-9_]*`?\s*\.\s*)?"
            r"`?property_code`?\s*=\s*(?P<quote>['\"])[^'\"]+(?P=quote)",
            sql,
            re.IGNORECASE,
        )
    )


def _is_aggregate_query(masked_sql: str) -> bool:
    lowered = masked_sql.lower()
    return " group by " in f" {lowered} " or bool(
        re.search(r"\b(count|sum|avg|min|max)\s*\(", lowered)
    )


def _validate_limit(masked_sql: str) -> tuple[bool, str]:
    if _is_aggregate_query(masked_sql):
        return True, "ok"

    limit_match = LIMIT_RE.search(masked_sql)
    if not limit_match:
        return False, "Row-level SQL drafts must include LIMIT <= 50."

    if int(limit_match.group(1)) > 50:
        return False, "Row-level SQL drafts must use LIMIT <= 50."

    return True, "ok"


def validate_drafted_sql(sql: str | None) -> tuple[bool, str]:
    """Validate an LLM-drafted approval query before it is shown in the UI.

    This function expects the SQL to still contain the placeholder
    `:property_code`. At this stage the SQL is only a proposal; it must not be
    executable yet because the backend has not injected the active property code.
    """
    if not sql or not sql.strip():
        return False, "The drafted SQL was empty."

    candidate = sql.strip()
    masked_sql = _mask_comments_and_string_literals(candidate)

    if ";" in candidate:
        return False, "Semicolons and multiple statements are not allowed."
    if "--" in candidate or "/*" in candidate or "*/" in candidate:
        return False, "SQL comments are not allowed."
    if not re.match(r"^(select|with)\b", candidate, re.IGNORECASE):
        return False, "Only read-only SELECT or WITH queries can be drafted."
    if FORBIDDEN_SQL_RE.search(masked_sql):
        return False, "The SQL contains a forbidden operation."
    if re.search(r"\bor\s+1\s*=\s*1\b", masked_sql, re.IGNORECASE):
        return False, "Always-true OR predicates are not allowed."
    if re.search(r"\bunion\b", masked_sql, re.IGNORECASE):
        return False, "UNION queries are not allowed for approval drafts."
    if _contains_pii_identifier(masked_sql):
        return False, "PII fields cannot be selected or referenced."
    if "property_code" not in masked_sql.lower():
        return False, "The SQL must filter by property_code."
    if ":property_code" not in candidate:
        return False, "The SQL must use the :property_code placeholder."
    if _has_hardcoded_property_code(candidate):
        return False, "The SQL must not hardcode a property code."

    ok, message = _validate_tables(masked_sql)
    if not ok:
        return False, message

    ok, message = _validate_qualified_columns(masked_sql)
    if not ok:
        return False, message

    if not _has_property_placeholder_filter(masked_sql):
        return False, "Missing property_code = :property_code filter."

    ok, message = _validate_limit(masked_sql)
    if not ok:
        return False, message

    return True, "ok"


def _bind_property_code(sql: str, property_code: str) -> str:
    safe_code = property_code.lower().replace("'", "''")
    return sql.replace(":property_code", f"'{safe_code}'")


def bind_property_code_for_execution(sql: str, property_code: str) -> str:
    """Inject the backend-controlled property code after approval validation."""
    return _bind_property_code(sql, property_code)


def validate_read_only_sql(
    sql: str,
    property_code: str,
    max_rows: int = 100,
) -> SqlValidationResult:
    """Validate and bind an approved read-only query immediately before execution.

    The approval card stores SQL with `:property_code`; this final gate reuses
    draft validation, injects the active property code server-side, and clamps any
    row limit before MySQL execution.
    """
    is_valid, reason = validate_drafted_sql(sql)
    if not is_valid:
        return SqlValidationResult(None, reason)

    bound_sql = bind_property_code_for_execution(sql.strip(), property_code)

    limit_match = LIMIT_RE.search(bound_sql)
    if limit_match and int(limit_match.group(1)) > max_rows:
        bound_sql = LIMIT_RE.sub(f"LIMIT {max_rows}", bound_sql)

    return SqlValidationResult(bound_sql)


def draft_sql_for_approval(
    message: str,
    property_code: str,
    property_name: str,
    sql_request: str,
    chat_model: Callable[[list[dict[str, str]]], str],
) -> DraftedSql | None:
    """Ask the LLM for a non-executed SQL draft to show as an approval card."""
    system_prompt = (
        "You are a schema-aware SQL drafter for a property-scoped rent-roll assistant.\n"
        "Return only valid JSON. Do not answer the user directly.\n\n"
        "First decide whether the user question is answerable using ONLY the provided "
        "schema. If required tables/columns are missing, return answerable=false.\n\n"
        "Rules:\n"
        "- Never execute SQL. Only draft SQL for approval.\n"
        "- If the request needs external data, reviews, crime, schools, resident "
        "satisfaction, maintenance, cap rate, NOI, demographics, market comps, or PII, "
        "return answerable=false.\n"
        "- If answerable=true, generate one SELECT-only MySQL query.\n"
        "- Use only allowed tables and columns from the schema.\n"
        "- Always scope to active property using property_code = :property_code.\n"
        "- Do not hardcode the property code.\n"
        "- Use latest report_month by default for snapshot/ranking questions unless the "
        "user asks for a specific period.\n"
        "- For latest unit-level rows, use rent_roll_units.report_id joined or filtered "
        "through rent_roll_reports so latest report month is scoped to the active property.\n"
        "- Add LIMIT <= 50 for row-level queries.\n"
        "- Do not include semicolons, comments, UNION, write operations, temp tables, or "
        "stored procedures.\n\n"
        f"{SQL_SCHEMA_DESCRIPTION}\n\n"
        "Return JSON exactly like one of these:\n"
        "{\n"
        '  "answerable": true,\n'
        '  "unavailable_reason": null,\n'
        '  "sql": "SELECT ... WHERE table.property_code = :property_code LIMIT 10",\n'
        '  "explanation": "Short user-facing explanation.",\n'
        '  "parameters": {"property_code": "active property code placeholder"},\n'
        '  "safety_notes": ["Read-only SELECT", "Scoped to active property"]\n'
        "}\n\n"
        "{\n"
        '  "answerable": false,\n'
        '  "unavailable_reason": "The requested data is not present in the current schema.",\n'
        '  "sql": null,\n'
        '  "explanation": "Short explanation.",\n'
        '  "parameters": {},\n'
        '  "safety_notes": []\n'
        "}"
    )
    user_prompt = (
        f"Active property: {property_name} ({property_code})\n"
        f"User message: {message}\n"
        f"SQL request: {sql_request}\n\n"
        "Decide if answerable from schema. If answerable, draft approval-ready SQL."
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

    if not drafted.answerable:
        return DraftedSql(
            answerable=False,
            unavailable_reason=(
                drafted.unavailable_reason
                or "This question is not answerable from the current MySQL schema."
            ),
            sql=None,
            explanation=drafted.explanation or "The requested data is unavailable.",
            parameters={},
            safety_notes=[],
        )

    return DraftedSql(
        answerable=True,
        unavailable_reason=None,
        sql=(drafted.sql or "").strip(),
        explanation=drafted.explanation.strip(),
        parameters={"property_code": property_code},
        safety_notes=drafted.safety_notes
        or ["Read-only SELECT", "Scoped to the active property with :property_code"],
    )


def execute_approved_sql(
    settings: Settings,
    sql: str,
    property_code: str,
) -> tuple[str, list[dict]]:
    validation = validate_read_only_sql(sql, property_code)
    if not validation.ok or not validation.sql:
        raise ValueError(validation.error or "The SQL could not be validated.")
    return validation.sql, MySQLDatabase(settings).fetch_sql(validation.sql)
