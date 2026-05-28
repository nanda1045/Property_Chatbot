from __future__ import annotations

import re
from dataclasses import dataclass

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
    r"alter|call|create|delete|drop|grant|insert|load|lock|replace|revoke|set|truncate|"
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

    if candidate.count(";") > 1 or (";" in candidate and not candidate.endswith(";")):
        return SqlValidationResult(None, "Only one SQL statement can be approved.")
    candidate = candidate.rstrip(";").strip()

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


def execute_approved_sql(settings: Settings, sql: str, property_code: str) -> tuple[str, list[dict]]:
    validation = validate_read_only_sql(sql, property_code)
    if not validation.ok or not validation.sql:
        raise ValueError(validation.error or "The SQL could not be validated.")
    return validation.sql, MySQLDatabase(settings).fetch_sql(validation.sql)
