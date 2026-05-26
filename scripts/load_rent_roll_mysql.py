#!/usr/bin/env python3
"""Parse rent-roll spreadsheets and load structured data into MySQL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


DATA_DIR = Path("Data/RentRoll_LeaseCharges_NamesRedacted copy")
PROPERTY_SOURCES = Path("config/property_sources.json")
SCHEMA_FILE = Path("sql/schema.sql")
DEFAULT_DATABASE = "aker_chatbot"
EXCEL_EPOCH = datetime(1899, 12, 30)
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sept": 9,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
SECTION_NAMES = {
    "current/notice/vacant residents",
    "future residents/applicants",
    "non-revenue units",
    "down units",
    "model units",
    "employee units",
}
NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class UnitRow:
    property_code: str
    resident_group: str | None
    unit: str
    unit_type: str | None
    sqft: int | None
    resident_id: str | None
    resident_name: str | None
    resident_status: str | None
    market_rent: Decimal | None
    resident_deposit: Decimal | None
    other_deposit: Decimal | None
    move_in_date: date | None
    lease_expiration_date: date | None
    move_out_date: date | None
    balance: Decimal | None
    source_row_number: int
    charges: list["ChargeRow"] = field(default_factory=list)


@dataclass
class ChargeRow:
    property_code: str
    charge_code: str
    amount: Decimal | None
    source_row_number: int


@dataclass
class SummaryGroupRow:
    property_code: str
    group_name: str
    square_footage: Decimal | None
    market_rent: Decimal | None
    lease_charges: Decimal | None
    security_deposit: Decimal | None
    other_deposits: Decimal | None
    unit_count: int | None
    unit_occupancy_pct: Decimal | None
    sqft_occupied_pct: Decimal | None
    balance: Decimal | None
    source_row_number: int


@dataclass
class ChargeSummaryRow:
    property_code: str
    charge_code: str
    amount: Decimal | None
    source_row_number: int


@dataclass
class RentRollReport:
    property_code: str
    property_name: str
    report_month: date
    as_of_date: date | None
    source_file: Path
    units: list[UnitRow]
    summary_groups: list[SummaryGroupRow]
    charge_summary: list[ChargeSummaryRow]

    @property
    def source_file_hash(self) -> str:
        return hashlib.sha256(str(self.source_file).encode("utf-8")).hexdigest()


def cell_ref_to_col(ref: str) -> str:
    return re.sub(r"\d+", "", ref)


def row_number(ref: str) -> int:
    match = re.search(r"\d+", ref)
    return int(match.group()) if match else 0


def read_sheet_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as workbook:
        root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    for row in root.findall(".//a:sheetData/a:row", NS):
        values: dict[str, str] = {"_row": row.attrib.get("r", "0")}
        for cell in row.findall("a:c", NS):
            ref = cell.attrib.get("r", "")
            inline_text = cell.find("a:is/a:t", NS)
            value = cell.find("a:v", NS)
            text = inline_text.text if inline_text is not None else value.text if value is not None else ""
            values[cell_ref_to_col(ref)] = text or ""
            values["_row"] = str(row_number(ref) or values["_row"])
        rows.append(values)
    return rows


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def normalized(value: Any) -> str:
    return (clean_text(value) or "").lower()


def parse_decimal(value: Any) -> Decimal | None:
    text = clean_text(value)
    if text is None:
        return None
    text = text.replace(",", "").replace("$", "").replace("%", "")
    if text in {"-", ""}:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def parse_int(value: Any) -> int | None:
    amount = parse_decimal(value)
    return int(amount) if amount is not None else None


def parse_date_value(value: Any) -> date | None:
    text = clean_text(value)
    if text is None:
        return None
    if re.fullmatch(r"\d+(\.0+)?", text):
        return (EXCEL_EPOCH + timedelta(days=int(float(text)))).date()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_report_date(text: str | None, prefix: str) -> date | None:
    if not text:
        return None
    cleaned = text.replace(prefix, "", 1).replace("=", "").strip()
    if prefix.lower().startswith("month"):
        try:
            month, year = cleaned.split("/")
            return date(int(year), int(month), 1)
        except (ValueError, IndexError):
            return None
    return parse_date_value(cleaned)


def month_from_filename(path: Path) -> date | None:
    match = re.match(r"([A-Za-z]+)_", path.name)
    if not match:
        return None
    month = MONTHS.get(match.group(1).lower())
    return date(2025, month, 1) if month else None


def parse_property_header(value: str | None, path: Path) -> tuple[str, str]:
    text = clean_text(value) or ""
    match = re.search(r"\(([^)]+)\)", text)
    if match:
        code = match.group(1).strip().lower()
        name = text[: match.start()].strip()
        return code, name

    code = path.stem.rsplit("_", 1)[-1].lower()
    return code, text or code


def is_section_row(row: dict[str, str]) -> bool:
    return normalized(row.get("A")) in SECTION_NAMES


def is_summary_start(row: dict[str, str]) -> bool:
    return normalized(row.get("A")) == "summary groups"


def is_charge_summary_start(row: dict[str, str]) -> bool:
    return normalized(row.get("A")).startswith("summary of charges by charge code")


def is_data_unit_row(row: dict[str, str]) -> bool:
    return bool(clean_text(row.get("A"))) and bool(clean_text(row.get("B")))


def add_charge_if_present(unit: UnitRow, row: dict[str, str]) -> None:
    charge_code = clean_text(row.get("G"))
    if not charge_code or charge_code.lower() == "total":
        return
    unit.charges.append(
        ChargeRow(
            property_code=unit.property_code,
            charge_code=charge_code,
            amount=parse_decimal(row.get("H")),
            source_row_number=int(row.get("_row", "0")),
        )
    )


def parse_rent_roll(path: Path) -> RentRollReport:
    rows = read_sheet_rows(path)
    header_by_row = {int(row.get("_row", 0)): row for row in rows}
    property_code, property_name = parse_property_header(header_by_row.get(2, {}).get("A"), path)
    as_of_date = parse_report_date(header_by_row.get(3, {}).get("A"), "As Of")
    report_month = parse_report_date(header_by_row.get(4, {}).get("A"), "Month Year")
    if report_month is None:
        report_month = month_from_filename(path)
    if report_month is None:
        raise ValueError(f"Could not infer report month for {path}")

    units: list[UnitRow] = []
    summary_groups: list[SummaryGroupRow] = []
    charge_summary: list[ChargeSummaryRow] = []
    current_group: str | None = None
    current_unit: UnitRow | None = None
    mode = "detail"

    for row in rows:
        source_row_number = int(row.get("_row", "0"))
        if source_row_number <= 7:
            continue

        if is_summary_start(row):
            mode = "summary"
            current_unit = None
            continue
        if is_charge_summary_start(row):
            mode = "charge_summary"
            current_unit = None
            continue

        if mode == "detail":
            if is_section_row(row):
                current_group = clean_text(row.get("A"))
                current_unit = None
                continue

            if is_data_unit_row(row):
                resident_id = clean_text(row.get("D"))
                resident_status = "VACANT" if normalized(resident_id) == "vacant" else "OCCUPIED"
                unit = UnitRow(
                    property_code=property_code,
                    resident_group=current_group,
                    unit=clean_text(row.get("A")) or "",
                    unit_type=clean_text(row.get("B")),
                    sqft=parse_int(row.get("C")),
                    resident_id=resident_id,
                    resident_name=clean_text(row.get("E")),
                    resident_status=resident_status,
                    market_rent=parse_decimal(row.get("F")),
                    resident_deposit=parse_decimal(row.get("I")),
                    other_deposit=parse_decimal(row.get("J")),
                    move_in_date=parse_date_value(row.get("K")),
                    lease_expiration_date=parse_date_value(row.get("L")),
                    move_out_date=parse_date_value(row.get("M")),
                    balance=parse_decimal(row.get("N")),
                    source_row_number=source_row_number,
                )
                add_charge_if_present(unit, row)
                units.append(unit)
                current_unit = unit
                continue

            if current_unit is not None:
                add_charge_if_present(current_unit, row)
            continue

        if mode == "summary":
            group_name = clean_text(row.get("A"))
            if not group_name or group_name.lower() in {"summary groups", "totals:"}:
                if group_name and group_name.lower() == "totals:":
                    summary_groups.append(
                        SummaryGroupRow(
                            property_code=property_code,
                            group_name="Totals",
                            square_footage=parse_decimal(row.get("F")),
                            market_rent=parse_decimal(row.get("G")),
                            lease_charges=parse_decimal(row.get("H")),
                            security_deposit=parse_decimal(row.get("I")),
                            other_deposits=parse_decimal(row.get("J")),
                            unit_count=parse_int(row.get("K")),
                            unit_occupancy_pct=parse_decimal(row.get("L")),
                            sqft_occupied_pct=parse_decimal(row.get("M")),
                            balance=parse_decimal(row.get("N")),
                            source_row_number=source_row_number,
                        )
                    )
                continue
            if group_name in {"Square", "Footage"} or normalized(row.get("F")) == "footage":
                continue
            if group_name.startswith("("):
                continue
            summary_groups.append(
                SummaryGroupRow(
                    property_code=property_code,
                    group_name=group_name,
                    square_footage=parse_decimal(row.get("F")),
                    market_rent=parse_decimal(row.get("G")),
                    lease_charges=parse_decimal(row.get("H")),
                    security_deposit=parse_decimal(row.get("I")),
                    other_deposits=parse_decimal(row.get("J")),
                    unit_count=parse_int(row.get("K")),
                    unit_occupancy_pct=parse_decimal(row.get("L")),
                    sqft_occupied_pct=parse_decimal(row.get("M")),
                    balance=parse_decimal(row.get("N")),
                    source_row_number=source_row_number,
                )
            )
            continue

        if mode == "charge_summary":
            charge_code = clean_text(row.get("A"))
            if not charge_code or charge_code.startswith("(") or charge_code.lower() == "charge code":
                continue
            if charge_code.lower() == "total":
                continue
            charge_summary.append(
                ChargeSummaryRow(
                    property_code=property_code,
                    charge_code=charge_code,
                    amount=parse_decimal(row.get("D")),
                    source_row_number=source_row_number,
                )
            )

    return RentRollReport(
        property_code=property_code,
        property_name=property_name,
        report_month=report_month,
        as_of_date=as_of_date,
        source_file=path,
        units=units,
        summary_groups=summary_groups,
        charge_summary=charge_summary,
    )


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, Decimal):
        return str(value.quantize(Decimal("0.01")))
    if isinstance(value, (date, datetime)):
        return f"'{value.isoformat()}'"
    if isinstance(value, bool):
        return "1" if value else "0"
    text = str(value)
    text = text.replace("\\", "\\\\").replace("'", "''")
    return f"'{text}'"


def values_tuple(values: list[Any]) -> str:
    return "(" + ", ".join(sql_literal(value) for value in values) + ")"


def batched_insert(table: str, columns: list[str], rows: list[list[Any]], batch_size: int = 500) -> list[str]:
    statements = []
    if not rows:
        return statements
    column_sql = ", ".join(f"`{column}`" for column in columns)
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        values_sql = ",\n".join(values_tuple(row) for row in batch)
        statements.append(f"INSERT INTO `{table}` ({column_sql}) VALUES\n{values_sql};")
    return statements


def load_property_sources(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as source_file:
        return json.load(source_file)


def parse_reports(data_dir: Path, codes: list[str] | None) -> list[RentRollReport]:
    paths = sorted(data_dir.glob("*.xls"))
    if codes:
        allowed = {code.lower() for code in codes}
        paths = [path for path in paths if path.stem.rsplit("_", 1)[-1].lower() in allowed]
    reports = []
    for path in paths:
        reports.append(parse_rent_roll(path))
    return reports


def generate_sql(
    reports: list[RentRollReport],
    property_sources: dict[str, dict[str, Any]],
    database: str,
    reset: bool,
) -> str:
    schema_sql = SCHEMA_FILE.read_text(encoding="utf-8")
    lines = [
        f"CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;",
        f"USE `{database}`;",
        "SET NAMES utf8mb4 COLLATE utf8mb4_unicode_ci;",
    ]
    if reset:
        lines.extend(
            [
                "SET FOREIGN_KEY_CHECKS = 0;",
                "DROP TABLE IF EXISTS lease_charges;",
                "DROP TABLE IF EXISTS rent_roll_charge_summary;",
                "DROP TABLE IF EXISTS rent_roll_summary_groups;",
                "DROP TABLE IF EXISTS rent_roll_units;",
                "DROP TABLE IF EXISTS rent_roll_reports;",
                "DROP TABLE IF EXISTS properties;",
                "SET FOREIGN_KEY_CHECKS = 1;",
            ]
        )
    lines.append(schema_sql)

    property_rows = []
    seen_properties: set[str] = set()
    for report in reports:
        if report.property_code in seen_properties:
            continue
        seen_properties.add(report.property_code)
        source_config = property_sources.get(report.property_code, {})
        property_rows.append(
            [
                report.property_code,
                source_config.get("property_name") or report.property_name,
                source_config.get("address"),
                source_config.get("primary_site"),
            ]
        )
    lines.extend(
        batched_insert(
            "properties",
            ["property_code", "property_name", "address", "source_site"],
            property_rows,
        )
    )

    report_rows = [
        [
            report.property_code,
            report.report_month,
            report.as_of_date,
            str(report.source_file),
            report.source_file_hash,
            report.source_file.name,
        ]
        for report in reports
    ]
    lines.extend(
        batched_insert(
            "rent_roll_reports",
            [
                "property_code",
                "report_month",
                "as_of_date",
                "source_file",
                "source_file_hash",
                "source_filename",
            ],
            report_rows,
        )
    )

    lines.append("CREATE TEMPORARY TABLE tmp_report_ids AS SELECT id, source_file_hash FROM rent_roll_reports;")

    unit_rows = []
    for report in reports:
        for unit in report.units:
            unit_rows.append(
                [
                    report.source_file_hash,
                    unit.property_code,
                    unit.resident_group,
                    unit.unit,
                    unit.unit_type,
                    unit.sqft,
                    unit.resident_id,
                    unit.resident_name,
                    unit.resident_status,
                    unit.market_rent,
                    unit.resident_deposit,
                    unit.other_deposit,
                    unit.move_in_date,
                    unit.lease_expiration_date,
                    unit.move_out_date,
                    unit.balance,
                    unit.source_row_number,
                ]
            )
    lines.append(
        """
CREATE TEMPORARY TABLE tmp_units (
  source_file_hash CHAR(64) NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  resident_group VARCHAR(128) NULL,
  unit VARCHAR(64) NOT NULL,
  unit_type VARCHAR(128) NULL,
  sqft INT NULL,
  resident_id VARCHAR(128) NULL,
  resident_name VARCHAR(255) NULL,
  resident_status VARCHAR(64) NULL,
  market_rent DECIMAL(12,2) NULL,
  resident_deposit DECIMAL(12,2) NULL,
  other_deposit DECIMAL(12,2) NULL,
  move_in_date DATE NULL,
  lease_expiration_date DATE NULL,
  move_out_date DATE NULL,
  balance DECIMAL(12,2) NULL,
  source_row_number INT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""".strip()
    )
    lines.extend(
        batched_insert(
            "tmp_units",
            [
                "source_file_hash",
                "property_code",
                "resident_group",
                "unit",
                "unit_type",
                "sqft",
                "resident_id",
                "resident_name",
                "resident_status",
                "market_rent",
                "resident_deposit",
                "other_deposit",
                "move_in_date",
                "lease_expiration_date",
                "move_out_date",
                "balance",
                "source_row_number",
            ],
            unit_rows,
        )
    )
    lines.append(
        """
INSERT INTO rent_roll_units (
  report_id, property_code, resident_group, unit, unit_type, sqft, resident_id,
  resident_name, resident_status, market_rent, resident_deposit, other_deposit,
  move_in_date, lease_expiration_date, move_out_date, balance, source_row_number
)
SELECT
  r.id, u.property_code, u.resident_group, u.unit, u.unit_type, u.sqft, u.resident_id,
  u.resident_name, u.resident_status, u.market_rent, u.resident_deposit, u.other_deposit,
  u.move_in_date, u.lease_expiration_date, u.move_out_date, u.balance, u.source_row_number
FROM tmp_units u
JOIN tmp_report_ids r ON r.source_file_hash = u.source_file_hash;
""".strip()
    )
    lines.append(
        """
CREATE TEMPORARY TABLE tmp_unit_ids AS
SELECT u.id, r.source_file_hash, u.source_row_number
FROM rent_roll_units u
JOIN rent_roll_reports r ON r.id = u.report_id;
""".strip()
    )

    charge_rows = []
    summary_rows = []
    charge_summary_rows = []
    for report in reports:
        for unit in report.units:
            for charge in unit.charges:
                charge_rows.append(
                    [
                        report.source_file_hash,
                        unit.source_row_number,
                        charge.property_code,
                        charge.charge_code,
                        charge.amount,
                        charge.source_row_number,
                    ]
                )
        for summary in report.summary_groups:
            summary_rows.append(
                [
                    report.source_file_hash,
                    summary.property_code,
                    summary.group_name,
                    summary.square_footage,
                    summary.market_rent,
                    summary.lease_charges,
                    summary.security_deposit,
                    summary.other_deposits,
                    summary.unit_count,
                    summary.unit_occupancy_pct,
                    summary.sqft_occupied_pct,
                    summary.balance,
                    summary.source_row_number,
                ]
            )
        for charge_summary in report.charge_summary:
            charge_summary_rows.append(
                [
                    report.source_file_hash,
                    charge_summary.property_code,
                    charge_summary.charge_code,
                    charge_summary.amount,
                    charge_summary.source_row_number,
                ]
            )

    lines.append(
        """
CREATE TEMPORARY TABLE tmp_charges (
  source_file_hash CHAR(64) NOT NULL,
  unit_source_row_number INT NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  charge_code VARCHAR(64) NOT NULL,
  amount DECIMAL(12,2) NULL,
  source_row_number INT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""".strip()
    )
    lines.extend(
        batched_insert(
            "tmp_charges",
            [
                "source_file_hash",
                "unit_source_row_number",
                "property_code",
                "charge_code",
                "amount",
                "source_row_number",
            ],
            charge_rows,
        )
    )
    lines.append(
        """
INSERT INTO lease_charges (
  rent_roll_unit_id, report_id, property_code, charge_code, amount, source_row_number
)
SELECT
  u.id, r.id, c.property_code, c.charge_code, c.amount, c.source_row_number
FROM tmp_charges c
JOIN tmp_report_ids r ON r.source_file_hash = c.source_file_hash
JOIN tmp_unit_ids u
  ON u.source_file_hash = c.source_file_hash
 AND u.source_row_number = c.unit_source_row_number;
""".strip()
    )

    lines.append(
        """
CREATE TEMPORARY TABLE tmp_summary_groups (
  source_file_hash CHAR(64) NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  group_name VARCHAR(128) NOT NULL,
  square_footage DECIMAL(14,2) NULL,
  market_rent DECIMAL(14,2) NULL,
  lease_charges DECIMAL(14,2) NULL,
  security_deposit DECIMAL(14,2) NULL,
  other_deposits DECIMAL(14,2) NULL,
  unit_count INT NULL,
  unit_occupancy_pct DECIMAL(7,2) NULL,
  sqft_occupied_pct DECIMAL(7,2) NULL,
  balance DECIMAL(14,2) NULL,
  source_row_number INT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""".strip()
    )
    lines.extend(
        batched_insert(
            "tmp_summary_groups",
            [
                "source_file_hash",
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
                "source_row_number",
            ],
            summary_rows,
        )
    )
    lines.append(
        """
INSERT INTO rent_roll_summary_groups (
  report_id, property_code, group_name, square_footage, market_rent, lease_charges,
  security_deposit, other_deposits, unit_count, unit_occupancy_pct,
  sqft_occupied_pct, balance, source_row_number
)
SELECT
  r.id, s.property_code, s.group_name, s.square_footage, s.market_rent, s.lease_charges,
  s.security_deposit, s.other_deposits, s.unit_count, s.unit_occupancy_pct,
  s.sqft_occupied_pct, s.balance, s.source_row_number
FROM tmp_summary_groups s
JOIN tmp_report_ids r ON r.source_file_hash = s.source_file_hash;
""".strip()
    )

    lines.append(
        """
CREATE TEMPORARY TABLE tmp_charge_summary (
  source_file_hash CHAR(64) NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  charge_code VARCHAR(64) NOT NULL,
  amount DECIMAL(14,2) NULL,
  source_row_number INT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""".strip()
    )
    lines.extend(
        batched_insert(
            "tmp_charge_summary",
            ["source_file_hash", "property_code", "charge_code", "amount", "source_row_number"],
            charge_summary_rows,
        )
    )
    lines.append(
        """
INSERT INTO rent_roll_charge_summary (
  report_id, property_code, charge_code, amount, source_row_number
)
SELECT
  r.id, c.property_code, c.charge_code, c.amount, c.source_row_number
FROM tmp_charge_summary c
JOIN tmp_report_ids r ON r.source_file_hash = c.source_file_hash;
""".strip()
    )

    return "\n\n".join(lines) + "\n"


def print_stats(reports: list[RentRollReport]) -> None:
    units = sum(len(report.units) for report in reports)
    charges = sum(len(unit.charges) for report in reports for unit in report.units)
    summary_groups = sum(len(report.summary_groups) for report in reports)
    charge_summary = sum(len(report.charge_summary) for report in reports)
    codes = sorted({report.property_code for report in reports})
    print(f"Parsed {len(reports)} report file(s) for {len(codes)} property code(s).")
    print(f"Units: {units:,}")
    print(f"Lease charges: {charges:,}")
    print(f"Summary group rows: {summary_groups:,}")
    print(f"Charge summary rows: {charge_summary:,}")
    print("Property codes:", ", ".join(codes))


def run_mysql(sql: str, args: argparse.Namespace) -> None:
    command = [
        args.mysql_bin,
        f"--host={args.host}",
        f"--port={args.port}",
        f"--user={args.user}",
        "--protocol=tcp",
    ]
    env = os.environ.copy()
    if args.password:
        env["MYSQL_PWD"] = args.password
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False, encoding="utf-8") as sql_file:
        sql_file.write(sql)
        sql_path = sql_file.name
    try:
        with open(sql_path, "r", encoding="utf-8") as sql_input:
            subprocess.run(command, stdin=sql_input, env=env, check=True)
    finally:
        Path(sql_path).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--property-sources", default=str(PROPERTY_SOURCES))
    parser.add_argument("--database", default=os.environ.get("MYSQL_DATABASE", DEFAULT_DATABASE))
    parser.add_argument("--host", default=os.environ.get("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("MYSQL_PORT", "3306"))
    parser.add_argument("--user", default=os.environ.get("MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.environ.get("MYSQL_PASSWORD", "root"))
    parser.add_argument("--mysql-bin", default=os.environ.get("MYSQL_BIN", "mysql"))
    parser.add_argument("--codes", nargs="*", help="Optional property codes to load.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the structured tables first.")
    parser.add_argument("--dry-run", action="store_true", help="Parse files and print counts without loading MySQL.")
    parser.add_argument("--write-sql", help="Write generated SQL to this path instead of, or before, loading.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = parse_reports(Path(args.data_dir), args.codes)
    if not reports:
        print("No rent-roll files found to load.", file=sys.stderr)
        return 1

    print_stats(reports)
    if args.dry_run:
        return 0

    property_sources = load_property_sources(Path(args.property_sources))
    sql = generate_sql(reports, property_sources, args.database, args.reset)
    if args.write_sql:
        Path(args.write_sql).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_sql).write_text(sql, encoding="utf-8")
        print(f"Wrote SQL to {args.write_sql}")

    if args.write_sql and os.environ.get("SKIP_MYSQL_LOAD") == "1":
        return 0

    run_mysql(sql, args)
    print(f"Loaded structured rent-roll data into MySQL database `{args.database}`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
