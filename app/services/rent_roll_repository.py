from __future__ import annotations

from app.db.mysql import MySQLDatabase


class RentRollRepository:
    def __init__(self, db: MySQLDatabase) -> None:
        self.db = db

    def list_properties(self) -> list[dict]:
        return self.db.fetch_all(
            """
            SELECT property_code, property_name, address, source_site
            FROM properties
            ORDER BY property_code
            """
        )

    def get_property_profile(self, property_code: str) -> dict | None:
        return self.db.fetch_one(
            """
            SELECT property_code, property_name, address, source_site
            FROM properties
            WHERE property_code = %s
            """,
            (property_code.lower(),),
        )

    def get_report_periods(self, property_code: str) -> dict:
        rows = self.db.fetch_all(
            """
            SELECT DISTINCT report_month
            FROM rent_roll_reports
            WHERE property_code = %s
            ORDER BY report_month
            """,
            (property_code.lower(),),
        )
        months = [row["report_month"] for row in rows]
        years = sorted({int(month[:4]) for month in months})
        return {
            "property_code": property_code.lower(),
            "min_report_month": months[0] if months else None,
            "max_report_month": months[-1] if months else None,
            "months": months,
            "years": years,
        }

    def get_latest_kpis(self, property_code: str) -> dict:
        return {
            "current": self.db.fetch_one(
                """
                SELECT
                  r.property_code,
                  r.report_month,
                  r.as_of_date,
                  s.group_name,
                  s.unit_count,
                  s.unit_occupancy_pct,
                  s.sqft_occupied_pct,
                  s.market_rent,
                  s.lease_charges,
                  s.security_deposit,
                  s.balance
                FROM rent_roll_reports r
                JOIN rent_roll_summary_groups s ON s.report_id = r.id
                WHERE r.property_code = %s
                  AND s.property_code = %s
                  AND s.group_name = 'Current/Notice/Vacant Residents'
                ORDER BY r.report_month DESC
                LIMIT 1
                """,
                (property_code.lower(), property_code.lower()),
            ),
            "vacant": self.db.fetch_one(
                """
                SELECT
                  r.property_code,
                  r.report_month,
                  s.group_name,
                  s.unit_count,
                  s.market_rent,
                  s.balance
                FROM rent_roll_reports r
                JOIN rent_roll_summary_groups s ON s.report_id = r.id
                WHERE r.property_code = %s
                  AND s.property_code = %s
                  AND s.group_name = 'Total Vacant Units'
                ORDER BY r.report_month DESC
                LIMIT 1
                """,
                (property_code.lower(), property_code.lower()),
            ),
        }

    def get_occupancy_trend(self, property_code: str, months: int = 12) -> list[dict]:
        return self.db.fetch_all(
            """
            SELECT *
            FROM (
              SELECT
                r.report_month,
                s.unit_count,
                s.unit_occupancy_pct,
                s.sqft_occupied_pct,
                s.market_rent,
                s.lease_charges,
                s.balance
              FROM rent_roll_reports r
              JOIN rent_roll_summary_groups s ON s.report_id = r.id
              WHERE r.property_code = %s
                AND s.property_code = %s
                AND s.group_name = 'Current/Notice/Vacant Residents'
              ORDER BY r.report_month DESC
              LIMIT %s
            ) latest
            ORDER BY report_month ASC
            """,
            (property_code.lower(), property_code.lower(), months),
        )

    def get_charge_breakdown(self, property_code: str, limit: int = 10) -> list[dict]:
        return self.db.fetch_all(
            """
            SELECT c.charge_code, c.amount, r.report_month
            FROM rent_roll_charge_summary c
            JOIN rent_roll_reports r ON r.id = c.report_id
            WHERE c.property_code = %s
              AND r.property_code = %s
              AND r.report_month = (
                SELECT MAX(report_month)
                FROM rent_roll_reports
                WHERE property_code = %s
              )
            ORDER BY ABS(c.amount) DESC
            LIMIT %s
            """,
            (property_code.lower(), property_code.lower(), property_code.lower(), limit),
        )

    def get_top_balances(self, property_code: str, limit: int = 10) -> list[dict]:
        return self.db.fetch_all(
            """
            SELECT
              u.unit,
              u.unit_type,
              u.sqft,
              u.resident_status,
              u.market_rent,
              u.balance,
              r.report_month
            FROM rent_roll_units u
            JOIN rent_roll_reports r ON r.id = u.report_id
            WHERE u.property_code = %s
              AND r.property_code = %s
              AND r.report_month = (
                SELECT MAX(report_month)
                FROM rent_roll_reports
                WHERE property_code = %s
              )
            ORDER BY u.balance DESC
            LIMIT %s
            """,
            (property_code.lower(), property_code.lower(), property_code.lower(), limit),
        )

    def get_vacant_units(self, property_code: str, limit: int = 20) -> list[dict]:
        return self.db.fetch_all(
            """
            SELECT
              u.unit,
              u.unit_type,
              u.sqft,
              u.market_rent,
              u.balance,
              r.report_month
            FROM rent_roll_units u
            JOIN rent_roll_reports r ON r.id = u.report_id
            WHERE u.property_code = %s
              AND r.property_code = %s
              AND u.resident_status = 'VACANT'
              AND r.report_month = (
                SELECT MAX(report_month)
                FROM rent_roll_reports
                WHERE property_code = %s
              )
            ORDER BY u.unit
            LIMIT %s
            """,
            (property_code.lower(), property_code.lower(), property_code.lower(), limit),
        )

    def get_rent_by_unit_type(self, property_code: str) -> list[dict]:
        return self.db.fetch_all(
            """
            SELECT
              u.unit_type,
              COUNT(*) AS unit_count,
              ROUND(AVG(u.market_rent), 2) AS avg_market_rent,
              MIN(u.market_rent) AS min_market_rent,
              MAX(u.market_rent) AS max_market_rent,
              r.report_month
            FROM rent_roll_units u
            JOIN rent_roll_reports r ON r.id = u.report_id
            WHERE u.property_code = %s
              AND r.property_code = %s
              AND r.report_month = (
                SELECT MAX(report_month)
                FROM rent_roll_reports
                WHERE property_code = %s
              )
            GROUP BY u.unit_type, r.report_month
            ORDER BY unit_count DESC, u.unit_type
            """,
            (property_code.lower(), property_code.lower(), property_code.lower()),
        )
