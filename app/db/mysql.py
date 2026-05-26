from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import mysql.connector

from app.core.config import Settings


def json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value


def row_to_json(row: dict[str, Any]) -> dict[str, Any]:
    return {key: json_safe(value) for key, value in row.items()}


class MySQLDatabase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        connection = mysql.connector.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            database=self.settings.mysql_database,
        )
        try:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(query, params)
                return [row_to_json(row) for row in cursor.fetchall()]
            finally:
                cursor.close()
        finally:
            connection.close()

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self.fetch_all(query, params)
        return rows[0] if rows else None
