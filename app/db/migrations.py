"""Small, deterministic SQL migration runner for the existing MySQL database."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mysql.connector

from app.core.config import Settings

MIGRATIONS_PATH = Path(__file__).resolve().parents[2] / "sql" / "migrations"

CREATE_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version VARCHAR(255) PRIMARY KEY,
  checksum CHAR(64) NOT NULL,
  applied_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def _statements(sql: str) -> list[str]:
    lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


def run_migrations(settings: Settings, migrations_path: Path = MIGRATIONS_PATH) -> list[str]:
    """Apply pending `.sql` files in filename order and return applied versions."""
    connection = mysql.connector.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )
    applied_now: list[str] = []
    try:
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(CREATE_MIGRATIONS_TABLE_SQL)
            connection.commit()
            cursor.execute("SELECT version, checksum FROM schema_migrations")
            applied = {row["version"]: row["checksum"] for row in cursor.fetchall()}

            for migration in sorted(migrations_path.glob("*.sql")):
                sql = migration.read_text(encoding="utf-8")
                checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                existing_checksum = applied.get(migration.name)
                if existing_checksum:
                    if existing_checksum != checksum:
                        raise RuntimeError(
                            f"applied migration checksum changed: {migration.name}"
                        )
                    continue

                for statement in _statements(sql):
                    cursor.execute(statement)
                cursor.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                    (migration.name, checksum),
                )
                connection.commit()
                applied_now.append(migration.name)
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
    finally:
        connection.close()
    return applied_now
