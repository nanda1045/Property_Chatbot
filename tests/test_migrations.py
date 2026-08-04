from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.core.config import Settings
from app.db.migrations import run_migrations


class MigrationRunnerTests(unittest.TestCase):
    def test_pending_migration_is_applied_and_recorded(self) -> None:
        connection = MagicMock()
        cursor = connection.cursor.return_value
        cursor.fetchall.return_value = []

        with tempfile.TemporaryDirectory() as directory:
            migration = Path(directory) / "001_test.sql"
            migration.write_text(
                "CREATE TABLE first_table (id INT);\n"
                "CREATE TABLE second_table (id INT);\n",
                encoding="utf-8",
            )
            with patch(
                "app.db.migrations.mysql.connector.connect",
                return_value=connection,
            ):
                applied = run_migrations(
                    Settings(_env_file=None),
                    migrations_path=Path(directory),
                )

        self.assertEqual(applied, ["001_test.sql"])
        executed_sql = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("CREATE TABLE first_table" in sql for sql in executed_sql))
        self.assertTrue(any("CREATE TABLE second_table" in sql for sql in executed_sql))
        self.assertTrue(any("INSERT INTO schema_migrations" in sql for sql in executed_sql))


if __name__ == "__main__":
    unittest.main()
