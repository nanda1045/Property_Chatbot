"""Apply pending database migrations."""

from app.core.config import get_settings
from app.db.migrations import run_migrations


def main() -> None:
    applied = run_migrations(get_settings())
    if applied:
        print(f"Applied migrations: {', '.join(applied)}")
    else:
        print("Database is up to date.")


if __name__ == "__main__":
    main()
