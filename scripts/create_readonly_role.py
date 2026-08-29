"""
Runs database/create_readonly_role.sql using the same SQLAlchemy engine
the rest of the application already uses (built from .env via
config.settings) -- this avoids needing a separate, manually-typed `psql`
session with its own credential handling.

Usage:
    python -m scripts.create_readonly_role
"""
from __future__ import annotations

from pathlib import Path

from config.logging_config import get_logger
from database.connection import execute_sql_file

logger = get_logger(__name__)

SQL_PATH = Path(__file__).parent.parent / "database" / "create_readonly_role.sql"


def main() -> None:
    execute_sql_file(str(SQL_PATH))
    print(f"Executed {SQL_PATH}. The 'supplysense_readonly' role is ready.")
    print("Run `pytest tests/test_readonly_role.py -v` to verify it.")


if __name__ == "__main__":
    main()
