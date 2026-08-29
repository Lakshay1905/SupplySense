"""
Integration test for database/create_readonly_role.sql -- verifies that,
if the read-only role has been provisioned, it can read but genuinely
cannot write or perform DDL. This is the defense-in-depth layer beneath
the application-level SQL safety checks in ai/sql_safety.py.

Skipped automatically if the role hasn't been created (it's an optional,
documented production hardening step, not required for the app to run).
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from config.settings import settings
from database.connection import get_engine


READONLY_USER = "supplysense_readonly"
READONLY_PASSWORD = "change_me_in_production"  # matches the script's default


def _readonly_role_exists() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = :role"),
                {"role": READONLY_USER},
            )
            return result.first() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _readonly_role_exists(),
    reason="supplysense_readonly role not provisioned (run database/create_readonly_role.sql to test this)",
)


def _readonly_engine():
    url = (
        f"postgresql+psycopg2://{READONLY_USER}:{READONLY_PASSWORD}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )
    return create_engine(url)


def test_readonly_role_can_select():
    engine = _readonly_engine()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) AS n FROM fact_sales"))
        row = result.first()
        assert row is not None
        assert row[0] >= 0


def test_readonly_role_cannot_delete():
    engine = _readonly_engine()
    with engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="permission denied"):
            conn.execute(text("DELETE FROM fact_sales WHERE store_id = 1"))


def test_readonly_role_cannot_insert():
    engine = _readonly_engine()
    with engine.connect() as conn:
        with pytest.raises(ProgrammingError, match="permission denied"):
            conn.execute(text(
                "INSERT INTO dim_region (state_code, state_name) VALUES ('ZZ', 'Fake')"
            ))


def test_readonly_role_cannot_drop_table():
    engine = _readonly_engine()
    with engine.connect() as conn:
        with pytest.raises(ProgrammingError):
            conn.execute(text("DROP TABLE fact_sales"))
