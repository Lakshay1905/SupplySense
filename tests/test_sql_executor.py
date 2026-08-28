"""Tests for ai.sql_executor."""
from __future__ import annotations

import pytest

from database.connection import get_engine, table_row_count
from ai.sql_executor import execute_readonly_query


def _db_ready() -> bool:
    try:
        engine = get_engine()
        with engine.connect():
            pass
        return table_row_count("fact_sales") > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="Database not populated")


def test_execute_readonly_query_valid_select():
    result = execute_readonly_query("SELECT COUNT(*) AS n FROM dim_store")
    assert result.success
    assert result.row_count == 1
    assert result.rows[0]["n"] > 0


def test_execute_readonly_query_rejects_write_operation():
    result = execute_readonly_query("DROP TABLE dim_store")
    assert not result.success
    assert result.error is not None
    assert result.rows is None


def test_execute_readonly_query_rejects_unknown_table():
    result = execute_readonly_query("SELECT * FROM pg_shadow")
    assert not result.success


def test_execute_readonly_query_handles_sql_error_gracefully():
    result = execute_readonly_query("SELECT nonexistent_column FROM dim_store")
    assert not result.success
    assert "failed" in result.error.lower() or "column" in result.error.lower()


def test_execute_readonly_query_respects_row_limit():
    result = execute_readonly_query("SELECT * FROM fact_sales", row_limit=5)
    assert result.success
    assert result.row_count <= 5
