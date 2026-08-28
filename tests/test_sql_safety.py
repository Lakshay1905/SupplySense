"""Tests for ai.sql_safety."""
from __future__ import annotations

from ai.sql_safety import validate_readonly_sql, DISALLOWED_KEYWORDS


def test_valid_select_is_accepted():
    result = validate_readonly_sql("SELECT * FROM fact_sales LIMIT 10")
    assert result.is_valid
    assert result.error is None


def test_select_without_limit_gets_limit_appended():
    result = validate_readonly_sql("SELECT store_id FROM dim_store")
    assert result.is_valid
    assert "LIMIT" in result.sanitized_query.upper()


def test_select_with_existing_limit_is_preserved():
    result = validate_readonly_sql("SELECT store_id FROM dim_store LIMIT 5")
    assert result.is_valid
    assert "LIMIT 5" in result.sanitized_query


def test_empty_query_rejected():
    result = validate_readonly_sql("")
    assert not result.is_valid
    assert "Empty" in result.error


def test_multiple_statements_rejected():
    result = validate_readonly_sql("SELECT * FROM fact_sales; DROP TABLE fact_sales;")
    assert not result.is_valid


def test_drop_table_rejected():
    result = validate_readonly_sql("DROP TABLE fact_sales")
    assert not result.is_valid


def test_delete_rejected():
    result = validate_readonly_sql("DELETE FROM fact_sales WHERE store_id = 1")
    assert not result.is_valid


def test_insert_rejected():
    result = validate_readonly_sql("INSERT INTO fact_sales VALUES (1,2,3)")
    assert not result.is_valid


def test_update_rejected():
    result = validate_readonly_sql("UPDATE fact_sales SET sales = 0")
    assert not result.is_valid


def test_alter_rejected():
    result = validate_readonly_sql("ALTER TABLE fact_sales ADD COLUMN x INT")
    assert not result.is_valid


def test_truncate_rejected():
    result = validate_readonly_sql("TRUNCATE TABLE fact_sales")
    assert not result.is_valid


def test_unknown_table_rejected():
    result = validate_readonly_sql("SELECT * FROM pg_shadow")
    assert not result.is_valid
    assert "unknown" in result.error.lower() or "disallowed" in result.error.lower()


def test_cte_query_is_accepted_and_cte_alias_not_flagged():
    result = validate_readonly_sql("WITH recent AS (SELECT * FROM fact_sales) SELECT * FROM recent")
    assert result.is_valid


def test_multi_cte_query_is_accepted():
    query = ("WITH a AS (SELECT * FROM fact_sales), b AS (SELECT * FROM dim_store) "
             "SELECT * FROM a JOIN b ON true")
    result = validate_readonly_sql(query)
    assert result.is_valid


def test_join_across_allowed_tables_is_accepted():
    result = validate_readonly_sql(
        "SELECT f.sales, s.store_type FROM fact_sales f JOIN dim_store s ON f.store_id = s.store_id"
    )
    assert result.is_valid


def test_dangerous_function_call_rejected():
    result = validate_readonly_sql("SELECT pg_sleep(10)")
    assert not result.is_valid


def test_set_config_rejected():
    result = validate_readonly_sql("SELECT set_config('x', 'y', false)")
    assert not result.is_valid


def test_all_disallowed_keywords_individually_rejected():
    for kw in DISALLOWED_KEYWORDS:
        query = f"{kw} something_that_looks_like_sql"
        result = validate_readonly_sql(query)
        assert not result.is_valid, f"Expected '{kw}' to be rejected"


def test_row_limit_is_capped_at_max():
    result = validate_readonly_sql("SELECT * FROM dim_store", row_limit=100000)
    assert result.is_valid
    assert "LIMIT 500" in result.sanitized_query  # capped at MAX_ROW_LIMIT
