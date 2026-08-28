"""
Executes AI-copilot-issued SQL only after passing it through
ai.sql_safety.validate_readonly_sql, and catches/reports DB errors
gracefully instead of letting them surface as raw exceptions to the LLM
(or the user).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai.sql_safety import validate_readonly_sql
from database.connection import read_sql
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SqlExecutionResult:
    success: bool
    rows: list[dict] | None
    row_count: int
    error: str | None
    query_executed: str | None


def execute_readonly_query(query: str, row_limit: int = 200) -> SqlExecutionResult:
    validation = validate_readonly_sql(query, row_limit=row_limit)
    if not validation.is_valid:
        logger.warning("AI copilot SQL rejected: %s | query=%s", validation.error, query)
        return SqlExecutionResult(False, None, 0, validation.error, None)

    try:
        df: pd.DataFrame = read_sql(validation.sanitized_query)
    except Exception as exc:  # noqa: BLE001
        logger.error("AI copilot SQL execution failed: %s | query=%s", exc, validation.sanitized_query)
        return SqlExecutionResult(False, None, 0, f"Query execution failed: {exc}", validation.sanitized_query)

    records = df.to_dict(orient="records")
    return SqlExecutionResult(True, records, len(records), None, validation.sanitized_query)
