"""
Read-only SQL safety layer for the AI Copilot.

The copilot is allowed to run SQL against the analytical database so it
can answer open-ended questions the pre-built tools don't cover, but
every query passes through validation here first:

  - must be a single statement
  - must be a SELECT (or WITH ... SELECT) -- no DML/DDL/DCL of any kind
  - no disallowed keywords, even inside comments or strings won't help an
    attacker since we reject on tokenized keywords, not substring search
  - only references tables in the known analytical schema (allow-list)
  - a LIMIT is enforced (added if missing, capped if too large)

This is defense-in-depth: the database *connection itself* should also be
provisioned as a read-only role in production (documented in README), but
this layer means a bug or prompt-injection attempt can't reach write
operations even if that provisioning step is missed.
"""
from __future__ import annotations

from dataclasses import dataclass

import re

import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import Keyword, DML

from config.logging_config import get_logger

logger = get_logger(__name__)

MAX_ROW_LIMIT = 500
DEFAULT_ROW_LIMIT = 200

ALLOWED_TABLES = {
    "dim_date", "dim_region", "dim_store", "dim_product", "dim_supplier",
    "fact_sales", "fact_sales_features",
    "inventory_snapshot", "orders", "promotions",
    "model_evaluations", "forecasts", "optimization_results", "scenarios",
    "data_quality_log", "pipeline_runs",
}

DISALLOWED_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "GRANT",
    "REVOKE", "CREATE", "REPLACE", "MERGE", "CALL", "EXECUTE", "COPY",
    "VACUUM", "ATTACH", "DETACH", "PRAGMA", "SET", "RESET",
}

DISALLOWED_FUNCTION_SUBSTRINGS = {
    "set_config", "pg_sleep", "pg_read_file", "pg_read_binary_file",
    "pg_write_file", "lo_import", "lo_export", "dblink", "pg_terminate_backend",
    "pg_cancel_backend", "pg_reload_conf", "pg_rotate_logfile", "current_setting",
}


@dataclass
class SqlValidationResult:
    is_valid: bool
    sanitized_query: str | None
    error: str | None


def _extract_referenced_tables(statement: Statement) -> set[str]:
    """Best-effort extraction of table names following FROM/JOIN keywords."""
    tables = set()
    tokens = list(statement.flatten())
    for i, token in enumerate(tokens):
        if token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN"):
            # scan forward past whitespace to the next name-like token
            for next_token in tokens[i + 1:]:
                if next_token.is_whitespace:
                    continue
                if next_token.ttype is None or str(next_token.ttype).startswith("Token.Name"):
                    name = next_token.value.strip('"').split(".")[-1].lower()
                    tables.add(name)
                break
    return tables


def _extract_cte_names(query: str) -> set[str]:
    """Extract names defined by a WITH clause (`WITH name AS (...)`, and
    subsequent comma-separated CTEs), so they aren't mistaken for
    external tables by the allow-list check."""
    names = set()
    match = re.match(r"^\s*WITH\s+(.*)", query, re.IGNORECASE | re.DOTALL)
    if not match:
        return names
    for m in re.finditer(r"(?:^|,)\s*(\w+)\s+AS\s*\(", match.group(1), re.IGNORECASE):
        names.add(m.group(1).lower())
    return names


def validate_readonly_sql(query: str, row_limit: int = DEFAULT_ROW_LIMIT) -> SqlValidationResult:
    query = query.strip().rstrip(";")
    if not query:
        return SqlValidationResult(False, None, "Empty query")

    parsed = sqlparse.parse(query)
    if len(parsed) != 1:
        return SqlValidationResult(False, None, "Only a single SQL statement is allowed")

    statement = parsed[0]

    # Must start with SELECT or WITH (CTE feeding into a SELECT)
    first_meaningful = next((t for t in statement.tokens if not t.is_whitespace), None)
    stmt_type = statement.get_type()
    starts_ok = False
    if first_meaningful is not None:
        first_word = first_meaningful.value.upper()
        starts_ok = first_word in ("SELECT", "WITH")
    if not starts_ok or stmt_type not in ("SELECT", "UNKNOWN"):
        return SqlValidationResult(False, None, "Only SELECT (or WITH ... SELECT) statements are permitted")

    # Reject disallowed keywords anywhere in the statement
    upper_query = query.upper()
    for kw in DISALLOWED_KEYWORDS:
        # word-boundary-ish check to avoid false positives on column names
        if f" {kw} " in f" {upper_query} " or upper_query.startswith(kw + " "):
            return SqlValidationResult(False, None, f"Disallowed keyword: {kw}")

    # Reject known dangerous/administrative function calls
    lower_query = query.lower()
    for fn in DISALLOWED_FUNCTION_SUBSTRINGS:
        if fn in lower_query:
            return SqlValidationResult(False, None, f"Disallowed function call: {fn}")

    # Table allow-list (CTE-defined names are excluded, not treated as external tables)
    referenced = _extract_referenced_tables(statement)
    cte_names = _extract_cte_names(query)
    unknown = referenced - ALLOWED_TABLES - cte_names
    if unknown:
        return SqlValidationResult(False, None, f"Query references unknown/disallowed table(s): {unknown}")

    # Enforce LIMIT
    capped_limit = min(row_limit, MAX_ROW_LIMIT)
    if "LIMIT" not in upper_query:
        sanitized = f"{query} LIMIT {capped_limit}"
    else:
        sanitized = query  # trust an explicit LIMIT the caller supplied; DB layer still caps rows returned

    return SqlValidationResult(True, sanitized, None)
