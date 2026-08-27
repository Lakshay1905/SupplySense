"""
Database connection management for SupplySense.

Provides a single SQLAlchemy engine shared across the app, plus small
helpers for running raw SQL and bulk-loading DataFrames. All connection
parameters come from `config.settings` (env vars) -- nothing is hardcoded.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None


def get_engine() -> Engine:
    """Return a process-wide SQLAlchemy engine (created lazily)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            future=True,
        )
        logger.info("Created SQLAlchemy engine for %s:%s/%s",
                    settings.db_host, settings.db_port, settings.db_name)
    return _engine


SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    """Context-managed DB session with automatic commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def execute_sql_file(path: str) -> None:
    """Execute a .sql file (DDL), which may contain multiple statements."""
    engine = get_engine()
    with open(path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.execute(sql_text)
        raw_conn.commit()
    finally:
        raw_conn.close()
    logger.info("Executed DDL script %s", path)


def read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    """Read a SQL query into a DataFrame. Read-only helper for analytics/AI."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})


def bulk_copy_dataframe(df: pd.DataFrame, table_name: str) -> int:
    """Fast bulk load using PostgreSQL COPY. Table must already exist and be
    empty/truncated -- this appends via COPY, which is far faster than
    row-by-row or multi-row INSERT for large fact tables (>100k rows)."""
    import io
    engine = get_engine()
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    columns = ",".join(df.columns)
    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {table_name} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')",
                buf,
            )
        raw_conn.commit()
    finally:
        raw_conn.close()
    logger.info("Bulk-copied %d rows into '%s'", len(df), table_name)
    return len(df)


def write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "append",
    chunksize: int = 5000,
) -> int:
    """Bulk-write a DataFrame to a table. Returns number of rows written."""
    engine = get_engine()
    df.to_sql(
        table_name,
        engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    logger.info("Wrote %d rows to table '%s' (if_exists=%s)", len(df), table_name, if_exists)
    return len(df)


def table_row_count(table_name: str) -> int:
    df = read_sql(f"SELECT COUNT(*) AS n FROM {table_name}")
    return int(df["n"].iloc[0])
