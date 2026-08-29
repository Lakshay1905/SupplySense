"""
Initialize (or reset) the SupplySense PostgreSQL schema.

Usage:
    python -m database.init_db            # create tables if not exist
    python -m database.init_db --reset    # drop + recreate everything
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import text

from config.logging_config import get_logger
from database.connection import get_engine, execute_sql_file

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Drop order matters because of FK constraints (children first).
DROP_ORDER = [
    "pipeline_runs", "data_quality_log", "scenarios", "optimization_results",
    "forecasts", "model_evaluations", "promotions", "orders",
    "inventory_snapshot", "fact_sales_features", "fact_sales",
    "dim_supplier", "dim_product", "dim_date", "dim_store", "dim_region",
]


def reset_schema() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for table in DROP_ORDER:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    logger.info("Dropped %d existing tables", len(DROP_ORDER))


def create_schema() -> None:
    execute_sql_file(str(SCHEMA_PATH))
    logger.info("Schema created/verified from %s", SCHEMA_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize SupplySense database schema")
    parser.add_argument("--reset", action="store_true", help="Drop all tables before creating")
    args = parser.parse_args()

    if args.reset:
        reset_schema()

    create_schema()
    print("Database schema ready.")


if __name__ == "__main__":
    main()
