"""
End-to-end Phase 1 pipeline orchestrator.

    Raw CSVs -> Validation -> Cleaning -> Transformation (star schema)
    -> Feature Engineering -> Load to PostgreSQL -> EDA report

Usage:
    python -m scripts.run_phase1_pipeline
"""
from __future__ import annotations

import uuid
import sys
from datetime import datetime


from config.logging_config import get_logger
from database.connection import get_engine, write_dataframe, bulk_copy_dataframe
from database.init_db import create_schema
from pipelines.ingestion.load_raw_data import load_all
from pipelines.validation.validators import (
    validate_dataframe, validate_referential_integrity, validate_anomalies,
)
from pipelines.transformation.clean import clean_train, clean_store, clean_store_states
from pipelines.transformation.transform import (
    build_dim_date, build_dim_region, build_dim_store, build_fact_sales,
)
from analytics.features.feature_engineering import build_feature_table
from analytics.metrics.data_quality import reports_to_log_rows, summarize_report
from analytics.eda.eda_report import run_eda
from data.schemas.raw_schema import TRAIN_SCHEMA, STORE_SCHEMA, STORE_STATES_SCHEMA

logger = get_logger(__name__)


def main() -> None:
    run_id = f"phase1_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    logger.info("=== Starting Phase 1 pipeline run: %s ===", run_id)

    engine = get_engine()
    create_schema()

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO pipeline_runs (run_id, started_at, status) VALUES (%s, NOW(), %s)",
            (run_id, "running"),
        )

    all_reports = []

    # ---------------- 1. INGESTION ----------------
    logger.info("Step 1/6: Ingestion")
    raw = load_all()
    train_raw, store_raw, states_raw = raw["train"], raw["store"], raw["store_states"]
    rows_ingested = len(train_raw) + len(store_raw) + len(states_raw)

    # ---------------- 2. VALIDATION (pre-clean, on raw data) ----------------
    logger.info("Step 2/6: Validation (raw)")
    report_train = validate_dataframe(train_raw, TRAIN_SCHEMA, "raw_train",
                                       dedupe_subset=["Store", "Date"])
    report_store = validate_dataframe(store_raw, STORE_SCHEMA, "raw_store",
                                       dedupe_subset=["Store"])
    report_states = validate_dataframe(states_raw, STORE_STATES_SCHEMA, "raw_store_states",
                                        dedupe_subset=["Store"])
    report_ref = validate_referential_integrity(train_raw, store_raw)
    report_anomaly = validate_anomalies(train_raw, ["Sales", "Customers"])
    all_reports += [report_train, report_store, report_states, report_ref, report_anomaly]

    for r in all_reports:
        s = summarize_report(r)
        logger.info("  [%s] checked=%d failed=%d pass=%d warn=%d fail=%d",
                     s.stage, s.records_checked, s.records_failed, s.n_pass, s.n_warn, s.n_fail)

    hard_failures = [r for r in [report_ref] if r.has_failures]
    if hard_failures:
        logger.error("Hard validation failures detected in referential integrity -- aborting load")
        sys.exit(1)

    # ---------------- 3. CLEANING ----------------
    logger.info("Step 3/6: Cleaning")
    train_clean, train_stats = clean_train(train_raw)
    store_clean, store_stats = clean_store(store_raw)
    states_clean, states_stats = clean_store_states(states_raw)
    logger.info("Cleaning stats -- train: %s", train_stats)
    logger.info("Cleaning stats -- store: %s", store_stats)

    # Post-clean validation to confirm cleaning actually fixed issues
    report_train_clean = validate_dataframe(train_clean, TRAIN_SCHEMA, "clean_train")
    all_reports.append(report_train_clean)

    # ---------------- 4. TRANSFORMATION (star schema) ----------------
    logger.info("Step 4/6: Transformation into dimensional model")
    dim_date = build_dim_date(train_clean["Date"])
    dim_region = build_dim_region(states_clean)
    dim_store = build_dim_store(store_clean, states_clean, dim_region)
    fact_sales = build_fact_sales(train_clean)

    # ---------------- 5. FEATURE ENGINEERING ----------------
    logger.info("Step 5/6: Feature engineering")
    features = build_feature_table(fact_sales, dim_store)

    # ---------------- 6. LOAD TO POSTGRES ----------------
    logger.info("Step 6/6: Loading to PostgreSQL")
    with engine.begin() as conn:
        for table in ["fact_sales_features", "fact_sales", "dim_store", "dim_region", "dim_date"]:
            conn.exec_driver_sql(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")

    write_dataframe(dim_region.rename(columns=str), "dim_region", if_exists="append")
    write_dataframe(dim_date, "dim_date", if_exists="append")
    write_dataframe(dim_store, "dim_store", if_exists="append")

    # Large fact tables use COPY for speed (orders of magnitude faster than
    # row-by-row / multi-value INSERT for 1M+ rows).
    rows_loaded = bulk_copy_dataframe(fact_sales, "fact_sales")
    bulk_copy_dataframe(features, "fact_sales_features")

    # ---------------- DATA QUALITY LOG ----------------
    dq_rows = reports_to_log_rows(run_id, all_reports)
    if len(dq_rows):
        write_dataframe(dq_rows, "data_quality_log", if_exists="append")

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "UPDATE pipeline_runs SET finished_at = NOW(), status = %s, "
            "rows_ingested = %s, rows_loaded = %s, notes = %s WHERE run_id = %s",
            ("success", rows_ingested, rows_loaded, "Phase 1 data foundation build", run_id),
        )

    # ---------------- EDA ----------------
    logger.info("Running EDA / demand pattern analysis")
    eda_summary = run_eda(fact_sales, dim_store, dim_region, features)

    logger.info("=== Phase 1 pipeline run %s complete ===", run_id)
    print(f"\nPipeline run '{run_id}' completed successfully.")
    print(f"  fact_sales rows loaded:          {rows_loaded:,}")
    print(f"  stores:                          {dim_store.shape[0]:,}")
    print(f"  regions:                         {dim_region.shape[0]:,}")
    print(f"  calendar days:                   {dim_date.shape[0]:,}")
    print(f"  date range:                      {eda_summary['overview']['date_min']} to "
          f"{eda_summary['overview']['date_max']}")
    print(f"  data quality checks logged:      {len(dq_rows):,}")


if __name__ == "__main__":
    main()
