"""
Phase 2, step 1: run the full model benchmark (baselines, statistical,
ML) via rolling-origin backtesting on a stratified store sample, and
persist results to `model_evaluations` plus a local CSV for inspection.

Usage:
    python -m scripts.run_phase2_benchmark
"""
from __future__ import annotations

from config.logging_config import get_logger
from database.connection import get_engine, write_dataframe
from forecasting.evaluation.run_benchmark import run_full_benchmark
from forecasting.evaluation.model_selection import summarize_model_performance, select_best_model_per_store
from config.settings import REPORTS_DIR

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Phase 2: Model Benchmark ===")
    results = run_full_benchmark(n_per_segment=15)

    engine = get_engine()
    with engine.begin() as conn:
        conn.exec_driver_sql("TRUNCATE TABLE model_evaluations RESTART IDENTITY")

    db_rows = results.rename(columns={})[
        ["store_id", "model_name", "fold", "train_end_date", "test_start_date",
         "test_end_date", "mae", "rmse", "mape", "wmape", "bias"]
    ]
    write_dataframe(db_rows, "model_evaluations", if_exists="append")

    out_dir = REPORTS_DIR / "forecasting"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "benchmark_raw_results.csv", index=False)

    overall = summarize_model_performance(results)
    overall.to_csv(out_dir / "model_comparison_summary.csv", index=False)

    per_store_best = select_best_model_per_store(results)
    per_store_best.to_csv(out_dir / "best_model_per_store.csv", index=False)

    print("\n=== Overall model comparison (avg across all stores/folds) ===")
    print(overall.to_string(index=False))

    print("\n=== Best model chosen per store (top 10 shown) ===")
    print(per_store_best.head(10).to_string(index=False))

    print(f"\nWinning model overall: {overall.iloc[0]['model_name']} "
          f"(avg WMAPE = {overall.iloc[0]['avg_wmape']:.2f}%)")


if __name__ == "__main__":
    main()
