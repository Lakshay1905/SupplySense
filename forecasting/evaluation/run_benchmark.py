"""
Benchmark runner.

Runs rolling-origin backtesting across baselines, statistical models, and
ML models on a representative, stratified sample of stores (full-scale
SARIMA/Holt-Winters per-store fitting across all 1,115 stores is not a
sensible use of compute for a periodic batch job -- see module docstring
in forecasting/statistical/statistical_models.py). The winning model
family is then applied at full scale in scripts/run_phase2_forecasting.py.

Sampling is stratified by demand_segment and store_type so the comparison
reflects the actual mix of demand patterns in the business, not just the
easy-to-forecast stable majority.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from database.connection import read_sql
from forecasting.baselines.baseline_models import (
    naive_forecast, seasonal_naive_forecast, moving_average_forecast,
)
from forecasting.statistical.statistical_models import holt_winters_forecast, sarima_forecast
from forecasting.ml.ml_models import prepare_ml_features, train_random_forest, train_xgboost
from forecasting.evaluation.backtesting import generate_rolling_folds
from forecasting.evaluation.metrics import compute_all_metrics

logger = get_logger(__name__)

HORIZON = 42          # matches Rossmann's original 6-week holdout
N_FOLDS = 3
SEASON_LENGTH = 7


def get_stratified_sample(n_per_segment: int = 15, seed: int = 42) -> list[int]:
    """Stratified sample of store_ids across demand_segment x store_type."""
    df = read_sql("""
        SELECT DISTINCT f.store_id, f.demand_segment, s.store_type
        FROM fact_sales_features f
        JOIN dim_store s ON f.store_id = s.store_id
    """)
    rng = np.random.default_rng(seed)
    sampled = []
    for (segment, store_type), group in df.groupby(["demand_segment", "store_type"]):
        n = min(len(group), max(1, n_per_segment // df["store_type"].nunique()))
        sampled.extend(rng.choice(group["store_id"].values, size=n, replace=False).tolist())
    sampled = sorted(set(int(s) for s in sampled))
    logger.info("Stratified benchmark sample: %d stores", len(sampled))
    return sampled


def _load_store_series(store_id: int) -> pd.Series:
    df = read_sql(
        "SELECT date_id, sales FROM fact_sales WHERE store_id = :sid AND is_open = true ORDER BY date_id",
        {"sid": store_id},
    )
    s = df.set_index("date_id")["sales"]
    s.index = pd.to_datetime(s.index)
    return s.asfreq("D").interpolate(limit_direction="both")


def run_baseline_and_statistical_backtest(store_ids: list[int]) -> pd.DataFrame:
    """Per-store backtest for naive/seasonal-naive/moving-average/Holt-Winters/SARIMA."""
    rows = []
    for store_id in store_ids:
        series = _load_store_series(store_id)
        if len(series) < HORIZON * (N_FOLDS + 1):
            logger.warning("Store %d has insufficient history for %d folds; skipping", store_id, N_FOLDS)
            continue
        folds = generate_rolling_folds(series.reset_index()["date_id"], HORIZON, N_FOLDS)

        for f in folds:
            train = series[series.index <= f.train_end_date]
            test = series[(series.index >= f.test_start_date) & (series.index <= f.test_end_date)]
            if len(test) < HORIZON // 2:
                continue
            y_true = test.values

            model_forecasts = {
                "naive": naive_forecast(train, len(test)),
                "seasonal_naive": seasonal_naive_forecast(train, len(test), SEASON_LENGTH),
                "moving_average_7": moving_average_forecast(train, len(test), 7),
                "holt_winters": holt_winters_forecast(train, len(test), SEASON_LENGTH),
                "sarima": sarima_forecast(train, len(test)),
            }
            for model_name, y_pred in model_forecasts.items():
                metrics = compute_all_metrics(y_true, y_pred)
                rows.append({
                    "store_id": store_id, "model_name": model_name, "fold": f.fold,
                    "train_end_date": f.train_end_date, "test_start_date": f.test_start_date,
                    "test_end_date": f.test_end_date, **metrics,
                })
        logger.info("Backtested baselines/statistical models for store %d", store_id)
    return pd.DataFrame(rows)


def run_ml_backtest(store_ids: list[int]) -> pd.DataFrame:
    """Backtest global ML models (RF, XGBoost) -- trained once per fold
    across ALL stores' feature data (not just the sample), then evaluated
    on the sample stores' test windows for comparability with the
    per-store statistical/baseline results above."""
    features = read_sql("SELECT * FROM fact_sales_features")
    dim_store = read_sql("SELECT store_id, store_type, assortment FROM dim_store")
    features = prepare_ml_features(features, dim_store)
    features["date_id"] = pd.to_datetime(features["date_id"])

    all_dates = features["date_id"]
    folds = generate_rolling_folds(all_dates, HORIZON, N_FOLDS)

    rows = []
    for f in folds:
        train_df = features[features["date_id"] <= f.train_end_date].dropna(subset=["lag_28"])
        test_df = features[
            (features["date_id"] >= f.test_start_date)
            & (features["date_id"] <= f.test_end_date)
            & (features["store_id"].isin(store_ids))
        ]
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        for trainer, model_name in [(train_random_forest, "random_forest"), (train_xgboost, "xgboost")]:
            model = trainer(train_df)
            for store_id, store_test in test_df.groupby("store_id"):
                y_true = store_test["sales"].values
                y_pred = model.predict(store_test)
                metrics = compute_all_metrics(y_true, y_pred)
                rows.append({
                    "store_id": int(store_id), "model_name": model_name, "fold": f.fold,
                    "train_end_date": f.train_end_date, "test_start_date": f.test_start_date,
                    "test_end_date": f.test_end_date, **metrics,
                })
        logger.info("Backtested ML models for fold %d (train_end=%s)", f.fold, f.train_end_date.date())
    return pd.DataFrame(rows)


def run_full_benchmark(n_per_segment: int = 15) -> pd.DataFrame:
    store_ids = get_stratified_sample(n_per_segment)
    baseline_stat_results = run_baseline_and_statistical_backtest(store_ids)
    ml_results = run_ml_backtest(store_ids)
    combined = pd.concat([baseline_stat_results, ml_results], ignore_index=True)
    logger.info("Benchmark complete: %d evaluation rows across %d models",
                len(combined), combined["model_name"].nunique())
    return combined
