"""
Phase 2, step 2: train the winning model (selected empirically by the
benchmark in scripts/run_phase2_benchmark.py -- XGBoost, avg WMAPE 9.90%
vs 12.8% for Random Forest and 17-26% for statistical/baseline models)
on ALL available history for ALL 1,115 stores, then generate a 42-day
forward forecast with P10/P50/P90 uncertainty bands for every store.

Uncertainty bands are derived from the model's own out-of-sample
residuals (last backtest fold), per demand segment -- this keeps bands
store-specific in aggregate character (volatile-segment stores get wider
bands than stable-segment stores) without needing per-store residual
history, most stores' individual residual samples being too small to
estimate stable quantiles from alone.

Usage:
    python -m scripts.run_phase2_forecast
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from database.connection import get_engine, read_sql, write_dataframe
from forecasting.ml.ml_models import prepare_ml_features, train_xgboost, FEATURE_COLUMNS
from forecasting.probabilistic import compute_residual_quantiles, apply_probabilistic_bands
from forecasting.evaluation.backtesting import generate_rolling_folds
from analytics.features.feature_engineering import (
    add_calendar_features, add_lag_and_rolling_features,
)

logger = get_logger(__name__)

HORIZON = 42
MODEL_NAME = "xgboost"


def _held_out_residuals_by_segment(features: pd.DataFrame, dim_store: pd.DataFrame) -> pd.DataFrame:
    """Compute residuals on the most recent held-out fold (same split the
    benchmark used for its last fold) to derive per-segment uncertainty
    bands for the production forecast."""
    fold = generate_rolling_folds(features["date_id"], HORIZON, 1)[0]
    train_df = features[features["date_id"] <= fold.train_end_date].dropna(subset=["lag_28"])
    test_df = features[(features["date_id"] >= fold.test_start_date)
                        & (features["date_id"] <= fold.test_end_date)]

    model = train_xgboost(train_df)
    test_df = test_df.copy()
    test_df["y_pred"] = model.predict(test_df)
    test_df["residual"] = test_df["sales"] - test_df["y_pred"]

    seg_residuals = test_df.groupby("demand_segment")["residual"].apply(lambda s: s.values)
    return seg_residuals


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")


def prepare_forecast_artifacts() -> dict:
    """Train the production model once and cache everything needed for
    per-store recursive forecasting, so the (slower) per-store loop can
    be run in batches without retraining."""
    logger.info("Loading feature table and store dimension for full-scale forecasting")
    features = read_sql("SELECT * FROM fact_sales_features")
    dim_store = read_sql("SELECT store_id, store_type, assortment, competition_open_since_year, "
                          "competition_open_since_month FROM dim_store")
    features["date_id"] = pd.to_datetime(features["date_id"])
    features_ml = prepare_ml_features(features, dim_store)

    logger.info("Training final production XGBoost model on full history (%d rows)", len(features_ml))
    full_train = features_ml.dropna(subset=["lag_28"])
    model = train_xgboost(full_train)

    logger.info("Computing per-segment residual bands from held-out fold")
    segment_residuals = _held_out_residuals_by_segment(features_ml, dim_store)
    segment_bands = {
        seg: compute_residual_quantiles(res) for seg, res in segment_residuals.items()
    }
    logger.info("Segment uncertainty bands (lower/upper offsets): %s",
                {k: (round(v[0], 1), round(v[1], 1)) for k, v in segment_bands.items()})

    return {
        "model": model,
        "features": features,
        "dim_store": dim_store,
        "segment_bands": segment_bands,
    }


def forecast_store_batch(artifacts: dict, store_ids: list[int]) -> pd.DataFrame:
    """Run the recursive multi-step forecast for a batch of stores using
    pre-trained artifacts from prepare_forecast_artifacts()."""
    model = artifacts["model"]
    features = artifacts["features"]
    dim_store = artifacts["dim_store"]
    segment_bands = artifacts["segment_bands"]

    last_date = features["date_id"].max()
    future_dates = _future_dates(last_date, HORIZON)

    dim_store_indexed = dim_store.set_index("store_id")
    store_type_map = {t: i for i, t in enumerate(sorted(dim_store["store_type"].dropna().unique()))}
    assortment_map = {a: i for i, a in enumerate(sorted(dim_store["assortment"].dropna().unique()))}
    segment_lookup = features[["store_id", "demand_segment"]].drop_duplicates().set_index("store_id")["demand_segment"]

    forecast_rows = []
    for store_id in store_ids:
        history = features[features["store_id"] == store_id][["date_id", "store_id", "sales"]]
        if len(history) < 28:
            continue

        srow = dim_store_indexed.loc[store_id]
        comp_open_date = pd.NaT
        if pd.notna(srow.get("competition_open_since_year")):
            comp_open_date = pd.Timestamp(
                year=int(srow["competition_open_since_year"]),
                month=int(srow.get("competition_open_since_month") or 1), day=1)

        extended = history.copy()
        preds = []
        for date in future_dates:
            temp = pd.concat(
                [extended, pd.DataFrame([{"date_id": date, "store_id": store_id, "sales": np.nan}])],
                ignore_index=True,
            )
            temp = add_calendar_features(temp)
            temp = add_lag_and_rolling_features(temp)
            row = temp.iloc[[-1]].copy()
            row["days_since_competition"] = (
                (date - comp_open_date).days if pd.notna(comp_open_date) and date >= comp_open_date else np.nan
            )
            row["is_promo"] = False
            row["is_school_holiday"] = False
            row["is_state_holiday"] = False
            row["store_type_code"] = store_type_map.get(srow["store_type"])
            row["assortment_code"] = assortment_map.get(srow["assortment"])

            X = row[FEATURE_COLUMNS]
            pred = float(model.predict(X)[0])
            preds.append(pred)

            extended = pd.concat(
                [extended, pd.DataFrame([{"date_id": date, "store_id": store_id, "sales": pred}])],
                ignore_index=True,
            )

        segment = segment_lookup.get(store_id, "stable")
        lower_off, upper_off = segment_bands.get(segment, (0.0, 0.0))
        bands = apply_probabilistic_bands(np.array(preds), lower_off, upper_off)
        bands["store_id"] = store_id
        bands["target_date"] = future_dates
        forecast_rows.append(bands)

    if not forecast_rows:
        return pd.DataFrame(columns=["store_id", "forecast_date", "target_date", "model_name", "p10", "p50", "p90"])

    result = pd.concat(forecast_rows, ignore_index=True)
    result["forecast_date"] = last_date
    result["model_name"] = MODEL_NAME
    return result[["store_id", "forecast_date", "target_date", "model_name", "p10", "p50", "p90"]]


def generate_forecasts_for_all_stores() -> pd.DataFrame:
    artifacts = prepare_forecast_artifacts()
    store_ids = sorted(artifacts["features"]["store_id"].unique())
    logger.info("Generating recursive %d-day forecasts for %d stores", HORIZON, len(store_ids))
    result = forecast_store_batch(artifacts, store_ids)
    logger.info("Generated %d forecast rows (%d stores x %d days)", len(result), len(store_ids), HORIZON)
    return result


ARTIFACT_CACHE_PATH = "/tmp/supplysense_phase2_artifacts.pkl"


def main() -> None:
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Train model + cache artifacts, then exit")
    parser.add_argument("--batch-start", type=int, default=None)
    parser.add_argument("--batch-end", type=int, default=None)
    parser.add_argument("--truncate", action="store_true", help="Truncate forecasts table before writing")
    args = parser.parse_args()

    if args.prepare:
        logger.info("=== Phase 2: preparing forecast artifacts (train once) ===")
        artifacts = prepare_forecast_artifacts()
        with open(ARTIFACT_CACHE_PATH, "wb") as f:
            pickle.dump(artifacts, f)
        print(f"Artifacts cached to {ARTIFACT_CACHE_PATH}. "
              f"Store count: {artifacts['features']['store_id'].nunique()}")
        return

    with open(ARTIFACT_CACHE_PATH, "rb") as f:
        artifacts = pickle.load(f)

    all_store_ids = sorted(artifacts["features"]["store_id"].unique())
    if args.batch_start is not None and args.batch_end is not None:
        store_ids = all_store_ids[args.batch_start:args.batch_end]
    else:
        store_ids = all_store_ids

    logger.info("Forecasting batch: %d stores (%s..%s)", len(store_ids),
                store_ids[0] if store_ids else None, store_ids[-1] if store_ids else None)
    forecasts = forecast_store_batch(artifacts, store_ids)

    engine = get_engine()
    if args.truncate:
        with engine.begin() as conn:
            conn.exec_driver_sql("TRUNCATE TABLE forecasts RESTART IDENTITY")
    write_dataframe(forecasts, "forecasts", if_exists="append")

    print(f"Batch done: {len(forecasts):,} forecast rows written for {len(store_ids)} stores "
          f"(range {args.batch_start}:{args.batch_end}).")


if __name__ == "__main__":
    main()
