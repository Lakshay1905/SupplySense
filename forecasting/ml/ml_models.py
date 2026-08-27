"""
Machine-learning forecasters.

Rather than fitting 1,115 separate per-store models, we train a single
*global* model across all stores using the engineered feature table
(lags, rolling stats, calendar features, store attributes). This is the
standard, scalable approach for large retail panels: the model learns
shared demand patterns (weekday effects, promo effects, competition
effects) across stores while `store_id`-derived and store-attribute
features let it specialize per store. It also trains once instead of
1,115 times, which matters for a periodic batch job.

Two model families are benchmarked: Random Forest and XGBoost (gradient
boosting). Both consume the same feature matrix so the comparison in
forecasting/evaluation is apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28",
    "day_of_week", "is_weekend", "week_of_year", "month",
    "is_promo", "is_school_holiday", "is_state_holiday",
    "days_since_competition",
    "store_id", "store_type_code", "assortment_code",
]

CATEGORICAL_AS_INT = ["is_weekend", "is_promo", "is_school_holiday", "is_state_holiday"]


def prepare_ml_features(features: pd.DataFrame, dim_store: pd.DataFrame) -> pd.DataFrame:
    """Join store attributes and encode categoricals for tree-based models."""
    df = features.merge(
        dim_store[["store_id", "store_type", "assortment"]], on="store_id", how="left"
    )
    store_type_map = {t: i for i, t in enumerate(sorted(df["store_type"].dropna().unique()))}
    assortment_map = {a: i for i, a in enumerate(sorted(df["assortment"].dropna().unique()))}
    df["store_type_code"] = df["store_type"].map(store_type_map)
    df["assortment_code"] = df["assortment"].map(assortment_map)
    for c in CATEGORICAL_AS_INT:
        df[c] = df[c].astype(int)
    return df


@dataclass
class MLForecastModel:
    model_name: str
    model: object
    feature_medians: pd.Series

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURE_COLUMNS].fillna(self.feature_medians)
        preds = self.model.predict(X)
        return np.clip(preds, a_min=0, a_max=None)


def train_random_forest(train_df: pd.DataFrame, target_col: str = "sales",
                         max_train_rows: int = 80_000) -> MLForecastModel:
    """Train a global Random Forest.

    NOTE ON SUBSAMPLING: this environment has a single CPU core, and
    RandomForest (unlike XGBoost's histogram method) does not scale well
    single-threaded on 1M+ rows. We subsample to `max_train_rows` rows
    (random, reproducible) and use a shallower forest -- a documented,
    reasonable engineering tradeoff for a batch benchmarking job. XGBoost
    (trained on full data, see train_xgboost) is unaffected and is the
    model actually used for full-scale production forecasting if it wins
    the benchmark.
    """
    if len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=42)

    X = train_df[FEATURE_COLUMNS]
    feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)
    y = train_df[target_col]

    model = RandomForestRegressor(
        n_estimators=40, max_depth=9, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    )
    model.fit(X, y)
    logger.info("Trained RandomForest on %d rows (subsampled), %d features", len(X), X.shape[1])
    return MLForecastModel("random_forest", model, feature_medians)


def train_xgboost(train_df: pd.DataFrame, target_col: str = "sales") -> MLForecastModel:
    X = train_df[FEATURE_COLUMNS]
    feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)
    y = train_df[target_col]

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", n_jobs=-1, random_state=42,
    )
    model.fit(X, y)
    logger.info("Trained XGBoost on %d rows, %d features", len(X), X.shape[1])
    return MLForecastModel("xgboost", model, feature_medians)


def train_xgboost_quantile(train_df: pd.DataFrame, quantile: float,
                            target_col: str = "sales") -> MLForecastModel:
    """Train an XGBoost model with quantile regression objective, used
    directly for probabilistic (P10/P90) forecasting as an alternative to
    the residual-based method in forecasting.probabilistic."""
    X = train_df[FEATURE_COLUMNS]
    feature_medians = X.median(numeric_only=True)
    X = X.fillna(feature_medians)
    y = train_df[target_col]

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:quantileerror", quantile_alpha=quantile,
        n_jobs=-1, random_state=42,
    )
    model.fit(X, y)
    return MLForecastModel(f"xgboost_q{quantile}", model, feature_medians)
