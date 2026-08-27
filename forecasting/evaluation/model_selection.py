"""
Model selection: given backtest evaluation results (one row per
store/model/fold), pick the best-performing model per store (or overall,
for the global ML model) using average WMAPE across folds -- WMAPE is
preferred over plain MAPE as the selection criterion because it is
robust to the zero-sales days present in this dataset (closed stores).
"""
from __future__ import annotations

import pandas as pd


def select_best_model_per_store(evaluations: pd.DataFrame) -> pd.DataFrame:
    """`evaluations` must have columns: store_id, model_name, wmape (one row
    per fold). Returns one row per store with the winning model and its
    average metrics across folds."""
    agg = (
        evaluations.groupby(["store_id", "model_name"])
        .agg(avg_wmape=("wmape", "mean"), avg_mae=("mae", "mean"),
             avg_rmse=("rmse", "mean"), avg_bias=("bias", "mean"), n_folds=("wmape", "count"))
        .reset_index()
    )
    best_idx = agg.groupby("store_id")["avg_wmape"].idxmin()
    best = agg.loc[best_idx].reset_index(drop=True)
    return best


def summarize_model_performance(evaluations: pd.DataFrame) -> pd.DataFrame:
    """Overall (all-store, all-fold) performance per model -- this is what
    justifies "why was model X selected" for the AI copilot / UI."""
    summary = (
        evaluations.groupby("model_name")
        .agg(avg_mae=("mae", "mean"), avg_rmse=("rmse", "mean"),
             avg_mape=("mape", "mean"), avg_wmape=("wmape", "mean"),
             avg_bias=("bias", "mean"), n_evaluations=("wmape", "count"))
        .reset_index()
        .sort_values("avg_wmape")
    )
    return summary
