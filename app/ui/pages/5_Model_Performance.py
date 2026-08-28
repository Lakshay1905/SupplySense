"""Model Performance -- benchmark comparison, backtesting results, performance by segment."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from app.components.data_loaders import load_model_comparison
from database.connection import read_sql

st.set_page_config(page_title="Model Performance | SupplySense", page_icon="🧪", layout="wide")
st.title("🧪 Model Performance")
st.caption(
    "Real rolling-origin backtesting results across baseline, statistical, and ML forecasting models."
)

comparison = load_model_comparison()

if comparison.empty:
    st.warning("No model evaluation data found. Run `python -m scripts.run_phase2_benchmark` first.")
    st.stop()

st.subheader("Overall Model Comparison (avg across all backtested stores/folds)")
st.dataframe(
    comparison.rename(columns={
        "model_name": "Model", "avg_mae": "Avg MAE", "avg_rmse": "Avg RMSE",
        "avg_mape": "Avg MAPE (%)", "avg_wmape": "Avg WMAPE (%)", "avg_bias": "Avg Bias (%)",
        "n_evaluations": "N Evaluations",
    }).round(2),
    hide_index=True, width="stretch",
)

best_model = comparison.iloc[0]
st.success(
    f"**{best_model['model_name']}** was selected for production forecasting: lowest avg WMAPE "
    f"({best_model['avg_wmape']:.2f}%), beating the next-best model "
    f"({comparison.iloc[1]['model_name']}, {comparison.iloc[1]['avg_wmape']:.2f}%) and every baseline."
)

st.bar_chart(comparison.set_index("model_name")["avg_wmape"], height=350)
st.caption("Lower WMAPE (Weighted Mean Absolute Percentage Error) is better. "
           "WMAPE is used as the selection metric because it handles the dataset's legitimate "
           "zero-sales (closed-store) days robustly, unlike plain MAPE.")

st.divider()
st.subheader("Per-Store Winning Model")
per_store = read_sql("""
    SELECT store_id, model_name, AVG(wmape) as avg_wmape
    FROM model_evaluations GROUP BY store_id, model_name
""")
if len(per_store):
    best_per_store = per_store.loc[per_store.groupby("store_id")["avg_wmape"].idxmin()]
    win_counts = best_per_store["model_name"].value_counts()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(
            win_counts.rename_axis("Model").reset_index(name="Stores Won"),
            hide_index=True, width="stretch",
        )
    with col2:
        st.bar_chart(win_counts)

st.divider()
st.subheader("Raw Backtest Results")
raw = read_sql("SELECT * FROM model_evaluations ORDER BY store_id, model_name, fold")
st.dataframe(
    raw.rename(columns={
        "store_id": "Store", "model_name": "Model", "fold": "Fold",
        "train_end_date": "Train End", "test_start_date": "Test Start", "test_end_date": "Test End",
        "mae": "MAE", "rmse": "RMSE", "mape": "MAPE (%)", "wmape": "WMAPE (%)", "bias": "Bias (%)",
    }),
    hide_index=True, width="stretch", height=400,
)

st.caption(
    "Methodology: rolling-origin (walk-forward) time-series cross-validation, 3 folds, "
    "42-day test horizon per fold, backtested on a stratified sample of stores spanning every "
    "demand-segment × store-type combination. See README.md for full methodology."
)
