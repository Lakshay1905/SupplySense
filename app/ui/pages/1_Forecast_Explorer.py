"""Forecast Explorer -- historical demand + probabilistic forecast per store."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from app.components.data_loaders import (
    load_store_list, load_store_history, load_store_forecast, load_per_store_best_model,
)

st.set_page_config(page_title="Forecast Explorer | SupplySense", page_icon="📈", layout="wide")
st.title("📈 Forecast Explorer")
st.caption("Historical demand, probabilistic forecast, seasonality, and model performance per store.")

stores = load_store_list()

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    region_filter = st.selectbox("Region", ["All"] + sorted(stores["state_name"].dropna().unique().tolist()))
with col_f2:
    type_filter = st.selectbox("Store Type (category proxy)", ["All"] + sorted(stores["store_type"].unique().tolist()))
with col_f3:
    history_days = st.slider("History window (days)", 30, 365, 180, step=30)

filtered = stores.copy()
if region_filter != "All":
    filtered = filtered[filtered["state_name"] == region_filter]
if type_filter != "All":
    filtered = filtered[filtered["store_type"] == type_filter]

if filtered.empty:
    st.warning("No stores match the selected filters.")
    st.stop()

store_id = st.selectbox("Store", filtered["store_id"].tolist())

st.divider()

history = load_store_history(store_id, days=history_days)
forecast = load_store_forecast(store_id)

if history.empty:
    st.warning("No historical data for this store.")
    st.stop()

chart_df = pd.DataFrame({
    "date": pd.to_datetime(history["date_id"]),
    "Historical Actual": history["sales"],
})
chart_df = chart_df.set_index("date")

if not forecast.empty:
    fc = forecast.copy()
    fc["target_date"] = pd.to_datetime(fc["target_date"])
    fc_chart = fc.set_index("target_date")[["p10", "p50", "p90"]].rename(
        columns={"p10": "Forecast P10", "p50": "Forecast P50 (median)", "p90": "Forecast P90"}
    )
    combined = pd.concat([chart_df, fc_chart], axis=0).sort_index()
else:
    combined = chart_df

st.subheader(f"Store {store_id}: Historical Demand + Probabilistic Forecast")
st.line_chart(combined, height=400)

if not forecast.empty:
    st.caption(
        f"Forecast model: **{forecast['model_name'].iloc[0]}** | "
        f"Horizon: {forecast['target_date'].min()} to {forecast['target_date'].max()} "
        f"({len(forecast)} days) | Shaded range represents the P10-P90 uncertainty band."
    )

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Day-of-Week Seasonality")
    hist_dow = history.copy()
    hist_dow["day_of_week"] = pd.to_datetime(hist_dow["date_id"]).dt.day_name()
    dow_avg = hist_dow[hist_dow["is_open"]].groupby("day_of_week")["sales"].mean()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg = dow_avg.reindex([d for d in dow_order if d in dow_avg.index])
    st.bar_chart(dow_avg)

with col_b:
    st.subheader("Model Performance for This Store")
    best_models = load_per_store_best_model()
    store_models = best_models[best_models["store_id"] == store_id].sort_values("avg_wmape")
    if len(store_models):
        st.dataframe(
            store_models.rename(columns={
                "model_name": "Model", "avg_wmape": "Avg WMAPE (%)",
            })[["Model", "Avg WMAPE (%)"]].round(2),
            hide_index=True, width="stretch",
        )
        st.caption(
            f"Lower WMAPE is better. This store was part of the Phase 2 backtesting sample; "
            f"**{store_models.iloc[0]['model_name']}** performed best here."
        )
    else:
        st.info(
            "This store wasn't part of the stratified benchmarking sample (Phase 2 benchmarks "
            "run on a representative subset of stores). The full-scale forecast still uses "
            "the overall winning model (XGBoost)."
        )

st.divider()
st.subheader("Recent History Data")
st.dataframe(
    history.rename(columns={
        "date_id": "Date", "sales": "Sales", "is_open": "Open", "is_promo": "Promo",
        "state_holiday": "State Holiday", "school_holiday": "School Holiday",
    }).sort_values("Date", ascending=False),
    hide_index=True, width="stretch", height=300,
)
