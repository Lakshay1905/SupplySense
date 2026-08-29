"""
SupplySense -- Overview Dashboard (main entry point).

Run with: streamlit run app/ui/Home.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `streamlit run app/ui/Home.py` from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.data_loaders import load_overview_kpis, load_optimization_results, load_store_list

st.set_page_config(page_title="SupplySense", page_icon="📦", layout="wide")

st.title("📦 SupplySense")
st.caption("Intelligent Demand Forecasting & Inventory Decision Engine")

try:
    kpis = load_overview_kpis()
except Exception as exc:
    st.error(
        f"Could not connect to the database or required tables are missing: {exc}\n\n"
        "Run the Phase 1-3 pipelines first: `python -m scripts.run_phase1_pipeline`, "
        "`python -m scripts.run_phase2_benchmark`, `python -m scripts.run_phase2_forecast --prepare` "
        "then `--batch-start 0 --batch-end 1115 --truncate`, and `python -m scripts.run_phase3_pipeline`."
    )
    st.stop()

st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Stores tracked", f"{kpis['total_stores']:,}")
col2.metric("Best forecasting model", kpis["best_model"] or "N/A",
            f"{kpis['best_model_wmape']:.1f}% WMAPE" if kpis["best_model_wmape"] else None)
col3.metric("Avg stockout risk (sampled stores)",
            f"{kpis['avg_stockout_risk']*100:.1f}%" if kpis["avg_stockout_risk"] is not None else "N/A")
col4.metric("Total expected cost (sampled stores)",
            f"€{kpis['total_expected_cost']:,.0f}" if kpis["total_expected_cost"] is not None else "N/A")

col5, col6, col7 = st.columns(3)
col5.metric("Stores recommended to order", f"{kpis['n_stores_ordering']} / {kpis['n_stores_evaluated']}")
if kpis["forecast_range"][0] is not None:
    col6.metric("Forecast horizon", f"{kpis['forecast_range'][0]} to {kpis['forecast_range'][1]}")
dq = kpis["data_quality"]
total_checks = sum(dq.values()) if dq else 0
col7.metric("Data quality checks passed", f"{dq.get('pass', 0)} / {total_checks}" if total_checks else "N/A")

st.divider()

st.subheader("Exceptions & Alerts")
try:
    opt = load_optimization_results()
    high_risk = opt[opt["stockout_probability"] > 0.10].sort_values("stockout_probability", ascending=False)
    budget_binding = opt[opt["drivers_json"].astype(str).str.contains("budget_binding", na=False)]

    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        st.markdown(f"**⚠️ High stockout risk (>10%): {len(high_risk)} stores**")
        if len(high_risk):
            st.dataframe(
                high_risk[["store_id", "recommended_order_qty", "stockout_probability", "expected_cost"]]
                .head(10).rename(columns={
                    "store_id": "Store", "recommended_order_qty": "Recommended Order",
                    "stockout_probability": "Stockout Risk", "expected_cost": "Expected Cost (€)",
                }),
                hide_index=True, width='stretch',
            )
        else:
            st.success("No stores currently exceed the 10% stockout-risk alert threshold.")
    with alert_col2:
        st.markdown(f"**💰 Budget-constrained stores: {len(budget_binding)} stores**")
        if len(budget_binding):
            st.dataframe(
                budget_binding[["store_id", "recommended_order_qty", "expected_cost"]]
                .head(10).rename(columns={
                    "store_id": "Store", "recommended_order_qty": "Recommended Order",
                    "expected_cost": "Expected Cost (€)",
                }),
                hide_index=True, width='stretch',
            )
        else:
            st.info("No stores are currently budget-constrained in the sampled recommendations.")
except Exception as exc:
    st.warning(f"Optimization results not available yet: {exc}")

st.divider()
st.subheader("Store Directory")
stores = load_store_list()
st.dataframe(stores.rename(columns={
    "store_id": "Store", "store_type": "Store Type (category proxy)",
    "assortment": "Assortment", "state_name": "Region",
}), hide_index=True, width='stretch', height=300)

st.caption(
    "Navigate using the sidebar: Forecast Explorer, Inventory Recommendations, Optimization, "
    "Scenario Simulator, Model Performance, Data Quality, and the AI Copilot."
)
