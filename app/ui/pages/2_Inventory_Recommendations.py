"""Inventory Recommendations -- live-computed order quantity, cost breakdown, and drivers."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from app.components.data_loaders import load_store_list
from optimization.recommendation_engine import (
    generate_recommendation, get_current_inventory, get_supplier_lead_time,
)
from config.settings import BUSINESS_DEFAULTS

st.set_page_config(page_title="Inventory Recommendations | SupplySense", page_icon="📋", layout="wide")
st.title("📋 Inventory Recommendations")
st.caption(
    "Live-computed recommendations from the Monte Carlo + constrained optimization engine "
    "(not a stored snapshot -- adjust inputs below and recompute)."
)

stores = load_store_list()
store_id = st.selectbox("Store", stores["store_id"].tolist())

with st.expander("Adjust assumptions (optional)", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        target_service_level = st.slider("Target service level", 0.80, 0.995,
                                          BUSINESS_DEFAULTS["target_service_level"], step=0.005)
    with col2:
        use_budget = st.checkbox("Apply a procurement budget cap")
        budget = st.number_input("Budget (€)", min_value=0.0, value=10000.0, step=500.0) if use_budget else None
    with col3:
        use_capacity = st.checkbox("Apply a warehouse capacity cap")
        capacity = st.number_input("Capacity (units)", min_value=0.0, value=5000.0, step=100.0) if use_capacity else None

try:
    on_hand, incoming = get_current_inventory(store_id)
    lead_time = get_supplier_lead_time(store_id)
    result = generate_recommendation(
        store_id=store_id, target_service_level=target_service_level,
        procurement_budget=budget, warehouse_capacity_units=capacity,
    )
except ValueError as exc:
    st.error(f"Could not generate a recommendation: {exc}")
    st.stop()

st.divider()

st.subheader(f"Recommendation: Store {store_id}")
big_col1, big_col2, big_col3 = st.columns([2, 1, 1])
with big_col1:
    st.metric("Recommended Order Quantity", f"{result.recommended_order_qty:,.0f} units")
with big_col2:
    st.metric("Achieved Service Level", f"{result.achieved_service_level*100:.1f}%",
               f"target {target_service_level*100:.1f}%")
with big_col3:
    st.metric("Stockout Risk", f"{result.stockout_probability*100:.1f}%")

if not result.within_target_service_level:
    st.warning(
        "⚠️ The target service level is **not achievable** under the current budget/capacity "
        "constraints. The recommendation shown is the lowest-cost feasible option instead."
    )

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Current State")
    st.dataframe(pd.DataFrame([
        {"Metric": "Current on-hand inventory", "Value": f"{on_hand:,.0f} units"},
        {"Metric": "Incoming inventory", "Value": f"{incoming:,.0f} units"},
        {"Metric": "Supplier lead time", "Value": f"{lead_time:.1f} days"},
        {"Metric": "Decision horizon (lead time + review period)", "Value": f"{result.horizon_days} days"},
        {"Metric": "Forecasted demand (P50, revenue-equivalent)", "Value": f"€{result.demand_forecast_units_p50 * result.drivers['unit_price_proxy']:,.0f}"},
    ]), hide_index=True, width="stretch")

with col_b:
    st.subheader("Cost Breakdown")
    st.dataframe(pd.DataFrame([
        {"Cost Component": "Procurement", "Amount (€)": f"{result.procurement_cost:,.2f}"},
        {"Cost Component": "Expected Holding Cost", "Amount (€)": f"{result.expected_holding_cost:,.2f}"},
        {"Cost Component": "Expected Stockout Cost", "Amount (€)": f"{result.expected_stockout_cost:,.2f}"},
        {"Cost Component": "Expected Total Cost", "Amount (€)": f"{result.expected_total_cost:,.2f}"},
    ]), hide_index=True, width="stretch")

st.divider()
st.subheader("Why This Recommendation? (Decision Drivers)")

driver_text = f"""
- **Demand change vs recent average:** {result.demand_change_pct:+.1f}%
- **Starting inventory position:** {result.starting_inventory:,.0f} units on-hand + {result.drivers['incoming_inventory']:,.0f} incoming
- **Supplier lead time:** {result.drivers['lead_time_days']:.1f} days (+ {result.drivers['review_period_days']}-day review period = {result.horizon_days}-day decision horizon)
- **Target service level:** {result.drivers['target_service_level_pct']:.1f}% → achieved stockout risk: {result.drivers['achieved_stockout_risk_pct']:.1f}%
- **Budget status:** {result.budget_status.replace('_', ' ')}
- **Capacity status:** {result.capacity_status.replace('_', ' ')}
- **Unit economics:** €{result.drivers['unit_cost']:.2f} cost / €{result.drivers['unit_price_proxy']:.2f} price (proxy, from this store's historical avg. revenue-per-transaction)
"""
st.markdown(driver_text)

st.divider()
st.subheader("Cost vs. Order Quantity (Monte Carlo simulation)")
curve = result.full_cost_curve.sort_values("order_qty")
chart_data = curve.set_index("order_qty")[["expected_total_cost"]].rename(
    columns={"expected_total_cost": "Expected Total Cost (€)"}
)
st.line_chart(chart_data, height=350)
st.caption(
    "This curve is generated by simulating 5,000 demand draws from the store's probabilistic "
    "forecast for every candidate order quantity -- not a formula shortcut. The chosen quantity "
    "is the cheapest one that also meets the target service level (see Decision Drivers above)."
)
