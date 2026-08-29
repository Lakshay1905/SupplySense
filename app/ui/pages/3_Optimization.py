"""Optimization -- configure cost/constraint assumptions and view portfolio allocation."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.data_loaders import load_store_list
from optimization.recommendation_engine import generate_recommendation
from optimization.portfolio_optimizer import optimize_portfolio_allocation
from config.settings import BUSINESS_DEFAULTS

st.set_page_config(page_title="Optimization | SupplySense", page_icon="⚙️", layout="wide")
st.title("⚙️ Optimization")
st.caption(
    "Configure shared assumptions and run the portfolio-level constrained optimizer "
    "(a real Mixed-Integer Program, solved with PuLP/CBC) across multiple stores."
)

stores = load_store_list()

st.subheader("Global Assumptions")
col1, col2, col3, col4 = st.columns(4)
with col1:
    target_service_level = st.slider("Target service level", 0.80, 0.995,
                                      BUSINESS_DEFAULTS["target_service_level"], step=0.005)
with col2:
    moq = st.number_input("MOQ (units)", min_value=1, value=BUSINESS_DEFAULTS["moq_units"])
with col3:
    order_multiple = st.number_input("Order multiple (units)", min_value=1,
                                      value=BUSINESS_DEFAULTS["order_multiple_units"])
with col4:
    lead_time_days = st.number_input("Lead time override (days, blank = use supplier data)",
                                      min_value=0.0, value=0.0, step=1.0)

st.divider()
st.subheader("Portfolio Budget Allocation")
st.caption(
    "Select a group of stores and a shared procurement budget. The optimizer decides how much "
    "each store should receive to minimize total portfolio cost -- not a proportional or greedy split."
)

region_options = sorted(stores["state_name"].dropna().unique().tolist())
selected_region = st.selectbox("Region", region_options)
region_stores = stores[stores["state_name"] == selected_region]["store_id"].tolist()

max_stores_for_demo = 30
if len(region_stores) > max_stores_for_demo:
    st.info(f"Region has {len(region_stores)} stores; using the first {max_stores_for_demo} for this "
             f"interactive demo to keep response times low.")
    region_stores = region_stores[:max_stores_for_demo]

run_portfolio = st.button("Compute portfolio allocation", type="primary")

if run_portfolio:
    with st.spinner(f"Running Monte Carlo optimization for {len(region_stores)} stores..."):
        curves = {}
        skipped = []
        for sid in region_stores:
            try:
                result = generate_recommendation(
                    store_id=sid, target_service_level=target_service_level,
                    lead_time_override=lead_time_days if lead_time_days > 0 else None,
                )
                curves[sid] = result.full_cost_curve
            except ValueError:
                skipped.append(sid)

        if len(curves) < 2:
            st.error("Not enough stores with valid data to run a portfolio allocation.")
            st.stop()

        unconstrained_need = sum(
            c.loc[c["expected_total_cost"].idxmin(), "procurement_cost"] for c in curves.values()
        )

    st.metric("Unconstrained procurement need (sum of per-store optima)", f"€{unconstrained_need:,.0f}")
    budget_pct = st.slider("Shared budget as % of unconstrained need", 10, 150, 60, step=5)
    shared_budget = unconstrained_need * budget_pct / 100

    with st.spinner("Solving portfolio MILP..."):
        allocation = optimize_portfolio_allocation(curves, total_budget=shared_budget, total_capacity=None)

    st.success(f"Solved: **{allocation.status}** | Budget: €{shared_budget:,.0f} "
               f"({budget_pct}% of unconstrained need)")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Expected Cost", f"€{allocation.total_cost:,.0f}")
    col_b.metric("Total Procurement Spend", f"€{allocation.total_procurement_spend:,.0f}")
    col_c.metric("Budget Utilization", f"{allocation.budget_utilization_pct:.1f}%")

    st.subheader("Store-Level Allocation")
    st.dataframe(
        allocation.allocations.rename(columns={
            "store_id": "Store", "order_qty": "Allocated Order Qty",
            "expected_total_cost": "Expected Cost (€)", "procurement_cost": "Procurement Cost (€)",
            "stockout_probability": "Stockout Risk",
        }),
        hide_index=True, width="stretch",
    )

    zero_alloc = allocation.allocations[allocation.allocations["order_qty"] == 0]
    if len(zero_alloc):
        st.info(
            f"{len(zero_alloc)} store(s) received zero allocation under this budget -- the "
            f"optimizer determined other stores had a better cost-reduction-per-euro at the margin."
        )

    if skipped:
        st.caption(f"Skipped {len(skipped)} store(s) with insufficient forecast/inventory data.")
else:
    st.info("Configure assumptions above and click **Compute portfolio allocation** to run the optimizer.")
