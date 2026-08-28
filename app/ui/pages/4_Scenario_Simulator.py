"""Scenario Simulator -- baseline vs what-if comparison."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from app.components.data_loaders import load_store_list
from scenarios.scenario_engine import run_scenario, ScenarioDefinition, PRESET_SCENARIOS

st.set_page_config(page_title="Scenario Simulator | SupplySense", page_icon="🔀", layout="wide")
st.title("🔀 Scenario Simulator")
st.caption(
    "Change business assumptions and compare the resulting recommendation to baseline. "
    "Every scenario re-runs the real optimizer with modified inputs -- not a multiplier on the baseline output."
)

stores = load_store_list()
store_id = st.selectbox("Store", stores["store_id"].tolist())

mode = st.radio("Scenario type", ["Preset", "Custom"], horizontal=True)

if mode == "Preset":
    preset_key = st.selectbox(
        "Preset scenario",
        options=list(PRESET_SCENARIOS.keys()),
        format_func=lambda k: PRESET_SCENARIOS[k].name,
    )
    scenario = PRESET_SCENARIOS[preset_key]
else:
    col1, col2 = st.columns(2)
    with col1:
        demand_multiplier = st.slider("Demand multiplier", 0.5, 2.0, 1.0, step=0.05)
        budget_multiplier = st.slider("Budget multiplier (relative)", 0.5, 1.5, 1.0, step=0.05)
        promo_uplift_pct = st.slider("Promotion uplift (%)", 0.0, 100.0, 0.0, step=5.0)
    with col2:
        lead_time_override = st.number_input("Lead time override (days, 0 = no change)",
                                              min_value=0.0, value=0.0, step=1.0)
        capacity_multiplier = st.slider("Capacity multiplier (relative)", 0.5, 2.0, 1.0, step=0.05)
        promo_duration_days = st.number_input("Promotion duration (days)", min_value=0, value=0, step=1)

    scenario = ScenarioDefinition(
        name="Custom scenario",
        demand_multiplier=demand_multiplier,
        lead_time_override=lead_time_override if lead_time_override > 0 else None,
        budget_multiplier=budget_multiplier if budget_multiplier != 1.0 else None,
        capacity_multiplier=capacity_multiplier if capacity_multiplier != 1.0 else None,
        promo_uplift_pct=promo_uplift_pct, promo_duration_days=int(promo_duration_days),
    )

run_button = st.button("Run scenario", type="primary")

if run_button:
    with st.spinner("Recomputing baseline and scenario recommendations..."):
        try:
            comparison = run_scenario(store_id, scenario)
        except ValueError as exc:
            st.error(f"Could not run scenario: {exc}")
            st.stop()

    st.divider()
    st.subheader(f"Baseline vs. {comparison.scenario_name} — Store {store_id}")

    metrics = [
        ("recommended_order_qty", "Recommended Order Qty", "{:.0f} units"),
        ("expected_total_cost", "Expected Total Cost", "€{:,.2f}"),
        ("stockout_probability_pct", "Stockout Risk", "{:.2f}%"),
        ("achieved_service_level_pct", "Achieved Service Level", "{:.2f}%"),
        ("procurement_cost", "Procurement Cost", "€{:,.2f}"),
    ]

    cols = st.columns(len(metrics))
    for col, (key, label, fmt) in zip(cols, metrics):
        baseline_val = comparison.baseline[key]
        scenario_val = comparison.scenario[key]
        delta = comparison.deltas[key]
        pct_change = comparison.deltas.get(f"{key}_pct_change")
        delta_str = f"{delta:+,.2f}" + (f" ({pct_change:+.1f}%)" if pct_change is not None else "")
        col.metric(label, fmt.format(scenario_val), delta_str)

    st.divider()
    comparison_table = pd.DataFrame([
        {"Metric": label, "Baseline": fmt.format(comparison.baseline[key]),
         "Scenario": fmt.format(comparison.scenario[key]),
         "Change": f"{comparison.deltas[key]:+,.2f}"}
        for key, label, fmt in metrics
    ])
    st.dataframe(comparison_table, hide_index=True, width="stretch")

    if not comparison.scenario["within_target_service_level"]:
        st.warning("⚠️ Under this scenario, the target service level is not achievable within constraints.")

    st.caption(
        f"Budget status: baseline={comparison.baseline['budget_status']}, "
        f"scenario={comparison.scenario['budget_status']} | "
        f"Capacity status: baseline={comparison.baseline['capacity_status']}, "
        f"scenario={comparison.scenario['capacity_status']}"
    )
else:
    st.info("Configure a scenario above and click **Run scenario** to compare against baseline.")
