"""Data Quality -- pipeline run history and validation check results."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from app.components.data_loaders import load_data_quality_log, load_pipeline_runs
from database.connection import read_sql

st.set_page_config(page_title="Data Quality | SupplySense", page_icon="🔍", layout="wide")
st.title("🔍 Data Quality")
st.caption("Real validation results and pipeline run history -- not a static status page.")

runs = load_pipeline_runs()
if runs.empty:
    st.warning("No pipeline runs found. Run `python -m scripts.run_phase1_pipeline` first.")
    st.stop()

latest_run = runs.iloc[0]
st.subheader("Latest Pipeline Run")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Run ID", latest_run["run_id"])
col2.metric("Status", latest_run["status"])
col3.metric("Rows Ingested", f"{latest_run['rows_ingested']:,}" if pd.notna(latest_run["rows_ingested"]) else "N/A")
col4.metric("Rows Loaded", f"{latest_run['rows_loaded']:,}" if pd.notna(latest_run["rows_loaded"]) else "N/A")

st.divider()
dq_log = load_data_quality_log()

if dq_log.empty:
    st.info("No data quality checks logged for the latest run.")
else:
    summary = dq_log.groupby("status").size()
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("✅ Pass", int(summary.get("pass", 0)))
    col_b.metric("⚠️ Warn", int(summary.get("warn", 0)))
    col_c.metric("❌ Fail", int(summary.get("fail", 0)))

    if summary.get("fail", 0) > 0:
        st.error(f"{summary['fail']} check(s) failed in the latest run -- investigate before trusting downstream results.")
    else:
        st.success("No failed checks in the latest run.")

    st.divider()
    st.subheader("Checks by Pipeline Stage")
    stage_summary = dq_log.groupby(["stage", "status"]).size().unstack(fill_value=0)
    st.bar_chart(stage_summary, height=350)

    st.divider()
    st.subheader("All Checks (Latest Run)")
    status_filter = st.multiselect("Filter by status", ["pass", "warn", "fail"], default=["warn", "fail"])
    filtered_log = dq_log[dq_log["status"].isin(status_filter)] if status_filter else dq_log
    st.dataframe(
        filtered_log.rename(columns={
            "stage": "Stage", "check_name": "Check", "status": "Status",
            "records_checked": "Records Checked", "records_failed": "Records Failed", "details": "Details",
        })[["Stage", "Check", "Status", "Records Checked", "Records Failed", "Details"]],
        hide_index=True, width="stretch", height=400,
    )

st.divider()
st.subheader("Pipeline Run History")
st.dataframe(
    runs.rename(columns={
        "run_id": "Run ID", "started_at": "Started", "finished_at": "Finished",
        "status": "Status", "rows_ingested": "Rows Ingested", "rows_loaded": "Rows Loaded", "notes": "Notes",
    }),
    hide_index=True, width="stretch",
)
