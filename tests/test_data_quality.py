"""Tests for analytics.metrics.data_quality."""
from __future__ import annotations

import pandas as pd

from pipelines.validation.validators import CheckResult, ValidationReport
from analytics.metrics.data_quality import summarize_report, reports_to_log_rows


def _make_report(stage: str) -> ValidationReport:
    checks = [
        CheckResult("nulls::A", "pass", 100, 0, "ok"),
        CheckResult("nulls::B", "warn", 100, 2, "minor issue"),
        CheckResult("range::C", "fail", 100, 10, "big issue"),
    ]
    return ValidationReport(stage=stage, checks=checks)


def test_summarize_report_counts_statuses_correctly():
    report = _make_report("test_stage")
    summary = summarize_report(report)
    assert summary.n_pass == 1
    assert summary.n_warn == 1
    assert summary.n_fail == 1
    assert summary.records_failed == 12
    assert summary.records_checked == 100


def test_summarize_report_pass_rate():
    report = _make_report("test_stage")
    summary = summarize_report(report)
    assert abs(summary.pass_rate - (1 / 3)) < 1e-9


def test_summarize_empty_report_has_full_pass_rate():
    empty_report = ValidationReport(stage="empty", checks=[])
    summary = summarize_report(empty_report)
    assert summary.pass_rate == 1.0
    assert summary.records_checked == 0


def test_reports_to_log_rows_flattens_multiple_reports():
    r1 = _make_report("stage1")
    r2 = _make_report("stage2")
    rows = reports_to_log_rows("run_123", [r1, r2])
    assert len(rows) == 6
    assert set(rows["stage"]) == {"stage1", "stage2"}
    assert (rows["run_id"] == "run_123").all()
    assert list(rows.columns) == ["run_id", "stage", "check_name", "status",
                                   "records_checked", "records_failed", "details"]


def test_reports_to_log_rows_handles_empty_list():
    rows = reports_to_log_rows("run_123", [])
    assert len(rows) == 0
    assert "run_id" in rows.columns
