"""
Aggregate data-quality metrics for the Data Quality UI tab (Phase 4) and
for persisting run history to `data_quality_log` / `pipeline_runs`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pipelines.validation.validators import ValidationReport


@dataclass
class DataQualitySummary:
    stage: str
    records_checked: int
    records_failed: int
    n_pass: int
    n_warn: int
    n_fail: int

    @property
    def pass_rate(self) -> float:
        total = self.n_pass + self.n_warn + self.n_fail
        return self.n_pass / total if total else 1.0


def summarize_report(report: ValidationReport) -> DataQualitySummary:
    frame = report.to_frame()
    return DataQualitySummary(
        stage=report.stage,
        records_checked=int(frame["records_checked"].max()) if len(frame) else 0,
        records_failed=int(frame["records_failed"].sum()) if len(frame) else 0,
        n_pass=int((frame["status"] == "pass").sum()) if len(frame) else 0,
        n_warn=int((frame["status"] == "warn").sum()) if len(frame) else 0,
        n_fail=int((frame["status"] == "fail").sum()) if len(frame) else 0,
    )


def reports_to_log_rows(run_id: str, reports: list[ValidationReport]) -> pd.DataFrame:
    """Flatten a list of ValidationReports into rows for data_quality_log."""
    frames = []
    for report in reports:
        f = report.to_frame()
        if len(f) == 0:
            continue
        f.insert(0, "run_id", run_id)
        f.insert(1, "stage", report.stage)
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=["run_id", "stage", "check_name", "status",
                                      "records_checked", "records_failed", "details"])
    out = pd.concat(frames, ignore_index=True)
    return out[["run_id", "stage", "check_name", "status", "records_checked",
                "records_failed", "details"]]
