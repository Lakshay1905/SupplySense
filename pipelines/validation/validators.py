"""
Data validation engine.

Runs a battery of checks (schema/type conformance, range checks, null
checks, duplicate checks, referential integrity, anomaly checks) against a
DataFrame and returns a structured, serializable report. Every check
produces a `CheckResult` so results can be logged to `data_quality_log`
and rendered in the "Data Quality" UI tab in Phase 4.

Design goal: validation never silently swallows problems. Every issue is
recorded with counts and a human-readable message, and the pipeline
decides (in pipelines/transformation) how to react (drop, impute, flag).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from data.schemas.raw_schema import ColumnSpec
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CheckResult:
    check_name: str
    status: str            # "pass" | "warn" | "fail"
    records_checked: int
    records_failed: int
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    stage: str
    checks: list[CheckResult]

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def summary(self) -> dict[str, int]:
        summary = {"pass": 0, "warn": 0, "fail": 0}
        for c in self.checks:
            summary[c.status] += 1
        return summary

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.checks])


def _coerce_check(df: pd.DataFrame, col: ColumnSpec) -> CheckResult:
    """Check a column can be coerced to its declared dtype."""
    n = len(df)
    if col.name not in df.columns:
        return CheckResult(f"dtype::{col.name}", "fail", n, n, f"Column '{col.name}' missing")

    series = df[col.name]
    if col.dtype == "int":
        coerced = pd.to_numeric(series, errors="coerce")
        failed = coerced.isna().sum() - series.isna().sum()
    elif col.dtype == "float":
        coerced = pd.to_numeric(series, errors="coerce")
        failed = coerced.isna().sum() - series.isna().sum()
    elif col.dtype == "date":
        coerced = pd.to_datetime(series, errors="coerce")
        failed = coerced.isna().sum() - series.isna().sum()
    else:
        failed = 0

    status = "pass" if failed == 0 else ("warn" if failed / max(n, 1) < 0.01 else "fail")
    return CheckResult(f"dtype::{col.name}", status, n, int(failed),
                        f"{failed} values in '{col.name}' could not be coerced to {col.dtype}")


def _null_check(df: pd.DataFrame, col: ColumnSpec) -> CheckResult:
    n = len(df)
    if col.name not in df.columns:
        return CheckResult(f"nulls::{col.name}", "fail", n, n, f"Column '{col.name}' missing")
    nulls = int(df[col.name].isna().sum())
    if col.nullable:
        status = "pass"
    else:
        status = "pass" if nulls == 0 else ("warn" if nulls / max(n, 1) < 0.01 else "fail")
    return CheckResult(f"nulls::{col.name}", status, n, nulls,
                        f"{nulls} null values in '{col.name}' (nullable={col.nullable})")


def _range_check(df: pd.DataFrame, col: ColumnSpec) -> CheckResult | None:
    if col.min_value is None and col.max_value is None:
        return None
    if col.name not in df.columns:
        return None
    n = len(df)
    series = pd.to_numeric(df[col.name], errors="coerce")
    mask_fail = pd.Series(False, index=df.index)
    if col.min_value is not None:
        mask_fail |= series < col.min_value
    if col.max_value is not None:
        mask_fail |= series > col.max_value
    failed = int(mask_fail.sum())
    status = "pass" if failed == 0 else ("warn" if failed / max(n, 1) < 0.01 else "fail")
    return CheckResult(f"range::{col.name}", status, n, failed,
                        f"{failed} values in '{col.name}' outside "
                        f"[{col.min_value}, {col.max_value}]")


def _allowed_values_check(df: pd.DataFrame, col: ColumnSpec) -> CheckResult | None:
    if col.allowed_values is None or col.name not in df.columns:
        return None
    n = len(df)
    mask_fail = ~df[col.name].astype(str).isin({str(v) for v in col.allowed_values}) & df[col.name].notna()
    failed = int(mask_fail.sum())
    status = "pass" if failed == 0 else ("warn" if failed / max(n, 1) < 0.01 else "fail")
    return CheckResult(f"allowed_values::{col.name}", status, n, failed,
                        f"{failed} values in '{col.name}' outside {col.allowed_values}")


def _duplicate_check(df: pd.DataFrame, subset: list[str]) -> CheckResult:
    n = len(df)
    dup_mask = df.duplicated(subset=subset, keep="first")
    failed = int(dup_mask.sum())
    status = "pass" if failed == 0 else "warn"
    return CheckResult(f"duplicates::{'+'.join(subset)}", status, n, failed,
                        f"{failed} duplicate rows on {subset}")


def _referential_check(child: pd.DataFrame, child_key: str, parent: pd.DataFrame,
                        parent_key: str, name: str) -> CheckResult:
    n = len(child)
    orphans = ~child[child_key].isin(parent[parent_key])
    failed = int(orphans.sum())
    status = "pass" if failed == 0 else "fail"
    return CheckResult(f"referential::{name}", status, n, failed,
                        f"{failed} rows in child reference a {parent_key} not present in parent")


def _anomaly_check_negative_or_extreme(df: pd.DataFrame, col: str, z_thresh: float = 8.0) -> CheckResult:
    """Flag statistically extreme values (|z-score| > threshold) as anomalies."""
    n = len(df)
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.std(ddof=0) == 0 or len(series) == 0:
        return CheckResult(f"anomaly::{col}", "pass", n, 0, "No variance to test")
    z = (series - series.mean()) / series.std(ddof=0)
    failed = int((z.abs() > z_thresh).sum())
    status = "pass" if failed == 0 else "warn"
    return CheckResult(f"anomaly::{col}", status, n, failed,
                        f"{failed} extreme outliers in '{col}' (|z| > {z_thresh})")


def validate_dataframe(df: pd.DataFrame, schema: list[ColumnSpec], stage: str,
                        dedupe_subset: list[str] | None = None) -> ValidationReport:
    """Run the full battery of schema checks against a DataFrame."""
    checks: list[CheckResult] = []
    for col in schema:
        checks.append(_coerce_check(df, col))
        checks.append(_null_check(df, col))
        rc = _range_check(df, col)
        if rc:
            checks.append(rc)
        avc = _allowed_values_check(df, col)
        if avc:
            checks.append(avc)

    if dedupe_subset:
        checks.append(_duplicate_check(df, dedupe_subset))

    report = ValidationReport(stage=stage, checks=checks)
    logger.info("Validation[%s]: %s", stage, report.summary)
    return report


def validate_referential_integrity(train: pd.DataFrame, store: pd.DataFrame) -> ValidationReport:
    checks = [_referential_check(train, "Store", store, "Store", "train_to_store")]
    report = ValidationReport(stage="referential_integrity", checks=checks)
    logger.info("Validation[referential_integrity]: %s", report.summary)
    return report


def validate_anomalies(df: pd.DataFrame, numeric_cols: list[str]) -> ValidationReport:
    checks = [_anomaly_check_negative_or_extreme(df, c) for c in numeric_cols if c in df.columns]
    report = ValidationReport(stage="anomaly_detection", checks=checks)
    logger.info("Validation[anomaly_detection]: %s", report.summary)
    return report
