"""Tests for pipelines.validation.validators."""
from __future__ import annotations

import pandas as pd
import numpy as np

from pipelines.validation.validators import (
    validate_dataframe, validate_referential_integrity, validate_anomalies,
    ValidationReport,
)
from data.schemas.raw_schema import TRAIN_SCHEMA, STORE_SCHEMA


def test_validate_dataframe_detects_null_in_non_nullable_column():
    df = pd.DataFrame({
        "Store": [1, None, 3],
        "DayOfWeek": [1, 2, 3],
        "Date": ["2015-01-01", "2015-01-02", "2015-01-03"],
        "Sales": [100, 200, 300],
        "Customers": [10, 20, 30],
        "Open": [1, 1, 1],
        "Promo": [0, 0, 0],
        "StateHoliday": ["0", "0", "0"],
        "SchoolHoliday": [0, 0, 0],
    })
    report = validate_dataframe(df, TRAIN_SCHEMA, "test_stage")
    store_null_checks = [c for c in report.checks if c.check_name == "nulls::Store"]
    assert len(store_null_checks) == 1
    # 1/3 null rate on a non-nullable column exceeds the 1% warn threshold -> fail
    assert store_null_checks[0].status == "fail"
    assert store_null_checks[0].records_failed == 1


def test_validate_dataframe_passes_clean_data():
    df = pd.DataFrame({
        "Store": [1, 2, 3],
        "DayOfWeek": [1, 2, 3],
        "Date": ["2015-01-01", "2015-01-02", "2015-01-03"],
        "Sales": [100, 200, 300],
        "Customers": [10, 20, 30],
        "Open": [1, 1, 1],
        "Promo": [0, 1, 0],
        "StateHoliday": ["0", "0", "a"],
        "SchoolHoliday": [0, 0, 1],
    })
    report = validate_dataframe(df, TRAIN_SCHEMA, "test_stage")
    assert not report.has_failures
    assert report.summary["fail"] == 0


def test_allowed_values_check_flags_invalid_category():
    df = pd.DataFrame({
        "Store": [1, 2],
        "StoreType": ["a", "z"],  # 'z' is invalid
        "Assortment": ["a", "b"],
        "CompetitionDistance": [100.0, 200.0],
        "CompetitionOpenSinceMonth": [1, 2],
        "CompetitionOpenSinceYear": [2010, 2011],
        "Promo2": [0, 1],
        "Promo2SinceWeek": [np.nan, 5],
        "Promo2SinceYear": [np.nan, 2015],
        "PromoInterval": [np.nan, "Jan,Apr"],
    })
    report = validate_dataframe(df, STORE_SCHEMA, "test_store")
    store_type_checks = [c for c in report.checks if c.check_name == "allowed_values::StoreType"]
    assert len(store_type_checks) == 1
    assert store_type_checks[0].records_failed == 1


def test_range_check_flags_out_of_range_values():
    df = pd.DataFrame({
        "Store": [1, 2],
        "DayOfWeek": [1, 9],  # 9 is out of [1,7] range
        "Date": ["2015-01-01", "2015-01-02"],
        "Sales": [100, 200],
        "Customers": [10, 20],
        "Open": [1, 1],
        "Promo": [0, 0],
        "StateHoliday": ["0", "0"],
        "SchoolHoliday": [0, 0],
    })
    report = validate_dataframe(df, TRAIN_SCHEMA, "test_stage")
    dow_range_checks = [c for c in report.checks if c.check_name == "range::DayOfWeek"]
    assert dow_range_checks[0].records_failed == 1


def test_duplicate_check_counts_duplicates():
    df = pd.DataFrame({
        "Store": [1, 1, 2],
        "DayOfWeek": [1, 1, 2],
        "Date": ["2015-01-01", "2015-01-01", "2015-01-02"],
        "Sales": [100, 100, 200],
        "Customers": [10, 10, 20],
        "Open": [1, 1, 1],
        "Promo": [0, 0, 0],
        "StateHoliday": ["0", "0", "0"],
        "SchoolHoliday": [0, 0, 0],
    })
    report = validate_dataframe(df, TRAIN_SCHEMA, "test_stage", dedupe_subset=["Store", "Date"])
    dup_checks = [c for c in report.checks if "duplicates" in c.check_name]
    assert len(dup_checks) == 1
    assert dup_checks[0].records_failed == 1


def test_referential_integrity_detects_orphan_store():
    train = pd.DataFrame({"Store": [1, 2, 99]})
    store = pd.DataFrame({"Store": [1, 2]})
    report = validate_referential_integrity(train, store)
    assert report.has_failures
    assert report.checks[0].records_failed == 1


def test_referential_integrity_passes_when_all_present():
    train = pd.DataFrame({"Store": [1, 2, 2]})
    store = pd.DataFrame({"Store": [1, 2]})
    report = validate_referential_integrity(train, store)
    assert not report.has_failures


def test_anomaly_detection_flags_extreme_outlier():
    values = [100] * 100 + [1_000_000]  # one massive outlier
    df = pd.DataFrame({"Sales": values, "Customers": [10] * 101})
    report = validate_anomalies(df, ["Sales", "Customers"])
    sales_check = [c for c in report.checks if c.check_name == "anomaly::Sales"][0]
    assert sales_check.records_failed >= 1


def test_validation_report_summary_and_to_frame():
    report = validate_dataframe(
        pd.DataFrame({
            "Store": [1], "DayOfWeek": [1], "Date": ["2015-01-01"], "Sales": [1],
            "Customers": [1], "Open": [1], "Promo": [0], "StateHoliday": ["0"], "SchoolHoliday": [0],
        }),
        TRAIN_SCHEMA, "test_stage",
    )
    frame = report.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert set(["check_name", "status", "records_checked", "records_failed", "details"]).issubset(frame.columns)
    assert isinstance(report.summary, dict)
