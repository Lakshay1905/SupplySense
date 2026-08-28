"""Tests for optimization.cost_model."""
from __future__ import annotations

import pytest

from optimization.cost_model import build_store_cost_profile, StoreCostProfile


def test_build_store_cost_profile_basic_math():
    profile = build_store_cost_profile(store_id=1, avg_sales_per_customer=10.0, margin_rate=0.35)
    assert profile.unit_price == pytest.approx(10.0)
    assert profile.unit_cost == pytest.approx(6.5)
    assert profile.margin_per_unit == pytest.approx(3.5)


def test_build_store_cost_profile_holding_cost_scales_with_unit_cost():
    cheap = build_store_cost_profile(1, avg_sales_per_customer=5.0)
    expensive = build_store_cost_profile(2, avg_sales_per_customer=50.0)
    assert expensive.holding_cost_per_unit_per_day > cheap.holding_cost_per_unit_per_day


def test_build_store_cost_profile_guards_against_zero_price():
    profile = build_store_cost_profile(store_id=1, avg_sales_per_customer=0.0)
    assert profile.unit_price > 0  # guarded, not zero/negative


def test_build_store_cost_profile_guards_against_negative_price():
    profile = build_store_cost_profile(store_id=1, avg_sales_per_customer=-5.0)
    assert profile.unit_price > 0


def test_revenue_to_units_conversion():
    profile = build_store_cost_profile(store_id=1, avg_sales_per_customer=10.0)
    assert profile.revenue_to_units(1000.0) == pytest.approx(100.0)


def test_revenue_to_units_zero_revenue():
    profile = build_store_cost_profile(store_id=1, avg_sales_per_customer=10.0)
    assert profile.revenue_to_units(0.0) == 0.0


def test_higher_stockout_multiplier_increases_stockout_cost():
    low = build_store_cost_profile(1, avg_sales_per_customer=10.0, stockout_cost_multiplier=1.0)
    high = build_store_cost_profile(1, avg_sales_per_customer=10.0, stockout_cost_multiplier=3.0)
    assert high.stockout_cost_per_unit > low.stockout_cost_per_unit
