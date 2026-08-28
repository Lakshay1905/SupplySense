"""Tests for optimization.inventory_optimizer."""
from __future__ import annotations

import numpy as np
import pytest

from optimization.cost_model import build_store_cost_profile
from optimization.inventory_optimizer import (
    OptimizationConstraints, optimize_order_quantity, _candidate_grid,
)


@pytest.fixture
def cost_profile():
    return build_store_cost_profile(store_id=1, avg_sales_per_customer=10.0)


def test_candidate_grid_includes_zero():
    grid = _candidate_grid(max_qty=100, order_multiple=10)
    assert grid[0] == 0.0


def test_candidate_grid_respects_order_multiple():
    grid = _candidate_grid(max_qty=95, order_multiple=10)
    assert np.all(grid % 10 == 0)


def test_candidate_grid_zero_max_qty_returns_only_zero():
    grid = _candidate_grid(max_qty=0, order_multiple=10)
    np.testing.assert_array_equal(grid, [0.0])


def test_optimizer_recommends_zero_when_inventory_already_covers_demand(cost_profile):
    daily = np.array([100.0] * 14)
    result = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.8, daily_p50_units=daily, daily_p90_units=daily * 1.2,
        starting_inventory=10000, incoming_inventory=0, cost_profile=cost_profile,
        constraints=OptimizationConstraints(), recent_avg_daily_units=100,
    )
    assert result.recommended_order_qty == 0.0
    assert result.achieved_service_level >= 0.95


def test_optimizer_recommends_order_when_inventory_insufficient(cost_profile):
    daily = np.array([1000.0] * 14)
    result = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.8, daily_p50_units=daily, daily_p90_units=daily * 1.2,
        starting_inventory=0, incoming_inventory=0, cost_profile=cost_profile,
        constraints=OptimizationConstraints(), recent_avg_daily_units=1000,
    )
    assert result.recommended_order_qty > 0
    assert result.achieved_service_level >= 0.90  # should target ~95%


def test_optimizer_respects_moq_constraint(cost_profile):
    """A tiny demand shortfall should either be rounded up to at least MOQ,
    or the optimizer should order nothing (never order below MOQ)."""
    daily = np.array([5.0] * 7)
    constraints = OptimizationConstraints(moq_units=200)
    result = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.9, daily_p50_units=daily, daily_p90_units=daily * 1.1,
        starting_inventory=0, incoming_inventory=0, cost_profile=cost_profile,
        constraints=constraints, recent_avg_daily_units=5,
    )
    assert result.recommended_order_qty == 0 or result.recommended_order_qty >= 200


def test_optimizer_respects_budget_constraint(cost_profile):
    daily = np.array([1000.0] * 14)
    unconstrained = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.8, daily_p50_units=daily, daily_p90_units=daily * 1.2,
        starting_inventory=0, incoming_inventory=0, cost_profile=cost_profile,
        constraints=OptimizationConstraints(), recent_avg_daily_units=1000,
    )
    tight_budget = unconstrained.procurement_cost * 0.3
    constrained = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.8, daily_p50_units=daily, daily_p90_units=daily * 1.2,
        starting_inventory=0, incoming_inventory=0, cost_profile=cost_profile,
        constraints=OptimizationConstraints(procurement_budget=tight_budget), recent_avg_daily_units=1000,
    )
    assert constrained.procurement_cost <= tight_budget + 1e-6
    assert constrained.recommended_order_qty <= unconstrained.recommended_order_qty


def test_optimizer_respects_capacity_constraint(cost_profile):
    daily = np.array([1000.0] * 14)
    tight_capacity = 500  # far below what would be needed
    result = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.8, daily_p50_units=daily, daily_p90_units=daily * 1.2,
        starting_inventory=0, incoming_inventory=0, cost_profile=cost_profile,
        constraints=OptimizationConstraints(warehouse_capacity_units=tight_capacity),
        recent_avg_daily_units=1000,
    )
    assert result.starting_inventory + result.recommended_order_qty <= tight_capacity + 1e-6


def test_optimizer_flags_unachievable_service_level_under_tight_budget(cost_profile):
    daily = np.array([1000.0] * 14)
    result = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.8, daily_p50_units=daily, daily_p90_units=daily * 1.2,
        starting_inventory=0, incoming_inventory=0, cost_profile=cost_profile,
        constraints=OptimizationConstraints(procurement_budget=1.0),  # essentially no budget
        recent_avg_daily_units=1000,
    )
    assert result.within_target_service_level is False
    assert result.drivers["service_level_achievable_under_constraints"] is False


def test_optimizer_drivers_contains_expected_keys(cost_profile):
    daily = np.array([500.0] * 14)
    result = optimize_order_quantity(
        store_id=1, daily_p10_units=daily * 0.8, daily_p50_units=daily, daily_p90_units=daily * 1.2,
        starting_inventory=1000, incoming_inventory=0, cost_profile=cost_profile,
        constraints=OptimizationConstraints(), recent_avg_daily_units=500,
    )
    for key in ["demand_change_pct", "starting_inventory", "target_service_level_pct",
                "achieved_stockout_risk_pct", "budget_status", "capacity_status"]:
        assert key in result.drivers
