"""Tests for optimization.portfolio_optimizer."""
from __future__ import annotations

import pandas as pd
import pytest

from optimization.portfolio_optimizer import optimize_portfolio_allocation


def _make_curve(order_qtys, unit_cost, base_holding=1.0, base_stockout=50.0):
    """Build a synthetic cost curve where cost decreases with qty up to a
    point (stockout reduction) then increases (holding cost dominates)."""
    rows = []
    for q in order_qtys:
        stockout_cost = max(base_stockout - q * 0.05, 0)
        holding_cost = q * base_holding * 0.01
        procurement_cost = q * unit_cost
        rows.append({
            "order_qty": q,
            "procurement_cost": procurement_cost,
            "expected_total_cost": procurement_cost + stockout_cost + holding_cost,
            "stockout_probability": max(0.5 - q * 0.0005, 0.0),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def two_store_curves():
    return {
        1: _make_curve([0, 100, 200, 300, 400], unit_cost=5.0),
        2: _make_curve([0, 100, 200, 300, 400], unit_cost=8.0),
    }


def test_portfolio_optimizer_selects_exactly_one_candidate_per_store(two_store_curves):
    result = optimize_portfolio_allocation(two_store_curves, total_budget=None, total_capacity=None)
    assert len(result.allocations) == len(two_store_curves)
    assert set(result.allocations["store_id"]) == set(two_store_curves.keys())


def test_portfolio_optimizer_respects_budget_constraint(two_store_curves):
    result = optimize_portfolio_allocation(two_store_curves, total_budget=1000, total_capacity=None)
    assert result.total_procurement_spend <= 1000 + 1e-6
    assert result.status == "Optimal"


def test_portfolio_optimizer_respects_capacity_constraint(two_store_curves):
    result = optimize_portfolio_allocation(two_store_curves, total_budget=None, total_capacity=300)
    assert result.total_units_ordered <= 300 + 1e-6


def test_portfolio_optimizer_unconstrained_picks_min_cost_per_store(two_store_curves):
    result = optimize_portfolio_allocation(two_store_curves, total_budget=None, total_capacity=None)
    for store_id, curve in two_store_curves.items():
        expected_min_cost = curve["expected_total_cost"].min()
        chosen_cost = result.allocations[result.allocations["store_id"] == store_id]["expected_total_cost"].iloc[0]
        assert chosen_cost == pytest.approx(expected_min_cost)


def test_portfolio_optimizer_tight_budget_forces_tradeoffs(two_store_curves):
    """With a budget too small for both stores to get their individually
    optimal quantity, the optimizer should still return a feasible,
    optimal (not just any feasible) solution."""
    result = optimize_portfolio_allocation(two_store_curves, total_budget=500, total_capacity=None)
    assert result.status == "Optimal"
    assert result.total_procurement_spend <= 500 + 1e-6


def test_portfolio_optimizer_downsamples_large_candidate_sets():
    big_curve = _make_curve(list(range(0, 2000, 10)), unit_cost=3.0)
    curves = {1: big_curve}
    result = optimize_portfolio_allocation(curves, total_budget=None, total_capacity=None,
                                            max_candidates_per_store=10)
    assert result.status == "Optimal"
    assert len(result.allocations) == 1
