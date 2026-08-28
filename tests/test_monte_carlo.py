"""Tests for simulation.monte_carlo."""
from __future__ import annotations

import numpy as np
import pytest

from simulation.monte_carlo import (
    fit_normal_from_quantiles, simulate_horizon_demand, evaluate_order_quantities,
)
from optimization.cost_model import build_store_cost_profile


def test_fit_normal_from_quantiles_mu_equals_p50():
    mu, sigma = fit_normal_from_quantiles(p10=80, p50=100, p90=120)
    assert mu == 100
    assert sigma > 0


def test_fit_normal_from_quantiles_wider_range_gives_larger_sigma():
    _, sigma_narrow = fit_normal_from_quantiles(90, 100, 110)
    _, sigma_wide = fit_normal_from_quantiles(50, 100, 150)
    assert sigma_wide > sigma_narrow


def test_simulate_horizon_demand_shape_and_nonnegativity():
    daily_p10 = np.array([80, 80, 80])
    daily_p50 = np.array([100, 100, 100])
    daily_p90 = np.array([120, 120, 120])
    total = simulate_horizon_demand(daily_p10, daily_p50, daily_p90, n_simulations=1000)
    assert total.shape == (1000,)
    assert (total >= 0).all()


def test_simulate_horizon_demand_reproducible_with_same_seed():
    daily_p10 = np.array([80, 80])
    daily_p50 = np.array([100, 100])
    daily_p90 = np.array([120, 120])
    total1 = simulate_horizon_demand(daily_p10, daily_p50, daily_p90, n_simulations=500, random_state=1)
    total2 = simulate_horizon_demand(daily_p10, daily_p50, daily_p90, n_simulations=500, random_state=1)
    np.testing.assert_array_equal(total1, total2)


def test_simulate_horizon_demand_mean_near_sum_of_p50():
    daily_p10 = np.array([80] * 10)
    daily_p50 = np.array([100] * 10)
    daily_p90 = np.array([120] * 10)
    total = simulate_horizon_demand(daily_p10, daily_p50, daily_p90, n_simulations=20000, random_state=1)
    # Mean of simulated total demand should be close to sum of daily medians
    assert total.mean() == pytest.approx(1000, rel=0.05)


def test_evaluate_order_quantities_higher_qty_reduces_stockout_risk():
    rng = np.random.default_rng(0)
    simulated_demand = rng.normal(1000, 150, size=5000)
    cost_profile = build_store_cost_profile(1, avg_sales_per_customer=10.0)
    candidates = np.array([0, 500, 1000, 1500, 2000])
    df = evaluate_order_quantities(simulated_demand, candidates, starting_inventory=0,
                                    cost_profile=cost_profile)
    # stockout probability should be monotonically non-increasing as order qty increases
    assert (df["stockout_probability"].diff().dropna() <= 1e-9).all()


def test_evaluate_order_quantities_higher_qty_increases_holding_cost():
    rng = np.random.default_rng(0)
    simulated_demand = rng.normal(1000, 150, size=5000)
    cost_profile = build_store_cost_profile(1, avg_sales_per_customer=10.0)
    candidates = np.array([0, 1000, 2000, 3000])
    df = evaluate_order_quantities(simulated_demand, candidates, starting_inventory=0,
                                    cost_profile=cost_profile)
    assert (df["expected_holding_cost"].diff().dropna() >= -1e-9).all()


def test_evaluate_order_quantities_zero_order_has_max_stockout_risk_given_zero_inventory():
    rng = np.random.default_rng(0)
    simulated_demand = rng.normal(1000, 150, size=5000)
    cost_profile = build_store_cost_profile(1, avg_sales_per_customer=10.0)
    candidates = np.array([0, 500, 1000])
    df = evaluate_order_quantities(simulated_demand, candidates, starting_inventory=0,
                                    cost_profile=cost_profile)
    zero_row = df[df["order_qty"] == 0].iloc[0]
    assert zero_row["stockout_probability"] > 0.9  # almost certain stockout with no inventory/order
