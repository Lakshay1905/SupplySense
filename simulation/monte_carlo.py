"""
Monte Carlo simulation engine.

For a store facing a lead-time-plus-review-period demand distribution
(derived from the Phase 2 probabilistic forecast, P10/P50/P90), simulate
many draws of total demand over that horizon, then evaluate a set of
candidate order quantities against those simulated draws to estimate:

    - stockout probability
    - achieved service level (1 - stockout probability, cycle service level)
    - expected holding cost
    - expected stockout cost
    - expected total cost

This is what lets the optimizer choose an order quantity based on actual
simulated risk/cost tradeoffs rather than a naive `forecast * safety_factor`
heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimization.cost_model import StoreCostProfile

N_SIMULATIONS_DEFAULT = 5000


def fit_normal_from_quantiles(p10: float, p50: float, p90: float) -> tuple[float, float]:
    """Approximate a Normal(mu, sigma) from P10/P50/P90 (used to sample
    simulated demand). z-scores for 10th/90th percentiles: +/-1.2816."""
    mu = p50
    z90 = 1.2816
    sigma = max((p90 - p10) / (2 * z90), 1e-6)
    return mu, sigma


def simulate_horizon_demand(daily_p10: np.ndarray, daily_p50: np.ndarray, daily_p90: np.ndarray,
                             n_simulations: int = N_SIMULATIONS_DEFAULT,
                             random_state: int = 42) -> np.ndarray:
    """Simulate total demand over a multi-day horizon by sampling each
    day's demand from its own fitted Normal(mu, sigma) (clipped at 0) and
    summing across days -- captures day-specific uncertainty (e.g. wider
    bands on volatile/promo days) rather than a single flat distribution.
    Returns an array of shape (n_simulations,) of total horizon demand.
    """
    rng = np.random.default_rng(random_state)
    horizon = len(daily_p50)
    total = np.zeros(n_simulations)
    for t in range(horizon):
        mu, sigma = fit_normal_from_quantiles(daily_p10[t], daily_p50[t], daily_p90[t])
        draws = rng.normal(mu, sigma, size=n_simulations)
        total += np.clip(draws, 0, None)
    return total


@dataclass
class SimulationResult:
    order_qty: float
    stockout_probability: float
    service_level: float
    expected_holding_cost: float
    expected_stockout_cost: float
    expected_total_cost: float
    expected_units_short: float
    expected_units_excess: float


def evaluate_order_quantities(
    simulated_demand: np.ndarray,
    candidate_qtys: np.ndarray,
    starting_inventory: float,
    cost_profile: StoreCostProfile,
    procurement_included: bool = True,
) -> pd.DataFrame:
    """For each candidate order quantity, compute simulated cost/risk
    metrics against the simulated demand draws.

    Available-to-sell for a given order qty Q = starting_inventory + Q.
    Shortfall = max(0, demand - available). Excess = max(0, available - demand).
    """
    rows = []
    for q in candidate_qtys:
        available = starting_inventory + q
        shortfall = np.clip(simulated_demand - available, 0, None)
        excess = np.clip(available - simulated_demand, 0, None)

        stockout_prob = float(np.mean(shortfall > 0))
        service_level = 1 - stockout_prob
        expected_short = float(np.mean(shortfall))
        expected_excess = float(np.mean(excess))

        expected_holding_cost = expected_excess * cost_profile.holding_cost_per_unit_per_day
        expected_stockout_cost = expected_short * cost_profile.stockout_cost_per_unit
        procurement_cost = q * cost_profile.unit_cost if procurement_included else 0.0
        expected_total_cost = procurement_cost + expected_holding_cost + expected_stockout_cost

        rows.append(SimulationResult(
            order_qty=q, stockout_probability=stockout_prob, service_level=service_level,
            expected_holding_cost=expected_holding_cost, expected_stockout_cost=expected_stockout_cost,
            expected_total_cost=expected_total_cost, expected_units_short=expected_short,
            expected_units_excess=expected_excess,
        ).__dict__)
    return pd.DataFrame(rows)
