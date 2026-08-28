"""
Per-store inventory optimization engine.

This is the core differentiator of SupplySense: NOT
`order_quantity = forecast * safety_factor`, but a genuine search over
candidate order quantities, evaluated by Monte Carlo simulation against
real cost/risk tradeoffs, subject to real operational constraints (MOQ,
order multiple, warehouse capacity, procurement budget).

Algorithm:
  1. Pull the store's P10/P50/P90 forecast for the decision horizon
     (lead time + review period) from the `forecasts` table.
  2. Monte Carlo simulate total horizon demand from those daily bands.
  3. Build a grid of candidate order quantities (0 .. max feasible,
     stepped by the order multiple).
  4. Evaluate each candidate's expected cost via simulation.
  5. Filter to feasible candidates (>= MOQ or exactly 0, within budget,
     within capacity headroom).
  6. Pick the feasible candidate with lowest expected total cost.
  7. Report the decision with explicit "drivers" (why this number).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import BUSINESS_DEFAULTS
from optimization.cost_model import StoreCostProfile
from simulation.monte_carlo import simulate_horizon_demand, evaluate_order_quantities


@dataclass
class OptimizationConstraints:
    moq_units: float = BUSINESS_DEFAULTS["moq_units"]
    order_multiple: float = BUSINESS_DEFAULTS["order_multiple_units"]
    warehouse_capacity_units: float | None = None   # None = unconstrained
    procurement_budget: float | None = None          # None = unconstrained
    target_service_level: float = BUSINESS_DEFAULTS["target_service_level"]


@dataclass
class OptimizationResult:
    store_id: int
    recommended_order_qty: float
    expected_total_cost: float
    expected_holding_cost: float
    expected_stockout_cost: float
    procurement_cost: float
    stockout_probability: float
    achieved_service_level: float
    starting_inventory: float
    horizon_days: int
    demand_forecast_units_p50: float
    demand_change_pct: float
    budget_status: str
    capacity_status: str
    within_target_service_level: bool
    drivers: dict = field(default_factory=dict)
    full_cost_curve: pd.DataFrame | None = None


def _candidate_grid(max_qty: float, order_multiple: float) -> np.ndarray:
    if max_qty <= 0:
        return np.array([0.0])
    n_steps = int(max_qty // order_multiple)
    grid = np.arange(0, n_steps + 1) * order_multiple
    return grid.astype(float)


def optimize_order_quantity(
    store_id: int,
    daily_p10_units: np.ndarray,
    daily_p50_units: np.ndarray,
    daily_p90_units: np.ndarray,
    starting_inventory: float,
    incoming_inventory: float,
    cost_profile: StoreCostProfile,
    constraints: OptimizationConstraints,
    recent_avg_daily_units: float,
    n_simulations: int = 5000,
    random_state: int = 42,
) -> OptimizationResult:
    horizon_days = len(daily_p50_units)
    available_now = starting_inventory + incoming_inventory

    simulated_demand = simulate_horizon_demand(
        daily_p10_units, daily_p50_units, daily_p90_units, n_simulations, random_state
    )

    # Determine the max order quantity considered, bounded by capacity headroom if given
    theoretical_max = max(simulated_demand.max() - available_now, constraints.moq_units * 3, 1.0)
    if constraints.warehouse_capacity_units is not None:
        capacity_headroom = max(constraints.warehouse_capacity_units - available_now, 0.0)
        max_qty = min(theoretical_max, capacity_headroom)
    else:
        max_qty = theoretical_max

    candidates = _candidate_grid(max_qty, constraints.order_multiple)

    cost_curve = evaluate_order_quantities(simulated_demand, candidates, available_now, cost_profile)
    cost_curve["procurement_cost"] = cost_curve["order_qty"] * cost_profile.unit_cost

    # --- feasibility filtering ---
    feasible = cost_curve.copy()
    feasible = feasible[(feasible["order_qty"] == 0) | (feasible["order_qty"] >= constraints.moq_units)]

    budget_status = "no_budget_constraint"
    if constraints.procurement_budget is not None:
        affordable = feasible["procurement_cost"] <= constraints.procurement_budget
        if affordable.any():
            feasible = feasible[affordable]
            budget_status = "within_budget"
        else:
            # Nothing affordable except possibly 0 -- fall back to the
            # cheapest affordable non-zero candidate below budget, or 0.
            feasible = feasible[feasible["order_qty"] == 0]
            budget_status = "budget_binding"

    if feasible.empty:
        feasible = cost_curve[cost_curve["order_qty"] == 0]

    # Service-level-aware selection: among feasible (budget/capacity/MOQ
    # respecting) candidates, prefer the cheapest one that ALSO meets the
    # target service level. Only fall back to pure cost-minimization
    # (which may under-shoot the target) if no feasible candidate can
    # reach it -- this is flagged explicitly via `within_target_service_level`
    # so the business always sees when the target isn't achievable under
    # current constraints, rather than the optimizer silently choosing a
    # cheaper, riskier quantity.
    meets_target = feasible[feasible["service_level"] >= constraints.target_service_level]
    service_level_achievable = not meets_target.empty
    if service_level_achievable:
        best = meets_target.loc[meets_target["expected_total_cost"].idxmin()]
    else:
        best = feasible.loc[feasible["expected_total_cost"].idxmin()]

    capacity_status = "no_capacity_constraint"
    if constraints.warehouse_capacity_units is not None:
        projected_inventory = available_now + best["order_qty"]
        capacity_status = (
            "within_capacity" if projected_inventory <= constraints.warehouse_capacity_units
            else "capacity_binding"
        )

    demand_p50_total = float(np.sum(daily_p50_units))
    demand_change_pct = (
        (np.mean(daily_p50_units) - recent_avg_daily_units) / recent_avg_daily_units * 100
        if recent_avg_daily_units > 0 else 0.0
    )

    drivers = {
        "demand_change_pct": round(float(demand_change_pct), 1),
        "starting_inventory": round(float(starting_inventory), 1),
        "incoming_inventory": round(float(incoming_inventory), 1),
        "horizon_days": int(horizon_days),
        "target_service_level_pct": round(float(constraints.target_service_level) * 100, 1),
        "achieved_stockout_risk_pct": round(float(best["stockout_probability"]) * 100, 1),
        "budget_status": budget_status,
        "capacity_status": capacity_status,
        "unit_cost": round(float(cost_profile.unit_cost), 2),
        "unit_price_proxy": round(float(cost_profile.unit_price), 2),
        "service_level_achievable_under_constraints": bool(service_level_achievable),
    }

    return OptimizationResult(
        store_id=store_id,
        recommended_order_qty=float(best["order_qty"]),
        expected_total_cost=float(best["expected_total_cost"]),
        expected_holding_cost=float(best["expected_holding_cost"]),
        expected_stockout_cost=float(best["expected_stockout_cost"]),
        procurement_cost=float(best["procurement_cost"]),
        stockout_probability=float(best["stockout_probability"]),
        achieved_service_level=float(best["service_level"]),
        starting_inventory=starting_inventory,
        horizon_days=horizon_days,
        demand_forecast_units_p50=demand_p50_total,
        demand_change_pct=demand_change_pct,
        budget_status=budget_status,
        capacity_status=capacity_status,
        within_target_service_level=bool(best["service_level"] >= constraints.target_service_level),
        drivers=drivers,
        full_cost_curve=cost_curve,
    )
