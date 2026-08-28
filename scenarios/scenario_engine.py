"""
Scenario / what-if engine.

Lets a business user change assumptions (demand shift, supplier lead
time, budget, capacity, promotion) and see how the recommended order
quantity, cost, and risk change relative to the baseline -- computed by
re-running the same optimization engine with modified inputs, not by
applying a hand-wavy multiplier to the baseline output.
"""
from __future__ import annotations

from dataclasses import dataclass

from optimization.inventory_optimizer import OptimizationResult
from optimization.recommendation_engine import generate_recommendation


@dataclass
class ScenarioDefinition:
    name: str
    demand_multiplier: float = 1.0
    lead_time_override: float | None = None
    budget_multiplier: float | None = None        # relative to baseline budget, if any
    capacity_multiplier: float | None = None      # relative to baseline capacity, if any
    promo_uplift_pct: float = 0.0
    promo_duration_days: int = 0


@dataclass
class ScenarioComparison:
    scenario_name: str
    store_id: int
    baseline: dict
    scenario: dict
    deltas: dict


def _result_to_dict(result: OptimizationResult) -> dict:
    return {
        "recommended_order_qty": round(result.recommended_order_qty, 1),
        "expected_total_cost": round(result.expected_total_cost, 2),
        "expected_holding_cost": round(result.expected_holding_cost, 2),
        "expected_stockout_cost": round(result.expected_stockout_cost, 2),
        "procurement_cost": round(result.procurement_cost, 2),
        "stockout_probability_pct": round(result.stockout_probability * 100, 2),
        "achieved_service_level_pct": round(result.achieved_service_level * 100, 2),
        "within_target_service_level": bool(result.within_target_service_level),
        "budget_status": result.budget_status,
        "capacity_status": result.capacity_status,
    }


def run_scenario(
    store_id: int,
    scenario: ScenarioDefinition,
    baseline_budget: float | None = None,
    baseline_capacity: float | None = None,
    target_service_level: float = 0.95,
) -> ScenarioComparison:
    baseline_result = generate_recommendation(
        store_id=store_id, warehouse_capacity_units=baseline_capacity,
        procurement_budget=baseline_budget, target_service_level=target_service_level,
    )

    scenario_budget = (
        baseline_budget * scenario.budget_multiplier
        if baseline_budget is not None and scenario.budget_multiplier is not None
        else baseline_budget
    )
    scenario_capacity = (
        baseline_capacity * scenario.capacity_multiplier
        if baseline_capacity is not None and scenario.capacity_multiplier is not None
        else baseline_capacity
    )

    scenario_result = generate_recommendation(
        store_id=store_id,
        warehouse_capacity_units=scenario_capacity,
        procurement_budget=scenario_budget,
        target_service_level=target_service_level,
        demand_multiplier=scenario.demand_multiplier,
        lead_time_override=scenario.lead_time_override,
        promo_uplift_pct=scenario.promo_uplift_pct,
        promo_duration_days=scenario.promo_duration_days,
    )

    baseline_dict = _result_to_dict(baseline_result)
    scenario_dict = _result_to_dict(scenario_result)

    deltas = {}
    for key in ["recommended_order_qty", "expected_total_cost", "expected_holding_cost",
                "expected_stockout_cost", "procurement_cost", "stockout_probability_pct",
                "achieved_service_level_pct"]:
        deltas[key] = round(scenario_dict[key] - baseline_dict[key], 2)
        base_val = baseline_dict[key]
        deltas[f"{key}_pct_change"] = (
            round(deltas[key] / base_val * 100, 1) if base_val not in (0, None) else None
        )

    return ScenarioComparison(
        scenario_name=scenario.name, store_id=store_id,
        baseline=baseline_dict, scenario=scenario_dict, deltas=deltas,
    )


# --- Pre-built common scenarios (used by the UI as quick-pick options) ---

PRESET_SCENARIOS = {
    "demand_up_20": ScenarioDefinition(name="Demand +20%", demand_multiplier=1.20),
    "demand_down_15": ScenarioDefinition(name="Demand -15%", demand_multiplier=0.85),
    "lead_time_doubled": ScenarioDefinition(name="Lead time 7 -> 14 days", lead_time_override=14.0),
    "budget_cut_15": ScenarioDefinition(name="Budget -15%", budget_multiplier=0.85),
    "budget_cut_20": ScenarioDefinition(name="Budget -20%", budget_multiplier=0.80),
    "capacity_up_25": ScenarioDefinition(name="Warehouse capacity +25%", capacity_multiplier=1.25),
    "promotion_2week_30pct": ScenarioDefinition(
        name="2-week promotion, +30% uplift", promo_uplift_pct=30.0, promo_duration_days=14,
    ),
}
