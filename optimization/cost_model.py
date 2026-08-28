"""
Cost model for inventory decisions.

IMPORTANT DATA LIMITATION, HANDLED EXPLICITLY:
Rossmann's `Sales` field is store *revenue* in EUR, not a unit count --
there is no SKU-level quantity data in the source dataset. To make
MOQ / order-multiple / warehouse-capacity constraints meaningful (they
are naturally expressed in units, not currency), we convert forecasted
revenue into an implied unit count using each store's own historical
average revenue-per-customer-transaction (`sales_per_customer`, computed
directly from real data in Phase 1) as a proxy "unit price" -- i.e. we
treat one customer transaction as approximately one demand "unit" of the
store's assortment. This is a documented, data-grounded modeling choice,
not fabricated data.

Margin, holding-cost rate, and stockout-cost multiplier are genuine
business *parameters* that any real retailer would configure based on
their own P&L -- they are not something a sales-history dataset would
ever contain. Defaults live in `config.settings.BUSINESS_DEFAULTS` and
are clearly surfaced as configurable assumptions everywhere they are
used (UI, reports, AI copilot).
"""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import BUSINESS_DEFAULTS

DEFAULT_MARGIN_RATE = 0.35  # gross margin assumption (documented business parameter)


@dataclass
class StoreCostProfile:
    store_id: int
    unit_price: float          # EUR per unit (proxy: avg revenue per customer)
    unit_cost: float           # EUR per unit procured
    margin_per_unit: float
    holding_cost_per_unit_per_day: float
    stockout_cost_per_unit: float

    def revenue_to_units(self, revenue: float) -> float:
        if self.unit_price <= 0:
            return 0.0
        return revenue / self.unit_price


def build_store_cost_profile(
    store_id: int,
    avg_sales_per_customer: float,
    margin_rate: float = DEFAULT_MARGIN_RATE,
    annual_holding_rate: float = BUSINESS_DEFAULTS["holding_cost_rate_annual"],
    stockout_cost_multiplier: float = BUSINESS_DEFAULTS["stockout_cost_multiplier"],
) -> StoreCostProfile:
    """Derive a per-store cost profile from real historical data
    (avg_sales_per_customer) plus documented business-parameter
    assumptions (margin, holding rate, stockout multiplier)."""
    unit_price = max(avg_sales_per_customer, 0.01)  # guard against 0/negative
    unit_cost = unit_price * (1 - margin_rate)
    margin_per_unit = unit_price - unit_cost
    holding_cost_per_unit_per_day = unit_cost * annual_holding_rate / 365.0
    stockout_cost_per_unit = margin_per_unit * stockout_cost_multiplier

    return StoreCostProfile(
        store_id=store_id,
        unit_price=unit_price,
        unit_cost=unit_cost,
        margin_per_unit=margin_per_unit,
        holding_cost_per_unit_per_day=holding_cost_per_unit_per_day,
        stockout_cost_per_unit=stockout_cost_per_unit,
    )
