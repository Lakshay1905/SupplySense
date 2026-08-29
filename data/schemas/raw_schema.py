"""
Declarative schema definitions for the raw Rossmann source files.

These are used by pipelines.validation to check that incoming data matches
expectations before it is cleaned/transformed/loaded. Keeping the schema
declarative (rather than hardcoded checks scattered through the pipeline)
makes it easy to see -- and unit test -- exactly what "valid" means.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ColumnSpec:
    name: str
    dtype: str                      # "int", "float", "str", "date", "bool"
    nullable: bool = False
    allowed_values: set[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None


TRAIN_SCHEMA: list[ColumnSpec] = [
    ColumnSpec("Store", "int", nullable=False, min_value=1),
    ColumnSpec("DayOfWeek", "int", nullable=False, min_value=1, max_value=7),
    ColumnSpec("Date", "date", nullable=False),
    ColumnSpec("Sales", "float", nullable=False, min_value=0),
    ColumnSpec("Customers", "float", nullable=False, min_value=0),
    ColumnSpec("Open", "int", nullable=False, allowed_values={0, 1}),
    ColumnSpec("Promo", "int", nullable=False, allowed_values={0, 1}),
    ColumnSpec("StateHoliday", "str", nullable=False, allowed_values={"0", "a", "b", "c"}),
    ColumnSpec("SchoolHoliday", "int", nullable=False, allowed_values={0, 1}),
]

STORE_SCHEMA: list[ColumnSpec] = [
    ColumnSpec("Store", "int", nullable=False, min_value=1),
    ColumnSpec("StoreType", "str", nullable=False, allowed_values={"a", "b", "c", "d"}),
    ColumnSpec("Assortment", "str", nullable=False, allowed_values={"a", "b", "c"}),
    ColumnSpec("CompetitionDistance", "float", nullable=True, min_value=0),
    ColumnSpec("CompetitionOpenSinceMonth", "float", nullable=True, min_value=1, max_value=12),
    ColumnSpec("CompetitionOpenSinceYear", "float", nullable=True, min_value=1900),
    ColumnSpec("Promo2", "int", nullable=False, allowed_values={0, 1}),
    ColumnSpec("Promo2SinceWeek", "float", nullable=True, min_value=1, max_value=53),
    ColumnSpec("Promo2SinceYear", "float", nullable=True, min_value=1900),
    ColumnSpec("PromoInterval", "str", nullable=True),
]

STORE_STATES_SCHEMA: list[ColumnSpec] = [
    ColumnSpec("Store", "int", nullable=False, min_value=1),
    ColumnSpec("State", "str", nullable=False),
]

GERMAN_STATE_NAMES = {
    "BW": "Baden-Wurttemberg", "BY": "Bavaria", "BE": "Berlin", "BB": "Brandenburg",
    "HB": "Bremen", "HH": "Hamburg", "HE": "Hesse", "MV": "Mecklenburg-Vorpommern",
    "NI": "Lower Saxony", "NW": "North Rhine-Westphalia", "RP": "Rhineland-Palatinate",
    "SL": "Saarland", "SN": "Saxony", "ST": "Saxony-Anhalt", "SH": "Schleswig-Holstein",
    "TH": "Thuringia", "HB,NI": "Bremen/Lower Saxony",
}
