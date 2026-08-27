"""
Ingestion layer: reads raw source files exactly as provided (no cleaning
here -- that happens in pipelines.transformation) and returns DataFrames.

Source: Rossmann Store Sales (public dataset, ~1.7M store-days across 1115
stores in Germany, 2013-01-01 to 2015-07-31). Includes daily sales,
customers, promo flags, holiday flags (train.csv), store attributes
(store.csv) and a store -> federal-state mapping (store_states.csv) used
here as the "region" dimension.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DATA_RAW_DIR
from config.logging_config import get_logger

logger = get_logger(__name__)

TRAIN_FILE = DATA_RAW_DIR / "train.csv"
STORE_FILE = DATA_RAW_DIR / "store.csv"
STORE_STATES_FILE = DATA_RAW_DIR / "store_states.csv"


def _check_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected raw data file not found: {path}. "
            "Run scripts/download_data.sh first."
        )


def load_train() -> pd.DataFrame:
    _check_exists(TRAIN_FILE)
    df = pd.read_csv(TRAIN_FILE, low_memory=False)
    logger.info("Loaded train.csv: %d rows, %d columns", *df.shape)
    return df


def load_store() -> pd.DataFrame:
    _check_exists(STORE_FILE)
    df = pd.read_csv(STORE_FILE, low_memory=False)
    logger.info("Loaded store.csv: %d rows, %d columns", *df.shape)
    return df


def load_store_states() -> pd.DataFrame:
    _check_exists(STORE_STATES_FILE)
    df = pd.read_csv(STORE_STATES_FILE)
    logger.info("Loaded store_states.csv: %d rows, %d columns", *df.shape)
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """Load every raw source table and return them keyed by name."""
    return {
        "train": load_train(),
        "store": load_store(),
        "store_states": load_store_states(),
    }
