"""
Central configuration for SupplySense.

All configuration is loaded from environment variables (via a `.env` file in
development). No secrets or connection strings are hardcoded anywhere else
in the codebase -- every module that needs configuration imports `settings`
from here.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (config/settings.py -> SupplySense/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SCHEMAS_DIR = PROJECT_ROOT / "data" / "schemas"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"


class Settings(BaseSettings):
    """Application-wide settings, populated from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "supplysense"
    db_user: str = "supplysense"
    db_password: str = "supplysense_dev_pw"

    # --- AI Copilot ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Business / domain configuration that is not a secret but is still
# centralized so every module reads the same assumptions.
BUSINESS_DEFAULTS = {
    "target_service_level": 0.95,       # default cycle service level
    "holding_cost_rate_annual": 0.22,   # % of unit cost held per year
    "default_lead_time_days": 7,        # supplier lead time proxy
    "review_period_days": 7,            # periodic review cycle
    "moq_units": 50,                    # minimum order quantity proxy
    "order_multiple_units": 10,         # orders must be placed in multiples of this
    "stockout_cost_multiplier": 2.5,    # stockout cost as multiple of unit margin
}
