"""
Central configuration for SupplySense.

All configuration is loaded from environment variables (via a `.env` file in
development). No secrets or connection strings are hardcoded anywhere else
in the codebase -- every module that needs configuration imports `settings`
from here.
"""
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Streamlit Cloud secrets bridge ---
# Streamlit Cloud injects secrets via `st.secrets` (from its own dashboard
# config), not real OS environment variables -- but pydantic-settings here
# only reads os.environ / a .env file. This bridges the two, so the exact
# same Settings class works unmodified whether running locally (.env),
# in Docker (env vars), or on Streamlit Cloud (st.secrets). It's a no-op
# (silently does nothing) for local pipeline scripts and tests run outside
# a Streamlit session, or when no secrets.toml exists -- it never raises.
try:
    import streamlit as st
    for _key in ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_SSLMODE",
                 "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY",
                 "LLM_PROVIDER", "ANTHROPIC_MODEL", "GEMINI_MODEL"]:
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

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
    db_sslmode: str | None = None   # e.g. "require" -- needed by most hosted
                                     # Postgres providers (Neon, Supabase, RDS).
                                     # Left unset for local/Docker Postgres,
                                     # which typically don't require SSL.

    # --- AI Copilot ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    llm_provider: str | None = None   # "anthropic" | "gemini" | None = auto-detect
    anthropic_model: str | None = None
    gemini_model: str | None = None

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        url = (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        if self.db_sslmode:
            url += f"?sslmode={self.db_sslmode}"
        return url

    @property
    def active_llm_provider(self) -> str | None:
        """Resolve which LLM provider to use: explicit LLM_PROVIDER env var
        wins; otherwise auto-detect from whichever API key is set
        (Anthropic takes precedence if both happen to be configured)."""
        if self.llm_provider:
            return self.llm_provider.lower()
        if self.anthropic_api_key:
            return "anthropic"
        if self.gemini_api_key:
            return "gemini"
        return None

    @property
    def has_llm_key(self) -> bool:
        return self.active_llm_provider is not None


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
