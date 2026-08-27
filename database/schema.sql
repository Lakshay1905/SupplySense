-- =====================================================================
-- SupplySense analytical schema (PostgreSQL)
--
-- Star-schema style: dimension tables describe "who/what/where/when",
-- fact tables hold measurable events, and result tables hold the outputs
-- of downstream analytical engines (forecasting, optimization,
-- simulation, scenarios). Phase 1 populates the dimension + raw fact
-- tables; later phases populate the *_forecasts / *_optimization /
-- *_scenarios / *_simulation tables.
-- =====================================================================

-- ---------------------------------------------------------------------
-- DIMENSIONS
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_region (
    region_id       SERIAL PRIMARY KEY,
    state_code      VARCHAR(10) UNIQUE NOT NULL,
    state_name      VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS dim_store (
    store_id                        INTEGER PRIMARY KEY,
    store_type                      VARCHAR(5) NOT NULL,   -- proxy for "category" (a/b/c/d)
    assortment                      VARCHAR(5) NOT NULL,   -- basic/extra/extended -> sub-category
    competition_distance_m          NUMERIC,
    competition_open_since_month    SMALLINT,
    competition_open_since_year     SMALLINT,
    promo2_active                   BOOLEAN NOT NULL DEFAULT FALSE,
    promo2_since_week               SMALLINT,
    promo2_since_year               SMALLINT,
    promo_interval                  VARCHAR(50),
    region_id                       INTEGER REFERENCES dim_region(region_id)
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id         DATE PRIMARY KEY,
    year            SMALLINT NOT NULL,
    quarter         SMALLINT NOT NULL,
    month           SMALLINT NOT NULL,
    week_of_year    SMALLINT NOT NULL,
    day_of_month    SMALLINT NOT NULL,
    day_of_week     SMALLINT NOT NULL,     -- 1 = Monday ... 7 = Sunday
    day_name        VARCHAR(10) NOT NULL,
    is_weekend      BOOLEAN NOT NULL,
    month_name      VARCHAR(10) NOT NULL
);

-- Product dimension kept for architectural completeness / future extension.
-- The public dataset used in Phase 1 (Rossmann) reports store-level demand
-- only (no SKU-level sales), so this table is not populated in Phase 1.
-- StoreType/Assortment on dim_store serve as the documented category proxy.
CREATE TABLE IF NOT EXISTS dim_product (
    product_id      SERIAL PRIMARY KEY,
    product_name    VARCHAR(200) NOT NULL,
    category        VARCHAR(100),
    sub_category    VARCHAR(100),
    unit_cost       NUMERIC,
    unit_price      NUMERIC
);

CREATE TABLE IF NOT EXISTS dim_supplier (
    supplier_id         SERIAL PRIMARY KEY,
    supplier_name       VARCHAR(200) NOT NULL,
    lead_time_days      NUMERIC NOT NULL,
    lead_time_std_days  NUMERIC NOT NULL DEFAULT 0,
    reliability_score   NUMERIC,          -- 0-1, on-time-in-full proxy
    moq_units           INTEGER NOT NULL DEFAULT 1,
    order_multiple      INTEGER NOT NULL DEFAULT 1,
    notes               TEXT
);

-- ---------------------------------------------------------------------
-- FACTS (Phase 1)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_sales (
    sales_id            BIGSERIAL PRIMARY KEY,
    date_id             DATE NOT NULL REFERENCES dim_date(date_id),
    store_id            INTEGER NOT NULL REFERENCES dim_store(store_id),
    sales               NUMERIC NOT NULL,
    customers           INTEGER,
    is_open             BOOLEAN NOT NULL,
    is_promo            BOOLEAN NOT NULL DEFAULT FALSE,
    state_holiday       VARCHAR(5) NOT NULL DEFAULT '0',
    school_holiday      BOOLEAN NOT NULL DEFAULT FALSE,
    sales_per_customer  NUMERIC,
    UNIQUE (date_id, store_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_sales_store ON fact_sales(store_id);
CREATE INDEX IF NOT EXISTS idx_fact_sales_date ON fact_sales(date_id);

-- Engineered features materialized for downstream forecasting (Phase 2).
CREATE TABLE IF NOT EXISTS fact_sales_features (
    feature_id              BIGSERIAL PRIMARY KEY,
    date_id                 DATE NOT NULL,
    store_id                INTEGER NOT NULL,
    sales                   NUMERIC NOT NULL,
    lag_1                   NUMERIC,
    lag_7                   NUMERIC,
    lag_14                  NUMERIC,
    lag_28                  NUMERIC,
    rolling_mean_7          NUMERIC,
    rolling_mean_14         NUMERIC,
    rolling_mean_28         NUMERIC,
    rolling_std_7           NUMERIC,
    rolling_std_28          NUMERIC,
    day_of_week             SMALLINT,
    is_weekend              BOOLEAN,
    week_of_year            SMALLINT,
    month                   SMALLINT,
    is_promo                BOOLEAN,
    is_school_holiday       BOOLEAN,
    is_state_holiday        BOOLEAN,
    days_since_competition  NUMERIC,
    demand_segment          VARCHAR(20),
    UNIQUE (date_id, store_id)
);

CREATE INDEX IF NOT EXISTS idx_features_store ON fact_sales_features(store_id);
CREATE INDEX IF NOT EXISTS idx_features_date ON fact_sales_features(date_id);

-- ---------------------------------------------------------------------
-- OPERATIONAL / SUPPLY-CHAIN TABLES (populated in Phase 3)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS inventory_snapshot (
    snapshot_id         BIGSERIAL PRIMARY KEY,
    date_id             DATE NOT NULL,
    store_id            INTEGER NOT NULL,
    on_hand_units       NUMERIC NOT NULL,
    incoming_units      NUMERIC NOT NULL DEFAULT 0,
    reorder_point       NUMERIC,
    safety_stock        NUMERIC,
    UNIQUE (date_id, store_id)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id            BIGSERIAL PRIMARY KEY,
    store_id            INTEGER NOT NULL,
    supplier_id         INTEGER,
    order_date          DATE NOT NULL,
    quantity_units      NUMERIC NOT NULL,
    unit_cost           NUMERIC,
    expected_arrival    DATE,
    status              VARCHAR(20) DEFAULT 'planned'
);

CREATE TABLE IF NOT EXISTS promotions (
    promotion_id        BIGSERIAL PRIMARY KEY,
    store_id            INTEGER NOT NULL,
    start_date          DATE NOT NULL,
    end_date            DATE NOT NULL,
    discount_pct        NUMERIC,
    expected_uplift_pct NUMERIC,
    notes               TEXT
);

-- ---------------------------------------------------------------------
-- ANALYTICAL RESULT TABLES (populated in Phases 2-3)
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_evaluations (
    evaluation_id       BIGSERIAL PRIMARY KEY,
    store_id            INTEGER NOT NULL,
    model_name          VARCHAR(50) NOT NULL,
    fold                INTEGER NOT NULL,
    train_end_date      DATE NOT NULL,
    test_start_date     DATE NOT NULL,
    test_end_date       DATE NOT NULL,
    mae                 NUMERIC,
    rmse                NUMERIC,
    mape                NUMERIC,
    wmape               NUMERIC,
    bias                NUMERIC,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id         BIGSERIAL PRIMARY KEY,
    store_id            INTEGER NOT NULL,
    forecast_date       DATE NOT NULL,
    target_date         DATE NOT NULL,
    model_name          VARCHAR(50) NOT NULL,
    p10                 NUMERIC,
    p50                 NUMERIC,
    p90                 NUMERIC,
    created_at          TIMESTAMP DEFAULT NOW(),
    UNIQUE (store_id, forecast_date, target_date, model_name)
);

CREATE TABLE IF NOT EXISTS optimization_results (
    result_id           BIGSERIAL PRIMARY KEY,
    store_id            INTEGER NOT NULL,
    run_date            DATE NOT NULL,
    recommended_order_qty NUMERIC,
    expected_cost        NUMERIC,
    stockout_probability NUMERIC,
    service_level_target NUMERIC,
    drivers_json          JSONB,
    created_at            TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id          BIGSERIAL PRIMARY KEY,
    scenario_name        VARCHAR(200) NOT NULL,
    parameters_json       JSONB NOT NULL,
    result_json            JSONB,
    created_at             TIMESTAMP DEFAULT NOW()
);

-- ---------------------------------------------------------------------
-- DATA QUALITY / PIPELINE OBSERVABILITY
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS data_quality_log (
    log_id              BIGSERIAL PRIMARY KEY,
    run_id              VARCHAR(50) NOT NULL,
    stage               VARCHAR(50) NOT NULL,
    check_name          VARCHAR(100) NOT NULL,
    status              VARCHAR(20) NOT NULL,   -- pass / warn / fail
    records_checked     BIGINT,
    records_failed      BIGINT,
    details             TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              VARCHAR(50) PRIMARY KEY,
    started_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMP,
    status              VARCHAR(20) NOT NULL DEFAULT 'running',
    rows_ingested       BIGINT,
    rows_loaded         BIGINT,
    notes               TEXT
);
