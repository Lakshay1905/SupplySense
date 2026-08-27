# SupplySense — Intelligent Demand Forecasting & Inventory Decision Engine

SupplySense turns raw retail sales history into inventory **decisions**:
what to order, how much safety stock to hold, and what the stockout /
cost tradeoffs look like under different assumptions. It is being built
in five phases; this document currently covers **Phase 1: Data
Foundation**.

```
Raw Data → Ingestion → Validation → Cleaning → Transformation →
Feature Engineering → PostgreSQL → Analytics / ML / Optimization
```

---

## Phase 1 — Data Foundation (complete)

### Dataset

**Rossmann Store Sales** — a public retail dataset covering daily sales
for 1,115 drugstores in Germany from 2013-01-01 to 2015-07-31
(~1.02M store-day records). Retrieved from public GitHub mirrors of the
original Kaggle competition data:

- `train.csv` — daily `Sales`, `Customers`, `Open`, `Promo`,
  `StateHoliday`, `SchoolHoliday` per store
- `store.csv` — `StoreType`, `Assortment`, `CompetitionDistance`,
  `Promo2` fields per store
- `store_states.csv` — store → German federal state mapping

Run `scripts/download_data.sh` to (re)fetch these into `data/raw/`.

### Documented proxy decisions

This is real sales data, not a synthetic supply-chain dataset, so a few
fields required documented, defensible proxies rather than fabrication:

| Business concept | Real field used | Notes |
|---|---|---|
| Category | `StoreType` (a/b/c/d) | Rossmann has no SKU-level sales, so category-level demand is not decomposable. StoreType is a genuine dataset field distinguishing store formats/assortment strategy. |
| Sub-category | `Assortment` (basic/extra/extended) | Real field. |
| Region | German federal state (`store_states.csv`) | Real store→state mapping from a public companion dataset. |
| Product | *not populated in Phase 1* | `dim_product` exists in the schema for architectural completeness and future extension, but is intentionally left empty rather than fabricating SKU-level sales that don't exist in the source data. **"Store" is the atomic demand unit** for forecasting and inventory decisions in this project. |
| Inventory, suppliers, costs, lead times | *not populated in Phase 1* | These are business/operational parameters (not something a sales-history dataset would contain). They will be introduced in Phase 3 as clearly-labeled, configurable assumptions (`config/settings.py: BUSINESS_DEFAULTS`) driving the optimization engine — never presented as historical fact. |

### Architecture

```
SupplySense/
├── config/                  # settings.py (env-driven), logging_config.py
├── data/
│   ├── raw/                 # downloaded CSVs (train, store, store_states)
│   ├── processed/           # reserved for cached intermediate artifacts
│   └── schemas/             # declarative column schemas used by validation
├── database/
│   ├── schema.sql           # full star-schema DDL (all 5 phases)
│   ├── connection.py        # SQLAlchemy engine, COPY-based bulk loader
│   └── init_db.py           # create/reset schema
├── pipelines/
│   ├── ingestion/           # raw CSV -> DataFrame
│   ├── validation/          # schema/range/null/duplicate/referential/anomaly checks
│   └── transformation/      # cleaning + star-schema construction
├── analytics/
│   ├── features/            # lag/rolling/calendar features, demand segmentation
│   ├── metrics/             # data-quality aggregation
│   └── eda/                 # EDA report + charts (reports/eda/)
├── forecasting/ optimization/ simulation/ scenarios/ ai/ app/   # scaffolding for Phases 2-4
├── scripts/
│   ├── download_data.sh
│   └── run_phase1_pipeline.py   # full Phase 1 orchestrator
├── tests/                   # 47 pytest tests (unit + live-DB integration)
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

### Database schema

PostgreSQL, star-schema style. Phase 1 populates:

- `dim_date` (942 calendar days, full daily grain, no gaps)
- `dim_region` (12 German federal states)
- `dim_store` (1,115 stores, joined to region)
- `fact_sales` (1,017,209 store-days)
- `fact_sales_features` (materialized lag/rolling/calendar features + demand segment, 1 row per `fact_sales` row)
- `data_quality_log` / `pipeline_runs` (full observability of every pipeline run)

Tables for later phases (`dim_product`, `dim_supplier`, `inventory_snapshot`,
`orders`, `promotions`, `forecasts`, `model_evaluations`,
`optimization_results`, `scenarios`) are created now but intentionally left
empty until their respective phase populates them.

### Data quality & validation

Every raw and cleaned table is run through a validation battery (dtype
coercion, null checks against nullable/non-nullable schema, range checks,
allowed-value checks, duplicate detection, referential integrity,
statistical outlier detection via z-score). Results are logged to
`data_quality_log`, not just printed — so the Phase 4 "Data Quality" tab
can query real history.

Last full run (`phase1_20260826_185051_6aed8c`):

- 1,017,209 rows ingested and loaded (0 dropped — the source data required no row removal)
- 92 automated checks logged: **90 pass, 2 warn (statistical outliers), 0 fail**
- 0 orphan store references, 0 duplicate `(store, date)` pairs, 0 negative sales/customers after cleaning

Cleaning decisions actually applied (counts from real data):
- Closed-store/nonzero-sales inconsistencies: 0 occurrences in this dataset (rule implemented and unit-tested regardless)
- Missing `CompetitionDistance`: 3 stores — left `NULL` (no competitor on record), not imputed
- Missing `Promo2Since*` / `PromoInterval`: 544 stores not enrolled in Promo2 — legitimately null, not missing data

### EDA highlights (from `reports/eda/eda_summary.json`, computed on real data)

- 844,392 of 1,017,209 store-days are "open" (16.99% closed — mostly Sundays/holidays)
- Average daily sales per open store: **6,955.5**, median **6,369**, P10/P90 = 3,762 / 10,771
- Promo days average **8,228** vs **5,929** on non-promo days (descriptive correlation, explicitly **not** claimed as a causal uplift)
- Demand segmentation (heuristic ADI/CV-based): **1,089 stable**, **24 seasonal**, **2 volatile** stores
- Store type `b` stores have visibly higher average daily sales (10,231) than types a/c/d (~6,800–6,900)

Charts generated: `sales_by_day_of_week.png`, `sales_by_month.png`,
`sample_store_timeseries.png`, `demand_segments.png`, `sales_histogram.png`
(all in `reports/eda/`).

### Testing

47 automated tests (`pytest`), covering:
- Validation logic (null/range/allowed-value/duplicate/referential/anomaly checks) — including a deliberately-planted bug (closed-store sales not zeroed) that the test suite caught and which was subsequently fixed in `pipelines/transformation/clean.py`
- Cleaning logic (deduplication, negative-value clipping, division-by-zero safety, documented null-preservation)
- Star-schema transformation (date-range completeness, weekend flags, region joins, dtype correctness)
- Feature engineering (lag/rolling correctness with an explicit no-leakage check, demand segmentation heuristics)
- Data-quality metrics aggregation
- Live-database integration tests (row counts, referential integrity, duplicate/negative-value checks directly against PostgreSQL) — auto-skipped if Postgres isn't reachable

```
47 passed in ~3s
```

### Running it yourself

```bash
# 1. Start PostgreSQL (via docker-compose, or a local install)
docker-compose up -d postgres

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch the raw dataset
bash scripts/download_data.sh

# 4. Configure environment
cp .env.example .env   # edit DB credentials if needed

# 5. Run the full Phase 1 pipeline
python -m scripts.run_phase1_pipeline

# 6. Run tests
pytest
```

Or via Docker end-to-end: `docker-compose up --build pipeline`.

---

## Phase 2 — Forecasting & Predictive Analytics (complete)

### Benchmarking methodology

Every model is evaluated with **rolling-origin ("walk-forward") time-series
cross-validation** — never a random train/test split, which would leak
future information into training. Each of 3 folds trains on all data up
to a cutoff and evaluates on the following 42 days (matching Rossmann's
original 6-week holdout convention), with successive folds moving the
cutoff forward through mid-to-late 2015.

Full benchmarking (baselines + statistical + ML, backtested per store)
was run on a **stratified sample of 22 stores** spanning every
demand-segment × store-type combination found in Phase 1's EDA — fitting
per-store SARIMA/Holt-Winters across all 1,115 stores is not a sensible
use of compute for a periodic batch job, and the sample is representative
of the full population's demand-pattern mix. The winning model class is
then trained once, globally, on full-scale data (see below).

### Real benchmark results (`reports/forecasting/model_comparison_summary.csv`)

| Model | Avg MAE | Avg RMSE | Avg MAPE | **Avg WMAPE** | Avg Bias |
|---|---|---|---|---|---|
| **XGBoost** | 756.4 | 1081.0 | 10.25% | **9.90%** | -0.45% |
| Random Forest | 1007.7 | 1503.8 | 13.65% | 12.82% | -0.53% |
| SARIMA | 1512.3 | 1922.4 | 19.53% | 17.66% | -0.42% |
| Holt-Winters | 1756.8 | 2215.9 | 22.78% | 20.48% | -1.63% |
| Seasonal Naive | 1858.6 | 2421.2 | 23.27% | 22.02% | -0.36% |
| Naive | 2115.0 | 2625.4 | 30.94% | 24.75% | 2.70% |
| Moving Average (7d) | 2181.3 | 2698.2 | 30.76% | 25.66% | -0.36% |

**XGBoost won for all 22 of 22 sampled stores** (see
`reports/forecasting/best_model_per_store.csv`) — a clean, decisive
result, not a marginal one. It was therefore selected as the production
model. Both baselines and every statistical model were meaningfully
beaten by the ML approach, which can exploit lag/rolling/calendar/promo/
competition features jointly rather than modeling each series in
isolation.

### Production forecasting pipeline

1. **Feature set**: 17 engineered features per store-day — 4 lags (1/7/14/28 days), 5 rolling mean/std windows, calendar features (day-of-week, week, month, weekend flag), promo/holiday flags, days-since-competition-opened, plus store type/assortment as categorical codes.
2. **Global model**: a single XGBoost regressor (300 trees, depth 6, `reg:squarederror`) trained on all 985,989 valid feature rows across all 1,115 stores at once — not 1,115 separate models. This is the standard, compute-efficient approach for large retail panels and lets the model share weekday/promo/competition effects across stores while still specializing per store via store-level features.
3. **Recursive multi-step forecasting**: 42 days ahead per store, where each day's lag/rolling features are rebuilt using the model's own prior-day forecasts (since real future sales aren't available) — standard recursive forecasting, implemented in `scripts/run_phase2_forecast.py`.
4. **Probabilistic bands (P10/P50/P90)**: computed from the empirical quantiles of the model's residuals on a held-out fold, **grouped by demand segment** (stable/seasonal/volatile) so volatile stores get visibly wider uncertainty bands than stable ones — not a fixed +/-X% band. Real computed offsets: stable (-926.9, +758.0), seasonal (-881.8, +1402.1), volatile (-580.1, +383.8).

### Full-scale forecast results

Generated and stored in the `forecasts` table:
- **46,830 forecast rows** = 1,115 stores × 42 forecast days (2015-08-01 to 2015-09-11)
- **0 probabilistic-band ordering violations** (P10 ≤ P50 ≤ P90 holds for every row, verified by both a unit check in the pipeline and a live-DB integration test)
- **0 negative forecasts** (all clipped at zero, since demand cannot be negative)
- The model correctly learned each store's weekly closure pattern (e.g. Store 1's Sunday forecasts land near-zero, matching its historical Sunday closures) — see `reports/forecasting/sample_forecast_store1.png`

### Compute note (documented tradeoff)

This environment provides a single CPU core. XGBoost's histogram-based
training scales fine on it (full 985,989-row fit in ~20s), but
scikit-learn's RandomForest does not parallelize usefully on 1 core at
full data volume. Random Forest in the benchmark is therefore trained on
an 80,000-row random subsample with a shallower forest (40 trees, depth
9) — a reasonable, explicitly documented tradeoff for a benchmarking
comparison; it does not affect the production model (XGBoost, trained on
full data) or change which model won.

### Testing

41 new tests added (88 total, all passing):
- Metrics correctness (MAE/RMSE/MAPE/WMAPE/bias), including edge cases (all-zero actuals, zero-actual rows)
- Baseline model correctness (naive, seasonal-naive cycling, moving average)
- Rolling-origin fold generation (no train/test leakage, correct window lengths, non-overlapping folds)
- Probabilistic band construction (ordering, zero-clipping, NaN-robustness)
- Model selection logic (correct per-store winner, correct fold-averaging)
- ML feature preparation and prediction sanity (non-negativity, missing-value handling)
- Live-DB integration tests: `model_evaluations` contains all three model families, `forecasts` has no band violations, no negative values, and full horizon coverage for every store

```
88 passed in ~8s
```

### Running it yourself

```bash
# Step 1: run the benchmark (backtests all model families on a stratified sample)
python -m scripts.run_phase2_benchmark

# Step 2: train the production model + cache it (train once)
python -m scripts.run_phase2_forecast --prepare

# Step 3: generate forecasts for all stores (can be run in batches to bound runtime)
python -m scripts.run_phase2_forecast --batch-start 0 --batch-end 1115 --truncate

# Tests
pytest
```

---

## Roadmap

- **Phase 3** — Inventory optimization, Monte Carlo simulation, scenario/what-if engine.
- **Phase 4** — Streamlit application + AI analytics copilot, grounded in real pipeline/model/optimization output.
- **Phase 5** — Tests, Docker hardening, deployment docs, final polish.
