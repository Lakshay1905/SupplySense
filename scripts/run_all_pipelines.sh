#!/usr/bin/env bash
# Runs the complete SupplySense data + analytics pipeline end-to-end:
# Phase 1 (data foundation) -> Phase 2 (forecasting) -> Phase 3 (optimization).
#
# Assumes PostgreSQL is already running and .env is configured (see README).
# The forecast-generation step is split into batches to keep any single
# step's runtime bounded and resumable on constrained hardware; on a
# typical modern machine this entire script takes roughly 10-20 minutes,
# dominated by the recursive 42-day-ahead forecast generation across all
# 1,115 stores.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Phase 1: Data Foundation ==="
python -m database.init_db --reset
python -m scripts.run_phase1_pipeline

echo ""
echo "=== Phase 2: Forecasting (benchmark) ==="
python -m scripts.run_phase2_benchmark

echo ""
echo "=== Phase 2: Forecasting (full-scale generation) ==="
python -m scripts.run_phase2_forecast --prepare
python -m scripts.run_phase2_forecast --batch-start 0    --batch-end 150  --truncate
python -m scripts.run_phase2_forecast --batch-start 150  --batch-end 300
python -m scripts.run_phase2_forecast --batch-start 300  --batch-end 450
python -m scripts.run_phase2_forecast --batch-start 450  --batch-end 600
python -m scripts.run_phase2_forecast --batch-start 600  --batch-end 750
python -m scripts.run_phase2_forecast --batch-start 750  --batch-end 900
python -m scripts.run_phase2_forecast --batch-start 900  --batch-end 1050
python -m scripts.run_phase2_forecast --batch-start 1050 --batch-end 1115

echo ""
echo "=== Phase 3: Optimization, Simulation & Decision Engine ==="
python -m scripts.run_phase3_pipeline

echo ""
echo "=== Running test suite ==="
pytest -q

echo ""
echo "=== All pipelines complete. Launch the app with: streamlit run app/ui/Home.py ==="
