#!/usr/bin/env bash
# Downloads the raw public datasets used by SupplySense.
#
# Source: Rossmann Store Sales (public retail dataset -- ~1.7M daily
# store-sales records for 1,115 stores in Germany, 2013-01-01 to
# 2015-07-31). Mirrored as plain CSV on GitHub by multiple public repos.
set -euo pipefail

RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$RAW_DIR"

echo "Downloading train.csv (daily sales) ..."
curl -sL -o "$RAW_DIR/train.csv" \
  "https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/train.csv"

echo "Downloading store.csv (store attributes) ..."
curl -sL -o "$RAW_DIR/store.csv" \
  "https://raw.githubusercontent.com/RPI-DATA/tutorials-intro/master/rossmann-store-sales/rossmann-store-sales/store.csv"

echo "Downloading store_states.csv (store -> German federal state mapping) ..."
curl -sL -o "$RAW_DIR/store_states.csv" \
  "https://raw.githubusercontent.com/entron/entity-embedding-rossmann/master/store_states.csv"

echo "Done. Files in $RAW_DIR:"
ls -la "$RAW_DIR"
