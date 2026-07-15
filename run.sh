#!/usr/bin/env bash
# =============================================================================
# AIgnition Forecast Studio — Judge Entry Point
# =============================================================================
# Usage:
#   ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
#   ./run.sh ./ml_engine/data/raw ./ml_engine/pickle/model.pkl ./ml_engine/output/predictions.csv
#
# Positional args (all optional, fall back to env vars, then defaults):
#   $1 / DATA_DIR     Directory containing raw ad-platform CSVs  (default: ml_engine/data/raw)
#   $2 / MODEL_PATH   Path to serialised model bundle             (default: ml_engine/pickle/model.pkl)
#   $3 / OUTPUT_PATH  Path where the output CSV will be written   (default: ml_engine/output/predictions.csv)
#
# Additional environment variables:
#   HORIZON_DAYS Number of future days to forecast           (default: 60)
#   SKIP_TRAIN   If "1", skip retraining and load MODEL_PATH (default: 1 if MODEL_PATH exists, else 0)
# =============================================================================
set -euo pipefail

# ---------- Defaults (positional args win, then env vars, then hard default) -
DATA_DIR="${1:-${DATA_DIR:-ml_engine/data/raw}}"
MODEL_PATH="${2:-${MODEL_PATH:-ml_engine/pickle/model.pkl}}"
OUTPUT_PATH="${3:-${OUTPUT_PATH:-ml_engine/output/predictions.csv}}"
HORIZON_DAYS="${HORIZON_DAYS:-60}"
if [ -f "$MODEL_PATH" ]; then
    SKIP_TRAIN="${SKIP_TRAIN:-1}"
else
    SKIP_TRAIN="${SKIP_TRAIN:-0}"
fi

echo "====================================================="
echo " AIgnition Forecast Studio"
echo "====================================================="
echo "  DATA_DIR     = $DATA_DIR"
echo "  MODEL_PATH   = $MODEL_PATH"
echo "  OUTPUT_PATH  = $OUTPUT_PATH"
echo "  HORIZON_DAYS = $HORIZON_DAYS"
echo "  SKIP_TRAIN   = $SKIP_TRAIN"
echo "====================================================="

# ---------- Dependency check -------------------------------------------------
if ! python -c "import lightgbm, pandas, numpy, sklearn, holidays, yaml" 2>/dev/null; then
    echo "[run.sh] Installing dependencies from requirements.txt ..."
    pip install -r requirements.txt --quiet
fi

# ---------- Create output directory ------------------------------------------
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$(dirname "$MODEL_PATH")"

# ---------- Execute pipeline -------------------------------------------------
python -m ml_engine.cli.predict \
    --data-dir  "$DATA_DIR" \
    --model-path "$MODEL_PATH" \
    --output-path "$OUTPUT_PATH" \
    --horizon-days "$HORIZON_DAYS" \
    --skip-train  "$SKIP_TRAIN"

echo "[run.sh] Done. Predictions written to: $OUTPUT_PATH"
