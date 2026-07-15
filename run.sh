#!/usr/bin/env bash
# =============================================================================
# AIgnition Forecast Studio — Judge Entry Point
# =============================================================================
# Usage:
#   DATA_DIR=data/raw MODEL_PATH=pickle/model.pkl OUTPUT_PATH=output/predictions.csv bash run.sh
#
# Environment variables (all have sensible defaults):
#   DATA_DIR     Directory containing raw ad-platform CSVs  (default: data/raw)
#   MODEL_PATH   Path to serialised model bundle             (default: pickle/model.pkl)
#   OUTPUT_PATH  Path where the output CSV will be written   (default: output/predictions.csv)
#   HORIZON_DAYS Number of future days to forecast           (default: 60)
#   SKIP_TRAIN   If "1", skip retraining and load MODEL_PATH (default: 0)
# =============================================================================
set -euo pipefail

# ---------- Defaults ---------------------------------------------------------
DATA_DIR="${DATA_DIR:-data/raw}"
MODEL_PATH="${MODEL_PATH:-pickle/model.pkl}"
OUTPUT_PATH="${OUTPUT_PATH:-output/predictions.csv}"
HORIZON_DAYS="${HORIZON_DAYS:-60}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

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
python -m src.cli.predict \
    --data-dir  "$DATA_DIR" \
    --model-path "$MODEL_PATH" \
    --output-path "$OUTPUT_PATH" \
    --horizon-days "$HORIZON_DAYS" \
    --skip-train  "$SKIP_TRAIN"

echo "[run.sh] Done. Predictions written to: $OUTPUT_PATH"
