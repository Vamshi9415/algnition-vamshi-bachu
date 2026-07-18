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

python_version="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

check_python_version() {
    case "$python_version" in
        3.10|3.11|3.12|3.13) ;;
        *)
            echo "[run.sh] Unsupported Python version: $python_version (need 3.10+)"
            exit 1
            ;;
    esac
}

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

check_python_version

if [ ! -d "$DATA_DIR" ]; then
    echo "[run.sh] DATA_DIR not found: $DATA_DIR"
    exit 1
fi

csv_count="$(find "$DATA_DIR" -maxdepth 1 -name '*.csv' | wc -l | tr -d ' ')"
if [ "$csv_count" = "0" ]; then
    echo "[run.sh] No CSV files found in $DATA_DIR"
    exit 1
fi
echo "[run.sh] ✓ Found $csv_count dataset(s)"

# ---------- Dependency check -------------------------------------------------
if ! python -c "import lightgbm, pandas, numpy, sklearn, holidays, yaml" 2>/dev/null; then
    echo "[run.sh] Installing dependencies from requirements.txt ..."
    pip install -r requirements.txt --quiet
fi

if [ "$SKIP_TRAIN" = "1" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "[run.sh] Requested SKIP_TRAIN=1 but model bundle is missing at $MODEL_PATH"
    exit 1
fi

echo "[run.sh] ✓ Runtime prerequisites satisfied"

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

if [ ! -f "$OUTPUT_PATH" ]; then
    echo "[run.sh] Output verification failed: $OUTPUT_PATH not created"
    exit 1
fi

echo "[run.sh] Done. Predictions written to: $OUTPUT_PATH"
