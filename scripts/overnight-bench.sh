#!/bin/bash
# Overnight benchmark runner — runs all available models on all devices.
# Deploy to each machine and run via nohup.
#
# Usage:
#   nohup bash scripts/overnight-bench.sh <CODENAME> <TDP> 2>&1 &
#   Example: nohup bash scripts/overnight-bench.sh LNL 17W 2>&1 &

set -euo pipefail

CODENAME="${1:?Usage: overnight-bench.sh <CODENAME> <TDP>}"
TDP="${2:?Usage: overnight-bench.sh <CODENAME> <TDP>}"
VENV_DIR="$HOME/intel-bench-venv"
REPO_DIR="$HOME/intel-bench/repo"
MODEL_DIR="$HOME/models/intel-bench"
DB_PATH="$HOME/intel-bench/results/benchmarks.db"
LOG_DIR="$HOME/intel-bench/logs"

mkdir -p "$LOG_DIR"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

# Detect available devices
DEVICES=$(python3 -c "
from openvino import Core
core = Core()
devs = [d for d in core.available_devices if d in ('GPU', 'NPU', 'CPU')]
print(' '.join(devs))
")
echo "$(date): Available devices: $DEVICES"

# Available precisions (check which models exist)
PRECISIONS=""
for prec in INT4 INT8 FP16; do
    model_path="$MODEL_DIR/Llama-3.1-8B-Instruct-$prec"
    if [ -d "$model_path" ] && ls "$model_path"/*.xml >/dev/null 2>&1; then
        PRECISIONS="$PRECISIONS $prec"
        echo "$(date): Found $prec model at $model_path"
    else
        echo "$(date): SKIP $prec — model not found at $model_path"
    fi
done

if [ -z "$PRECISIONS" ]; then
    echo "$(date): ERROR — No models found in $MODEL_DIR"
    exit 1
fi

echo "$(date): Starting overnight benchmark"
echo "  Codename: $CODENAME"
echo "  TDP: $TDP"
echo "  Devices: $DEVICES"
echo "  Precisions:$PRECISIONS"
echo "  DB: $DB_PATH"
echo ""

TOTAL_RUNS=0
FAILED_RUNS=0

for device in $DEVICES; do
    for prec in $PRECISIONS; do
        logfile="$LOG_DIR/${CODENAME}-${device}-${prec}.log"
        echo "============================================================"
        echo "$(date): START — $CODENAME $device $prec"
        echo "  Log: $logfile"
        echo "============================================================"

        if python3 benchmark.py \
            --precision "$prec" \
            --device "$device" \
            --codename "$CODENAME" \
            --tdp "$TDP" \
            --temperature 0.0 0.7 \
            --db "$DB_PATH" \
            --notes "Overnight run: $CODENAME $device $prec" \
            > "$logfile" 2>&1; then
            echo "$(date): DONE  — $CODENAME $device $prec (success)"
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        else
            echo "$(date): FAIL  — $CODENAME $device $prec (exit code $?)"
            echo "  Check $logfile for details"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            # Continue to next — don't stop the whole overnight run
        fi
        echo ""
    done
done

echo "============================================================"
echo "$(date): OVERNIGHT COMPLETE"
echo "  Total device/precision combos run: $TOTAL_RUNS"
echo "  Failed: $FAILED_RUNS"
echo "  Results in: $DB_PATH"
echo "============================================================"
