#!/usr/bin/env bash
# Optimization Experiments — Overnight Runner for PTL-FAIRCHILD
#
# Runs all 3 optimization experiments sequentially on GPU.
# Each experiment gets its own log file.
#
# Usage:
#   bash scripts/optimization-experiments.sh
#   nohup bash scripts/optimization-experiments.sh > optimization-run.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB="${HOME}/intel-bench/results/benchmarks.db"
LOG_DIR="${PROJECT_DIR}/logs/optimization-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$LOG_DIR"

cd "$PROJECT_DIR"

echo "=== Optimization Experiments ==="
echo "Started: $(date)"
echo "Project: $PROJECT_DIR"
echo "Database: $DB"
echo "Logs: $LOG_DIR"
echo ""

# --- Experiment 1: ov_config runtime flags ---
echo "=== Experiment 1: ov_config runtime flags ==="
echo "Started: $(date)"

python3 benchmark.py \
    --precision FP16 INT8 INT4 --device GPU \
    --temperature 0.0 0.7 --codename PTL --tdp 25W \
    --experiment "PTL-ov-config-DDR5-5600" \
    --ov-config "KV_CACHE_PRECISION=u8,DYNAMIC_QUANTIZATION_GROUP_SIZE=64,PERFORMANCE_HINT=LATENCY" \
    --perfspect --db "$DB" \
    2>&1 | tee "$LOG_DIR/exp1-ov-config.log"

echo ""
echo "Experiment 1 completed: $(date)"
echo ""

# --- Experiment 2: GPTQ INT4 ---
# NOTE: Requires GPTQ model to be pre-exported. If not present, this will fail.
# Export with: python3 scripts/export_model.py --model meta-llama/Llama-3.1-8B-Instruct \
#              --precision INT4 --algorithm gptq --dataset wikitext2 --num-samples 128 --suffix "-gptq"
echo "=== Experiment 2: GPTQ INT4 ==="
echo "Started: $(date)"

python3 benchmark.py \
    --precision INT4 --device GPU \
    --temperature 0.0 0.7 --codename PTL --tdp 25W \
    --experiment "PTL-gptq-INT4-DDR5-5600" \
    --model-suffix "-gptq" \
    --perfspect --db "$DB" \
    2>&1 | tee "$LOG_DIR/exp2-gptq.log"

echo ""
echo "Experiment 2 completed: $(date)"
echo ""

# --- Experiment 3: OpenVINO GenAI pipeline ---
echo "=== Experiment 3: OpenVINO GenAI LLMPipeline ==="
echo "Started: $(date)"

python3 benchmark.py \
    --precision FP16 INT8 INT4 --device GPU \
    --temperature 0.0 0.7 --codename PTL --tdp 25W \
    --experiment "PTL-genai-DDR5-5600" \
    --backend genai \
    --perfspect --db "$DB" \
    2>&1 | tee "$LOG_DIR/exp3-genai.log"

echo ""
echo "Experiment 3 completed: $(date)"
echo ""

# --- Summary ---
echo "=== All Experiments Complete ==="
echo "Finished: $(date)"
echo "Logs: $LOG_DIR"
echo ""
echo "View results:"
echo "  python3 benchmark.py --results --db $DB"
