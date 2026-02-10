#!/usr/bin/env bash
# Intel AI Benchmarking — Machine Setup
#
# One-command setup for any Intel machine.
# Run: curl -sSL <raw-url> | bash
# Or:  bash scripts/setup.sh
#
# Installs: Python 3.10+, OpenVINO, optimum-intel, project dependencies

set -euo pipefail

echo "=== Intel AI Benchmarking — Setup ==="
echo "Machine: $(hostname)"
echo "OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""

# --- System packages ---
echo "[1/5] Checking system packages..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Installing system packages (requires sudo)..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3 python3-pip python3-venv git
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3 python3-pip git
    else
        echo "ERROR: Python 3 not found and unknown package manager."
        exit 1
    fi
else
    echo "  Python 3 found: $(python3 --version)"
fi

# --- Python venv ---
VENV_DIR="${HOME}/intel-bench-venv"
echo "[2/5] Creating Python venv at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# --- Core dependencies ---
echo "[3/5] Installing core Python packages..."
pip install --upgrade pip setuptools wheel -q

pip install -q \
    openvino \
    optimum-intel[openvino] \
    transformers \
    torch \
    toml \
    nncf

# --- Verify OpenVINO ---
echo "[4/5] Verifying OpenVINO installation..."
python3 -c "
import openvino as ov
print(f'  OpenVINO version: {ov.__version__}')
core = ov.Core()
devices = core.available_devices
print(f'  Available devices: {devices}')
"

# --- Project setup ---
echo "[5/5] Setting up project directories..."
BENCH_DIR="${HOME}/intel-bench"
mkdir -p "${BENCH_DIR}/models"
mkdir -p "${BENCH_DIR}/results"

echo ""
echo "=== Setup Complete ==="
echo "  Venv: source ${VENV_DIR}/bin/activate"
echo "  Models: ${BENCH_DIR}/models/"
echo "  Results: ${BENCH_DIR}/results/"
echo ""
echo "Next steps:"
echo "  1. Clone the repo: git clone <repo-url> ${BENCH_DIR}/repo"
echo "  2. Activate venv: source ${VENV_DIR}/bin/activate"
echo "  3. Run benchmark: python ${BENCH_DIR}/repo/benchmark.py --help"
