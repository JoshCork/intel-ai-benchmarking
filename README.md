# Intel AI Benchmarking

Standardized LLM inference benchmarking across Intel hardware. Designed for comparing Llama 3.1 8B (and other models) across different Intel platforms — Alder Lake + Arc dGPU, Lunar Lake iGPU, Meteor Lake iGPU.

## Hardware Fleet

| Machine | Codename | CPU | GPU | RAM | NPU |
|---------|----------|-----|-----|-----|-----|
| friday-cork | ADL | i7-12700H | Arc A770M 16GB dGPU + Iris Xe iGPU | 64GB | GNA |
| LNL-GROVE | LNL | Core Ultra 7 258V | Xe2 iGPU | 32GB | Lunar Lake NPU |
| MTL-NOYCE | MTL | Core Ultra 5 125H | Intel Arc iGPU | 16GB | Meteor Lake NPU |

## Quick Start

```bash
# 1. Setup a machine (installs OpenVINO, creates venv)
ssh grove@LNL-GROVE.local
bash scripts/setup.sh
source ~/intel-bench-venv/bin/activate

# 2. Export model (or put pre-exported model on USB at /intel-ai-models/)
python scripts/export_model.py --model meta-llama/Llama-3.1-8B-Instruct --precision INT4

# 3. Run benchmarks
python benchmark.py --precision INT4 --temperature 0.0 0.7

# 4. View results
python benchmark.py --results
```

## Model Loading Priority

1. **USB drive** — checks `/media/<user>/*/intel-ai-models/` for pre-exported models
2. **Local cache** — checks `~/models/intel-bench/`
3. **HuggingFace download** — downloads and exports to OpenVINO IR automatically

OpenVINO models are **hardware-portable**: export once at any precision, run on any Intel GPU/CPU. Device-specific kernels are compiled at load time.

## Test Scenarios

Kiosk-style interactions mimicking real customer conversations:

| Scenario | Type | Description |
|----------|------|-------------|
| greeting | greeting | Simple "Hi there!" — baseline latency |
| store_hours | simple_task | Store hours inquiry — short factual response |
| product_lookup | simple_task | Product search — structured response |
| return_policy | complex | Multi-part policy question — detailed response |
| loyalty_program | complex | Program benefits inquiry — comprehensive response |
| multi_turn_directions | multi_turn | 3-turn conversation about finding items |
| multi_turn_troubleshoot | multi_turn | 3-turn troubleshooting dialogue |

## Benchmark Protocol

- **Warmup**: 3 runs (GPU kernel compilation + cache warming)
- **Measured**: 10 runs per scenario
- **Temperatures**: 0.0 (deterministic) and 0.7 (realistic sampling)
- **System prompt**: Kiosk assistant persona included for realism

## Metrics Captured

Per run:
- **TTFT** (Time to First Token) — ms
- **Token throughput** — tokens/sec
- **Total latency** — ms
- **Input/output token counts**
- **Prompt and response text snapshots**

Aggregates:
- Mean, median, P5, P95, standard deviation for TTFT, TPS, total latency

Machine fingerprint:
- CPU model, codename, cores/threads
- GPU model, type (dGPU/iGPU), VRAM
- NPU model
- Memory: total GB, type (DDR4/DDR5/LPDDR5x), speed (MT/s), channels
- OS, kernel, OpenVINO version

## Database

SQLite database on friday-cork (Tailscale-accessible):
`/home/friday/intel-bench/results/benchmarks.db`

Tables: `machines`, `benchmark_runs`, `run_metrics`, `run_aggregates`

## Project Structure

```
intel-ai-benchmarking/
├── benchmark.py              # Main CLI
├── config.toml               # Default configuration
├── lib/
│   ├── hardware.py           # Hardware fingerprinting
│   ├── model_loader.py       # USB → local → HuggingFace model loading
│   ├── inference.py          # OpenVINO LLM wrapper with timing
│   ├── db.py                 # SQLite database
│   └── metrics.py            # Statistical aggregation
├── scenarios/
│   └── kiosk.py              # Kiosk test scenarios
├── scripts/
│   ├── setup.sh              # Machine setup script
│   └── export_model.py       # HuggingFace → OpenVINO export
└── results/                  # Local results (gitignored)
```
