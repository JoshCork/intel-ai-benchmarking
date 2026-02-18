# Benchmarking Procedures

Standard operating procedures for running LLM inference benchmarks across the Intel NUC fleet.

---

## 1. Machine Topology

| Machine | Hostname | Role | CPU | GPU | RAM | LAN IP |
|---------|----------|------|-----|-----|-----|--------|
| Fairchild | PTL-FAIRCHILD | Primary benchmark host | Core Ultra 9 288V (Panther Lake H) | Xe3-LPG iGPU | 62GB DDR5-5600 | 192.168.18.236 |
| Noyce | MTL-NOYCE | Secondary / model export | Core Ultra 5 125H (Meteor Lake) | Intel Arc iGPU | 16GB | 192.168.18.210 |
| Grove | LNL-GROVE | Model storage / secondary | Core Ultra 7 258V (Lunar Lake) | Xe2 iGPU | 32GB LPDDR5x-7200 | 192.168.18.247 |
| Friday | friday-cork | Portable workstation (intermittent) | i7-12700H (Alder Lake) | Arc A770M 16GB dGPU | 64GB DDR5 | Variable |
| SKELETOR-03 | SKELETOR-03 | Central DB host | — | — | — | 192.168.18.145 |

### Network

- **LAN**: All machines on 192.168.18.0/20 — always use LAN IPs for inter-machine transfers
- **Tailscale**: Available as fallback (100.x.x.x) but routes through SFO relay — ~50% slower
- **Rule**: Never relay transfers through the Mac. Always SCP directly between NUCs over LAN.

### SSH Access

Each NUC uses its codename as the username:
- `ssh fairchild@192.168.18.236`
- `ssh noyce@192.168.18.210`
- `ssh grove@192.168.18.247`
- `ssh skeletor@192.168.18.145`

SSH keys are pre-configured between all machines. Password auth is also available (sudo password = standard lab password).

---

## 2. Environment Setup (New Machine)

Run the setup script on any new machine:

```bash
ssh <user>@<host>
git clone https://github.com/JoshCork/intel-ai-benchmarking.git ~/intel-bench/repo
bash ~/intel-bench/repo/scripts/setup.sh
```

This installs:
- Python 3.10+ venv at `~/intel-bench-venv/`
- OpenVINO + optimum-intel + openvino-genai
- Project directories: `~/intel-bench/models/`, `~/intel-bench/results/`

Verify the install:

```bash
source ~/intel-bench-venv/bin/activate
python3 -c "import openvino; print(openvino.__version__)"
python3 -c "from openvino import Core; print(Core().available_devices)"
```

---

## 3. Model Export

Models are exported from HuggingFace to OpenVINO IR format. Export once, run anywhere — the same INT4 model works on any Intel GPU/CPU.

### 3.1 Standard Export (AWQ — Default Quantization)

```bash
source ~/intel-bench-venv/bin/activate
cd ~/intel-bench/repo

# Single precision
python3 scripts/export_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --precision INT4

# All precisions (FP16, INT8, INT4)
python3 scripts/export_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --precision all

# Qwen model
python3 scripts/export_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --precision all
```

Output: `~/models/intel-bench/<ModelName>-<PRECISION>/`

**Time estimates**: FP16 ~15min, INT8 ~20min, INT4 ~25min (varies by machine RAM/CPU).

### 3.2 GPTQ Export (Scale Estimation)

GPTQ with scale estimation produces higher-quality INT4 weights by running calibration on a dataset. The result is a mixed INT4/INT8 model (~87% INT4, ~13% INT8 layers).

```bash
# GPTQ export — takes 4-8 hours depending on machine
python3 scripts/export_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --precision INT4 \
    --algorithm gptq \
    --dataset wikitext2 \
    --num-samples 128 \
    --suffix "-gptq"
```

Output: `~/models/intel-bench/Llama-3.1-8B-Instruct-INT4-gptq/`

**Important**: GPTQ export is CPU/RAM intensive and takes many hours. Run on a machine with at least 16GB RAM. Use `nohup` for unattended operation:

```bash
nohup python3 scripts/export_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --precision INT4 --algorithm gptq \
    --dataset wikitext2 --num-samples 128 \
    --suffix "-gptq" > ~/intel-bench/logs/gptq-export.log 2>&1 &
```

**Note**: There is no INT8 or FP16 variant of GPTQ — it only applies to INT4 quantization.

### 3.3 Transferring Models Between Machines

Always transfer over LAN, never through the Mac:

```bash
# From any NUC to another (example: Grove → Fairchild)
scp -r ~/models/intel-bench/Qwen2.5-7B-Instruct-INT4 \
    fairchild@192.168.18.236:~/models/intel-bench/

# Bulk transfer all models for a given base model
scp -r ~/models/intel-bench/Llama-3.1-8B-Instruct-* \
    fairchild@192.168.18.236:~/models/intel-bench/
```

LAN transfer speed: ~1GB/min. Tailscale relay: ~500MB/min.

---

## 4. Running Benchmarks

### 4.1 Single Benchmark Run

```bash
source ~/intel-bench-venv/bin/activate
cd ~/intel-bench/repo

python3 benchmark.py \
    --precision INT4 \
    --device GPU \
    --codename PTL \
    --tdp 25W \
    --temperature 0.0 0.7 \
    --db ~/intel-bench/results/benchmarks.db
```

Key flags:
- `--precision`: FP16, INT8, INT4
- `--device`: GPU, CPU, NPU
- `--temperature`: Space-separated list (0.0 for deterministic, 0.7 for realistic)
- `--backend`: `optimum` (default Python wrapper) or `genai` (C++ LLMPipeline, ~9% faster)
- `--model-suffix`: For GPTQ models, use `--model-suffix="-gptq"` (note: use `=` to avoid argparse parsing `-gptq` as a flag)
- `--experiment`: Tag for the experiment name in the database
- `--ov-config`: Comma-separated OpenVINO runtime config (e.g., `PERFORMANCE_HINT=LATENCY`)
- `--perfspect`: Run Intel PerfSpect system profiler (once per session)
- `--model`: Override model (default: Llama, also supports `qwen`)

### 4.2 Overnight Batch Run

Run all available model/device/precision combinations unattended:

```bash
ssh fairchild@192.168.18.236
source ~/intel-bench-venv/bin/activate
cd ~/intel-bench/repo

nohup bash scripts/overnight-bench.sh PTL 25W "Baseline DDR5-5600" \
    > ~/intel-bench/logs/overnight.log 2>&1 &

# Monitor progress
tail -f ~/intel-bench/logs/overnight.log
```

The overnight script:
1. Detects available devices (GPU, NPU, CPU)
2. Finds all exported models in `~/models/intel-bench/`
3. Runs each device/precision combo with warmup + 10 measured iterations
4. Logs each combo to its own log file
5. Continues on failure (doesn't stop the whole batch)

### 4.3 Optimization Experiments

Run the full experiment suite (ov_config flags, GPTQ, GenAI pipeline):

```bash
nohup bash scripts/optimization-experiments.sh \
    > ~/intel-bench/logs/optimization.log 2>&1 &
```

This runs three experiments sequentially:
1. **Experiment 1**: `ov_config` runtime flags (KV_CACHE_PRECISION, PERFORMANCE_HINT)
2. **Experiment 2**: GPTQ INT4 vs AWQ INT4 baseline
3. **Experiment 3**: OpenVINO GenAI C++ pipeline vs Python optimum wrapper

### 4.4 Adding a New Model

To benchmark a new model:

1. Export it on a machine with sufficient RAM (Fairchild recommended):
   ```bash
   python3 scripts/export_model.py --model <HuggingFace-ID> --precision all
   ```
2. Transfer to other machines if needed (Section 3.3)
3. Add model detection to `overnight-bench.sh` or run manually with `--model` flag

---

## 5. Results Collection & Central Database

### 5.1 Architecture

Each NUC stores results locally in `~/intel-bench/results/benchmarks.db` (SQLite). SKELETOR-03 is the central consolidation host.

```
  Fairchild ──SCP──→
  Noyce     ──SCP──→  SKELETOR-03 (master DB)
  Grove     ──SCP──→  ~/intel-bench/results/master.db
  Friday    ──SCP──→
```

### 5.2 Pushing Results to SKELETOR

After a benchmark run completes, push results to the central DB:

```bash
bash scripts/push-results.sh
# Or specify a custom DB path:
bash scripts/push-results.sh ~/intel-bench/results/benchmarks.db
```

This SCPs the local database to `skeletor@192.168.18.145:~/intel-bench/results/incoming/<hostname>-<timestamp>.db`.

**Automation**: Add this to the end of your overnight scripts:

```bash
# At the end of overnight-bench.sh or optimization-experiments.sh:
bash scripts/push-results.sh
```

### 5.3 Consolidating the Master DB

On SKELETOR, merge incoming databases into the master:

```bash
ssh skeletor@192.168.18.145
cd ~/intel-bench

# Merge all incoming DBs into master
python3 repo/scripts/sync_db.py \
    --local-files results/incoming/*.db \
    --target results/master.db
```

The sync script handles:
- **ID remapping**: Each NUC has its own auto-increment IDs; these get remapped during merge
- **Deduplication**: Uses `(hostname, started_at)` as natural key — duplicate runs are skipped
- **Schema migration**: Automatically adds missing columns (e.g., `experiment_name`)

### 5.4 Querying the Database

View results from the command line:

```bash
python3 benchmark.py --results --db ~/intel-bench/results/benchmarks.db
```

Direct SQLite queries:

```bash
sqlite3 ~/intel-bench/results/benchmarks.db <<'SQL'
SELECT
    m.hostname,
    r.model_name,
    r.precision,
    r.backend,
    r.experiment_name,
    a.median_tps,
    a.median_ttft_ms,
    COUNT(*) as runs
FROM run_aggregates a
JOIN benchmark_runs r ON a.run_id = r.id
JOIN machines m ON r.machine_id = m.id
GROUP BY m.hostname, r.model_name, r.precision, r.backend, r.experiment_name
ORDER BY a.median_tps DESC;
SQL
```

---

## 6. Whitepaper Workflow

### 6.1 Documents

All whitepapers live in `docs/` and are version-controlled in the private GitHub repo:

| File | Purpose |
|------|---------|
| `docs/optimization-whitepaper.md` | Performance optimization experiments (ov_config, GPTQ, GenAI pipeline) |
| `docs/quality-comparison-whitepaper.md` | Semantic quality comparison (Qwen vs Llama, FP16 vs INT4) |
| `docs/whitepaper.md` | Original benchmarking overview (legacy) |

### 6.2 Updating Whitepapers

1. Run benchmarks and collect data in the database
2. Query the database for relevant metrics
3. Update the appropriate whitepaper with new results
4. Commit and push to GitHub:

```bash
cd ~/Source/JoshCork/intel-ai-benchmarking  # Mac repo checkout
git add docs/*.md
git commit -m "docs: update whitepaper with <description>"
git push
```

### 6.3 Publishing

The final publication will synthesize all whitepapers into a single paper. The workflow is:

1. Complete all optimization experiments → `optimization-whitepaper.md`
2. Complete quality analysis → `quality-comparison-whitepaper.md`
3. Synthesize into final publishable paper (future task)

---

## 7. Key Results Reference

### Best Configurations (Panther Lake H, Xe3-LPG, 25W)

| Configuration | TPS | TTFT | vs Baseline |
|---------------|-----|------|-------------|
| Qwen2.5-7B INT4 AWQ + GenAI | 18.7 | 65ms | +38.5% |
| Llama 3.1-8B INT4 GPTQ + GenAI | 17.6 | 77ms | +30.4% |
| Qwen2.5-7B INT4 AWQ + optimum | 17.2 | — | +27.4% |
| Llama 3.1-8B INT4 GPTQ + optimum | 16.3 | — | +20.7% |
| Llama 3.1-8B INT4 AWQ + GenAI | 14.7 | 86ms | +8.9% |
| **Llama 3.1-8B INT4 AWQ + optimum (baseline)** | **13.5** | **—** | **—** |

### Optimization Stack

Optimizations are **additive** and can be combined:

1. **GPTQ** (+20.7%): Better weight quantization via calibration
2. **GenAI pipeline** (+8.9%): C++ inference path, avoids Python overhead
3. **Model selection** (+27.4%): Qwen2.5-7B's GQA architecture is more bandwidth-efficient
4. **Combined GPTQ + GenAI** (+30.4%): Best Llama configuration

### Dead Ends

- `DYNAMIC_QUANTIZATION_GROUP_SIZE=64`: Crashes with `bad_function_call` on OpenVINO 2025.4.1
- `KV_CACHE_PRECISION=u8`: Hurts INT4 performance (already quantized, double-quantizing KV cache degrades quality)

---

## 8. Troubleshooting

### Model not found
Ensure models are in `~/models/intel-bench/<ModelName>-<PRECISION>/` with a `.xml` file present.

### SSH connection refused
Check LAN IP is correct. Machines may get new DHCP leases. Use `arp -a` or check router DHCP table.

### GPTQ export hangs / OOM
GPTQ requires ~2x model size in RAM for calibration. Use a machine with at least 16GB free. Monitor with `htop`.

### argparse treats --model-suffix value as flag
Use `=` syntax: `--model-suffix="-gptq"` instead of `--model-suffix -gptq`.

### Benchmark fails with "No devices found"
Ensure the GPU driver is loaded: `ls /dev/dri/`. If missing, reboot or check kernel modules.

### Overnight script stuck
Check if a single benchmark is hanging. Look at the log files in `~/intel-bench/logs/`. Kill and restart if needed.
