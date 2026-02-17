# GPU Runtime Optimization for Intel Xe3 iGPU LLM Inference

**Optimizing Llama 3.1 8B Instruct on Panther Lake Xe3-LPG with OpenVINO**

*February 2026*

---

## Abstract

*This section will be completed after all experiments are run.*

Building on our baseline benchmarking results (see `whitepaper.md`), this paper investigates three concrete runtime optimization strategies for improving LLM inference throughput on Intel Xe3-LPG integrated GPUs. We test each optimization independently on a Panther Lake H system (25W, DDR5-5600) using standardized kiosk conversation scenarios, comparing against the unoptimized baseline to isolate the effect of each lever.

---

## 1. Introduction

### 1.1 Background

Our baseline benchmarking established that Panther Lake (Xe3-LPG) with DDR5-5600 achieves approximately **13.9 TPS at INT4** with zero runtime optimizations — using `optimum-intel`'s `OVModelForCausalLM.from_pretrained()` with default settings. This leaves performance on the table: OpenVINO exposes several runtime configuration knobs that can improve throughput, and alternative inference pipelines exist that may outperform the Python-based optimum-intel wrapper.

### 1.2 Optimization Hypotheses

We identified three independent optimization levers through analysis of OpenVINO documentation and community benchmarks:

1. **Runtime configuration flags** (`ov_config`): KV cache quantization, dynamic quantization, and latency-optimized scheduling may reduce memory pressure and improve GPU utilization.

2. **GPTQ quantization with scale estimation**: The default INT4 export uses AWQ quantization. GPTQ with scale estimation may produce a more GPU-friendly weight layout, potentially improving throughput at the same precision.

3. **Native OpenVINO GenAI pipeline**: Replacing the Python-based `optimum-intel` wrapper with the C++ `openvino-genai` `LLMPipeline` may reduce per-token overhead and improve throughput, especially for the decode phase.

### 1.3 Goals

- Measure the throughput impact of each optimization **independently**
- Identify which optimizations are additive vs. redundant
- Determine whether any optimization meaningfully closes the gap to the DDR5-7200 configuration (16.4 TPS baseline)
- Provide reproduction commands for all experiments

---

## 2. Experimental Design

### 2.1 Test Platform

| Component | Specification |
|-----------|--------------|
| **System** | PTL-FAIRCHILD |
| **CPU** | Intel Core Ultra (Panther Lake H), 25W TDP |
| **GPU** | Intel Xe3-LPG (integrated) |
| **Memory** | 2×32GB DDR5-5600 (64GB total) |
| **OS** | Ubuntu 24.04 LTS |
| **Runtime** | OpenVINO 2024.6+ via optimum-intel |

### 2.2 Baseline Configuration

The baseline uses the existing benchmark code with **zero runtime optimizations**:

```python
# Baseline: no ov_config, no special flags
model = OVModelForCausalLM.from_pretrained(model_path, device="GPU")
```

Baseline experiment name: `PTL-FAIRCHILD-DDR5-5600` (already collected).

### 2.3 Experiment Matrix

#### Experiment 1: `ov_config` Runtime Flags

| Parameter | Value |
|-----------|-------|
| **Change** | Pass `ov_config` dict to `OVModelForCausalLM.from_pretrained()` |
| **Flags** | `KV_CACHE_PRECISION=u8`, `DYNAMIC_QUANTIZATION_GROUP_SIZE=64`, `PERFORMANCE_HINT=LATENCY` |
| **Precisions** | FP16, INT8, INT4 |
| **Temperatures** | 0.0, 0.7 |
| **Re-export required?** | No |
| **New dependencies?** | No |

**Hypothesis**: KV cache quantization (u8) reduces memory bandwidth demand during decode. Dynamic quantization may improve INT8/FP16 throughput. Latency hint tells the scheduler to optimize for single-stream inference.

#### Experiment 2: GPTQ INT4 with Scale Estimation

| Parameter | Value |
|-----------|-------|
| **Change** | Re-export INT4 model using `--gptq --scale-estimation` |
| **Export flags** | `--gptq --scale-estimation --dataset wikitext2 --group-size 128 --num-samples 128` |
| **Precisions** | INT4 only (different quantization algorithm) |
| **Temperatures** | 0.0, 0.7 |
| **Re-export required?** | Yes (INT4 only) |
| **New dependencies?** | No |

**Hypothesis**: GPTQ with scale estimation may produce better weight quantization than the default AWQ approach, leading to higher throughput and/or better output quality at INT4.

#### Experiment 3: OpenVINO GenAI `LLMPipeline`

| Parameter | Value |
|-----------|-------|
| **Change** | Replace `optimum-intel` Python wrapper with `openvino-genai` C++ pipeline |
| **Backend** | `openvino_genai.LLMPipeline` |
| **Precisions** | FP16, INT8, INT4 |
| **Temperatures** | 0.0, 0.7 |
| **Re-export required?** | No |
| **New dependencies?** | `openvino-genai` pip package |

**Hypothesis**: The C++ pipeline eliminates Python overhead in the token generation loop, potentially improving both TTFT and sustained TPS. Effect should be most visible on faster configurations (INT4) where per-token overhead is a larger fraction of total time.

### 2.4 Control Variables

All experiments hold constant:
- **Model**: Llama 3.1 8B Instruct (same HuggingFace base model)
- **Device**: GPU only (Xe3-LPG iGPU)
- **Scenarios**: All 7 kiosk conversation scenarios
- **Warmup**: 3 runs per scenario (discarded)
- **Measured runs**: 10 per scenario
- **Max tokens**: 256
- **System prompt**: Same across all experiments (from `config.toml`)
- **System state**: PerfSpect config captured before each experiment

### 2.5 Metrics

For each experiment, we report:
- **TPS** (tokens/sec): Mean, P5, P95 across measured runs
- **TTFT** (time to first token): Mean in milliseconds
- **Δ vs. baseline**: Percentage change from unoptimized baseline

---

## 3. Experiment 1 Results: `ov_config` Runtime Flags

*To be filled after running experiment.*

### 3.1 Configuration

```bash
python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-ov-config-DDR5-5600" \
  --ov-config "KV_CACHE_PRECISION=u8,DYNAMIC_QUANTIZATION_GROUP_SIZE=64,PERFORMANCE_HINT=LATENCY" \
  --perfspect --db ~/intel-bench/results/benchmarks.db
```

### 3.2 Results

| Precision | Temp | Baseline TPS | Optimized TPS | Δ TPS | Baseline TTFT | Optimized TTFT | Δ TTFT |
|-----------|------|-------------|--------------|-------|--------------|----------------|--------|
| FP16 | 0.0 | | | | | | |
| FP16 | 0.7 | | | | | | |
| INT8 | 0.0 | | | | | | |
| INT8 | 0.7 | | | | | | |
| INT4 | 0.0 | | | | | | |
| INT4 | 0.7 | | | | | | |

### 3.3 Analysis

*To be filled after results.*

---

## 4. Experiment 2 Results: GPTQ INT4 with Scale Estimation

*To be filled after running experiment.*

### 4.1 Configuration

```bash
# Export
python3 scripts/export_model.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --precision INT4 --algorithm gptq \
  --dataset wikitext2 --num-samples 128 --suffix "-gptq"

# Benchmark
python3 benchmark.py --precision INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-gptq-INT4-DDR5-5600" \
  --model-suffix "-gptq" \
  --db ~/intel-bench/results/benchmarks.db
```

### 4.2 Results

| Precision | Temp | Baseline TPS (AWQ) | GPTQ TPS | Δ TPS | Baseline TTFT | GPTQ TTFT | Δ TTFT |
|-----------|------|-------------------|----------|-------|--------------|-----------|--------|
| INT4 | 0.0 | | | | | | |
| INT4 | 0.7 | | | | | | |

### 4.3 Analysis

*To be filled after results.*

---

## 5. Experiment 3 Results: OpenVINO GenAI `LLMPipeline`

*To be filled after running experiment.*

### 5.1 Configuration

```bash
python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-genai-DDR5-5600" \
  --backend genai \
  --db ~/intel-bench/results/benchmarks.db
```

### 5.2 Results

| Precision | Temp | Baseline TPS (optimum) | GenAI TPS | Δ TPS | Baseline TTFT | GenAI TTFT | Δ TTFT |
|-----------|------|----------------------|-----------|-------|--------------|------------|--------|
| FP16 | 0.0 | | | | | | |
| FP16 | 0.7 | | | | | | |
| INT8 | 0.0 | | | | | | |
| INT8 | 0.7 | | | | | | |
| INT4 | 0.0 | | | | | | |
| INT4 | 0.7 | | | | | | |

### 5.3 Analysis

*To be filled after results.*

---

## 6. Cross-Experiment Comparison

*To be filled after all experiments.*

### 6.1 Summary Table

| Optimization | Best INT4 TPS | Δ vs. Baseline | TTFT Impact | Complexity |
|-------------|--------------|----------------|-------------|------------|
| Baseline (none) | | — | — | None |
| Exp 1: ov_config | | | | Low (runtime flags) |
| Exp 2: GPTQ INT4 | | | | Medium (re-export) |
| Exp 3: GenAI pipeline | | | | Medium (new dep) |

### 6.2 Additive Potential

*Can optimizations be combined? E.g., GenAI pipeline + ov_config flags, or GenAI + GPTQ model.*

---

## 7. Discussion

*To be filled after all experiments.*

---

## 8. Conclusions

*To be filled after all experiments.*

---

## Appendix A: Reproduction Commands

### A.1 Environment Setup

```bash
# Activate venv
source ~/intel-bench-venv/bin/activate

# Verify OpenVINO
python3 -c "import openvino; print(openvino.__version__)"
```

### A.2 Baseline Run

```bash
python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-FAIRCHILD-DDR5-5600" \
  --perfspect --db ~/intel-bench/results/benchmarks.db
```

### A.3 Experiment 1: ov_config

```bash
python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-ov-config-DDR5-5600" \
  --ov-config "KV_CACHE_PRECISION=u8,DYNAMIC_QUANTIZATION_GROUP_SIZE=64,PERFORMANCE_HINT=LATENCY" \
  --perfspect --db ~/intel-bench/results/benchmarks.db
```

### A.4 Experiment 2: GPTQ Export + Benchmark

```bash
# Export GPTQ INT4 model
python3 scripts/export_model.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --precision INT4 --algorithm gptq \
  --dataset wikitext2 --num-samples 128 --suffix "-gptq"

# Benchmark
python3 benchmark.py --precision INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-gptq-INT4-DDR5-5600" \
  --model-suffix "-gptq" \
  --db ~/intel-bench/results/benchmarks.db
```

### A.5 Experiment 3: GenAI Pipeline

```bash
# Install dependency
pip install openvino-genai

# Benchmark
python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-genai-DDR5-5600" \
  --backend genai \
  --db ~/intel-bench/results/benchmarks.db
```

### A.6 Overnight Runner (All Experiments)

```bash
nohup bash scripts/optimization-experiments.sh > optimization-run.log 2>&1 &
```
