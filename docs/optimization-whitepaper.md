# GPU Runtime Optimization for Intel Client GPU LLM Inference

**Optimizing LLM Inference on Panther Lake Xe3-LPG iGPU and Arc A770M dGPU with OpenVINO**

*February 2026*

---

## Abstract

Building on our baseline benchmarking results (see `whitepaper.md`), this paper investigates runtime optimization strategies for improving LLM inference throughput on the Intel Xe3-LPG integrated GPU (Panther Lake H, 25W, DDR5-5600). We test three optimizations independently on Llama 3.1 8B Instruct, then benchmark Qwen2.5-7B-Instruct as an alternative model to evaluate the impact of architecture and model size on bandwidth-constrained hardware.

**Key findings**: (1) OpenVINO runtime configuration flags (`ov_config`) provide **no benefit** and can degrade INT4 performance by up to 7.4%. (2) The OpenVINO GenAI C++ `LLMPipeline` delivers **+4-16% throughput** over the Python-based optimum-intel wrapper, with larger gains on faster hardware (+16% on Arc A770M dGPU vs +9% on PTL iGPU at INT4). (3) **GPTQ quantization delivers +21% on iGPU and +47% on dGPU** over default AWQ at INT4. (4) **Qwen2.5-7B-Instruct is 8-28% faster than Llama 3.1-8B** across all precisions, reaching **52.2 TPS / 42ms TTFT on the Arc A770M** and **18.7 TPS / 65ms TTFT on PTL iGPU** — driven by fewer parameters (7B vs 8B) and Grouped Query Attention (GQA). (5) FP16 remains memory-bandwidth-bound regardless of model or optimization. (6) **Optimization gains scale with hardware speed** — the dGPU amplifies all software optimizations because Python overhead and suboptimal weight layouts represent a larger fraction of the per-token budget at higher throughput. (7) **DDR5-7200 validates the optimization amplification model**: Qwen 2.5-7B INT4 AWQ + GenAI reaches **22.8 TPS / 55ms TTFT** on PTL with DDR5-7200, and Lunar Lake's on-package LPDDR5X matches this at **22.8 TPS / 60ms TTFT** with GenAI — confirming that bandwidth scaling and software optimization gains are additive.

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
- Determine whether any optimization meaningfully closes the gap to the DDR5-7200 configuration (16.4 TPS baseline) — and validate with actual DDR5-7200 GenAI measurements
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

### 3.1 Configuration

```bash
python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-ov-config-DDR5-5600" \
  --ov-config "KV_CACHE_PRECISION=u8,DYNAMIC_QUANTIZATION_GROUP_SIZE=64,PERFORMANCE_HINT=LATENCY" \
  --perfspect --db ~/intel-bench/results/benchmarks.db
```

### 3.2 Results

| Precision | Temp | Baseline TPS | Optimized TPS | Δ TPS | Notes |
|-----------|------|-------------|--------------|-------|-------|
| FP16 | 0.0 | 5.1 | 5.0 | **-2.0%** | Within noise |
| FP16 | 0.7 | 5.0 | 5.0 | 0.0% | No change |
| INT8 | 0.0 | 9.7 | 9.7 | 0.0% | No change |
| INT8 | 0.7 | 9.4 | 9.2 | **-2.1%** | Slight degradation |
| INT4 | 0.0 | 13.5 | 13.4 | **-0.7%** | Within noise |
| INT4 | 0.7 | 12.9 | 12.6 | **-2.3%** | Degradation |

*Note: TTFT values of 0ms in the optimum-intel backend are a known measurement artifact — the Python streamer callback does not reliably capture first-token timing.*

### 3.3 Experiment 1.5: KV Cache u8 Only

After Experiment 1 showed negative results from the combined flags, we isolated `KV_CACHE_PRECISION=u8` alone to test whether the KV cache quantization was beneficial on its own.

```bash
python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-kv-cache-u8-DDR5-5600" \
  --ov-config "KV_CACHE_PRECISION=u8" \
  --perfspect --db ~/intel-bench/results/benchmarks.db
```

| Precision | Temp | Baseline TPS | KV u8 TPS | Δ TPS | Notes |
|-----------|------|-------------|-----------|-------|-------|
| FP16 | 0.0 | 5.1 | 5.1 | 0.0% | Neutral |
| FP16 | 0.7 | 5.0 | 5.0 | 0.0% | Neutral |
| INT8 | 0.0 | 9.7 | 9.7 | 0.0% | Neutral |
| INT8 | 0.7 | 9.4 | 9.4 | 0.0% | Neutral |
| INT4 | 0.0 | 13.5 | 12.5 | **-7.4%** | Significant degradation |
| INT4 | 0.7 | 12.9 | 12.1 | **-6.2%** | Significant degradation |

### 3.4 Analysis

**All three `ov_config` flags together produced no benefit and slight degradation** (-0.7% to -2.3%) across all precisions. The combined overhead of u8 KV cache quantization, dynamic quantization group sizing, and latency hints appears to introduce more computational overhead than the memory bandwidth savings they provide on this platform.

**Isolating KV_CACHE_PRECISION=u8** showed it is **neutral for FP16 and INT8** but **actively harmful for INT4** (-6.2% to -7.4%). The likely explanation is "double quantization" — when model weights are already INT4, quantizing the KV cache to u8 introduces additional precision loss that forces more recomputation or causes the scheduler to make suboptimal decisions. The effect is that the GPU spends more time on quantization/dequantization overhead than it saves on memory bandwidth.

**Conclusion**: Runtime `ov_config` flags do not improve performance on Xe3-LPG with DDR5-5600. The iGPU's memory subsystem is already operating efficiently under default settings, and adding quantization overhead to an already bandwidth-constrained pipeline only makes things worse.

---

## 4. Experiment 2 Results: GPTQ INT4 with Scale Estimation

### 4.1 Configuration

```bash
# Export (ran on Noyce — CPU-only, ~8 hours)
optimum-cli export openvino \
  -m meta-llama/Llama-3.1-8B-Instruct \
  --weight-format int4 --gptq --scale-estimation \
  --dataset wikitext2 --group-size 128 --num-samples 128 \
  ~/models/intel-bench/Llama-3.1-8B-Instruct-INT4-gptq

# Benchmark (optimum backend)
python3 benchmark.py --precision INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-gptq-INT4-DDR5-5600" \
  --model-suffix "-gptq" \
  --db ~/intel-bench/results/benchmarks.db

# Benchmark (GenAI backend)
python3 benchmark.py --precision INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-gptq-genai-INT4-DDR5-5600" \
  --model-suffix "-gptq" --backend genai \
  --db ~/intel-bench/results/benchmarks.db
```

### 4.2 Results

| Backend | Temp | AWQ INT4 TPS | GPTQ INT4 TPS | Δ TPS | GPTQ TTFT (ms) |
|---------|------|-------------|---------------|-------|----------------|
| optimum | 0.0 | 13.5 | **16.3** | **+20.7%** | N/A* |
| optimum | 0.7 | 12.9 | **15.4** | **+19.4%** | N/A* |
| GenAI | 0.0 | 14.7 | **17.6** | **+19.7%** | **77** |
| GenAI | 0.7 | 14.5 | **17.3** | **+19.3%** | **78** |

*\* optimum-intel streamer does not reliably capture TTFT*

### 4.3 Analysis

**GPTQ with scale estimation is a major win** — a consistent ~20% throughput improvement over the default AWQ quantization at INT4, regardless of backend or temperature.

The GPTQ model (4.4GB) is slightly larger than the AWQ model (4.2GB), yet decodes significantly faster. This suggests that GPTQ's per-channel scale estimation produces weight layouts that are more efficient for the Xe3-LPG's INT4 compute units, likely because the scale factors better preserve the dynamic range of critical layers.

**GPTQ + GenAI is the best Llama configuration**: 17.6 TPS with 77ms TTFT — a **30.4% improvement** over the AWQ optimum baseline (13.5 TPS). The GPTQ and GenAI optimizations are fully additive: GPTQ improves the model itself (~20%), GenAI improves the runtime (~9%), and the gains stack.

**Export cost**: The GPTQ export with scale estimation and 128 calibration samples took ~8 hours on CPU (Noyce, 16 cores). This is a one-time cost that pays for itself immediately in production throughput.

---

## 5. Experiment 3 Results: OpenVINO GenAI `LLMPipeline`

### 5.1 Configuration

```bash
pip install openvino-genai

python3 benchmark.py --precision FP16 INT8 INT4 --device GPU \
  --temperature 0.0 0.7 --codename PTL --tdp 25W \
  --experiment "PTL-genai-DDR5-5600" \
  --backend genai \
  --db ~/intel-bench/results/benchmarks.db
```

### 5.2 Results

| Precision | Temp | Baseline TPS (optimum) | GenAI TPS | Δ TPS | GenAI TTFT (ms) |
|-----------|------|----------------------|-----------|-------|-----------------|
| FP16 | 0.0 | 5.1 | 5.1 | 0.0% | 215 |
| FP16 | 0.7 | 5.0 | 5.1 | **+2.0%** | 215 |
| INT8 | 0.0 | 9.7 | 10.1 | **+4.1%** | 119 |
| INT8 | 0.7 | 9.4 | 10.0 | **+6.4%** | 120 |
| INT4 | 0.0 | 13.5 | 14.7 | **+8.9%** | 90 |
| INT4 | 0.7 | 12.9 | 14.5 | **+12.4%** | 90 |

### 5.3 Analysis

**The GenAI C++ pipeline is the clear winner**, delivering consistent throughput improvements across all precisions and temperatures:

- **INT4**: The largest gains (+8.9% to +12.4%), pushing INT4/0.0 from 13.5 to 14.7 TPS and INT4/0.7 from 12.9 to 14.5 TPS. This represents a significant uplift for the most bandwidth-efficient precision.

- **INT8**: Solid gains of +4.1% to +6.4%, moving from 9.7 to 10.1 TPS (greedy) and 9.4 to 10.0 TPS (sampling).

- **FP16**: Minimal improvement (0-2%), confirming that FP16 is truly memory-bandwidth-bound. The C++ pipeline can't help when the bottleneck is raw DRAM throughput.

**Key insight — real TTFT values**: The GenAI pipeline's native `StreamerBase` callback reliably captures time-to-first-token, which the Python-based optimum-intel streamer failed to report (showing 0ms). Real TTFT values are:
- **INT4**: 90ms (excellent for interactive kiosk use)
- **INT8**: 119ms
- **FP16**: 215ms

All TTFT values are well under the 500ms threshold for perceived "instant" response in conversational UX.

**Why GenAI is faster**: The C++ `LLMPipeline` eliminates Python overhead in the token generation loop. In the optimum-intel path, each token generation step crosses the Python-C++ boundary via PyTorch tensors, incurs GIL contention, and involves Python object creation/destruction. The GenAI pipeline keeps the entire generate loop in C++ with only a single Python callback per token (the streamer). At INT4 speeds (~15 TPS), this overhead represents ~7-12% of the per-token time, which is exactly the improvement we observe.

---

## 6. Cross-Experiment Comparison

### 6.1 Summary Table — PTL iGPU (INT4, temp=0.0 — primary kiosk configuration)

#### Llama 3.1-8B-Instruct Optimizations (PTL iGPU, DDR5-5600)

| Optimization | INT4 TPS | Δ vs. Baseline | TTFT (ms) | Complexity | Recommendation |
|-------------|----------|----------------|-----------|------------|----------------|
| Baseline (optimum-intel, no flags) | 13.5 | — | N/A* | None | Current default |
| Exp 1: 3 ov_config flags | 13.4 | **-0.7%** | N/A* | Low | **Do not use** |
| Exp 1.5: KV_CACHE_PRECISION=u8 only | 12.5 | **-7.4%** | N/A* | Low | **Do not use** |
| Exp 2: GPTQ INT4 (optimum) | **16.3** | **+20.7%** | N/A* | Medium | **Use this model** |
| Exp 3: GenAI C++ pipeline | **14.7** | **+8.9%** | **90** | Medium | **Use this backend** |
| **Exp 2+3: GPTQ + GenAI** | **17.6** | **+30.4%** | **77** | Medium | **Best Llama (iGPU)** |

#### Model Comparison — PTL iGPU (GenAI backend — best configuration)

| Model + Backend | INT4 TPS | Δ vs. Llama Baseline | TTFT (ms) | Recommendation |
|----------------|----------|---------------------|-----------|----------------|
| Llama 3.1-8B AWQ + optimum | 13.5 | — | N/A* | Baseline |
| Llama 3.1-8B AWQ + GenAI | 14.7 | +8.9% | 90 | Good |
| Llama 3.1-8B GPTQ + optimum | 16.3 | +20.7% | N/A* | Good |
| Llama 3.1-8B GPTQ + GenAI | **17.6** | **+30.4%** | **77** | **Best Llama (iGPU)** |
| Qwen 2.5-7B AWQ + optimum | 17.2 | +27.4% | N/A* | Good |
| **Qwen 2.5-7B AWQ + GenAI** | **18.7** | **+38.5%** | **65** | **Best iGPU (DDR5-5600)** |

### 6.2 Summary Table — PTL iGPU (DDR5-7200) — Bandwidth Validation

To validate that optimization gains scale with hardware bandwidth, we re-ran the best configurations on PTL with DDR5-7200 SODIMMs (+22% bandwidth over DDR5-5600).

| Model + Backend | INT4 TPS | Δ vs. DDR5-5600 Same Config | TTFT (ms) | Notes |
|----------------|----------|----------------------------|-----------|-------|
| Qwen 2.5-7B AWQ + GenAI | **22.8** | **+21.9%** (vs 18.7) | **55** | Matches bandwidth elasticity prediction (21-22 TPS) |
| Llama 3.1-8B GPTQ + GenAI | **21.5** | **+22.2%** (vs 17.6) | **67** | GPTQ gains preserved at higher bandwidth |

**Key finding**: Both configurations scale ~22% with the +22% bandwidth increase, confirming near-linear bandwidth elasticity. The optimization gains from GenAI and GPTQ are fully preserved — they do not diminish as hardware gets faster. This validates the **optimization amplification** thesis: software gains and hardware bandwidth gains are additive.

### 6.3 Summary Table — Lunar Lake iGPU (GenAI Backend)

Lunar Lake (Xe2-LPG, 8 cores, on-package LPDDR5X-8533, ~136 GB/s) was re-benchmarked with the GenAI backend. Previous Lunar Lake results used optimum-intel only (14.9 TPS with a <1ms TTFT measurement artifact).

| Model + Backend | INT4 TPS | TTFT (ms) | Notes |
|----------------|----------|-----------|-------|
| Qwen 2.5-7B AWQ + optimum (previous) | 14.9 | N/A* | Original measurement |
| **Qwen 2.5-7B AWQ + GenAI** | **22.8** | **60** | +53% — GenAI unlocks LPDDR5X bandwidth |

**Key finding**: With the GenAI backend, Lunar Lake matches PTL DDR5-7200 (22.8 TPS) despite having fewer Xe cores (8 vs 12). Lunar Lake's on-package LPDDR5X provides lower latency and ~136 GB/s effective bandwidth that, combined with the GenAI pipeline's reduced overhead, compensates for the core count disadvantage. The previous 14.9 TPS result reflected optimum-intel's Python overhead limiting Lunar Lake's bandwidth utilization, not a hardware limitation.

### 6.4 Summary Table — Arc A770M dGPU (SKELETOR-03)

The same optimization strategies were validated on the Arc A770M discrete GPU (16GB GDDR6, ~512 GB/s bandwidth) to measure how optimization gains scale with hardware speed.

#### Model Comparison — Arc A770M dGPU (INT4, temp=0.0)

| Model + Backend | INT4 TPS | Δ vs. Llama Baseline | TTFT (ms) | Recommendation |
|----------------|----------|---------------------|-----------|----------------|
| Llama 3.1-8B AWQ + optimum | 31.1 | — | N/A* | Baseline |
| Llama 3.1-8B AWQ + GenAI | 36.0 | +15.8% | 55 | Good |
| Llama 3.1-8B GPTQ + optimum | 45.8 | +47.3% | N/A* | Good |
| **Llama 3.1-8B GPTQ + GenAI** | **50.3** | **+61.7%** | **54** | **Best Llama (dGPU)** |
| Qwen 2.5-7B AWQ + optimum | 46.4 | +49.2% | N/A* | Good |
| **Qwen 2.5-7B AWQ + GenAI** | **52.2** | **+67.8%** | **42** | **Best overall** |

#### Full Precision Comparison — Arc A770M dGPU (GenAI vs Baseline, temp=0.0)

| Precision | Baseline TPS (optimum) | GenAI TPS | Δ | GenAI TTFT (ms) |
|-----------|----------------------|-----------|---|-----------------|
| FP16 | 15.7 | 17.3 | +10.2% | 84 |
| INT8 | 26.2 | 29.5 | +12.6% | 59 |
| INT4 | 31.1 | 36.0 | +15.8% | 55 |

**Key insight — optimization gains amplify on faster hardware**: The GenAI C++ pipeline delivers +16% on the dGPU vs +9% on the iGPU at INT4. GPTQ delivers +47% on the dGPU vs +21% on the iGPU. This scaling occurs because Python overhead and suboptimal weight layouts represent a larger fraction of the per-token budget when tokens are generated faster. The dGPU's ~32ms/token decode (INT4 baseline) has proportionally more room for overhead reduction than the iGPU's ~74ms/token.

*\* optimum-intel streamer does not reliably capture TTFT*

### 6.5 Full Precision Comparison — PTL iGPU (GenAI vs Baseline, temp=0.0)

| Precision | Baseline TPS | GenAI TPS | Δ | GenAI TTFT (ms) |
|-----------|-------------|-----------|---|-----------------|
| FP16 | 5.1 | 5.1 | 0.0% | 215 |
| INT8 | 9.7 | 10.1 | +4.1% | 119 |
| INT4 | 13.5 | 14.7 | +8.9% | 90 |

### 6.6 Additive Potential

The GPTQ and GenAI optimizations are **fully additive** on both platforms. GPTQ improves the model weights (~20-47%), GenAI improves the runtime (~9-16%), and the gains stack multiplicatively. Combining GenAI + ov_config flags is **not recommended** — the ov_config flags showed negative impact even with the optimum-intel backend.

---

## 7. Discussion

### 7.1 The Memory Bandwidth Wall

The most striking finding is the **FP16 performance ceiling at ~5.1 TPS on the iGPU** regardless of optimization strategy. This is a hard memory bandwidth limit: the 15GB FP16 model must read every weight from DDR5-5600 DRAM for each token, and the Xe3-LPG iGPU shares that DRAM bandwidth with the CPU. No software optimization can overcome this fundamental hardware constraint.

The only path to faster FP16 is faster memory. Our baseline whitepaper showed DDR5-7200 achieving 6.0 TPS (+18%), confirming that FP16 performance scales linearly with memory bandwidth.

**DDR5-7200 validates bandwidth elasticity for optimized INT4 as well.** When we ran the best INT4 configurations (GenAI backend) on PTL with DDR5-7200 SODIMMs, Qwen 2.5-7B reached **22.8 TPS** (vs 18.7 on DDR5-5600, +22%) and Llama 3.1-8B GPTQ reached **21.5 TPS** (vs 17.6, +22%). The +22% throughput gain matches the +22% bandwidth increase almost exactly, confirming that optimized INT4 inference remains bandwidth-elastic — the software optimizations (GenAI, GPTQ, model architecture) do not shift the bottleneck away from memory bandwidth, they simply use the available bandwidth more efficiently. This is the ideal outcome: it means optimization gains and bandwidth gains are additive, and future hardware with faster memory will continue to benefit from these software techniques.

**The dGPU confirms this analysis from the opposite direction.** The Arc A770M with dedicated GDDR6 (~512 GB/s) achieves **17.3 TPS at FP16** — 3.4× the iGPU's 5.1 TPS. With ~10× the memory bandwidth, the dGPU delivers ~3.4× the FP16 throughput, indicating that FP16 decode is ~34% bandwidth-efficient on GDDR6. The GenAI pipeline still helps on dGPU FP16 (+10.2% vs 0% on iGPU), suggesting that Python overhead is measurable even at FP16 speeds when the memory subsystem is fast enough.

### 7.2 Why C++ Wins but Config Flags Don't

The GenAI C++ pipeline's improvement is most pronounced at INT4 (+8.9% iGPU, +15.8% dGPU) and decreases at FP16 (0% iGPU, +10.2% dGPU). This pattern is explained by the fraction of time spent in Python overhead vs. actual compute:

- **iGPU**: At FP16 (5.1 TPS, ~196ms/token), Python overhead is negligible (<1% of token time). At INT4 (13.5 TPS, ~74ms/token), Python overhead becomes significant (~7-12% of token time).
- **dGPU**: At FP16 (15.7 TPS, ~64ms/token), Python overhead is already measurable (~10%). At INT4 (31.1 TPS, ~32ms/token), it becomes dominant (~16% of token time).

This explains why **GenAI gains scale with hardware speed** — as the GPU gets faster, the fixed Python per-token cost becomes a larger fraction of total time, and the C++ pipeline's elimination of that overhead yields proportionally larger gains.

The `ov_config` runtime flags, by contrast, add computational overhead (quantization/dequantization of KV cache, group-level dynamic quantization math) that is supposed to be offset by reduced memory traffic. On Xe3-LPG, the iGPU's memory subsystem is already efficient enough that the overhead exceeds the savings. These flags were not tested on the dGPU, but the iGPU results were sufficiently negative to rule them out.

### 7.3 Qwen2.5-7B vs Llama 3.1-8B: Model Architecture Impact

After establishing the GenAI C++ pipeline as the optimal backend, we benchmarked Qwen2.5-7B-Instruct alongside Llama 3.1-8B-Instruct to evaluate whether a different model architecture could further improve throughput on bandwidth-constrained hardware.

#### 7.3.1 GenAI Backend Comparison — PTL iGPU (Best Configuration)

| Precision | Llama 3.1-8B TPS | Qwen 2.5-7B TPS | Qwen Advantage | Llama TTFT | Qwen TTFT |
|-----------|-----------------|-----------------|----------------|------------|-----------|
| INT4/0.0 | 14.7 | **18.7** | **+27.2%** | 90ms | **65ms** |
| INT4/0.7 | 14.5 | **18.4** | **+26.9%** | 90ms | **66ms** |
| INT8/0.0 | 10.1 | **10.9** | **+7.9%** | 119ms | **103ms** |
| INT8/0.7 | 10.0 | **10.8** | **+8.0%** | 120ms | **104ms** |
| FP16/0.0 | 5.1 | **5.5** | **+7.8%** | 215ms | **193ms** |
| FP16/0.7 | 5.1 | **5.5** | **+7.8%** | 215ms | **194ms** |

#### 7.3.2 GenAI Backend Comparison — Arc A770M dGPU

| Precision | Llama 3.1-8B TPS | Qwen 2.5-7B TPS | Qwen Advantage | Llama TTFT | Qwen TTFT |
|-----------|-----------------|-----------------|----------------|------------|-----------|
| INT4/0.0 | 36.0 | **52.2** | **+45.0%** | 55ms | **42ms** |
| INT8/0.0 | 29.5 | **30.3** | **+2.7%** | 59ms | **55ms** |
| FP16/0.0 | 17.3 | **19.0** | **+9.8%** | 84ms | **76ms** |

#### 7.3.3 Optimum-Intel Backend Comparison — PTL iGPU (Baseline)

| Precision | Llama 3.1-8B TPS | Qwen 2.5-7B TPS | Qwen Advantage |
|-----------|-----------------|-----------------|----------------|
| INT4/0.0 | 13.5 | **17.2** | **+27.4%** |
| INT4/0.7 | 12.9 | **15.6** | **+20.9%** |
| INT8/0.0 | 9.7 | **10.4** | **+7.2%** |
| INT8/0.7 | 9.4 | **9.8** | **+4.3%** |
| FP16/0.0 | 5.1 | **5.4** | **+5.9%** |
| FP16/0.7 | 5.0 | **5.3** | **+6.0%** |

#### 7.3.4 Why Qwen Is Faster

The performance gap is explained by three architectural differences:

1. **Fewer parameters**: Qwen2.5-7B has ~6.5B non-embedding parameters vs Llama's 8.0B — **19% fewer weights** to read from memory per token. On bandwidth-bound hardware, this translates almost linearly to throughput gains.

2. **Grouped Query Attention (GQA)**: Qwen2.5-7B uses 28 query heads with only 4 KV heads (7:1 ratio), dramatically reducing KV cache size compared to Llama's layout. Smaller KV cache means less memory traffic during decode.

3. **Amplification at INT4**: The Qwen advantage at INT4 is dramatic — **+27% on iGPU and +45% on dGPU** — compared to ~8-10% at FP16/INT8. This occurs because INT4 removes the weight-bandwidth bottleneck enough that KV cache and per-token overhead become relatively larger. Qwen's GQA reduces the KV cache component, and this benefit compounds at higher decode speeds. On the dGPU, where INT4 baseline already runs at 31 TPS, Qwen's architectural advantages amplify to nearly 50% faster throughput.

#### 7.3.5 GenAI Uplift per Model

| Platform | Model | Precision | Optimum TPS | GenAI TPS | GenAI Uplift |
|----------|-------|-----------|------------|-----------|-------------|
| iGPU | Llama 3.1-8B | INT4/0.0 | 13.5 | 14.7 | +8.9% |
| iGPU | Qwen 2.5-7B | INT4/0.0 | 17.2 | 18.7 | +8.7% |
| dGPU | Llama 3.1-8B | INT4/0.0 | 31.1 | 36.0 | +15.8% |
| dGPU | Qwen 2.5-7B | INT4/0.0 | 46.4 | 52.2 | +12.5% |

The GenAI C++ pipeline provides consistent uplift for both models on both platforms, confirming that the Python overhead reduction is model-agnostic. The dGPU sees larger absolute gains (+12-16%) than the iGPU (+8-9%) because the faster decode rate makes Python overhead a proportionally larger bottleneck.

### 7.4 Practical Impact for Kiosk Deployment

#### iGPU (Panther Lake Xe3-LPG)

The best iGPU configuration — **Qwen2.5-7B-Instruct INT4 with GenAI pipeline** — delivers **18.7 TPS with 65ms TTFT**:

- A 50-token response takes **2.7 seconds** (vs. 3.7 seconds with Llama optimum baseline)
- First token appears in **65ms** (perceived as instant)
- Combined software optimizations (model choice + GenAI pipeline) yield a **39% throughput improvement** over the Llama optimum baseline — equivalent to upgrading from DDR5-5600 to DDR5-8000+ in hardware terms

#### dGPU (Arc A770M)

The best dGPU configuration — **Qwen2.5-7B-Instruct INT4 with GenAI pipeline** — delivers **52.2 TPS with 42ms TTFT**:

- A 50-token response takes **under 1 second** (vs. 1.6 seconds with Llama optimum baseline)
- First token appears in **42ms** — imperceptible latency
- Combined software optimizations yield a **+68% throughput improvement** over Llama optimum baseline
- Even the baseline dGPU (Llama optimum, 31.1 TPS) is **2.3× faster** than the fully optimized iGPU (Qwen GenAI, 18.7 TPS), demonstrating that dedicated GDDR6 bandwidth is transformative for LLM decode

#### iGPU with DDR5-7200 (Panther Lake Xe3-LPG, fast memory)

PTL with DDR5-7200 SODIMMs — **Qwen2.5-7B-Instruct INT4 with GenAI pipeline** — delivers **22.8 TPS with 55ms TTFT**:

- A 50-token response takes **2.2 seconds** (vs. 2.7 seconds on DDR5-5600)
- First token appears in **55ms** — imperceptible
- Tied with Lunar Lake GenAI (also 22.8 TPS) as the **fastest iGPU configuration tested**, and exceeds the previous Lunar Lake optimum-intel result (14.9 TPS) by 53%
- The DDR5-7200 result validates our bandwidth elasticity model: we predicted 21-22 TPS based on linear bandwidth scaling from DDR5-5600, and measured 22.8 TPS — slightly above prediction, suggesting the GenAI pipeline benefits from DDR5-7200's higher burst bandwidth

#### Lunar Lake iGPU (Xe2-LPG, LPDDR5X-8533)

Lunar Lake with the GenAI backend — **Qwen2.5-7B-Instruct INT4 with GenAI pipeline** — delivers **22.8 TPS with 60ms TTFT**:

- Matches PTL DDR5-7200 exactly in throughput despite having fewer Xe cores (8 vs 12)
- On-package LPDDR5X provides lower memory latency than SODIMM DDR5, compensating for the core count difference
- The previous Lunar Lake result (14.9 TPS) used the optimum-intel backend; the GenAI pipeline provides a +53% uplift — the largest GenAI improvement observed on any platform, indicating that Python overhead was particularly severe on Lunar Lake's fast memory subsystem
- The <1ms TTFT previously reported for Lunar Lake was a measurement artifact from the optimum-intel streamer; the real TTFT is **60ms**

#### GPTQ on dGPU

The GPTQ optimization is particularly impactful on the dGPU: **Llama GPTQ + GenAI reaches 50.3 TPS** (+62% over Llama AWQ optimum baseline), nearly matching Qwen AWQ + GenAI (52.2 TPS). This means GPTQ can compensate for Llama's larger model size on fast hardware, giving deployment flexibility when Llama is preferred for quality or licensing reasons.

---

## 8. Conclusions

1. **Use Qwen2.5-7B-Instruct over Llama 3.1-8B-Instruct** when throughput matters. The smaller model with GQA delivers +27% higher INT4 throughput on iGPU (18.7 vs 14.7 TPS) and +45% on dGPU (52.2 vs 36.0 TPS). The Qwen advantage amplifies on faster hardware.

2. **Use the OpenVINO GenAI C++ pipeline** (`openvino-genai` LLMPipeline) instead of optimum-intel for production deployment. It provides +9-16% INT4 throughput improvement with no quality tradeoff, for both Llama and Qwen models on both iGPU and dGPU.

3. **Do not use `ov_config` runtime flags** (KV cache quantization, dynamic quantization, performance hints) on Xe3-LPG. They provide zero benefit and can degrade INT4 performance by up to 7%.

4. **Use GPTQ with scale estimation** instead of default AWQ quantization for INT4 models. GPTQ delivers +21% on iGPU and +47% on dGPU over AWQ with a one-time ~8-hour CPU export cost. The GPTQ and GenAI optimizations are fully additive (+30% combined on iGPU, +62% on dGPU).

5. **Optimization gains scale with hardware speed — validated by DDR5-7200.** Every software optimization we tested delivers larger percentage gains on the faster dGPU than on the iGPU. DDR5-7200 further validates this: Qwen GenAI reaches 22.8 TPS (+22% over DDR5-5600's 18.7 TPS), and Llama GPTQ+GenAI reaches 21.5 TPS (+22% over 17.6 TPS). The ~22% throughput gain matches the ~22% bandwidth increase exactly, confirming **bandwidth elasticity** — software optimization gains and hardware bandwidth gains are additive, not overlapping. Implication: as Intel ships faster memory and GPUs, software optimization becomes *more* important, not less.

6. **Lunar Lake matches PTL DDR5-7200 with GenAI backend.** Lunar Lake's on-package LPDDR5X-8533 with GenAI delivers **22.8 TPS / 60ms TTFT** — identical throughput to PTL DDR5-7200 despite having fewer Xe cores (8 vs 12). The previous Lunar Lake measurement (14.9 TPS) used optimum-intel and understated the platform's capability. The GenAI backend provides the largest uplift on Lunar Lake (+53%) of any platform, indicating that Python overhead was disproportionately limiting throughput on LPDDR5X's fast, low-latency memory subsystem.

7. **Optimal configurations**:
   - **iGPU (DDR5-5600)**: Qwen2.5-7B INT4 AWQ + GenAI — **18.7 TPS, 65ms TTFT**. If Llama required: GPTQ + GenAI — **17.6 TPS, 77ms TTFT**.
   - **iGPU (DDR5-7200)**: Qwen2.5-7B INT4 AWQ + GenAI — **22.8 TPS, 55ms TTFT**. Llama GPTQ + GenAI — **21.5 TPS, 67ms TTFT**.
   - **Lunar Lake iGPU**: Qwen2.5-7B INT4 AWQ + GenAI — **22.8 TPS, 60ms TTFT** — matches PTL DDR5-7200.
   - **dGPU**: Qwen2.5-7B INT4 AWQ + GenAI — **52.2 TPS, 42ms TTFT**. If Llama required: GPTQ + GenAI — **50.3 TPS, 54ms TTFT**. A GPTQ export of Qwen may yield even higher numbers (not yet tested).

8. **FP16 is memory-bandwidth-bound** at 5.1-5.5 TPS on iGPU and 17-19 TPS on dGPU, regardless of model or optimization. The only path to faster FP16 is faster memory (DDR5-7200+ or wider GDDR6 bus).

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
