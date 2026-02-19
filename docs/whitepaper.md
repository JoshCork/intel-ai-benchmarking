# Intel Edge AI Inference Benchmarking

**LLM Inference on Intel Client GPUs — Llama 3.1 8B & Qwen 2.5 7B**

*February 2026*

---

## Abstract

This paper presents a standardized benchmarking methodology for evaluating large language model (LLM) inference performance on Intel client-class hardware using the OpenVINO runtime. We benchmark **Meta's Llama 3.1 8B Instruct** and **Alibaba's Qwen2.5-7B-Instruct** across three weight precision formats (FP16, INT8, INT4) on Intel discrete and integrated GPUs, using realistic retail kiosk conversation scenarios as test workloads. We further evaluate runtime optimization strategies — GPTQ quantization, the OpenVINO GenAI C++ pipeline, and `ov_config` runtime flags — and conduct a semantic quality comparison across models and precisions.

Our results demonstrate that INT4 quantization on an Intel Arc A770M discrete GPU achieves **31-33 tokens/sec** with the baseline optimum-intel backend and up to **52.2 tokens/sec** with Qwen2.5-7B + GenAI pipeline — exceeding the 10 TPS interactive threshold by 5x — while maintaining conversational quality indistinguishable from FP16 inference. The full precision sweep across platforms reveals that **quantization yields proportionally larger gains on bandwidth-constrained iGPUs** (2.33x INT4/FP16 speedup on Lunar Lake vs 2.03x on Arc A770M).

On integrated GPUs with INT4, **Panther Lake (Xe3-LPG) with DDR5-7200 achieves 22.8 TPS** with Qwen + GenAI (16.4 TPS baseline) — the fastest iGPU configuration tested — while **Lunar Lake (Xe2) achieves 22.8 TPS** with Qwen + GenAI (matching PTL DDR5-7200 thanks to on-package LPDDR5X bandwidth) and **Meteor Lake (Xe-LPG) reaches only ~6.7 TPS**. A controlled memory bandwidth experiment on Panther Lake — same silicon with DDR5-7200 vs DDR5-5600 — reveals an **18% throughput drop** from 22% less bandwidth, confirming that iGPU LLM inference is overwhelmingly memory-bandwidth bound. A follow-up DDR5-7200 GenAI experiment measured 22.8 TPS — validating the 0.82 bandwidth elasticity coefficient (which predicted 21-22 TPS). At FP16, Meteor Lake manages only **3.3 TPS** — every query takes 10-26 seconds, rendering it unusable for interactive applications. NPU inference is not currently viable for autoregressive LLM workloads due to dynamic shape constraints in the Intel NPU compiler.

**Optimization findings**: Software optimizations deliver substantial gains beyond baseline hardware performance. The OpenVINO GenAI C++ pipeline provides **+4-16% throughput** over the Python-based optimum-intel wrapper, with larger gains on faster hardware (up to +16% on Arc A770M vs +9% on PTL iGPU at INT4). GPTQ quantization with scale estimation achieves **+47% on dGPU** and **+21% on iGPU** over default AWQ at INT4. Combined, GPTQ + GenAI delivers **+62%** on the Arc A770M — pushing Llama 3.1-8B from 31.1 to **50.3 TPS / 54ms TTFT**. Switching to **Qwen2.5-7B-Instruct** yields the highest throughput: **52.2 TPS / 42ms TTFT** on the Arc A770M and **18.7 TPS / 65ms TTFT** on the Panther Lake iGPU. Runtime `ov_config` flags provided no benefit and degraded INT4 performance by up to 7.4%.

**Quality findings**: INT4 quantization causes no meaningful quality degradation for kiosk conversations. Both models produce excellent, contextually appropriate responses at all precision levels. Qwen wins 3 of 7 scenarios head-to-head, Llama wins 2, with 2 ties. Both models hallucinate product names — RAG is essential for production deployment. The quality difference between models exceeds the quality difference between precisions, making model selection more important than precision choice.

---

## 1. Introduction

### 1.1 Motivation

Edge deployment of LLMs requires balancing three constraints: **latency** (time to first token and sustained throughput), **quality** (coherent, contextually appropriate responses), and **hardware cost** (running on available client-class silicon rather than datacenter GPUs).

Intel's OpenVINO toolkit enables hardware-portable model deployment: a single quantized model runs on Arc discrete GPUs, Lunar Lake integrated GPUs, and Meteor Lake integrated GPUs without recompilation. OpenVINO compiles device-specific compute kernels at load time, making the same `.xml/.bin` model artifact portable across Intel GPU generations.

This benchmark establishes a repeatable methodology for measuring real-world conversational inference performance on Intel client hardware.

### 1.2 Target Use Case

We target a **retail kiosk assistant** — an interactive terminal where customers ask questions about store hours, product availability, return policies, and loyalty programs. This use case demands:

- **Time to First Token (TTFT) < 100ms** — the screen should start responding immediately
- **Sustained throughput > 10 tokens/sec** — text appears at readable speed
- **Conversational quality** — responses should be helpful, accurate, and appropriately scoped
- **Multi-turn context** — the system maintains conversation state across turns

### 1.3 Hardware Fleet

| Machine | CPU | Codename | GPU | GPU Type | TDP | RAM |
|---------|-----|----------|-----|----------|-----|-----|
| friday-cork | 12th Gen Intel Core i7-12700H | ADL (Alder Lake) | Intel Arc A770M (16GB GDDR6) | Discrete (Xe-HPG) | 45W CPU / 120-150W TGP | 62 GB DDR4 |
| LNL-GROVE | Intel Core Ultra 7 258V | LNL (Lunar Lake) | Intel Arc 140V (Xe2-LPG, integrated) | Integrated | 17W SoC | 31 GB LPDDR5X-8533 |
| MTL-NOYCE | Intel Core Ultra 5 125H | MTL (Meteor Lake) | Intel Arc (Xe-LPG, integrated) | Integrated | 28W SoC | 62 GB DDR5-5600 |
| PTL-FAIRCHILD | Intel Core Ultra (Panther Lake-H) | PTL (Panther Lake) | Intel Arc (Xe3-LPG, integrated) | Integrated | 25W SoC | 16 GB DDR5-7200 / 64 GB DDR5-5600 |
| SKELETOR-03 | 12th Gen Intel Core i7-12700H | ADL (Alder Lake) | Intel Arc A770M (16GB GDDR6) | Discrete (Xe-HPG) | 45W CPU / 120-150W TGP | 62 GB DDR4 |

All machines run Ubuntu 24.04 LTS with OpenVINO 2025.4.1, Python 3.12, and the Intel GPU compute runtime (NEO). SKELETOR-03 is identical hardware to friday-cork and serves as the central benchmarking database host. Both dGPU machines produce identical baseline results (within measurement noise), confirming hardware reproducibility.

#### GPU Architecture Comparison

| Attribute | Arc A770M (Xe-HPG) | Arc 140V / LNL (Xe2-LPG) | PTL iGPU (Xe3-LPG) | MTL iGPU (Xe-LPG) |
|-----------|-------------------|--------------------------|---------------------|-------------------|
| **Xe Cores** | 32 | 8 | 12 | 8 |
| **XMX Matrix Engines** | 512 | 64 (8/core) | 96 (8/core) | **None** |
| **INT8 TOPS (GPU)** | 223 | 67 | ~85 (estimated) | 18 (vector only) |
| **Memory Type** | 16 GB GDDR6 (dedicated) | LPDDR5X-8533 (shared) | DDR5-7200 (SODIMM) / DDR5-5600 (SODIMM) | DDR5-5600 (shared) |
| **Memory Bus** | 256-bit | 128-bit | 128-bit | 128-bit |
| **Theoretical Bandwidth** | ~512 GB/s | ~136.5 GB/s | ~115 GB/s (7200) / ~90 GB/s (5600) | ~89.6 GB/s |
| **Effective Bandwidth** | ~400+ GB/s | ~95-110 GB/s | ~80-90 GB/s (7200) / ~60-70 GB/s (5600) | ~60-70 GB/s |
| **NPU** | N/A | NPU4 (48 TOPS) | NPU5 | NPU3 (11 TOPS) |

**Key architectural difference**: Meteor Lake's iGPU **lacks XMX matrix engines** entirely — all INT8/INT4 computation runs on standard vector units via DP4a instructions. Lunar Lake's Xe2 introduces XMX to the iGPU for the first time, providing ~3.7x the INT8 AI compute at the same core count. Panther Lake's Xe3-LPG increases to 12 cores with 96 XMX engines, providing 50% more compute than Lunar Lake. The Arc A770M has 4x the Xe cores *and* XMX, giving it overwhelming compute density.

**Memory bandwidth is the primary bottleneck** for LLM autoregressive decode. Each token requires reading the full model weights (~3.5 GB for INT4). The theoretical decode ceiling per platform:

| Platform | Effective BW | Theoretical Max TPS (INT4) | Observed TPS | Efficiency |
|----------|-------------|---------------------------|-------------|------------|
| Arc A770M | ~400 GB/s | ~114 TPS | 31.6 TPS | ~28% |
| PTL iGPU (DDR5-7200) | ~85 GB/s | ~24 TPS | 16.4 TPS (22.8 w/ GenAI) | ~68% (~95% w/ GenAI) |
| PTL iGPU (DDR5-5600) | ~65 GB/s | ~19 TPS | 13.5 TPS | ~71% |
| Lunar Lake iGPU | ~100 GB/s | ~29 TPS | 14.9 TPS (22.8 w/ GenAI) | ~51% (~79% w/ GenAI) |
| Meteor Lake iGPU | ~65 GB/s | ~19 TPS | 6.7 TPS | ~35% |

*GenAI figures use Qwen 2.5-7B INT4 AWQ + GenAI C++ backend; baseline figures use Llama 3.1-8B INT4 AWQ + optimum-intel.*

Panther Lake achieves the highest memory bandwidth efficiency (68-71% baseline, up to ~95% with Qwen + GenAI), likely due to a combination of Xe3's improved memory access patterns and the SODIMM interface characteristics. Lunar Lake achieves strong efficiency (51% baseline, ~79% with GenAI) due to on-package LPDDR5X providing lower latency — and with Qwen + GenAI, Lunar Lake matches PTL DDR5-7200 at 22.8 TPS despite lower theoretical bandwidth, suggesting its on-package memory advantage compensates for fewer Xe cores. Meteor Lake's moderate efficiency (35%) reflects the compute bottleneck from missing XMX engines — the vector-only pipeline cannot fully saturate available bandwidth for INT4 matrix operations.

### 1.4 Companion Papers

This paper is part of a three-paper series. The main results and cross-platform analysis are presented here; two companion papers provide deep dives into specific topics:

- **`optimization-whitepaper.md`** — *GPU Runtime Optimization for Intel Xe3 iGPU LLM Inference*. Detailed experimental results for three optimization strategies on Panther Lake: `ov_config` runtime flags (no benefit), GPTQ quantization (+20.7%), and the OpenVINO GenAI C++ pipeline (+8.9%). Includes Qwen vs Llama model architecture comparison and reproduction commands for all experiments.

- **`quality-comparison-whitepaper.md`** — *LLM Output Quality Comparison: Qwen2.5-7B vs Llama 3.1-8B on Intel Xe3*. Semantic quality analysis of actual generated responses across both models, all precision levels (FP16/INT8/INT4), and GPTQ variants. Includes full response transcripts, per-scenario head-to-head scoring, and the finding that INT4 quality equals FP16 for kiosk use cases.

---

## 2. Methodology

### 2.1 Model Preparation

**Base Models**:
- `meta-llama/Llama-3.1-8B-Instruct` (8.0B parameters) from HuggingFace
- `Qwen/Qwen2.5-7B-Instruct` (~6.5B non-embedding parameters) from HuggingFace

Qwen2.5-7B-Instruct is 19% smaller than Llama 3.1-8B in non-embedding parameters and uses Grouped Query Attention (GQA) with a 7:1 query-to-KV head ratio, reducing KV cache size during decode. These architectural differences have meaningful impact on bandwidth-constrained iGPU inference (see Section 8).

Models are exported from HuggingFace Transformers format to OpenVINO Intermediate Representation (IR) using the `optimum-intel` library:

```bash
optimum-cli export openvino -m <model_name> \
    --weight-format <precision> <output_dir>
```

Three precision variants are produced for each model:

| Precision | Weight Format | Model Size (Llama/Qwen) | Quantization Method |
|-----------|---------------|-------------------------|---------------------|
| **FP16** | Half-precision floating point | ~15 GB / ~14 GB | None (native export) |
| **INT8** | 8-bit integer, per-channel asymmetric | ~7.5 GB / ~7 GB | NNCF post-training quantization |
| **INT4** | Mixed INT4/INT8, group size 64 | ~5.2 GB / ~4.8 GB | NNCF with AWQ (Activation-aware Weight Quantization) |

The INT4 model uses NNCF's mixed-precision assignment: 80% of weight layers use INT4 asymmetric quantization with group size 64, while 20% (typically attention output projections and final layers) remain INT8 for accuracy preservation. Activation-aware Weight Quantization (AWQ) adjusts quantization parameters based on activation distributions to minimize accuracy loss.

**Alternative INT4 quantization — GPTQ with scale estimation**: For Llama 3.1-8B, we also exported an INT4 variant using GPTQ with scale estimation and 128 calibration samples from WikiText-2. The GPTQ model (4.4 GB) is slightly larger than the AWQ model (4.2 GB) but achieves significantly higher throughput (see Section 9). Export command:

```bash
optimum-cli export openvino -m meta-llama/Llama-3.1-8B-Instruct \
    --weight-format int4 --gptq --scale-estimation \
    --dataset wikitext2 --group-size 128 --num-samples 128 \
    ~/models/intel-bench/Llama-3.1-8B-Instruct-INT4-gptq
```

### 2.2 Inference Engine

Two inference backends are used:

**1. optimum-intel (Python)**: The baseline backend, using `OVModelForCausalLM` which wraps the OpenVINO runtime for autoregressive text generation.

```python
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

model = OVModelForCausalLM.from_pretrained(model_path, device="GPU")
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

**2. OpenVINO GenAI (C++)**: The optimized backend, using `openvino_genai.LLMPipeline` which keeps the entire generate loop in C++ with only a single Python callback per token. This eliminates Python overhead in the token generation loop and provides reliable TTFT measurement (see Section 9).

```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline(model_path, "GPU")
```

**Tokenization**: Each model's native chat template is applied via `tokenizer.apply_chat_template()`, producing properly formatted input sequences (Llama 3.1 Instruct format for Llama, ChatML format for Qwen).

**Decoding Strategies**:
- **Greedy (temperature=0.0)**: Deterministic; selects highest-probability token at each step. `do_sample=False`.
- **Sampling (temperature=0.7)**: Stochastic; applies temperature scaling and nucleus sampling with `top_p=0.9`. `do_sample=True`.

### 2.3 Timing Methodology

A custom `TimingStreamer` callback captures per-token timing during generation:

```python
class TimingStreamer:
    def put(self, value):
        now = time.perf_counter()
        if self.first_token_time is None:
            self.first_token_time = now
        self.token_times.append(now)
```

**Metrics captured per inference run:**

| Metric | Definition | Unit |
|--------|-----------|------|
| **TTFT** | Time from `model.generate()` call to first token callback | milliseconds |
| **Total Latency** | Wall-clock time from generate start to completion | milliseconds |
| **TPS** | `output_tokens / (total_ms / 1000)` | tokens/sec |
| **Output Tokens** | Number of tokens generated (excluding input) | count |
| **Per-Token Latency** | Delta between consecutive `TimingStreamer.put()` calls | milliseconds |

**Clock source**: `time.perf_counter()` — monotonic, sub-microsecond resolution on Linux.

### 2.4 Benchmark Protocol

Each scenario is run through a structured protocol:

1. **Warmup phase**: 3 runs discarded (allows GPU shader compilation, memory allocation, thermal stabilization)
2. **Measurement phase**: 10 timed runs recorded
3. **Token limit**: `max_new_tokens = 256`
4. **Temperature sweep**: Each scenario runs at both 0.0 (greedy) and 0.7 (sampling)

**Total runs per scenario**: 3 warmup + 10 measured = 13 inference calls
**Total runs per precision per temperature**: 7 scenarios x 13 = 91 inference calls
**Total runs per precision**: 91 x 2 temperatures = 182 inference calls

### 2.5 Statistical Aggregation

For each scenario/precision/temperature combination, we compute:

| Statistic | Algorithm |
|-----------|-----------|
| **Mean** | Arithmetic mean: `sum(values) / n` |
| **Median** | 50th percentile (linear interpolation) |
| **P5** | 5th percentile — best-case performance bound |
| **P95** | 95th percentile — worst-case performance bound |
| **Stddev** | Sample standard deviation: `sqrt(sum((x - mean)^2) / (n-1))` using Bessel's correction |

Percentiles use linear interpolation between adjacent sorted values:

```python
k = (p / 100.0) * (n - 1)
f, c = floor(k), ceil(k)
result = sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
```

---

## 3. Test Scenarios

All scenarios use a shared system prompt establishing the kiosk assistant persona:

> *You are a helpful kiosk assistant at a retail location. You help customers with finding products, answering questions about store policies, providing directions within the store, and assisting with loyalty program inquiries. Be concise, friendly, and professional. Keep responses under 3 sentences unless the customer asks for details.*

### 3.1 Scenario Definitions

| # | Name | Type | Description | User Prompt |
|---|------|------|-------------|-------------|
| 1 | `greeting` | Greeting | Minimal prompt — tests baseline latency | "Hi there! Can you help me?" |
| 2 | `store_hours` | Simple Task | Short factual response | "What are your store hours today?" |
| 3 | `product_lookup` | Simple Task | Structured product query | "I'm looking for wireless earbuds under $50. What do you have in stock?" |
| 4 | `return_policy` | Complex | Multi-constraint policy question | "I bought a laptop here 3 weeks ago and it's been crashing. I don't have the receipt but I paid with my credit card. Can I return it or get it fixed? What are my options?" |
| 5 | `loyalty_program` | Complex | Multi-part information request | "I want to sign up for your loyalty program. How does it work? What are the benefits, and do I get a discount on my purchase today?" |
| 6 | `multi_turn_directions` | Multi-Turn | 3-turn conversation about finding items | Turn 1: "Where can I find phone cases?" → Assistant responds → Turn 3: "Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?" |
| 7 | `multi_turn_troubleshoot` | Multi-Turn | 3-turn troubleshooting conversation | Turn 1: "The self-checkout machine isn't reading my credit card." → Assistant responds → Turn 3: "I tried that, it still says 'card read error'. I've used this card here before with no issues." |

---

## 4. Results — Intel Arc A770M (friday-cork)

**Machine**: Intel Core i7-12700H + Arc A770M (16GB VRAM), 62GB RAM, Ubuntu 24.04
**OpenVINO**: 2025.4.1 | **Device**: GPU.1 (discrete Arc A770M)

### 4.1 Performance Summary

#### INT4 (5.2 GB model — NNCF INT4_ASYM, group size 64, AWQ)

| Scenario | Temp | TPS Mean | TPS Median | TPS P5 | TPS P95 | TTFT Mean | Total Mean | Avg Tokens |
|----------|------|----------|------------|--------|---------|-----------|------------|------------|
| greeting | 0.0 | 30.2 | 30.2 | 28.4 | 31.2 | 0.9 ms | 432 ms | 13 |
| store_hours | 0.0 | 31.8 | 31.9 | 30.0 | 33.1 | 0.9 ms | 1,198 ms | 38 |
| product_lookup | 0.0 | 32.2 | 32.3 | 30.4 | 33.6 | 0.8 ms | 1,805 ms | 58 |
| return_policy | 0.0 | 31.2 | 31.2 | 29.6 | 33.0 | 1.2 ms | 1,733 ms | 54 |
| loyalty_program | 0.0 | 32.3 | 32.1 | 29.8 | 34.3 | 1.2 ms | 2,359 ms | 76 |
| multi_turn_directions | 0.0 | 33.1 | 33.2 | 31.9 | 34.1 | 0.8 ms | 1,600 ms | 53 |
| multi_turn_troubleshoot | 0.0 | 32.5 | 32.5 | 31.5 | 33.7 | 0.9 ms | 1,508 ms | 49 |
| | | | | | | | | |
| greeting | 0.7 | 22.8 | 22.7 | 21.4 | 24.2 | 0.6 ms | 860 ms | 20 |
| store_hours | 0.7 | 23.3 | 23.7 | 21.8 | 24.3 | 0.7 ms | 1,587 ms | 37 |
| product_lookup | 0.7 | 23.3 | 23.4 | 21.8 | 24.4 | 1.2 ms | 2,550 ms | 59 |
| return_policy | 0.7 | 23.8 | 24.1 | 22.8 | 24.5 | 1.1 ms | 2,948 ms | 70 |
| loyalty_program | 0.7 | 23.6 | 23.5 | 22.9 | 24.3 | 1.1 ms | 3,041 ms | 72 |
| multi_turn_directions | 0.7 | 23.7 | 23.7 | 23.3 | 24.2 | 1.0 ms | 2,288 ms | 54 |
| multi_turn_troubleshoot | 0.7 | 23.3 | 23.6 | 21.6 | 24.2 | 0.9 ms | 1,953 ms | 45 |

**Key findings — INT4:**
- Greedy decoding: **31.6 TPS mean** across all scenarios (3.2x kiosk target)
- Sampling (t=0.7): **23.4 TPS mean** across all scenarios (2.3x kiosk target)
- TTFT consistently < 2ms — effectively instant first-token response
- Temperature overhead: ~30% throughput reduction from greedy to sampling
- Low variance (stddev < 1.5 TPS) indicates stable, predictable performance

#### FP16 (15 GB model — native half-precision)

| Scenario | Temp | TPS Mean | TPS Median | TPS P5 | TPS P95 | TTFT Mean | Total Mean | Avg Tokens |
|----------|------|----------|------------|--------|---------|-----------|------------|------------|
| greeting | 0.0 | 15.3 | 15.3 | 14.9 | 15.5 | 1.3 ms | 852 ms | 13 |
| store_hours | 0.0 | 15.7 | 15.8 | 15.5 | 15.9 | 1.0 ms | 2,479 ms | 39 |
| product_lookup | 0.0 | 15.8 | 15.8 | 15.6 | 16.0 | 1.2 ms | 3,228 ms | 51 |
| return_policy | 0.0 | 15.6 | 15.6 | 15.3 | 15.8 | 1.3 ms | 4,629 ms | 72 |
| loyalty_program | 0.0 | 15.6 | 15.6 | 15.4 | 15.7 | 1.9 ms | 4,942 ms | 77 |
| multi_turn_directions | 0.0 | 15.5 | 15.5 | 15.4 | 15.7 | 1.5 ms | 3,290 ms | 51 |
| multi_turn_troubleshoot | 0.0 | 15.4 | 15.4 | 15.2 | 15.7 | 1.9 ms | 3,303 ms | 51 |
| | | | | | | | | |
| greeting | 0.7 | 10.9 | 10.8 | 10.4 | 11.7 | 1.7 ms | 1,395 ms | 15 |
| store_hours | 0.7 | 10.6 | 10.5 | 10.3 | 11.1 | 1.6 ms | 4,131 ms | 44 |
| product_lookup | 0.7 | 11.0 | 10.9 | 10.5 | 11.9 | 1.7 ms | 6,353 ms | 70 |
| return_policy | 0.7 | 10.7 | 10.7 | 10.4 | 11.0 | 1.8 ms | 6,356 ms | 68 |
| loyalty_program | 0.7 | 10.9 | 10.9 | 10.6 | 11.4 | 1.3 ms | 6,906 ms | 75 |
| multi_turn_directions | 0.7 | 10.9 | 10.8 | 10.5 | 11.4 | 1.7 ms | 5,013 ms | 55 |
| multi_turn_troubleshoot | 0.7 | 10.8 | 10.7 | 10.5 | 11.3 | 1.6 ms | 3,945 ms | 43 |

**Key findings — FP16:**
- Greedy decoding: **15.6 TPS mean** — consistent and extremely low variance (stddev ~0.2)
- Sampling (t=0.7): **10.8 TPS mean** — right at the kiosk target threshold
- TTFT: 1-2ms — slightly higher than INT4 but still sub-perceptible
- Remarkably stable: P5-P95 spread of only ~0.5 TPS indicates deterministic-like behavior
- Memory bound: 15GB model fully utilizes A770M's 16GB VRAM

#### INT8 (7.5 GB model — NNCF INT8_ASYM, per-channel)

| Scenario | Temp | TPS Mean | TPS Median | TPS P5 | TPS P95 | TTFT Mean | Total Mean | Avg Tokens |
|----------|------|----------|------------|--------|---------|-----------|------------|------------|
| greeting | 0.0 | 25.2 | 25.5 | 23.2 | 26.2 | 1.2 ms | 557 ms | 14 |
| store_hours | 0.0 | 26.2 | 26.2 | 25.2 | 27.1 | 1.3 ms | 1,489 ms | 39 |
| product_lookup | 0.0 | 26.5 | 26.6 | 25.6 | 27.3 | 1.0 ms | 2,081 ms | 55 |
| return_policy | 0.0 | 26.1 | 25.9 | 25.4 | 27.1 | 1.0 ms | 2,573 ms | 67 |
| loyalty_program | 0.0 | 26.8 | 26.7 | 26.1 | 27.5 | 1.2 ms | 2,838 ms | 76 |
| multi_turn_directions | 0.0 | 26.2 | 26.6 | 24.2 | 27.1 | 1.3 ms | 2,025 ms | 53 |
| multi_turn_troubleshoot | 0.0 | 25.8 | 26.0 | 24.5 | 27.0 | 1.1 ms | 1,978 ms | 51 |
| | | | | | | | | |
| greeting | 0.7 | 16.3 | 16.2 | 15.9 | 17.0 | 0.9 ms | 894 ms | 15 |
| store_hours | 0.7 | 18.2 | 18.7 | 16.5 | 19.1 | 0.8 ms | 2,190 ms | 40 |
| product_lookup | 0.7 | 17.1 | 16.9 | 16.3 | 18.1 | 0.9 ms | 3,295 ms | 56 |
| return_policy | 0.7 | 18.1 | 18.2 | 17.2 | 18.8 | 0.8 ms | 3,707 ms | 67 |
| loyalty_program | 0.7 | 18.4 | 18.6 | 17.1 | 18.9 | 0.8 ms | 4,097 ms | 75 |
| multi_turn_directions | 0.7 | 17.5 | 17.2 | 16.8 | 18.6 | 0.9 ms | 3,359 ms | 59 |
| multi_turn_troubleshoot | 0.7 | 17.3 | 17.3 | 16.6 | 18.0 | 0.9 ms | 2,577 ms | 45 |

**Key findings — INT8:**
- Greedy decoding: **26.1 TPS mean** — 67% faster than FP16, 17% slower than INT4
- Sampling (t=0.7): **17.6 TPS mean** — comfortably above the kiosk target
- TTFT: ~1ms — consistent with other precisions
- Moderate variance (stddev ~0.8 TPS) — between FP16's tight clustering and INT4's wider spread
- Sweet spot for quality-conscious deployments where INT4 quantization is a concern

### 4.2 Precision Comparison (Greedy Decoding, temp=0.0)

| Metric | FP16 | INT8 | INT4 | INT4 vs FP16 |
|--------|------|------|------|--------------|
| **Mean TPS** | 15.6 | 26.1 | 31.6 | **+103% (2.03x)** |
| **TPS Stddev** | 0.17 | 0.82 | 1.18 | Higher variance |
| **Mean TTFT** | 1.4 ms | 1.2 ms | 0.9 ms | 36% faster |
| **Model Size** | 15 GB | 7.5 GB | 5.2 GB | **65% smaller** |
| **Memory Headroom** | ~1 GB | ~8.5 GB | ~11 GB | |
| **vs Kiosk Target (10 TPS)** | 1.6x | 2.6x | 3.2x | |

#### Precision Comparison (Sampling, temp=0.7)

| Metric | FP16 | INT8 | INT4 |
|--------|------|------|------|
| **Mean TPS** | 10.8 | 17.6 | 23.4 |
| **TPS Stddev** | 0.35 | 0.67 | 0.78 |
| **vs Kiosk Target** | 1.1x | 1.8x | 2.3x |

### 4.3 Fastest Run Transcripts — Quality Assessment

The following are the **fastest TPS run** for each scenario, included for human quality grading and LLM-assisted evaluation.

#### 4.3.1 INT4, Greedy (temp=0.0) — Fastest Runs

**greeting** — 31.2 TPS | TTFT: 0.7ms | 416ms total | 13 tokens
> **Customer**: Hi there! Can you help me?
> **Kiosk**: Welcome to our store. How can I assist you today?

**store_hours** — 33.5 TPS | TTFT: 0.9ms | 1,133ms total | 38 tokens
> **Customer**: What are your store hours today?
> **Kiosk**: Our store hours today are from 9:00 AM to 9:00 PM. We're open for the rest of the evening, so you have plenty of time to shop.

**product_lookup** — 33.8 TPS | TTFT: 0.7ms | 1,717ms total | 58 tokens
> **Customer**: I'm looking for wireless earbuds under $50. What do you have in stock?
> **Kiosk**: We have a few options in our wireless earbuds section. Our current selection includes the "Eclipse" earbuds for $39.99 and the "Aurora" earbuds for $49.99. Would you like me to show you where they are?

**return_policy** — 33.4 TPS | TTFT: 1.2ms | 1,617ms total | 54 tokens
> **Customer**: I bought a laptop here 3 weeks ago and it's been crashing. I don't have the receipt but I paid with my credit card. Can I return it or get it fixed? What are my options?
> **Kiosk**: I'd be happy to help you with that. Since you paid with a credit card, you can still return or exchange the laptop within 30 days of purchase, but I'll need to verify the purchase with our system. Let me check on that for you.

**loyalty_program** — 34.3 TPS | TTFT: 1.4ms | 2,215ms total | 76 tokens
> **Customer**: I want to sign up for your loyalty program. How does it work? What are the benefits, and do I get a discount on my purchase today?
> **Kiosk**: Our loyalty program is called "Rewards Plus." By signing up, you'll earn points for every dollar you spend, which can be redeemed for discounts, free items, and exclusive offers. As a new member, you'll receive 10% off your purchase today, and you'll also get a $5 bonus for signing up. Would you like to sign up now?

**multi_turn_directions** — 34.2 TPS | TTFT: 0.7ms | 1,549ms total | 53 tokens
> **Customer** (turn 3): Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?
> **Kiosk**: We have a variety of waterproof cases for the iPhone 15. You can find them in our Electronics department, aisle 3, just down that way. We also have a selection of screen protectors in the same aisle, right next to the phone cases.

**multi_turn_troubleshoot** — 33.8 TPS | TTFT: 0.8ms | 1,450ms total | 49 tokens
> **Customer** (turn 3): I tried that, it still says 'card read error'. I've used this card here before with no issues.
> **Kiosk**: I'd be happy to help you with that. Can you please try swiping the card again, and if the issue persists, I can try to assist you with a manual transaction or check if there's a problem with the card reader.

#### 4.3.2 INT4, Sampling (temp=0.7) — Fastest Runs

**greeting** — 24.3 TPS | TTFT: 0.5ms | 1,275ms total | 31 tokens
> **Customer**: Hi there! Can you help me?
> **Kiosk**: Welcome to our store! What seems to be the reason for your visit today? Are you looking for a specific product or need assistance with something else?

**store_hours** — 24.4 TPS | TTFT: 0.6ms | 1,515ms total | 37 tokens
> **Customer**: What are your store hours today?
> **Kiosk**: Our store hours today are from 9:00 AM to 9:00 PM. We're open for the rest of the day, so feel free to shop with us!

**product_lookup** — 24.4 TPS | TTFT: 1.4ms | 2,415ms total | 59 tokens
> **Customer**: I'm looking for wireless earbuds under $50. What do you have in stock?
> **Kiosk**: We have several options available in our electronics department. Our current selection includes the X-Tech Wireless Earbuds ($29.99), the SonicBuds ($39.99), and the Aurora Earbuds ($49.99). Would you like me to show you where they are?

**return_policy** — 24.6 TPS | TTFT: 1.3ms | 2,972ms total | 73 tokens
> **Customer**: I bought a laptop here 3 weeks ago and it's been crashing. I don't have the receipt but I paid with my credit card. Can I return it or get it fixed? What are my options?
> **Kiosk**: I'd be happy to help you with that. Since you paid with your credit card, you can definitely return or exchange the laptop as long as it's within our return policy window of 30 days. We can also try to assist you with a repair, but to proceed with either option, we'll need to verify your purchase using your credit card statement.

**loyalty_program** — 24.3 TPS | TTFT: 1.3ms | 2,876ms total | 70 tokens
> **Customer**: I want to sign up for your loyalty program. How does it work? What are the benefits, and do I get a discount on my purchase today?
> **Kiosk**: Our loyalty program is called Rewards+. By signing up, you'll earn points on every purchase, get exclusive discounts, and receive early access to sales events. As a new member, you'll receive 10% off your purchase today, plus 100 bonus points to get you started. Would you like to sign up now at the kiosk?

**multi_turn_directions** — 24.3 TPS | TTFT: 0.5ms | 1,978ms total | 48 tokens
> **Customer** (turn 3): Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?
> **Kiosk**: We have a wide selection of waterproof cases for iPhone 15. Our waterproof cases are located on aisle 3, near the phone accessories section. We also have screen protectors on aisle 2, next to the phone display cases.

**multi_turn_troubleshoot** — 24.3 TPS | TTFT: 0.6ms | 2,260ms total | 55 tokens
> **Customer** (turn 3): I tried that, it still says 'card read error'. I've used this card here before with no issues.
> **Kiosk**: I'm happy to help you with that. Let me try scanning your card again for you. If the issue persists, we can also try resetting the card reader or checking if there's an issue with the card itself. Would you like me to try a different scanner?

#### 4.3.3 INT8, Greedy (temp=0.0) — Fastest Runs

**greeting** — 26.3 TPS | TTFT: 0.8ms | 533ms total | 14 tokens
> **Customer**: Hi there! Can you help me?
> **Kiosk**: Welcome to our store. What can I assist you with today?

**store_hours** — 27.4 TPS | TTFT: 1.1ms | 1,424ms total | 39 tokens
> **Customer**: What are your store hours today?
> **Kiosk**: Our store hours today are 9:00 AM to 9:00 PM. We're open for the rest of the day, so you have plenty of time to shop with us.

**product_lookup** — 27.5 TPS | TTFT: 1.2ms | 1,997ms total | 55 tokens
> **Customer**: I'm looking for wireless earbuds under $50. What do you have in stock?
> **Kiosk**: We have a few options available in our electronics section. Our current selection includes the X-Tech Wireless Earbuds for $29.99 and the SonicBuds Wireless Earbuds for $39.99. Would you like me to show you where they are?

**return_policy** — 27.5 TPS | TTFT: 1.2ms | 2,436ms total | 67 tokens
> **Customer**: I bought a laptop here 3 weeks ago and it's been crashing. I don't have the receipt but I paid with my credit card. Can I return it or get it fixed? What are my options?
> **Kiosk**: I'd be happy to help you with that. Since you paid with a credit card, you can still return or exchange the laptop within 30 days of purchase. However, without the receipt, we'll need to verify the purchase through your credit card statement. Can I see your credit card and the statement to confirm the purchase?

**loyalty_program** — 27.5 TPS | TTFT: 0.4ms | 2,768ms total | 76 tokens
> **Customer**: I want to sign up for your loyalty program. How does it work? What are the benefits, and do I get a discount on my purchase today?
> **Kiosk**: Our loyalty program is called "Rewards Plus." By signing up, you'll earn points for every dollar you spend, which can be redeemed for discounts, exclusive offers, and free items. As a new member, you'll receive 10% off your purchase today, and you'll also get a $5 bonus for signing up. Would you like to sign up now?

**multi_turn_directions** — 27.1 TPS | TTFT: 1.5ms | 1,952ms total | 53 tokens
> **Customer** (turn 3): Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?
> **Kiosk**: We have a variety of waterproof cases for the iPhone 15. You can find them in the Electronics department, aisle 3, just down that way. We also have a selection of screen protectors in the same aisle, right next to the phone cases.

**multi_turn_troubleshoot** — 27.2 TPS | TTFT: 0.7ms | 1,878ms total | 51 tokens
> **Customer** (turn 3): I tried that, it still says 'card read error'. I've used this card here before with no issues.
> **Kiosk**: I'd be happy to help you with that. Can you please try swiping the card again, and if the issue persists, I can try to assist you with an alternative payment method or contact our customer service team to look into the issue further.

#### 4.3.4 INT8, Sampling (temp=0.7) — Fastest Runs

**greeting** — 17.4 TPS | TTFT: 0.8ms | 747ms total | 13 tokens
> **Customer**: Hi there! Can you help me?
> **Kiosk**: Welcome to our store. How can I assist you today?

**store_hours** — 19.1 TPS | TTFT: 1.1ms | 2,040ms total | 39 tokens
> **Customer**: What are your store hours today?
> **Kiosk**: Our store hours today are 9:00 AM to 9:00 PM. We're open for the rest of the day, so you have plenty of time to shop with us.

**product_lookup** — 18.3 TPS | TTFT: 0.8ms | 2,899ms total | 53 tokens
> **Customer**: I'm looking for wireless earbuds under $50. What do you have in stock?
> **Kiosk**: We have several options available in our electronics department. Our current selection includes the X500 wireless earbuds for $39.99 and the ZONE wireless earbuds for $49.99. Would you like me to show you where they are located?

**return_policy** — 18.9 TPS | TTFT: 0.7ms | 3,752ms total | 71 tokens
> **Customer**: I bought a laptop here 3 weeks ago and it's been crashing. I don't have the receipt but I paid with my credit card. Can I return it or get it fixed? What are my options?
> **Kiosk**: I'd be happy to help you with that. Since you paid with your credit card, you can still return or exchange your laptop within 30 days of purchase. I can look up the purchase on our system using your credit card information, and we can proceed with the return or exchange process. Would you like me to check on that for you?

**loyalty_program** — 18.9 TPS | TTFT: 0.4ms | 3,757ms total | 71 tokens
> **Customer**: I want to sign up for your loyalty program. How does it work? What are the benefits, and do I get a discount on my purchase today?
> **Kiosk**: Our loyalty program is called "Rewards Plus." Members earn points for every purchase, which can be redeemed for discounts, free items, and exclusive offers. As a new member, you'll get 10% off your first purchase today, and we'll also give you 50 bonus points for signing up. Would you like to sign up now?

**multi_turn_directions** — 18.8 TPS | TTFT: 0.7ms | 2,238ms total | 42 tokens
> **Customer** (turn 3): Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?
> **Kiosk**: We have several waterproof case options for the iPhone 15. You can find them on the electronics aisle, just down that way. Our screen protectors are located right next to the phone cases.

**multi_turn_troubleshoot** — 18.0 TPS | TTFT: 1.0ms | 2,493ms total | 45 tokens
> **Customer** (turn 3): I tried that, it still says 'card read error'. I've used this card here before with no issues.
> **Kiosk**: I'd be happy to help you with that. Let me try to re-swipe your card for you or we can also try an alternate payment method if you'd like. Can I see your card again, please?

#### 4.3.5 FP16, Greedy (temp=0.0) — Fastest Runs

**greeting** — 15.5 TPS | TTFT: 1.5ms | 841ms total | 13 tokens
> **Customer**: Hi there! Can you help me?
> **Kiosk**: Welcome to our store. How can I assist you today?

**store_hours** — 15.9 TPS | TTFT: 1.1ms | 2,453ms total | 39 tokens
> **Customer**: What are your store hours today?
> **Kiosk**: Our store hours today are 9:00 AM to 9:00 PM. We're open for the rest of the day, so you have plenty of time to shop with us.

**product_lookup** — 16.0 TPS | TTFT: 1.5ms | 3,191ms total | 51 tokens
> **Customer**: I'm looking for wireless earbuds under $50. What do you have in stock?
> **Kiosk**: We have a few options available in our electronics section. Our current selection includes the X-Tech Wireless Earbuds for $29.99 and the SonicBuds for $39.99. Would you like me to show you where they are?

**return_policy** — 15.8 TPS | TTFT: 1.2ms | 4,543ms total | 72 tokens
> **Customer**: I bought a laptop here 3 weeks ago and it's been crashing. I don't have the receipt but I paid with my credit card. Can I return it or get it fixed? What are my options?
> **Kiosk**: I'd be happy to help you with that. Since you paid with a credit card, you can still return or exchange the laptop within 30 days of purchase, but I'll need to verify the purchase with our system. Can you please provide me with your credit card information or the card number associated with the purchase so I can look up the transaction?

**loyalty_program** — 15.7 TPS | TTFT: 1.6ms | 4,897ms total | 77 tokens
> **Customer**: I want to sign up for your loyalty program. How does it work? What are the benefits, and do I get a discount on my purchase today?
> **Kiosk**: Our loyalty program is called "Rewards Plus." By signing up, you'll earn points for every dollar you spend, which can be redeemed for discounts, free items, and exclusive offers. As a new member, you'll receive 10% off your first purchase today, and you'll also get a $5 bonus for signing up. Would you like to sign up now?

**multi_turn_directions** — 15.7 TPS | TTFT: 1.8ms | 3,244ms total | 51 tokens
> **Customer** (turn 3): Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?
> **Kiosk**: We have a variety of waterproof cases for the iPhone 15. You can find them in the Electronics department, just down that aisle to your left. We also have a selection of screen protectors on the same aisle, near the phone accessories section.

**multi_turn_troubleshoot** — 15.7 TPS | TTFT: 1.8ms | 3,242ms total | 51 tokens
> **Customer** (turn 3): I tried that, it still says 'card read error'. I've used this card here before with no issues.
> **Kiosk**: I'd be happy to help you with that. Can you please try swiping the card again, and if the issue persists, I can try to assist you with an alternative payment method or contact our customer service team to look into the issue further.

#### 4.3.6 FP16, Sampling (temp=0.7) — Fastest Runs

**greeting** — 11.9 TPS | TTFT: 1.7ms | 1,175ms total | 14 tokens
> **Customer**: Hi there! Can you help me?
> **Kiosk**: Welcome to our store. What can I assist you with today?

**store_hours** — 11.2 TPS | TTFT: 1.8ms | 4,094ms total | 46 tokens
> **Customer**: What are your store hours today?
> **Kiosk**: Our store hours today are 9:00 AM to 9:00 PM. We're open regular hours, with no special closures or events scheduled. Would you like to know the hours for any specific day or location?

**product_lookup** — 12.0 TPS | TTFT: 1.9ms | 6,185ms total | 74 tokens
> **Customer**: I'm looking for wireless earbuds under $50. What do you have in stock?
> **Kiosk**: We have a variety of wireless earbuds available within your budget. Our current selection includes the "Eclipse" earbuds for $39.99, the "Vibe" earbuds for $44.99, and the "Pulse" earbuds for $49.99. Would you like me to show you where they're located?

**return_policy** — 11.0 TPS | TTFT: 1.9ms | 6,821ms total | 75 tokens
> **Customer**: I bought a laptop here 3 weeks ago and it's been crashing. I don't have the receipt but I paid with my credit card. Can I return it or get it fixed? What are my options?
> **Kiosk**: I'd be happy to help you with that. Since you paid with your credit card, you can still return or exchange the laptop within 30 days of the purchase date, and we can look up the purchase on your credit card information. I can check the warranty status and see what options are available for you. Would you like me to check on that for you?

**loyalty_program** — 11.5 TPS | TTFT: 0.6ms | 5,489ms total | 63 tokens
> **Customer**: I want to sign up for your loyalty program. How does it work? What are the benefits, and do I get a discount on my purchase today?
> **Kiosk**: Our loyalty program is called Rewards Plus. Members earn points for every dollar spent, and when you reach 500 points, you get a $5 reward. As a new member, you'll receive 10% off your purchase today, and we'll also give you 100 bonus points to get you started.

**multi_turn_directions** — 11.5 TPS | TTFT: 2.2ms | 4,168ms total | 48 tokens
> **Customer** (turn 3): Yes, I need a waterproof case for an iPhone 15. Also, where are your screen protectors?
> **Kiosk**: We have a variety of waterproof cases for the iPhone 15. You can find them in our Electronics department, aisle 3. Our screen protectors are also located there, and I can give you a map if you'd like.

**multi_turn_troubleshoot** — 11.3 TPS | TTFT: 2.1ms | 3,722ms total | 42 tokens
> **Customer** (turn 3): I tried that, it still says 'card read error'. I've used this card here before with no issues.
> **Kiosk**: I'd be happy to help you with that. Can you please try swiping the card again, and if the issue persists, I can check the card reader or assist you with an alternative payment method.

---

## 5. Results — Intel Lunar Lake iGPU (LNL-GROVE)

**Machine**: Intel Core Ultra 7 258V, 31 GB LPDDR5X-8533 (on-package), Ubuntu 24.04
**OpenVINO**: 2025.4.1 | **GPU**: Intel Arc 140V (Xe2-LPG, 8 Xe2 cores, 64 XMX engines) | **NPU**: Lunar Lake NPU4

### 5.1 INT4 Performance — CPU vs GPU vs NPU

#### CPU (INT4)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 4.6 | 4.6 | 4.6 | 0.5 ms | 2,838 ms | 10 |
| store_hours | 0.0 | 7.9 | 7.9 | 7.9 | 0.6 ms | 4,810 ms | 10 |
| product_lookup | 0.0 | 8.7 | 8.7 | 8.7 | 0.4 ms | 7,261 ms | 10 |
| return_policy | 0.0 | 9.0 | 9.0 | 9.0 | 0.4 ms | 8,704 ms | 10 |
| loyalty_program | 0.0 | 9.0 | 9.0 | 9.1 | 0.6 ms | 8,408 ms | 10 |
| multi_turn_directions | 0.0 | 8.1 | 8.1 | 8.1 | 0.5 ms | 6,323 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 8.0 | 8.0 | 8.0 | 0.5 ms | 6,367 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 4.9 | 4.7 | 7.0 | 0.5 ms | 3,198 ms | 10 |
| store_hours | 0.7 | 7.6 | 7.7 | 8.3 | 0.4 ms | 5,085 ms | 10 |
| product_lookup | 0.7 | 8.2 | 8.2 | 8.6 | 0.4 ms | 7,382 ms | 10 |
| return_policy | 0.7 | 8.1 | 8.0 | 8.6 | 0.4 ms | 7,967 ms | 10 |
| loyalty_program | 0.7 | 8.3 | 8.3 | 8.8 | 0.5 ms | 8,017 ms | 10 |
| multi_turn_directions | 0.7 | 7.9 | 8.0 | 8.5 | 0.4 ms | 6,851 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 7.5 | 7.5 | 7.9 | 0.5 ms | 6,363 ms | 10 |

**Key findings — LNL CPU INT4:**
- Greedy: **8.0 TPS mean** (excluding greeting warmup outlier) — approaches the 10 TPS kiosk target on CPU alone
- Sampling: **7.5 TPS mean** — usable for non-latency-critical applications
- Lunar Lake's Lion Cove P-cores deliver notably strong CPU inference

#### GPU (INT4)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 14.3 | 14.3 | 14.6 | 0.6 ms | 1,330 ms | 10 |
| store_hours | 0.0 | 15.1 | 15.0 | 15.4 | 0.6 ms | 2,515 ms | 10 |
| product_lookup | 0.0 | 15.1 | 15.0 | 15.4 | 0.6 ms | 4,171 ms | 10 |
| return_policy | 0.0 | 14.2 | 14.2 | 14.3 | 0.7 ms | 3,793 ms | 10 |
| loyalty_program | 0.0 | 15.0 | 15.0 | 15.0 | 0.8 ms | 5,079 ms | 10 |
| multi_turn_directions | 0.0 | 14.7 | 14.7 | 14.7 | 0.7 ms | 3,408 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 14.9 | 14.9 | 14.9 | 0.5 ms | 2,551 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 13.4 | 13.5 | 13.9 | 0.9 ms | 1,212 ms | 10 |
| store_hours | 0.7 | 14.7 | 14.7 | 14.9 | 0.4 ms | 2,878 ms | 10 |
| product_lookup | 0.7 | 14.3 | 14.3 | 14.5 | 0.5 ms | 4,131 ms | 10 |
| return_policy | 0.7 | 14.1 | 14.2 | 14.3 | 0.4 ms | 4,774 ms | 10 |
| loyalty_program | 0.7 | 14.3 | 14.3 | 14.5 | 0.5 ms | 4,713 ms | 10 |
| multi_turn_directions | 0.7 | 13.6 | 13.6 | 13.8 | 0.7 ms | 3,989 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 13.6 | 13.6 | 14.0 | 0.7 ms | 3,530 ms | 10 |

**Key findings — LNL GPU INT4 (optimum-intel baseline):**
- Greedy: **14.9 TPS mean** — comfortably above the kiosk target at 17W
- Sampling: **14.0 TPS mean** — only ~6% throughput penalty vs greedy (much less than Arc A770M's ~26%)
- TTFT: sub-1ms reported — **this is a measurement artifact** of the optimum-intel Python streamer, not actual TTFT (see GenAI results below)
- GPU provides **1.86x speedup** over CPU on Lunar Lake — significant value from Xe2 XMX engines
- Remarkably tight variance (P5-P95 spread < 1 TPS)

> **Update — GenAI backend rerun**: Lunar Lake was re-benchmarked with the OpenVINO GenAI C++ backend, which provides accurate TTFT measurement. With **Qwen 2.5-7B INT4 AWQ + GenAI**, Lunar Lake achieves **22.8 TPS, 60ms TTFT** — a dramatic improvement over the 14.9 TPS optimum-intel baseline. The sub-1ms TTFT values reported above are artifacts of the optimum-intel streamer; the real TTFT is 60ms, which is still well under the 100ms perceptual threshold. At 22.8 TPS, Lunar Lake matches Panther Lake DDR5-7200 performance, confirming that on-package LPDDR5X-8533 bandwidth effectively compensates for Lunar Lake's fewer Xe cores (8 vs 12) and XMX engines (64 vs 96).

#### NPU (INT4) — Failed

The NPU failed to compile the model on both Lunar Lake and Meteor Lake:

```
RuntimeError: to_shape was called on a dynamic shape
```

The Intel NPU compiler requires **static tensor shapes** at compilation time. LLM autoregressive generation uses dynamic sequence lengths — each token extends the input sequence by one — which the NPU's static compilation pipeline cannot handle. This is a fundamental architectural limitation of current Intel NPUs (NPU3/NPU4) for LLM inference, not a driver or software bug.

**Workarounds** (not tested): Re-exporting the model with fixed `max_sequence_length` static shapes may allow NPU compilation, but would sacrifice the flexibility needed for variable-length conversations. Research projects like Agent.xpu demonstrate NPU+GPU heterogeneous inference is possible but not yet productized in OpenVINO.

### 5.2 INT8 Performance — GPU

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 9.7 | 10.2 | 10.3 | 0.6 ms | 1,356 ms | 10 |
| store_hours | 0.0 | 10.2 | 10.3 | 10.4 | 0.9 ms | 3,810 ms | 10 |
| product_lookup | 0.0 | 10.2 | 10.2 | 10.4 | 0.7 ms | 4,997 ms | 10 |
| return_policy | 0.0 | 10.4 | 10.4 | 10.4 | 0.6 ms | 6,437 ms | 10 |
| loyalty_program | 0.0 | 10.4 | 10.4 | 10.6 | 0.6 ms | 6,906 ms | 10 |
| multi_turn_directions | 0.0 | 10.1 | 10.1 | 10.4 | 0.7 ms | 4,948 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 10.1 | 10.1 | 10.4 | 0.7 ms | 4,943 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 9.9 | 9.9 | 10.1 | 0.5 ms | 1,624 ms | 10 |
| store_hours | 0.7 | 9.9 | 9.9 | 10.2 | 0.7 ms | 3,740 ms | 10 |
| product_lookup | 0.7 | 9.7 | 9.7 | 9.7 | 0.7 ms | 6,882 ms | 10 |
| return_policy | 0.7 | 9.9 | 9.9 | 9.9 | 0.8 ms | 6,289 ms | 10 |
| loyalty_program | 0.7 | 9.7 | 9.7 | 10.0 | 0.6 ms | 7,406 ms | 10 |
| multi_turn_directions | 0.7 | 9.6 | 9.6 | 9.8 | 0.8 ms | 5,613 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 9.6 | 9.7 | 9.9 | 0.5 ms | 4,180 ms | 10 |

**Key findings — LNL GPU INT8:**
- Greedy: **10.2 TPS mean** — right at the 10 TPS kiosk target
- Sampling: **9.7 TPS mean** — just below target but still usable
- INT8→INT4 speedup: **1.46x** (14.9 vs 10.2) — quantization pays off on Xe2
- Very low variance (P5-P95 spread < 1 TPS)

### 5.3 FP16 Performance — GPU

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 6.2 | 6.3 | 6.3 | 0.7 ms | 2,101 ms | 10 |
| store_hours | 0.0 | 6.4 | 6.4 | 6.4 | 0.7 ms | 6,121 ms | 10 |
| product_lookup | 0.0 | 6.4 | 6.4 | 6.5 | 0.5 ms | 7,970 ms | 10 |
| return_policy | 0.0 | 6.4 | 6.4 | 6.5 | 0.6 ms | 11,230 ms | 10 |
| loyalty_program | 0.0 | 6.4 | 6.4 | 6.4 | 0.8 ms | 12,069 ms | 10 |
| multi_turn_directions | 0.0 | 6.4 | 6.4 | 6.4 | 0.9 ms | 8,010 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 6.4 | 6.4 | 6.4 | 0.9 ms | 8,014 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 5.9 | 5.9 | 6.1 | 0.7 ms | 2,574 ms | 10 |
| store_hours | 0.7 | 6.1 | 6.1 | 6.2 | 0.7 ms | 6,595 ms | 10 |
| product_lookup | 0.7 | 6.2 | 6.2 | 6.3 | 0.6 ms | 9,898 ms | 10 |
| return_policy | 0.7 | 6.2 | 6.2 | 6.3 | 0.6 ms | 11,646 ms | 10 |
| loyalty_program | 0.7 | 6.2 | 6.2 | 6.3 | 0.7 ms | 11,934 ms | 10 |
| multi_turn_directions | 0.7 | 6.2 | 6.2 | 6.3 | 0.6 ms | 8,790 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 6.2 | 6.2 | 6.3 | 0.7 ms | 7,828 ms | 10 |

**Key findings — LNL GPU FP16:**
- Greedy: **6.4 TPS mean** — well below the kiosk target; the 15GB model saturates shared memory bandwidth
- Sampling: **6.1 TPS mean** — only ~4% penalty, consistent with other iGPU configurations
- FP16 on Lunar Lake iGPU is **slower than INT4 on CPU** (6.4 vs 8.0 TPS) — demonstrating that memory bandwidth is the absolute bottleneck when weight size exceeds what the shared memory subsystem can feed

### 5.4 Precision Comparison (LNL-GROVE, GPU, Greedy)

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | vs Kiosk Target |
|-----------|-----------|-----------|------------------|-----------------|
| FP16 | 15 GB | 6.4 | 1.0x (baseline) | 0.64x |
| INT8 | 7.5 GB | 10.2 | **1.59x** | **1.02x** |
| INT4 | 5.2 GB | 14.9 | **2.33x** | **1.49x** |

Quantization delivers a **larger speedup on shared-memory iGPUs** than on dedicated-VRAM dGPUs (2.33x vs 2.03x on Arc A770M). This is because the shared LPDDR5X memory subsystem is the tighter bottleneck — reducing model size has a proportionally larger impact when bandwidth is more constrained.

**With GenAI backend**: Qwen 2.5-7B INT4 AWQ + GenAI achieves **22.8 TPS / 60ms TTFT** on Lunar Lake — a +53% uplift over the Llama optimum-intel baseline, combining model architecture advantages and C++ pipeline efficiency.

---

## 6. Results — Intel Meteor Lake iGPU (MTL-NOYCE)

**Machine**: Intel Core Ultra 5 125H, 62 GB DDR5-5600, Ubuntu 24.04
**OpenVINO**: 2025.4.1 | **GPU**: Intel Arc (Xe-LPG, 8 Xe cores, **no XMX**) | **NPU**: Meteor Lake NPU3

> **Note**: MTL-NOYCE was upgraded from 15 GB to 62 GB DDR5-5600 during testing. INT4 and INT8 benchmarks were collected on the original 15 GB configuration. FP16 benchmarks required the RAM upgrade (the 15 GB FP16 model could not fit alongside OS/runtime in 15 GB). The DDR5-5600 provides ~89.6 GB/s theoretical bandwidth (lower than LPDDR5X), but the key architectural limitation — no XMX engines — remains the dominant factor in performance.

### 6.1 INT4 Performance — CPU vs GPU

#### CPU (INT4)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 3.3 | 3.3 | 3.4 | 0.5 ms | 3,919 ms | 10 |
| store_hours | 0.0 | 5.4 | 5.4 | 5.5 | 0.5 ms | 7,019 ms | 10 |
| product_lookup | 0.0 | 5.8 | 5.8 | 5.8 | 1.6 ms | 10,966 ms | 10 |
| return_policy | 0.0 | 5.8 | 5.8 | 5.9 | 0.8 ms | 13,513 ms | 10 |
| loyalty_program | 0.0 | 5.9 | 5.9 | 6.0 | 0.8 ms | 12,839 ms | 10 |
| multi_turn_directions | 0.0 | 5.3 | 5.3 | 5.4 | 0.6 ms | 9,587 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 5.3 | 5.3 | 5.3 | 0.6 ms | 9,661 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 3.3 | 3.2 | 3.8 | 0.6 ms | 4,368 ms | 10 |
| store_hours | 0.7 | 4.9 | 5.0 | 5.2 | 0.8 ms | 7,116 ms | 10 |
| product_lookup | 0.7 | 5.4 | 5.4 | 5.9 | 0.6 ms | 11,790 ms | 10 |
| return_policy | 0.7 | 5.5 | 5.5 | 5.7 | 0.6 ms | 13,152 ms | 10 |
| loyalty_program | 0.7 | 5.6 | 5.6 | 5.9 | 0.8 ms | 12,872 ms | 10 |
| multi_turn_directions | 0.7 | 5.3 | 5.3 | 5.5 | 0.4 ms | 9,856 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 5.0 | 5.1 | 5.2 | 0.6 ms | 9,375 ms | 10 |

#### GPU (INT4)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 5.5 | 5.6 | 5.6 | 0.6 ms | 2,353 ms | 10 |
| store_hours | 0.0 | 6.7 | 6.7 | 6.8 | 0.7 ms | 5,677 ms | 10 |
| product_lookup | 0.0 | 6.9 | 6.9 | 7.0 | 0.6 ms | 9,160 ms | 10 |
| return_policy | 0.0 | 6.9 | 6.9 | 7.0 | 0.5 ms | 9,977 ms | 10 |
| loyalty_program | 0.0 | 6.9 | 7.0 | 7.0 | 0.7 ms | 10,950 ms | 10 |
| multi_turn_directions | 0.0 | 6.7 | 6.7 | 6.8 | 0.7 ms | 7,445 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 6.7 | 6.7 | 6.7 | 0.7 ms | 7,038 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 5.4 | 5.4 | 6.0 | 0.7 ms | 2,796 ms | 10 |
| store_hours | 0.7 | 6.3 | 6.3 | 6.6 | 0.5 ms | 5,798 ms | 10 |
| product_lookup | 0.7 | 6.5 | 6.5 | 6.7 | 0.5 ms | 9,752 ms | 10 |
| return_policy | 0.7 | 6.6 | 6.6 | 6.8 | 0.4 ms | 10,577 ms | 10 |
| loyalty_program | 0.7 | 6.5 | 6.5 | 6.6 | 0.6 ms | 10,915 ms | 10 |
| multi_turn_directions | 0.7 | 6.4 | 6.4 | 6.6 | 0.9 ms | 8,474 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 6.3 | 6.3 | 6.5 | 0.6 ms | 7,267 ms | 10 |

**Key findings — MTL INT4:**
- Greedy GPU: **6.7 TPS mean** — below the kiosk target. Meteor Lake's GPU without XMX is compute-constrained
- Greedy CPU: **5.4 TPS mean** — GPU provides only **1.24x speedup** over CPU (vs Lunar Lake's 1.86x)
- The minimal GPU/CPU gap confirms the lack of XMX engines as the bottleneck — the GPU cannot accelerate INT4 matrix operations significantly beyond what the CPU vector units achieve
- Sampling penalty is only ~5-6%, similar to Lunar Lake — less than the ~26-33% on Arc A770M

#### NPU (INT4) — Failed

Same dynamic shape error as Lunar Lake. See Section 5.1 NPU analysis.

### 6.2 INT8 Performance — GPU

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 3.3 | 3.3 | 3.4 | 0.6 ms | 3,900 ms | 10 |
| store_hours | 0.0 | 5.1 | 5.1 | 5.2 | 0.8 ms | 7,633 ms | 10 |
| product_lookup | 0.0 | 5.0 | 5.0 | 5.0 | 0.7 ms | 10,293 ms | 10 |
| return_policy | 0.0 | 5.2 | 5.2 | 5.2 | 0.9 ms | 12,948 ms | 10 |
| loyalty_program | 0.0 | 5.2 | 5.1 | 5.2 | 0.8 ms | 14,771 ms | 10 |
| multi_turn_directions | 0.0 | 5.3 | 5.3 | 5.4 | 0.8 ms | 9,603 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 5.3 | 5.3 | 5.3 | 0.7 ms | 9,530 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 3.3 | 3.3 | 3.7 | 0.8 ms | 4,156 ms | 10 |
| store_hours | 0.7 | 4.8 | 4.8 | 5.1 | 1.0 ms | 8,030 ms | 10 |
| product_lookup | 0.7 | 4.8 | 4.8 | 5.0 | 0.9 ms | 12,937 ms | 10 |
| return_policy | 0.7 | 5.0 | 5.0 | 5.1 | 0.7 ms | 13,924 ms | 10 |
| loyalty_program | 0.7 | 4.9 | 4.9 | 5.0 | 0.7 ms | 14,286 ms | 10 |
| multi_turn_directions | 0.7 | 5.1 | 5.1 | 5.2 | 0.7 ms | 10,225 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 5.0 | 5.0 | 5.1 | 0.7 ms | 8,899 ms | 10 |

**Key findings — MTL GPU INT8:**
- Greedy: **5.2 TPS mean** (excluding greeting) — below both the kiosk target and INT4 GPU performance
- Sampling: **4.9 TPS mean** — minimal penalty vs greedy, consistent with other iGPU patterns
- **INT8 GPU is slower than INT4 CPU** on Meteor Lake (5.2 vs 5.4 TPS) — a remarkable finding that demonstrates the GPU provides no advantage when: (a) the model is larger (7.5 vs 5.2 GB), (b) bandwidth is shared, and (c) no XMX engines accelerate the matrix ops

### 6.3 FP16 Performance — GPU

> *Collected after RAM upgrade to 62 GB DDR5-5600. The 15 GB FP16 model could not fit in the original 15 GB configuration.*

| Scenario | Temp | TPS Mean | TPS Median | TPS P5 | TPS P95 | TTFT Mean | Total Mean | Avg Tokens |
|----------|------|----------|------------|--------|---------|-----------|------------|------------|
| greeting | 0.0 | 2.1 | 2.1 | 2.1 | 2.2 | 0.7 ms | 6,145 ms | 13 |
| store_hours | 0.0 | 3.7 | 3.7 | 3.7 | 3.8 | 1.0 ms | 10,519 ms | 39 |
| product_lookup | 0.0 | 3.1 | 3.1 | 3.1 | 3.1 | 1.0 ms | 16,540 ms | 51 |
| return_policy | 0.0 | 3.7 | 3.7 | 3.6 | 3.7 | 0.9 ms | 19,644 ms | 72 |
| loyalty_program | 0.0 | 3.3 | 3.3 | 3.3 | 3.3 | 1.0 ms | 23,262 ms | 77 |
| multi_turn_directions | 0.0 | 3.8 | 3.8 | 3.7 | 3.8 | 1.0 ms | 13,521 ms | 51 |
| multi_turn_troubleshoot | 0.0 | 3.5 | 3.5 | 3.4 | 3.5 | 0.7 ms | 14,670 ms | 51 |
| | | | | | | | | |
| greeting | 0.7 | 2.2 | 2.3 | 1.6 | 2.5 | 1.0 ms | 6,770 ms | 15 |
| store_hours | 0.7 | 3.6 | 3.6 | 3.5 | 3.7 | 0.7 ms | 12,338 ms | 45 |
| product_lookup | 0.7 | 3.2 | 3.2 | 3.0 | 3.3 | 0.8 ms | 20,369 ms | 65 |
| return_policy | 0.7 | 3.4 | 3.4 | 3.3 | 3.6 | 0.7 ms | 20,678 ms | 71 |
| loyalty_program | 0.7 | 3.0 | 3.0 | 2.9 | 3.1 | 0.7 ms | 26,184 ms | 78 |
| multi_turn_directions | 0.7 | 3.4 | 3.4 | 3.4 | 3.5 | 1.1 ms | 15,031 ms | 51 |
| multi_turn_troubleshoot | 0.7 | 3.0 | 3.1 | 2.8 | 3.2 | 0.7 ms | 15,636 ms | 48 |

**Key findings — MTL GPU FP16:**
- Greedy: **3.3 TPS mean** — the slowest GPU configuration across all platforms, 2x below the next-slowest (MTL INT8 at 5.2 TPS)
- TTFT: <1ms — still sub-perceptible despite the massive model
- The 15 GB FP16 model on DDR5-5600 (~90 GB/s bandwidth) yields a theoretical ceiling of ~6 TPS; achieving 3.3 TPS represents ~55% bandwidth efficiency, actually the highest of any MTL configuration — suggesting the compute overhead (no XMX) impacts smaller quantized models more than raw bandwidth-bound FP16
- **Total latency is prohibitive**: Simple queries take 10-16 seconds, complex queries 20-26 seconds — unusable for interactive kiosk
- INT4 quantization provides **2.03x speedup** over FP16 on Meteor Lake, identical to the Arc A770M ratio despite completely different architectural bottlenecks
- Sampling penalty: ~6%, consistent with the bandwidth-bound regime of other iGPU configurations

### 6.4 Precision Comparison (MTL-NOYCE, GPU, Greedy)

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | vs Kiosk Target |
|-----------|-----------|-----------|------------------|-----------------|
| FP16 | 15 GB | 3.3 | 1.0x (baseline) | 0.33x |
| INT8 | 7.5 GB | 5.2 | **1.58x** | 0.52x |
| INT4 | 5.2 GB | 6.7 | **2.03x** | 0.67x |

Even with INT4, Meteor Lake's GPU cannot reach the 10 TPS kiosk target. The combination of limited bandwidth (~90 GB/s DDR5-5600) and no XMX matrix engines creates a hard ceiling for LLM inference on this platform. Notably, the INT4/FP16 speedup ratio (2.03x) matches the Arc A770M exactly, despite completely different architectural profiles — suggesting this ratio is primarily determined by the model size reduction (2.9x) and is relatively hardware-independent.

---

## 7. Results — Intel Panther Lake iGPU (PTL-FAIRCHILD)

**Machine**: Intel Core Ultra (Panther Lake-H), 25W, Ubuntu 24.04
**OpenVINO**: 2025.4.1 | **GPU**: Intel Arc (Xe3-LPG, 12 Xe3 cores, 96 XMX engines) | **NPU**: Panther Lake NPU5

> **Note**: PTL-FAIRCHILD was tested with two memory configurations to measure memory bandwidth sensitivity:
> - **Config A**: 2x8GB DDR5-7200 Micron SODIMM (16GB total, ~115 GB/s)
> - **Config B**: 2x32GB DDR5-5600 SODIMM (64GB total, ~90 GB/s)
> INT4 and INT8 results are available for both configs. FP16 (15 GB model) was only tested with Config B (64GB) as Config A (16GB) lacked sufficient headroom.

### 7.1 INT4 Performance — Config A (DDR5-7200) vs Config B (DDR5-5600)

#### GPU INT4 — Config A (DDR5-7200)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 16.2 | 16.3 | 16.3 | 0 ms | 1,177 ms | 10 |
| store_hours | 0.0 | 16.5 | 16.6 | 16.7 | 0 ms | 2,300 ms | 10 |
| product_lookup | 0.0 | 16.3 | 16.4 | 16.5 | 0 ms | 3,876 ms | 10 |
| return_policy | 0.0 | 16.4 | 16.4 | 16.4 | 0 ms | 3,294 ms | 10 |
| loyalty_program | 0.0 | 16.6 | 16.6 | 16.6 | 0 ms | 4,592 ms | 10 |
| multi_turn_directions | 0.0 | 16.4 | 16.4 | 16.5 | 0 ms | 3,050 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 16.2 | 16.2 | 16.3 | 0 ms | 2,344 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 15.0 | 14.9 | 15.7 | 0 ms | 855 ms | 10 |
| store_hours | 0.7 | 15.7 | 15.7 | 16.0 | 0 ms | 2,317 ms | 10 |
| product_lookup | 0.7 | 15.7 | 15.7 | 15.9 | 0 ms | 4,029 ms | 10 |
| return_policy | 0.7 | 15.4 | 15.4 | 15.7 | 0 ms | 4,614 ms | 10 |
| loyalty_program | 0.7 | 15.5 | 15.5 | 15.8 | 0 ms | 4,896 ms | 10 |
| multi_turn_directions | 0.7 | 15.5 | 15.5 | 15.8 | 0 ms | 3,396 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 15.5 | 15.6 | 15.7 | 0 ms | 2,755 ms | 10 |

#### GPU INT4 — Config B (DDR5-5600)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 13.3 | 13.3 | 13.3 | 0 ms | 977 ms | 10 |
| store_hours | 0.0 | 13.7 | 13.7 | 13.7 | 0 ms | 2,778 ms | 10 |
| product_lookup | 0.0 | 13.5 | 13.5 | 13.6 | 0 ms | 3,844 ms | 10 |
| return_policy | 0.0 | 13.6 | 13.6 | 13.6 | 0 ms | 4,712 ms | 10 |
| loyalty_program | 0.0 | 13.6 | 13.6 | 13.7 | 0 ms | 5,572 ms | 10 |
| multi_turn_directions | 0.0 | 13.5 | 13.5 | 13.5 | 0 ms | 3,696 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 13.4 | 13.4 | 13.4 | 0 ms | 2,834 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 12.7 | 12.8 | 13.0 | 0 ms | 970 ms | 10 |
| store_hours | 0.7 | 13.2 | 13.2 | 13.3 | 0 ms | 2,984 ms | 10 |
| product_lookup | 0.7 | 13.0 | 13.1 | 13.2 | 0 ms | 4,751 ms | 10 |
| return_policy | 0.7 | 12.8 | 12.8 | 13.0 | 0 ms | 5,427 ms | 10 |
| loyalty_program | 0.7 | 12.9 | 12.8 | 13.0 | 0 ms | 5,986 ms | 10 |
| multi_turn_directions | 0.7 | 13.0 | 13.0 | 13.2 | 0 ms | 4,334 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 13.0 | 13.0 | 13.1 | 0 ms | 3,555 ms | 10 |

**Key findings — PTL GPU INT4:**
- **Config A (DDR5-7200)**: Greedy **16.4 TPS mean** — the fastest iGPU configuration tested, exceeding Lunar Lake by 10%
- **Config B (DDR5-5600)**: Greedy **13.5 TPS mean** — an **18% drop** from Config A, demonstrating clear memory bandwidth sensitivity
- PTL with DDR5-7200 benefits from both more Xe cores (12 vs 8) and faster memory; with DDR5-5600, the bandwidth reduction negates the architectural advantages
- Sampling penalty: ~6% on both configs, consistent with other iGPU platforms
- GPU provides **~2.1x speedup** over CPU on PTL (16.4 vs ~7.9 TPS with DDR5-7200)

### 7.2 INT8 Performance — GPU

#### GPU INT8 — Config A (DDR5-7200)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 11.8 | 11.8 | 11.8 | 0 ms | 1,106 ms | 10 |
| store_hours | 0.0 | 11.9 | 11.9 | 12.0 | 0 ms | 3,273 ms | 10 |
| product_lookup | 0.0 | 11.9 | 11.9 | 11.9 | 0 ms | 4,296 ms | 10 |
| return_policy | 0.0 | 11.9 | 11.9 | 11.9 | 0 ms | 5,647 ms | 10 |
| loyalty_program | 0.0 | 12.0 | 12.0 | 12.0 | 0 ms | 6,016 ms | 10 |
| multi_turn_directions | 0.0 | 12.0 | 12.0 | 12.0 | 0 ms | 4,178 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 12.0 | 12.0 | 12.0 | 0 ms | 4,180 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 11.4 | 11.5 | 11.6 | 0 ms | 1,241 ms | 10 |
| store_hours | 0.7 | 11.6 | 11.7 | 11.7 | 0 ms | 3,475 ms | 10 |
| product_lookup | 0.7 | 11.7 | 11.7 | 11.7 | 0 ms | 5,043 ms | 10 |
| return_policy | 0.7 | 11.5 | 11.6 | 11.7 | 0 ms | 5,753 ms | 10 |
| loyalty_program | 0.7 | 11.5 | 11.7 | 11.7 | 0 ms | 6,616 ms | 10 |
| multi_turn_directions | 0.7 | 11.6 | 11.7 | 11.7 | 0 ms | 4,818 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 11.6 | 11.7 | 11.7 | 0 ms | 4,082 ms | 10 |

#### GPU INT8 — Config B (DDR5-5600)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 9.6 | 9.6 | 9.7 | 0 ms | 1,350 ms | 10 |
| store_hours | 0.0 | 9.8 | 9.8 | 9.8 | 0 ms | 4,001 ms | 10 |
| product_lookup | 0.0 | 9.7 | 9.7 | 9.8 | 0 ms | 5,236 ms | 10 |
| return_policy | 0.0 | 9.8 | 9.8 | 9.8 | 0 ms | 6,869 ms | 10 |
| loyalty_program | 0.0 | 9.8 | 9.8 | 9.8 | 0 ms | 7,378 ms | 10 |
| multi_turn_directions | 0.0 | 9.8 | 9.8 | 9.8 | 0 ms | 5,129 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 9.7 | 9.7 | 9.8 | 0 ms | 5,132 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 9.2 | 9.3 | 9.4 | 0 ms | 1,542 ms | 10 |
| store_hours | 0.7 | 9.4 | 9.3 | 9.5 | 0 ms | 4,568 ms | 10 |
| product_lookup | 0.7 | 9.5 | 9.5 | 9.5 | 0 ms | 6,414 ms | 10 |
| return_policy | 0.7 | 9.4 | 9.4 | 9.5 | 0 ms | 7,399 ms | 10 |
| loyalty_program | 0.7 | 9.4 | 9.4 | 9.4 | 0 ms | 7,755 ms | 10 |
| multi_turn_directions | 0.7 | 9.4 | 9.5 | 9.5 | 0 ms | 5,909 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 9.4 | 9.5 | 9.5 | 0 ms | 4,985 ms | 10 |

**Key findings — PTL GPU INT8:**
- **Config A (DDR5-7200)**: Greedy **11.9 TPS mean** — comfortably above the 10 TPS kiosk target, and faster than LNL (10.2 TPS)
- **Config B (DDR5-5600)**: Greedy **9.7 TPS mean** — an **18% drop**, falling just below the kiosk target
- INT4/INT8 speedup: 1.38x (Config A) and 1.39x (Config B) — consistent across memory configs

### 7.3 FP16 Performance — GPU (Config B only)

> *FP16 (15 GB model) was only tested with Config B (2x32GB DDR5-5600, 64GB total). Config A (2x8GB DDR5-7200, 16GB total) lacked sufficient memory headroom.*

#### GPU FP16 — Config B (DDR5-5600)

| Scenario | Temp | TPS Mean | TPS Median | TPS P95 | TTFT Mean | Total Mean | Runs |
|----------|------|----------|------------|---------|-----------|------------|------|
| greeting | 0.0 | 5.1 | 5.1 | 5.1 | 0 ms | 2,559 ms | 10 |
| store_hours | 0.0 | 5.1 | 5.1 | 5.1 | 0 ms | 7,618 ms | 10 |
| product_lookup | 0.0 | 5.1 | 5.1 | 5.1 | 0 ms | 9,960 ms | 10 |
| return_policy | 0.0 | 5.1 | 5.1 | 5.1 | 0 ms | 14,065 ms | 10 |
| loyalty_program | 0.0 | 5.1 | 5.1 | 5.1 | 0 ms | 15,035 ms | 10 |
| multi_turn_directions | 0.0 | 5.1 | 5.1 | 5.1 | 0 ms | 9,957 ms | 10 |
| multi_turn_troubleshoot | 0.0 | 5.1 | 5.1 | 5.1 | 0 ms | 9,959 ms | 10 |
| | | | | | | | |
| greeting | 0.7 | 5.0 | 5.0 | 5.0 | 0 ms | 3,091 ms | 10 |
| store_hours | 0.7 | 5.0 | 5.0 | 5.0 | 0 ms | 8,751 ms | 10 |
| product_lookup | 0.7 | 5.0 | 5.0 | 5.0 | 0 ms | 12,714 ms | 10 |
| return_policy | 0.7 | 5.0 | 5.0 | 5.0 | 0 ms | 13,967 ms | 10 |
| loyalty_program | 0.7 | 5.0 | 5.0 | 5.0 | 0 ms | 13,982 ms | 10 |
| multi_turn_directions | 0.7 | 5.0 | 5.0 | 5.0 | 0 ms | 12,786 ms | 10 |
| multi_turn_troubleshoot | 0.7 | 5.0 | 5.0 | 5.0 | 0 ms | 9,599 ms | 10 |

**Key findings — PTL GPU FP16:**
- Greedy: **5.1 TPS mean** — below the kiosk target but faster than MTL (3.3 TPS) and slower than LNL (6.4 TPS)
- The DDR5-5600 bandwidth bottleneck is severe — PTL's 12 Xe3 cores and 96 XMX engines are heavily underutilized
- Extremely low variance (all scenarios near-identical TPS) — the pipeline is completely bandwidth-saturated

### 7.4 Memory Bandwidth Sensitivity

This is the key new insight from PTL testing. The same GPU architecture with different DRAM speeds:

| Precision | DDR5-7200 TPS | DDR5-5600 TPS | Delta | Bandwidth Ratio |
|-----------|:------------:|:------------:|:-----:|:---------------:|
| INT4 | 16.4 | 13.5 | **-18%** | 7200/5600 = 1.29x |
| INT8 | 11.9 | 9.7 | **-18%** | 7200/5600 = 1.29x |
| FP16 | N/A (16GB) | 5.1 | — | — |

The 22% reduction in memory bandwidth (DDR5-7200 → DDR5-5600) translates to a consistent **18% throughput reduction** across both INT4 and INT8. This near-linear relationship (18% TPS drop / 22% bandwidth drop = 0.82 elasticity) confirms that iGPU LLM inference is overwhelmingly memory-bandwidth bound. The slightly sub-linear elasticity suggests a small portion of execution time is compute-bound or involves fixed overhead.

**DDR5-7200 GenAI validation**: A follow-up experiment with the GenAI C++ backend on DDR5-7200 further validates the bandwidth elasticity model:

| Configuration | DDR5-7200 TPS | DDR5-5600 TPS | Predicted DDR5-7200 TPS* | Prediction Error |
|--------------|:------------:|:------------:|:-----------------------:|:---------------:|
| Qwen 2.5-7B INT4 AWQ + GenAI | **22.8** | 18.7 | 21-22 | Model slightly underestimates |
| Llama 3.1-8B INT4 GPTQ + GenAI | **21.5** | 17.6 | 20-21 | Model slightly underestimates |

*Predicted using 0.82 elasticity coefficient: DDR5-5600 TPS × (7200/5600)^0.82*

The measured DDR5-7200 results (22.8 and 21.5 TPS) slightly exceed the elasticity model's predictions, suggesting the 0.82 coefficient is conservative — the actual bandwidth sensitivity may be slightly higher at the top end of the DDR5 speed range. Regardless, this controlled experiment confirms that **DDR5-7200 provides ~22% higher throughput** than DDR5-5600 with identical silicon, making memory speed selection a first-order system design variable for edge AI.

### 7.5 Precision Comparison (PTL-FAIRCHILD, GPU, Greedy)

**Config A (DDR5-7200):**

| Precision | Model Size | Greedy TPS | Speedup vs FP16* | vs Kiosk Target |
|-----------|-----------|-----------|------------------|-----------------|
| INT8 | 7.5 GB | 11.9 | — | **1.19x** |
| INT4 | 5.2 GB | 16.4 | — | **1.64x** |

*FP16 not tested on Config A (insufficient RAM)

**Config B (DDR5-5600):**

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | vs Kiosk Target |
|-----------|-----------|-----------|------------------|-----------------|
| FP16 | 15 GB | 5.1 | 1.0x (baseline) | 0.51x |
| INT8 | 7.5 GB | 9.7 | **1.90x** | 0.97x |
| INT4 | 5.2 GB | 13.5 | **2.65x** | **1.35x** |

PTL with DDR5-5600 shows the **highest INT4/FP16 speedup ratio** of any platform tested (2.65x), exceeding Lunar Lake's 2.33x. This makes sense: the tighter bandwidth constraint (DDR5-5600 < LPDDR5X-8533) amplifies the benefit of model size reduction. More Xe cores (12 vs 8) with XMX engines mean the GPU has excess compute capacity — further shifting the bottleneck to bandwidth and rewarding quantization.

---

## 8. Results — Qwen 2.5 7B on Panther Lake (PTL-FAIRCHILD)

After establishing baseline Llama 3.1-8B results across all platforms, we benchmarked **Qwen2.5-7B-Instruct** on Panther Lake (DDR5-5600) to evaluate whether a smaller model with different attention architecture could improve throughput on bandwidth-constrained hardware. All Qwen benchmarks used the GenAI C++ backend (established as optimal in Section 9) and the optimum-intel baseline backend for comparison.

### 8.1 GenAI Backend Results (Best Configuration)

| Precision | Temp | TPS Mean | TTFT (ms) | Notes |
|-----------|------|----------|-----------|-------|
| INT4 | 0.0 | **18.7** | **65** | Fastest iGPU config tested |
| INT4 | 0.7 | 18.4 | 66 | |
| INT8 | 0.0 | 10.9 | 103 | |
| INT8 | 0.7 | 10.8 | 104 | |
| FP16 | 0.0 | 5.5 | 193 | |
| FP16 | 0.7 | 5.5 | 194 | |

### 8.2 Optimum-Intel Backend Results (Baseline)

| Precision | Temp | TPS Mean | Notes |
|-----------|------|----------|-------|
| INT4 | 0.0 | 17.2 | +27.4% over Llama baseline |
| INT4 | 0.7 | 15.6 | |
| INT8 | 0.0 | 10.4 | |
| INT8 | 0.7 | 9.8 | |
| FP16 | 0.0 | 5.4 | |
| FP16 | 0.7 | 5.3 | |

### 8.3 Side-by-Side: Qwen vs Llama on Panther Lake (GenAI Backend, Greedy)

| Precision | Llama 3.1-8B TPS | Qwen 2.5-7B TPS | Qwen Advantage | Llama TTFT | Qwen TTFT |
|-----------|-----------------|-----------------|----------------|------------|-----------|
| INT4 | 14.7 | **18.7** | **+27.2%** | 90ms | **65ms** |
| INT8 | 10.1 | **10.9** | **+7.9%** | 119ms | **103ms** |
| FP16 | 5.1 | **5.5** | **+7.8%** | 215ms | **193ms** |

**Key findings:**
- Qwen2.5-7B is **8-27% faster** than Llama 3.1-8B across all precisions on Panther Lake
- The advantage is largest at INT4 (+27%) because Qwen's smaller model size and GQA reduce both weight bandwidth and KV cache bandwidth — effects that compound when weight bandwidth is no longer the sole bottleneck
- At FP16 and INT8, the advantage narrows to ~8% — proportional to the parameter count difference (7B vs 8B), confirming the memory-bandwidth-bound regime
- GenAI uplift is model-agnostic: ~8-9% at INT4 for both models

---

## 9. Optimization Experiments on Panther Lake

Building on baseline results, we tested three independent optimization strategies on PTL-FAIRCHILD (DDR5-5600). Full experimental details are in the companion paper (`optimization-whitepaper.md`).

### 9.1 Experiment 1: `ov_config` Runtime Flags — No Benefit

We tested OpenVINO runtime configuration flags (`KV_CACHE_PRECISION=u8`, `DYNAMIC_QUANTIZATION_GROUP_SIZE=64`, `PERFORMANCE_HINT=LATENCY`) both combined and individually.

| Configuration | INT4 TPS | Δ vs Baseline | Recommendation |
|--------------|----------|---------------|----------------|
| Baseline (no flags) | 13.5 | — | Current default |
| All 3 flags combined | 13.4 | **-0.7%** | **Do not use** |
| KV_CACHE_PRECISION=u8 only | 12.5 | **-7.4%** | **Do not use** |

**Conclusion**: Runtime `ov_config` flags do not improve performance on Xe3-LPG. KV cache u8 quantization is actively harmful for INT4 — "double quantization" introduces more overhead than the memory savings it provides. The iGPU's memory subsystem is already operating efficiently under default settings.

### 9.2 Experiment 2: GPTQ INT4 with Scale Estimation — Major Win

Re-exporting the Llama INT4 model using GPTQ with scale estimation (128 calibration samples from WikiText-2) instead of the default AWQ quantization.

#### PTL iGPU (DDR5-5600)

| Backend | AWQ INT4 TPS | GPTQ INT4 TPS | Δ TPS | GPTQ TTFT (ms) |
|---------|-------------|---------------|-------|----------------|
| optimum | 13.5 | **16.3** | **+20.7%** | N/A* |
| GenAI | 14.7 | **17.6** | **+19.7%** | **77** |

#### Arc A770M dGPU (SKELETOR-03)

| Backend | AWQ INT4 TPS | GPTQ INT4 TPS | Δ TPS | GPTQ TTFT (ms) |
|---------|-------------|---------------|-------|----------------|
| optimum | 31.1 | **45.8** | **+47.3%** | N/A* |
| GenAI | 36.0 | **50.3** | **+39.7%** | **54** |

*\* optimum-intel streamer does not reliably capture TTFT*

GPTQ's per-channel scale estimation produces weight layouts that are more efficient for INT4 compute units. The GPTQ model (4.4 GB) is slightly larger than AWQ (4.2 GB) yet decodes dramatically faster. The **GPTQ gain is even larger on the dGPU (+47%)** than the iGPU (+21%), likely because the Arc A770M's higher bandwidth allows GPTQ's better weight layout to be exploited more fully. Export cost: ~8 hours on CPU (one-time).

### 9.3 Experiment 3: OpenVINO GenAI C++ Pipeline — Clear Winner

Replacing the Python-based `optimum-intel` wrapper with the C++ `openvino-genai` `LLMPipeline`.

#### PTL iGPU (DDR5-5600) — Llama 3.1-8B

| Precision | Baseline TPS (optimum) | GenAI TPS | Δ TPS | GenAI TTFT (ms) |
|-----------|----------------------|-----------|-------|-----------------|
| FP16 | 5.1 | 5.1 | 0.0% | 215 |
| INT8 | 9.7 | 10.1 | **+4.1%** | 119 |
| INT4 | 13.5 | 14.7 | **+8.9%** | 90 |

#### Arc A770M dGPU (SKELETOR-03) — Llama 3.1-8B

| Precision | Baseline TPS (optimum) | GenAI TPS | Δ TPS | GenAI TTFT (ms) |
|-----------|----------------------|-----------|-------|-----------------|
| FP16 | 15.7 | 17.3 | **+10.2%** | 84 |
| INT8 | 26.2 | 29.5 | **+12.6%** | 59 |
| INT4 | 31.1 | 36.0 | **+15.8%** | 55 |

The C++ pipeline eliminates Python overhead in the token generation loop. **The GenAI uplift is significantly larger on the dGPU** (+10-16%) than the iGPU (+0-9%) because the Arc A770M generates tokens fast enough that Python overhead becomes a meaningful fraction of per-token time even at FP16. On the iGPU, FP16 sees no improvement because the ~196ms/token decode time is dominated by memory bandwidth, not Python overhead. On the dGPU, the ~64ms/token FP16 decode time is fast enough for the C++ pipeline to provide a 10% uplift.

### 9.4 Combined Optimization Summary (INT4, Greedy)

#### PTL iGPU (DDR5-5600)

| Configuration | INT4 TPS | Δ vs Baseline | TTFT (ms) | Recommendation |
|-------------|----------|----------------|-----------|----------------|
| Llama AWQ + optimum (baseline) | 13.5 | — | N/A* | Baseline |
| Llama AWQ + GenAI | 14.7 | +8.9% | 90 | Good |
| Llama GPTQ + optimum | 16.3 | +20.7% | N/A* | Good |
| **Llama GPTQ + GenAI** | **17.6** | **+30.4%** | **77** | **Best Llama (iGPU)** |
| Qwen AWQ + optimum | 17.2 | +27.4% | N/A* | Good |
| **Qwen AWQ + GenAI** | **18.7** | **+38.5%** | **65** | **Best overall (iGPU)** |

#### Arc A770M dGPU (SKELETOR-03)

| Configuration | INT4 TPS | Δ vs Baseline | TTFT (ms) | Recommendation |
|-------------|----------|----------------|-----------|----------------|
| Llama AWQ + optimum (baseline) | 31.1 | — | N/A* | Baseline |
| Llama AWQ + GenAI | 36.0 | +15.8% | 55 | Good |
| Llama GPTQ + optimum | 45.8 | +47.3% | N/A* | Good |
| **Llama GPTQ + GenAI** | **50.3** | **+61.7%** | **54** | **Best Llama (dGPU)** |
| Qwen AWQ + optimum | 46.4 | +49.2% | N/A* | Good |
| **Qwen AWQ + GenAI** | **52.2** | **+67.8%** | **42** | **Best overall** |

*\* optimum-intel streamer does not reliably capture TTFT*

The GPTQ and GenAI optimizations are fully additive on both platforms. On the iGPU, combined software optimizations deliver **+30-39%**. On the dGPU, the gains are even larger: **+62-68%** — because the higher-bandwidth A770M can better exploit GPTQ's improved weight layout and the faster token rate makes Python overhead elimination more impactful. The dGPU's best configuration (**Qwen INT4 GenAI = 52.2 TPS**) is 2.8x faster than the iGPU's best (**18.7 TPS**).

> See `optimization-whitepaper.md` for full experimental details, reproduction commands, and analysis.

---

## 10. Quality Analysis

We conducted a semantic quality comparison of Qwen2.5-7B-Instruct and Llama 3.1-8B-Instruct across all 7 kiosk scenarios at FP16, INT8, and INT4 precisions, including GPTQ variants. Full analysis is in the companion paper (`quality-comparison-whitepaper.md`).

### 10.1 Quantization Impact on Quality

**INT4 = FP16 quality for kiosk conversations.** Both models produce equally helpful, coherent, and appropriate responses at INT4 as at FP16. Variations between precisions are limited to trivial word-level differences (e.g., "our store is open" vs "we're open") and different hallucinated product names — neither of which affects response quality.

| Model | FP16→INT8 | FP16→INT4 | Assessment |
|-------|-----------|-----------|------------|
| Qwen 2.5-7B | No degradation | No degradation (arguably improved in 2/7 scenarios) | Safe to deploy INT4 |
| Llama 3.1-8B | No degradation | No degradation | Safe to deploy INT4 |
| Llama 3.1-8B (GPTQ) | — | Equal or slightly better than AWQ INT4 | GPTQ recommended |

### 10.2 Model Comparison (INT4, Head-to-Head)

Using the fastest configuration for each model: Qwen INT4 AWQ GenAI (18.7 TPS) vs Llama INT4 GPTQ GenAI (17.6 TPS).

| Scenario | Winner | Key Difference |
|----------|--------|---------------|
| greeting | **Qwen** | More proactive — asks what type of help needed |
| store_hours | Tie | Both accurate and concise |
| product_lookup | **Qwen** | Cites real brand names (JBL, Anker) vs fictional brands |
| return_policy | **Qwen** | More thorough — mentions warranty, asks for details |
| loyalty_program | **Llama** | More structured with specific numbers |
| multi_turn_directions | Tie | Both give clear directions |
| multi_turn_troubleshoot | **Llama** | Better diagnostic approach |

**Overall: Qwen 3, Llama 2, Tie 2**

### 10.3 Key Quality Findings

1. **Model choice > precision choice**: The quality difference between Qwen and Llama (different response styles, different strengths) is much larger than the quality difference between FP16 and INT4 within the same model. Model selection should be driven by response style preference; precision should be driven purely by throughput.

2. **Both models hallucinate**: Both confidently cite product names, prices, and brand names not in the system prompt. Qwen's hallucinations are more plausible (real brand names like JBL), while Llama invents fictional brands. **RAG is essential** for any production kiosk deployment.

3. **No reason to run FP16 or INT8**: INT4 is strictly better — faster speed, equal quality. There is no quality-speed tradeoff to make at kiosk-level task complexity.

> See `quality-comparison-whitepaper.md` for full response transcripts, per-scenario analysis, and detailed comparison.

---

## 11. Analysis

### 11.1 Throughput Scaling Across Precisions

The three precision levels demonstrate clear throughput hierarchies, with quantization gains amplified on bandwidth-constrained platforms:

#### Arc A770M (Discrete GPU, ~512 GB/s dedicated GDDR6)

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | Bandwidth Reduction |
|-----------|-----------|-----------|------------------|---------------------|
| FP16 | 15 GB | 15.6 | 1.0x (baseline) | — |
| INT8 | 7.5 GB | 26.1 | **1.67x** | 2.0x |
| INT4 | 5.2 GB | 31.6 | **2.03x** | 2.9x |

#### Lunar Lake iGPU (~136 GB/s shared LPDDR5X)

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | Bandwidth Reduction |
|-----------|-----------|-----------|------------------|---------------------|
| FP16 | 15 GB | 6.4 | 1.0x (baseline) | — |
| INT8 | 7.5 GB | 10.2 | **1.59x** | 2.0x |
| INT4 | 5.2 GB | 14.9 (22.8†) | **2.33x** (3.56x†) | 2.9x |

*†Qwen 2.5-7B INT4 AWQ + GenAI backend*

#### PTL iGPU — Config A (~115 GB/s DDR5-7200)

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | Bandwidth Reduction |
|-----------|-----------|-----------|------------------|---------------------|
| INT8 | 7.5 GB | 11.9 | — | 2.0x |
| INT4 | 5.2 GB | 16.4 | — | 2.9x |

#### PTL iGPU — Config B (~90 GB/s DDR5-5600)

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | Bandwidth Reduction |
|-----------|-----------|-----------|------------------|---------------------|
| FP16 | 15 GB | 5.1 | 1.0x (baseline) | — |
| INT8 | 7.5 GB | 9.7 | **1.90x** | 2.0x |
| INT4 | 5.2 GB | 13.5 | **2.65x** | 2.9x |

#### Meteor Lake iGPU (~90 GB/s DDR5-5600, no XMX)

| Precision | Model Size | Greedy TPS | Speedup vs FP16 | Bandwidth Reduction |
|-----------|-----------|-----------|------------------|---------------------|
| FP16 | 15 GB | 3.3 | 1.0x (baseline) | — |
| INT8 | 7.5 GB | 5.2 | **1.58x** | 2.0x |
| INT4 | 5.2 GB | 6.7 | **2.03x** | 2.9x |

LLM inference on GPUs is primarily **memory-bandwidth bound** during autoregressive decoding — each token requires reading all model weights once. Reducing weight precision directly reduces bandwidth requirements.

**Quantization gains scale inversely with available bandwidth.** Panther Lake with DDR5-5600 shows the highest INT4/FP16 ratio (2.65x), followed by Lunar Lake (2.33x) and the Arc A770M (2.03x). The tighter the bandwidth constraint, the more every byte saved in model weights delivers proportionally more throughput.

With the complete precision sweep now available across four platforms, Meteor Lake shows an INT4/FP16 speedup of **2.03x** — identical to the Arc A770M despite having no XMX engines and completely different bandwidth characteristics. This suggests the ~2x INT4/FP16 ratio is a baseline driven primarily by model size reduction (15→5.2 GB). Platforms with tighter bandwidth constraints exceed this baseline: Lunar Lake (2.33x) and Panther Lake with DDR5-5600 (2.65x) — with Panther Lake's high ratio reflecting both its constrained bandwidth and excess compute capacity from 96 XMX engines.

On the Arc A770M, INT8 achieves **1.67x** speedup with 2x bandwidth reduction — nearly linear scaling, indicating minimal dequantization overhead for 8-bit weights. INT4 achieves **2.03x** speedup with 2.9x bandwidth reduction — sub-linear scaling because:
1. The INT4 model uses mixed precision (80% INT4, 20% INT8) — not pure INT4
2. Compute overhead from dequantization is higher for 4-bit weights
3. Input embedding and output projection layers remain at higher precision
4. Diminishing returns: the gap from INT8 to INT4 (+21%) is much smaller than FP16 to INT8 (+67%)

### 11.2 Variance Characteristics

| Precision | TPS Stddev (greedy) | Coefficient of Variation |
|-----------|--------------------:|------------------------:|
| FP16 | 0.17 | < 2% |
| INT8 | 0.82 | ~3% |
| INT4 | 1.18 | ~4% |

FP16 shows remarkably low variance (CoV < 2%), suggesting the GPU pipeline is fully deterministic during greedy decoding — each token generation takes a nearly identical amount of time.

Variance increases with quantization aggressiveness, likely due to:
- Dequantization overhead varying by layer group size alignment
- Mixed-precision routing decisions at runtime (particularly for INT4's 80/20 INT4/INT8 split)
- Kernel scheduling differences for quantized vs non-quantized layers

All three precisions are well within acceptable bounds for real-time applications.

### 11.3 Temperature Impact

Sampling (temperature=0.7, top_p=0.9) reduces throughput, but the penalty varies dramatically by platform:

#### Arc A770M (Discrete GPU)

| Precision | Greedy TPS | Sampling TPS | Delta |
|-----------|-----------|--------------|-------|
| INT4 | 31.6 | 23.4 | **-26%** |
| INT8 | 26.1 | 17.6 | **-33%** |
| FP16 | 15.6 | 10.8 | **-31%** |

#### Lunar Lake iGPU

| Precision | Greedy TPS | Sampling TPS | Delta |
|-----------|-----------|--------------|-------|
| INT4 | 14.9 | 14.0 | **-6%** |
| INT8 | 10.2 | 9.7 | **-5%** |
| FP16 | 6.4 | 6.1 | **-4%** |

#### Panther Lake iGPU

| Precision | Greedy TPS | Sampling TPS | Delta |
|-----------|-----------|--------------|-------|
| INT4 (7200) | 16.4 | 15.5 | **-5%** |
| INT4 (5600) | 13.5 | 12.9 | **-4%** |
| INT8 (7200) | 11.9 | 11.6 | **-3%** |
| INT8 (5600) | 9.7 | 9.4 | **-3%** |
| FP16 (5600) | 5.1 | 5.0 | **-2%** |

#### Meteor Lake iGPU

| Precision | Greedy TPS | Sampling TPS | Delta |
|-----------|-----------|--------------|-------|
| INT4 | 6.7 | 6.4 | **-4%** |
| INT8 | 5.2 | 4.9 | **-5%** |
| FP16 | 3.3 | 3.1 | **-6%** |

The **sampling penalty is 5-6x smaller on iGPUs** (~4-6%) compared to the discrete GPU (~26-33%). This is because the iGPU pipeline is already throughput-constrained by memory bandwidth — the CPU-side sampling overhead (softmax, top-p filtering, random sampling) occurs while the GPU is still feeding weights through the memory subsystem, effectively hiding the sampling latency.

On the Arc A770M, the GPU generates tokens fast enough that the sampling overhead becomes a visible bottleneck. The throughput penalty comes from:
1. **Softmax computation**: Sampling requires computing the full probability distribution, while greedy can use argmax shortcuts
2. **Top-p filtering**: Nucleus sampling requires sorting logits and computing cumulative probabilities
3. **Random sampling**: The sampling step itself adds overhead
4. **Variable output length**: Temperature sampling produces more variable-length responses (higher average token count)

### 11.4 Quality Assessment

**Greedy across all precisions**: FP16, INT8, and INT4 produce nearly identical responses for the same prompts on the Arc A770M. The `greeting` and `store_hours` scenarios generate word-for-word identical or near-identical text across all three precisions. Longer scenarios show minor phrasing differences (e.g., "10% off your purchase today" vs "10% off your first purchase today") but equivalent quality and factual consistency.

**INT8 vs FP16**: INT8 responses are essentially indistinguishable from FP16. The greedy transcripts show identical structure, tone, and factual content. This is expected — INT8 per-channel asymmetric quantization has minimal impact on transformer output distributions.

**INT4 vs FP16**: INT4 shows slightly more variation in product names (e.g., "Eclipse earbuds" vs "X-Tech Wireless Earbuds") and occasional minor wording differences, but overall quality, helpfulness, and accuracy remain equivalent. No hallucinations or degraded responses observed.

**Sampling**: Temperature=0.7 produces more varied and often slightly more engaging responses (e.g., "Welcome to our store!" vs "Welcome to our store.") with additional detail in product listings. The quality floor remains high across all three precisions.

**Key observation**: Neither INT8 nor INT4 quantization causes perceptible quality degradation for conversational kiosk tasks. The NNCF calibration (AWQ for INT4, per-channel asymmetric for INT8) effectively preserves model quality while delivering substantial throughput gains. A comprehensive semantic quality analysis across both models and all precisions — including GPTQ variants — is presented in Section 10 and detailed in `quality-comparison-whitepaper.md`.

### 11.5 TTFT Performance

With the optimum-intel backend, all configurations report TTFT < 2ms — but this is a **measurement artifact** of the Python streamer callback, which does not reliably capture first-token timing.

The GenAI C++ backend provides accurate TTFT measurement. Real TTFT values on the Arc A770M dGPU range from **42ms** (Qwen INT4) to **84ms** (Llama FP16) — all well under the 100ms perceptual threshold for "instant" response. On the PTL iGPU, TTFT ranges from **55ms** (Qwen INT4, DDR5-7200) to **215ms** (Llama FP16, DDR5-5600). On Lunar Lake, TTFT is **60ms** with Qwen INT4 + GenAI. Even the slowest TTFT (FP16 on iGPU) is below the 500ms threshold for perceived conversational responsiveness.

### 11.6 Cross-Platform Comparison

#### By Precision — Llama 3.1-8B, GPU Greedy (temp=0.0, optimum-intel baseline)

| Platform | INT4 TPS | INT8 TPS | FP16 TPS | INT4/FP16 Ratio |
|----------|----------|----------|----------|-----------------|
| **Arc A770M** (friday-cork) | **31.6** | **26.1** | **15.6** | 2.03x |
| **Panther Lake** (PTL, DDR5-7200) | **16.4** | **11.9** | — | — |
| **Panther Lake** (PTL, DDR5-5600) | **13.5** | **9.7** | **5.1** | 2.65x |
| **Lunar Lake** (LNL-GROVE) | **14.9** | **10.2** | **6.4** | 2.33x |
| **Meteor Lake** (MTL-NOYCE) | **6.7** | **5.2** | **3.3** | 2.03x |

#### Best Configuration per Platform — INT4, GPU Greedy (Qwen 2.5-7B AWQ or Llama GPTQ + GenAI)

| Platform | Best INT4 TPS | TTFT (ms) | Model + Quant | vs Baseline |
|----------|:------------:|:---------:|---------------|:-----------:|
| **Arc A770M** (SKELETOR-03) | **52.2** | **42** | Qwen AWQ + GenAI | +67.8% |
| **Panther Lake** (DDR5-7200) | **22.8** | **55** | Qwen AWQ + GenAI | +39.0% |
| **Panther Lake** (DDR5-5600) | **18.7** | **65** | Qwen AWQ + GenAI | +38.5% |
| **Lunar Lake** (LNL-GROVE) | **22.8** | **60** | Qwen AWQ + GenAI | +53.0% |
| **Meteor Lake** (MTL-NOYCE) | **6.7** | — | Llama AWQ + optimum | baseline |

Lunar Lake and Panther Lake DDR5-7200 converge at **22.8 TPS** with Qwen + GenAI despite different architectures (8 Xe2 cores vs 12 Xe3 cores). Lunar Lake's on-package LPDDR5X-8533 provides higher effective bandwidth (~100+ GB/s with lower latency) that compensates for fewer compute resources, producing identical throughput at the best configuration.

#### By Model + Backend — INT4 Greedy (temp=0.0)

**Panther Lake iGPU (DDR5-7200):**

| Configuration | INT4 TPS | TTFT (ms) | Δ vs DDR5-5600 Baseline |
|--------------|----------|-----------|--------------------------|
| **Qwen 2.5-7B INT4 AWQ + GenAI** | **22.8** | **55** | **+68.9%** |
| Llama 3.1-8B INT4 GPTQ + GenAI | **21.5** | **67** | +59.3% |

**Panther Lake iGPU (DDR5-5600):**

| Configuration | INT4 TPS | TTFT (ms) | Δ vs Baseline |
|--------------|----------|-----------|---------------|
| Llama 3.1-8B AWQ + optimum (baseline) | 13.5 | N/A* | — |
| Llama 3.1-8B AWQ + GenAI | 14.7 | 90 | +8.9% |
| Llama 3.1-8B GPTQ + GenAI | **17.6** | **77** | +30.4% |
| Qwen 2.5-7B AWQ + optimum | 17.2 | N/A* | +27.4% |
| **Qwen 2.5-7B AWQ + GenAI** | **18.7** | **65** | **+38.5%** |

**Lunar Lake iGPU (LPDDR5X-8533):**

| Configuration | INT4 TPS | TTFT (ms) | Δ vs Baseline |
|--------------|----------|-----------|---------------|
| Llama 3.1-8B AWQ + optimum (baseline) | 14.9 | N/A* | — |
| **Qwen 2.5-7B INT4 AWQ + GenAI** | **22.8** | **60** | **+53.0%** |

**Arc A770M dGPU (SKELETOR-03):**

| Configuration | INT4 TPS | TTFT (ms) | Δ vs Baseline |
|--------------|----------|-----------|---------------|
| Llama 3.1-8B AWQ + optimum (baseline) | 31.1 | N/A* | — |
| Llama 3.1-8B AWQ + GenAI | 36.0 | 55 | +15.8% |
| Llama 3.1-8B GPTQ + GenAI | **50.3** | **54** | +61.7% |
| Qwen 2.5-7B AWQ + optimum | 46.4 | N/A* | +49.2% |
| **Qwen 2.5-7B AWQ + GenAI** | **52.2** | **42** | **+67.8%** |

*\* optimum-intel streamer does not reliably capture TTFT*

#### Architecture Comparison (INT4 AWQ Optimum Baseline, Greedy)

| Platform | GPU Architecture | XMX | Memory BW | GPU TPS | CPU TPS | GPU/CPU Ratio | TPS/Watt |
|----------|-----------------|-----|-----------|---------|---------|---------------|----------|
| **Arc A770M** (friday-cork) | Xe-HPG, 32 cores | 512 engines | ~512 GB/s dedicated | **31.6** | — | — | ~0.23 |
| **Panther Lake** (DDR5-7200) | Xe3-LPG, 12 cores | 96 engines | ~115 GB/s shared | **16.4** | 7.9 | 2.08x | ~0.66 |
| **Panther Lake** (DDR5-5600) | Xe3-LPG, 12 cores | 96 engines | ~90 GB/s shared | **13.5** | 7.3 | 1.85x | ~0.54 |
| **Lunar Lake** (LNL-GROVE) | Xe2-LPG, 8 cores | 64 engines | ~136 GB/s shared | **14.9** (22.8†) | 8.0 | 1.86x | ~0.88 (~1.34†) |
| **Meteor Lake** (MTL-NOYCE) | Xe-LPG, 8 cores | **None** | ~90 GB/s shared | **6.7** | 5.4 | 1.24x | ~0.24 |

*†Qwen 2.5-7B INT4 AWQ + GenAI backend*

**Key insights:**

1. **Discrete vs integrated**: The Arc A770M delivers **2.3x** the throughput of the best iGPU configurations (52.2 vs 22.8 TPS with GenAI) and **7.8x** Meteor Lake's iGPU. The primary differentiator is dedicated GDDR6 bandwidth (~512 GB/s vs ~90-136 GB/s shared).

2. **Panther Lake (DDR5-7200) ties Lunar Lake at best config**: With Qwen + GenAI, both PTL DDR5-7200 and Lunar Lake achieve **22.8 TPS** — Lunar Lake's on-package LPDDR5X bandwidth advantage compensates for fewer Xe cores. At baseline (Llama + optimum), PTL DDR5-7200 leads (16.4 vs 14.9 TPS). With DDR5-5600, PTL drops to 13.5 TPS — below Lunar Lake — demonstrating that memory bandwidth can negate architectural advantages.

3. **XMX is transformative for iGPU LLM inference**: Lunar Lake (with XMX) achieves **2.2x** the GPU throughput of Meteor Lake (without XMX) despite having the same number of Xe cores and similar memory bandwidth. The absence of XMX engines is the single biggest factor limiting Meteor Lake's LLM performance.

4. **GPU/CPU ratios track XMX presence and core count**: Panther Lake (2.08x with DDR5-7200), Lunar Lake (1.86x), and Meteor Lake (1.24x) show that GPU acceleration scales with both XMX availability and core count.

5. **Power efficiency**: With Qwen + GenAI, Lunar Lake achieves **~1.34 TPS/W** at the SoC level (22.8 TPS / 17W) — **5.8x more efficient** than the Arc A770M (~0.23 TPS/W baseline). Panther Lake with DDR5-7200 achieves ~0.91 TPS/W at 25W (22.8 TPS) — strong efficiency at matched absolute performance. For battery-powered or thermally constrained deployments, Lunar Lake offers the best inference per watt.

6. **Temperature sensitivity varies by platform**: Sampling (t=0.7) causes a ~26-33% throughput hit on the Arc A770M but only ~2-6% on the iGPU platforms. This suggests the sampling overhead (softmax, top-p filtering) is proportionally smaller when the overall pipeline is slower.

7. **Quantization yields bigger gains on bandwidth-constrained platforms**: The INT4/FP16 speedup ratio reaches **2.65x on Panther Lake (DDR5-5600)**, **2.33x on Lunar Lake**, and **2.03x on Arc A770M and Meteor Lake**. Tighter bandwidth constraints amplify the benefit of model size reduction.

8. **Memory bandwidth is the dominant variable**: The PTL dual-config experiment (same silicon, different DRAM) shows a 0.82 elasticity coefficient — 22% less bandwidth produces 18% less throughput, consistently across INT4 and INT8. The DDR5-7200 GenAI validation (22.8 TPS measured vs 21-22 predicted) confirms this coefficient holds across different models, quantization methods, and backends.

9. **On Meteor Lake, INT8 GPU (5.2 TPS) is slower than INT4 CPU (5.4 TPS)**: Without XMX, the GPU cannot compensate for the 44% larger model size. This remains the strongest evidence that the architectural bottleneck varies by platform.

10. **Model architecture matters as much as hardware**: Switching from Llama 3.1-8B to Qwen 2.5-7B on the same PTL hardware yields +27% at INT4 — comparable to the gain from upgrading DDR5-5600 to DDR5-7200. Combined with software optimizations (GenAI + GPTQ/model choice), software delivers **+39-69%** over the Llama optimum baseline on the iGPU — exceeding the gain from a DRAM speed tier upgrade. On Lunar Lake, Qwen + GenAI delivers +53% over baseline (14.9 to 22.8 TPS).

### 11.7 NPU Limitations for LLM Inference

Both Lunar Lake (NPU4, 48 TOPS) and Meteor Lake (NPU3, 11 TOPS) failed to run the Llama 3.1 8B model. The root cause is architectural: Intel's NPU compiler (`vpux-compiler`) requires static tensor shapes at compilation time, but autoregressive LLM generation uses dynamic sequence lengths that grow with each token.

The error — `to_shape was called on a dynamic shape` — occurs when the NPU compiler attempts to lower the model graph and encounters unbounded dimension `9223372036854775807` (INT64_MAX) in tensor shapes used by the embedding layer, attention masks, and layer norms.

This is consistent with public research and Intel's own documentation: NPUs are optimized for **fixed-shape, batch-oriented workloads** (vision models, speech encoders, BERT-style models) — not the iterative, variable-length generation pattern of autoregressive LLMs.

**Future outlook**: Intel's Panther Lake (PTL) with NPU5 and the research project Agent.xpu demonstrate that heterogeneous GPU+NPU inference for LLMs is technically feasible, with the NPU handling prefill while the GPU handles decode. However, this is not yet available in production OpenVINO releases.

### 11.8 Panther Lake: Projections vs Reality

The original projection for Panther Lake was **20-25 TPS** for INT4, based on an assumed LPDDR5X-9600 memory subsystem (~154 GB/s). The actual result with DDR5-7200 SODIMMs: **16.4 TPS** — below the projected range.

However, the projection assumed LPDDR5X-9600 (~154 GB/s), while the CRB (Customer Reference Board) uses DDR5 SODIMMs instead. When bandwidth-adjusted, the result aligns well: 16.4 TPS * (154/115) = **~22 TPS**, squarely within the original projection range.

The memory bandwidth sensitivity experiment confirms this interpretation: PTL's Xe3 architecture has the compute headroom to deliver higher throughput, but is starved by SODIMM bandwidth. The 18% throughput drop from DDR5-7200 to DDR5-5600 — with identical silicon — proves the bottleneck is memory, not compute.

**Validated with GenAI backend**: The DDR5-7200 + Qwen GenAI experiment measured **22.8 TPS** — nearly exactly matching the bandwidth-adjusted projection of ~22 TPS. This validates both the projection methodology and the 0.82 elasticity coefficient. With Llama GPTQ + GenAI, DDR5-7200 achieves **21.5 TPS**, also within the projected range.

**Projection for production Panther Lake devices**: Systems with LPDDR5X-8533 (matching Lunar Lake's memory type) should achieve **~24-26 TPS** with Qwen + GenAI based on the bandwidth scaling observed. Notably, Lunar Lake already achieves 22.8 TPS with Qwen + GenAI — matching PTL DDR5-7200 — suggesting that on-package LPDDR5X's lower latency provides an outsized benefit that partially compensates for Lunar Lake's fewer Xe cores.

### 11.9 Memory Bandwidth as the Dominant Variable

The PTL dual-configuration experiment provides the first controlled test in this study isolating memory bandwidth from all other variables. Same silicon, same firmware, same drivers, same OpenVINO version — only the DRAM modules changed.

**Baseline results (optimum-intel):**
- 22% bandwidth reduction (DDR5-7200 → DDR5-5600) produced an **18% throughput drop**
- The **0.82 elasticity coefficient** (18% / 22%) was identical for both INT4 and INT8
- This near-linear relationship confirms that autoregressive LLM decode on iGPUs is overwhelmingly memory-bandwidth bound

**GenAI validation results:**
- DDR5-7200 with Qwen AWQ + GenAI: **22.8 TPS** (predicted 21-22 TPS using 0.82 coefficient from DDR5-5600 baseline of 18.7 TPS)
- DDR5-7200 with Llama GPTQ + GenAI: **21.5 TPS** (predicted 20-21 TPS from DDR5-5600 baseline of 17.6 TPS)
- Measured values slightly exceed predictions, suggesting the 0.82 coefficient is conservative at the high end of DDR5 speeds
- The elasticity relationship holds across different models, quantization methods, and backends — confirming it is a fundamental property of the memory subsystem, not an artifact of any particular software configuration

**Implications for system design:**
- Memory speed selection is as important as GPU architecture for iGPU LLM inference
- DDR5-7200 SODIMMs provide a meaningful performance boost over DDR5-5600 for edge AI workloads — a 29% bandwidth advantage translates to ~22% higher throughput
- For OEMs designing edge AI systems, the memory subsystem deserves as much attention as the SoC selection
- The slightly sub-linear elasticity (0.82 < 1.0) suggests a small fixed-overhead component, but the dominant factor is unmistakably bandwidth
- Lunar Lake's on-package LPDDR5X achieves parity with PTL DDR5-7200 (both 22.8 TPS at best config), demonstrating that memory latency — not just bandwidth — contributes to effective throughput

---

## 12. Conclusions

1. **Best dGPU configuration**: Qwen 2.5-7B INT4 AWQ + GenAI pipeline on Arc A770M = **52.2 TPS, 42ms TTFT**. A 50-token response completes in under 1 second — 5x the kiosk target.

2. **Best iGPU configurations**: Qwen 2.5-7B INT4 AWQ + GenAI pipeline achieves **22.8 TPS** on both **Panther Lake DDR5-7200 (55ms TTFT)** and **Lunar Lake LPDDR5X-8533 (60ms TTFT)**. On Panther Lake DDR5-5600: **18.7 TPS, 65ms TTFT**. A 50-token response takes 2.2 seconds on the fastest iGPU configs with perceived-instant first token.

3. **Best Llama configuration**: Llama 3.1-8B INT4 GPTQ + GenAI pipeline = **50.3 TPS / 54ms TTFT** on dGPU, **17.6 TPS / 77ms TTFT** on iGPU. GPTQ and GenAI optimizations are fully additive.

4. **Model selection matters more than precision for quality**: The difference between Qwen and Llama response styles is larger than the difference between FP16 and INT4 within either model. Choose your model based on brand voice preference; choose INT4 unconditionally for throughput.

5. **INT4 quantization is safe**: No meaningful quality degradation at any precision level for kiosk conversations. There is no quality-speed tradeoff — INT4 is strictly better (faster, equal quality).

6. **Software optimizations deliver hardware-equivalent gains**: Combined software changes (model choice + GPTQ + GenAI) deliver **+68% on dGPU** and **+39% on iGPU** over the Llama optimum baseline. On the dGPU, this is equivalent to upgrading from a mid-range to a high-end GPU — achieved purely through software. These are free performance gains with no hardware cost.

7. **Optimization gains scale with hardware speed**: The GenAI C++ pipeline delivers +16% on the dGPU vs +9% on the iGPU. GPTQ delivers +47% on the dGPU vs +21% on the iGPU. Faster hardware amplifies software optimization benefits because overhead becomes a larger fraction of the per-token budget.

8. **Memory bandwidth remains the dominant variable**: The PTL dual-config experiment confirms 0.82 elasticity — 22% less bandwidth = 18% less throughput. A follow-up DDR5-7200 GenAI experiment (22.8 TPS measured vs 21-22 TPS predicted) validates this coefficient across different models and backends. FP16 is memory-bandwidth-bound at ~5.1-5.5 TPS on iGPU regardless of model or optimization, and ~15.7-19.0 TPS on dGPU. Lunar Lake's on-package LPDDR5X allows it to match PTL DDR5-7200 at 22.8 TPS despite fewer Xe cores.

9. **RAG is essential**: Both models hallucinate product names and policy details. A production kiosk must use retrieval-augmented generation to ground responses in actual inventory/policy data.

---

## 13. Results Status

### Llama 3.1-8B-Instruct (Baseline)

| Machine | Device | Precision | Status |
|---------|--------|-----------|--------|
| friday-cork (Arc A770M) | GPU | FP16, INT8, INT4 | **Complete** |
| LNL-GROVE (Lunar Lake) | CPU | INT4 | **Complete** |
| LNL-GROVE (Lunar Lake) | GPU | INT4 | **Complete** |
| LNL-GROVE (Lunar Lake) | GPU | INT8 | **Complete** |
| LNL-GROVE (Lunar Lake) | GPU | FP16 | **Complete** |
| LNL-GROVE (Lunar Lake) | NPU | INT4 | **Failed** (dynamic shapes) |
| MTL-NOYCE (Meteor Lake) | CPU | INT4 | **Complete** |
| MTL-NOYCE (Meteor Lake) | GPU | INT4 | **Complete** |
| MTL-NOYCE (Meteor Lake) | GPU | INT8 | **Complete** |
| MTL-NOYCE (Meteor Lake) | NPU | INT4 | **Failed** (dynamic shapes) |
| MTL-NOYCE (Meteor Lake) | GPU | FP16 | **Complete** — after RAM upgrade to 62 GB DDR5-5600 |
| PTL-FAIRCHILD (Panther Lake) | CPU | INT4, INT8, FP16 | **Complete** — DDR5-7200 + DDR5-5600 |
| PTL-FAIRCHILD (Panther Lake) | GPU | INT4, INT8 | **Complete** — DDR5-7200 + DDR5-5600 |
| PTL-FAIRCHILD (Panther Lake) | GPU | FP16 | **Complete** — DDR5-5600 only (16GB insufficient for FP16) |

### Qwen2.5-7B-Instruct

| Machine | Device | Backend | Precision | Status |
|---------|--------|---------|-----------|--------|
| PTL-FAIRCHILD (Panther Lake) | GPU | optimum | FP16, INT8, INT4 | **Complete** — DDR5-5600 |
| PTL-FAIRCHILD (Panther Lake) | GPU | GenAI | FP16, INT8, INT4 | **Complete** — DDR5-5600 |
| PTL-FAIRCHILD (Panther Lake) | GPU | GenAI | INT4 | **Complete** — DDR5-7200, 22.8 TPS / 55ms TTFT |
| LNL-GROVE (Lunar Lake) | GPU | GenAI | INT4 | **Complete** — 22.8 TPS / 60ms TTFT |

### Optimization Experiments (PTL-FAIRCHILD, DDR5-5600)

| Experiment | Model | Status |
|-----------|-------|--------|
| ov_config runtime flags | Llama 3.1-8B | **Complete** — no benefit |
| KV_CACHE_PRECISION=u8 only | Llama 3.1-8B | **Complete** — -7.4% INT4 |
| GPTQ INT4 (optimum backend) | Llama 3.1-8B | **Complete** — +20.7% |
| GPTQ INT4 (GenAI backend) | Llama 3.1-8B | **Complete** — +19.7% |
| GenAI C++ pipeline | Llama 3.1-8B | **Complete** — +8.9% INT4 |
| Quality comparison | Both models | **Complete** — see Section 10 |
| DDR5-7200 GenAI validation | Qwen 2.5-7B AWQ | **Complete** — 22.8 TPS / 55ms TTFT |
| DDR5-7200 GenAI validation | Llama 3.1-8B GPTQ | **Complete** — 21.5 TPS / 67ms TTFT |

### SKELETOR-03 (Arc A770M dGPU) — New Configurations

| Model | Backend | Precision | Status |
|-------|---------|-----------|--------|
| Llama 3.1-8B AWQ | optimum | FP16, INT8, INT4 | **Complete** — matches friday-cork baseline |
| Llama 3.1-8B AWQ | GenAI | FP16, INT8, INT4 | **Complete** — +10-16% over optimum |
| Llama 3.1-8B GPTQ | optimum | INT4 | **Complete** — 45.8 TPS (+47% over AWQ) |
| Llama 3.1-8B GPTQ | GenAI | INT4 | **Complete** — 50.3 TPS (+62% over AWQ baseline) |
| Qwen 2.5-7B AWQ | optimum | FP16, INT8, INT4 | **Complete** — Qwen faster than Llama at all precisions |
| Qwen 2.5-7B AWQ | GenAI | FP16, INT8, INT4 | **Complete** — 52.2 TPS INT4 (best overall) |

---

## 14. Software Stack

| Component | Version |
|-----------|---------|
| **OS** | Ubuntu 24.04.3 LTS |
| **Kernel** | 6.14.0-37-generic |
| **Python** | 3.12.3 |
| **OpenVINO** | 2025.4.1-20426-82bbf0292c5-releases/2025/4 |
| **optimum-intel** | Latest (pip install) |
| **openvino-genai** | Latest (pip install) — C++ LLMPipeline backend |
| **transformers** | Latest (pip install) |
| **NNCF** | Latest (for quantization) |
| **Intel GPU Runtime** | NEO 25.18+ (Lunar Lake), 23.43+ (Meteor Lake) |

---

## 15. Reproducibility

All benchmark code, scenarios, and configuration are open source:

**Repository**: [github.com/JoshCork/intel-ai-benchmarking](https://github.com/JoshCork/intel-ai-benchmarking)

To reproduce these results:

```bash
# 1. Clone and setup
git clone https://github.com/JoshCork/intel-ai-benchmarking.git
cd intel-ai-benchmarking
python3 -m venv venv && source venv/bin/activate
pip install openvino optimum-intel[openvino] openvino-genai transformers torch nncf

# 2. Export models (INT4 examples)
python scripts/export_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --precision INT4

python scripts/export_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --precision INT4

# 3. Run benchmark (optimum backend)
python benchmark.py \
    --precision INT4 \
    --device GPU \
    --temperature 0.0 0.7 \
    --codename ADL \
    --tdp 45W

# 4. Run benchmark (GenAI C++ backend)
python benchmark.py \
    --precision INT4 \
    --device GPU \
    --temperature 0.0 0.7 \
    --codename PTL \
    --tdp 25W \
    --backend genai

# 5. (Optional) Capture system config with PerfSpect
python benchmark.py --perfspect --precision INT4 --device GPU

# Or just capture config without benchmarking
python benchmark.py --perfspect-only
```

Results are stored in a SQLite database for cross-machine comparison and historical tracking. System configurations (power governor, BIOS, memory, turbo boost) are optionally captured via Intel PerfSpect and linked to benchmark runs for full reproducibility.

---

## Appendix A: Raw Data Tables

### A.1 INT4 — Per-Run Metrics (temp=0.0)

*Full per-run metrics are stored in the SQLite database at `results/benchmarks.db` and can be exported with:*

```sql
SELECT r.scenario_name, m.run_number, m.tokens_per_sec, m.ttft_ms,
       m.total_ms, m.output_tokens
FROM run_metrics m
JOIN benchmark_runs r ON r.id = m.run_id
WHERE r.model_precision = 'INT4' AND r.temperature = 0.0 AND m.is_warmup = 0
ORDER BY r.scenario_name, m.run_number;
```

---

*All GPU benchmark configurations across five machines (Arc A770M x2, Lunar Lake, Meteor Lake, and Panther Lake), two models (Llama 3.1-8B, Qwen 2.5-7B), three precision levels, and two backends (optimum-intel, GenAI) are now complete. Optimization experiments (ov_config, GPTQ, GenAI pipeline) and quality analysis are complete on both Panther Lake iGPU and Arc A770M dGPU. DDR5-7200 bandwidth validation and Lunar Lake GenAI rerun confirm the 0.82 elasticity coefficient and establish 22.8 TPS as the iGPU performance ceiling for current hardware. Best overall: Qwen 2.5-7B INT4 AWQ + GenAI = 52.2 TPS / 42ms TTFT (dGPU), 22.8 TPS / 55-60ms TTFT (iGPU — both PTL DDR5-7200 and Lunar Lake).*
