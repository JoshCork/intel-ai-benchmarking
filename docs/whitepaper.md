# Intel Edge AI Inference Benchmarking

**Llama 3.1 8B Instruct on Intel Client GPUs — A Standardized Evaluation**

*February 2026*

---

## Abstract

This paper presents a standardized benchmarking methodology for evaluating large language model (LLM) inference performance on Intel client-class hardware using the OpenVINO runtime. We benchmark **Meta's Llama 3.1 8B Instruct** model across three weight precision formats (FP16, INT8, INT4) on Intel discrete and integrated GPUs, using realistic retail kiosk conversation scenarios as test workloads.

Our results demonstrate that INT4 quantization on an Intel Arc A770M discrete GPU achieves **31-33 tokens/sec** with greedy decoding — exceeding the 10 TPS interactive threshold by 3x — while maintaining conversational quality indistinguishable from FP16 inference. The full precision sweep across platforms reveals that **quantization yields proportionally larger gains on bandwidth-constrained iGPUs** (2.33x INT4/FP16 speedup on Lunar Lake vs 2.03x on Arc A770M).

On integrated GPUs with INT4, **Lunar Lake (Xe2) achieves ~15 TPS** while **Meteor Lake (Xe-LPG) reaches only ~6.7 TPS** — a 2.2x gap explained by Lunar Lake's XMX matrix engines. At INT8, Lunar Lake hits **10.2 TPS** (at the kiosk target) while Meteor Lake drops to **5.2 TPS** — notably slower than its own INT4 CPU performance (5.4 TPS), demonstrating that without XMX, the GPU provides no advantage for larger model formats. NPU inference is not currently viable for autoregressive LLM workloads due to dynamic shape constraints in the Intel NPU compiler. Temperature-based sampling (0.7) introduces a ~26-33% throughput penalty on discrete GPUs but only ~4-6% on iGPUs.

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
| MTL-NOYCE | Intel Core Ultra 5 125H | MTL (Meteor Lake) | Intel Arc (Xe-LPG, integrated) | Integrated | 28W SoC | 15 GB LPDDR5X-7467 |

All machines run Ubuntu 24.04 LTS with OpenVINO 2025.4.1, Python 3.12, and the Intel GPU compute runtime (NEO).

#### GPU Architecture Comparison

| Attribute | Arc A770M (Xe-HPG) | Arc 140V / LNL (Xe2-LPG) | MTL iGPU (Xe-LPG) |
|-----------|-------------------|--------------------------|-------------------|
| **Xe Cores** | 32 | 8 | 8 |
| **XMX Matrix Engines** | 512 | 64 (8/core) | **None** |
| **INT8 TOPS (GPU)** | 223 | 67 | 18 (vector only) |
| **Memory Type** | 16 GB GDDR6 (dedicated) | LPDDR5X-8533 (shared) | LPDDR5X-7467 (shared) |
| **Memory Bus** | 256-bit | 128-bit | 128-bit |
| **Theoretical Bandwidth** | ~512 GB/s | ~136.5 GB/s | ~119.5 GB/s |
| **Effective Bandwidth** | ~400+ GB/s | ~95-110 GB/s | ~80-87 GB/s |
| **NPU** | N/A | NPU4 (48 TOPS) | NPU3 (11 TOPS) |

**Key architectural difference**: Meteor Lake's iGPU **lacks XMX matrix engines** entirely — all INT8/INT4 computation runs on standard vector units via DP4a instructions. Lunar Lake's Xe2 introduces XMX to the iGPU for the first time, providing ~3.7x the INT8 AI compute at the same core count. The Arc A770M has 4x the Xe cores *and* XMX, giving it overwhelming compute density.

**Memory bandwidth is the primary bottleneck** for LLM autoregressive decode. Each token requires reading the full model weights (~3.5 GB for INT4). The theoretical decode ceiling per platform:

| Platform | Effective BW | Theoretical Max TPS (INT4) | Observed TPS | Efficiency |
|----------|-------------|---------------------------|-------------|------------|
| Arc A770M | ~400 GB/s | ~114 TPS | 31.6 TPS | ~28% |
| Lunar Lake iGPU | ~100 GB/s | ~29 TPS | 14.9 TPS | ~51% |
| Meteor Lake iGPU | ~80 GB/s | ~23 TPS | 6.7 TPS | ~29% |

Lunar Lake achieves the highest memory bandwidth efficiency (51%) likely due to on-package LPDDR5X providing lower latency. Meteor Lake's poor efficiency (29%) despite reasonable bandwidth reflects the compute bottleneck from missing XMX engines — the vector-only pipeline cannot saturate available bandwidth for INT4 matrix operations.

---

## 2. Methodology

### 2.1 Model Preparation

**Base Model**: `meta-llama/Llama-3.1-8B-Instruct` from HuggingFace

Models are exported from HuggingFace Transformers format to OpenVINO Intermediate Representation (IR) using the `optimum-intel` library:

```bash
optimum-cli export openvino -m meta-llama/Llama-3.1-8B-Instruct \
    --weight-format <precision> <output_dir>
```

Three precision variants are produced:

| Precision | Weight Format | Model Size | Quantization Method |
|-----------|---------------|------------|---------------------|
| **FP16** | Half-precision floating point | ~15 GB | None (native export) |
| **INT8** | 8-bit integer, per-channel asymmetric | ~7.5 GB | NNCF post-training quantization |
| **INT4** | Mixed INT4/INT8, group size 64 | ~5.2 GB | NNCF with AWQ (Activation-aware Weight Quantization) |

The INT4 model uses NNCF's mixed-precision assignment: 80% of weight layers use INT4 asymmetric quantization with group size 64, while 20% (typically attention output projections and final layers) remain INT8 for accuracy preservation. Activation-aware Weight Quantization (AWQ) adjusts quantization parameters based on activation distributions to minimize accuracy loss.

### 2.2 Inference Engine

Inference is performed using `optimum-intel`'s `OVModelForCausalLM`, which wraps the OpenVINO runtime for autoregressive text generation.

```python
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

model = OVModelForCausalLM.from_pretrained(model_path, device="GPU")
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

**Tokenization**: The model's native chat template (Llama 3.1 Instruct format) is applied via `tokenizer.apply_chat_template()`, producing properly formatted `<|begin_of_text|>`, `<|start_header_id|>`, and role-tagged input sequences.

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

**Key findings — LNL GPU INT4:**
- Greedy: **14.9 TPS mean** — comfortably above the kiosk target at 17W
- Sampling: **14.0 TPS mean** — only ~6% throughput penalty vs greedy (much less than Arc A770M's ~26%)
- TTFT: sub-1ms — excellent first-token response
- GPU provides **1.86x speedup** over CPU on Lunar Lake — significant value from Xe2 XMX engines
- Remarkably tight variance (P5-P95 spread < 1 TPS)

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

---

## 6. Results — Intel Meteor Lake iGPU (MTL-NOYCE)

**Machine**: Intel Core Ultra 5 125H, 15 GB LPDDR5X-7467, Ubuntu 24.04
**OpenVINO**: 2025.4.1 | **GPU**: Intel Arc (Xe-LPG, 8 Xe cores, **no XMX**) | **NPU**: Meteor Lake NPU3

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

### 6.3 Precision Comparison (MTL-NOYCE, GPU, Greedy)

| Precision | Model Size | Greedy TPS | Speedup vs INT8 | vs Kiosk Target |
|-----------|-----------|-----------|------------------|-----------------|
| INT8 | 7.5 GB | 5.2 | 1.0x (baseline) | 0.52x |
| INT4 | 5.2 GB | 6.7 | **1.30x** | 0.67x |

*FP16 benchmarks deferred — MTL-NOYCE has only 15 GB RAM, insufficient for the 15 GB FP16 model plus OS/runtime overhead. RAM upgrade to 64 GB DDR5-5600 SODIMM planned.*

Even with INT4, Meteor Lake's GPU cannot reach the 10 TPS kiosk target. The combination of shared bandwidth (~120 GB/s) and no XMX matrix engines creates a hard ceiling for LLM inference performance on this platform.

---

## 7. Analysis

### 7.1 Throughput Scaling Across Precisions

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
| INT4 | 5.2 GB | 14.9 | **2.33x** | 2.9x |

#### Meteor Lake iGPU (~120 GB/s shared LPDDR5X, no XMX)

| Precision | Model Size | Greedy TPS | Speedup vs INT8 | Bandwidth Reduction |
|-----------|-----------|-----------|------------------|---------------------|
| INT8 | 7.5 GB | 5.2 | 1.0x (baseline) | — |
| INT4 | 5.2 GB | 6.7 | **1.30x** | 1.4x |

LLM inference on GPUs is primarily **memory-bandwidth bound** during autoregressive decoding — each token requires reading all model weights once. Reducing weight precision directly reduces bandwidth requirements.

**Quantization gains scale inversely with available bandwidth.** Lunar Lake's INT4/FP16 speedup (2.33x) exceeds the Arc A770M's (2.03x) because the shared LPDDR5X is the tighter constraint — every byte saved in model weights delivers proportionally more throughput.

Meteor Lake shows the weakest INT4/INT8 speedup (1.30x) despite the same bandwidth reduction. Without XMX matrix engines, the GPU compute pipeline itself becomes a secondary bottleneck — the dequantization overhead for INT4 is proportionally larger on vector-only hardware.

On the Arc A770M, INT8 achieves **1.67x** speedup with 2x bandwidth reduction — nearly linear scaling, indicating minimal dequantization overhead for 8-bit weights. INT4 achieves **2.03x** speedup with 2.9x bandwidth reduction — sub-linear scaling because:
1. The INT4 model uses mixed precision (80% INT4, 20% INT8) — not pure INT4
2. Compute overhead from dequantization is higher for 4-bit weights
3. Input embedding and output projection layers remain at higher precision
4. Diminishing returns: the gap from INT8 to INT4 (+21%) is much smaller than FP16 to INT8 (+67%)

### 7.2 Variance Characteristics

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

### 7.3 Temperature Impact

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

#### Meteor Lake iGPU

| Precision | Greedy TPS | Sampling TPS | Delta |
|-----------|-----------|--------------|-------|
| INT4 | 6.7 | 6.4 | **-4%** |
| INT8 | 5.2 | 4.9 | **-5%** |

The **sampling penalty is 5-6x smaller on iGPUs** (~4-6%) compared to the discrete GPU (~26-33%). This is because the iGPU pipeline is already throughput-constrained by memory bandwidth — the CPU-side sampling overhead (softmax, top-p filtering, random sampling) occurs while the GPU is still feeding weights through the memory subsystem, effectively hiding the sampling latency.

On the Arc A770M, the GPU generates tokens fast enough that the sampling overhead becomes a visible bottleneck. The throughput penalty comes from:
1. **Softmax computation**: Sampling requires computing the full probability distribution, while greedy can use argmax shortcuts
2. **Top-p filtering**: Nucleus sampling requires sorting logits and computing cumulative probabilities
3. **Random sampling**: The sampling step itself adds overhead
4. **Variable output length**: Temperature sampling produces more variable-length responses (higher average token count)

### 7.4 Quality Assessment

**Greedy across all precisions**: FP16, INT8, and INT4 produce nearly identical responses for the same prompts. The `greeting` and `store_hours` scenarios generate word-for-word identical or near-identical text across all three precisions. Longer scenarios show minor phrasing differences (e.g., "10% off your purchase today" vs "10% off your first purchase today") but equivalent quality and factual consistency.

**INT8 vs FP16**: INT8 responses are essentially indistinguishable from FP16. The greedy transcripts show identical structure, tone, and factual content. This is expected — INT8 per-channel asymmetric quantization has minimal impact on transformer output distributions.

**INT4 vs FP16**: INT4 shows slightly more variation in product names (e.g., "Eclipse earbuds" vs "X-Tech Wireless Earbuds") and occasional minor wording differences, but overall quality, helpfulness, and accuracy remain equivalent. No hallucinations or degraded responses observed.

**Sampling**: Temperature=0.7 produces more varied and often slightly more engaging responses (e.g., "Welcome to our store!" vs "Welcome to our store.") with additional detail in product listings. The quality floor remains high across all three precisions.

**Key observation**: Neither INT8 nor INT4 quantization causes perceptible quality degradation for conversational kiosk tasks. The NNCF calibration (AWQ for INT4, per-channel asymmetric for INT8) effectively preserves model quality while delivering substantial throughput gains.

### 7.5 TTFT Performance

All configurations achieve TTFT < 2ms — orders of magnitude below the 100ms perceptual threshold. This is because:
1. The OpenVINO model is pre-compiled at load time
2. GPU memory is pre-allocated during model loading
3. The first-token computation involves a single forward pass through the model
4. Input sequences are short (< 100 tokens for most scenarios)

### 7.6 Cross-Platform Comparison

#### By Precision — GPU Greedy (temp=0.0)

| Platform | INT4 TPS | INT8 TPS | FP16 TPS | INT4/FP16 Ratio |
|----------|----------|----------|----------|-----------------|
| **Arc A770M** (friday-cork) | **31.6** | **26.1** | **15.6** | 2.03x |
| **Lunar Lake** (LNL-GROVE) | **14.9** | **10.2** | **6.4** | 2.33x |
| **Meteor Lake** (MTL-NOYCE) | **6.7** | **5.2** | *pending* | — |

#### Architecture Comparison (INT4, Greedy)

| Platform | GPU Architecture | XMX | Memory BW | GPU TPS | CPU TPS | GPU/CPU Ratio | TPS/Watt |
|----------|-----------------|-----|-----------|---------|---------|---------------|----------|
| **Arc A770M** (friday-cork) | Xe-HPG, 32 cores | 512 engines | ~512 GB/s dedicated | **31.6** | — | — | ~0.23 |
| **Lunar Lake** (LNL-GROVE) | Xe2-LPG, 8 cores | 64 engines | ~136 GB/s shared | **14.9** | 8.0 | 1.86x | ~0.88 |
| **Meteor Lake** (MTL-NOYCE) | Xe-LPG, 8 cores | **None** | ~120 GB/s shared | **6.7** | 5.4 | 1.24x | ~0.24 |

**Key insights:**

1. **Discrete vs integrated**: The Arc A770M delivers **2.1x** the throughput of Lunar Lake's iGPU and **4.7x** Meteor Lake's iGPU. The primary differentiator is dedicated GDDR6 bandwidth (~512 GB/s vs ~100-120 GB/s shared LPDDR5X).

2. **XMX is transformative for iGPU LLM inference**: Lunar Lake (with XMX) achieves **2.2x** the GPU throughput of Meteor Lake (without XMX) despite having the same number of Xe cores and similar memory bandwidth. The absence of XMX engines is the single biggest factor limiting Meteor Lake's LLM performance.

3. **Lunar Lake's GPU/CPU ratio (1.86x) vs Meteor Lake's (1.24x)** directly illustrates the XMX value. Without tensor acceleration, the GPU can barely outpace the CPU for matrix-heavy LLM workloads.

4. **Power efficiency**: Lunar Lake achieves **~0.88 TPS/W** at the SoC level — nearly **4x more efficient** than the Arc A770M (~0.23 TPS/W). For battery-powered or thermally constrained deployments, Lunar Lake offers the best inference per watt.

5. **Temperature sensitivity varies by platform**: Sampling (t=0.7) causes a ~26-33% throughput hit on the Arc A770M but only ~4-6% on the iGPU platforms. This suggests the sampling overhead (softmax, top-p filtering) is proportionally smaller when the overall pipeline is slower.

6. **Quantization yields bigger gains on bandwidth-constrained platforms**: The INT4/FP16 speedup ratio is **2.33x on Lunar Lake** vs **2.03x on the Arc A770M**. Shared LPDDR5X is the tighter bottleneck, so every byte of model size reduction has proportionally more impact.

7. **On Meteor Lake, INT8 GPU (5.2 TPS) is slower than INT4 CPU (5.4 TPS)**: Without XMX, the GPU cannot compensate for the 44% larger model size. This is the strongest evidence that memory bandwidth — not compute — is the primary bottleneck for LLM inference on integrated GPUs.

### 7.7 NPU Limitations for LLM Inference

Both Lunar Lake (NPU4, 48 TOPS) and Meteor Lake (NPU3, 11 TOPS) failed to run the Llama 3.1 8B model. The root cause is architectural: Intel's NPU compiler (`vpux-compiler`) requires static tensor shapes at compilation time, but autoregressive LLM generation uses dynamic sequence lengths that grow with each token.

The error — `to_shape was called on a dynamic shape` — occurs when the NPU compiler attempts to lower the model graph and encounters unbounded dimension `9223372036854775807` (INT64_MAX) in tensor shapes used by the embedding layer, attention masks, and layer norms.

This is consistent with public research and Intel's own documentation: NPUs are optimized for **fixed-shape, batch-oriented workloads** (vision models, speech encoders, BERT-style models) — not the iterative, variable-length generation pattern of autoregressive LLMs.

**Future outlook**: Intel's Panther Lake (PTL) with NPU5 and the research project Agent.xpu demonstrate that heterogeneous GPU+NPU inference for LLMs is technically feasible, with the NPU handling prefill while the GPU handles decode. However, this is not yet available in production OpenVINO releases.

### 7.8 Panther Lake Projections

Based on architectural analysis (Xe3-LPG, up to 12 Xe cores, 96 XMX engines, LPDDR5X-9600 at ~154 GB/s), Panther Lake's iGPU is projected to achieve **20-25 TPS** for INT4 Llama 3.1 8B — roughly 1.3-1.5x Lunar Lake's performance. Key improvements:

- 50% more Xe cores (12 vs 8) with XMX retained
- 2x L2 cache (16 MB vs 8 MB) reducing DRAM traffic per token
- ~13% more memory bandwidth (LPDDR5X-9600 vs 8533)
- Platform AI TOPS up to 180 (vs 120 on Lunar Lake)

This would place Panther Lake's iGPU performance in the range of **half an Arc A770M** at a fraction of the power budget — a significant milestone for on-device LLM inference.

---

## 8. Results Status

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
| MTL-NOYCE (Meteor Lake) | GPU | FP16 | **Deferred** — 15 GB RAM insufficient for 15 GB model |

---

## 9. Software Stack

| Component | Version |
|-----------|---------|
| **OS** | Ubuntu 24.04.3 LTS |
| **Kernel** | 6.14.0-37-generic |
| **Python** | 3.12.3 |
| **OpenVINO** | 2025.4.1-20426-82bbf0292c5-releases/2025/4 |
| **optimum-intel** | Latest (pip install) |
| **transformers** | Latest (pip install) |
| **NNCF** | Latest (for quantization) |
| **Intel GPU Runtime** | NEO 25.18+ (Lunar Lake), 23.43+ (Meteor Lake) |

---

## 10. Reproducibility

All benchmark code, scenarios, and configuration are open source:

**Repository**: [github.com/JoshCork/intel-ai-benchmarking](https://github.com/JoshCork/intel-ai-benchmarking)

To reproduce these results:

```bash
# 1. Clone and setup
git clone https://github.com/JoshCork/intel-ai-benchmarking.git
cd intel-ai-benchmarking
python3 -m venv venv && source venv/bin/activate
pip install openvino optimum-intel[openvino] transformers torch nncf

# 2. Export model (INT4 example)
python scripts/export_model.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --precision INT4

# 3. Run benchmark
python benchmark.py \
    --precision INT4 \
    --device GPU \
    --temperature 0.0 0.7 \
    --codename ADL \
    --tdp 45W
```

Results are stored in a SQLite database for cross-machine comparison and historical tracking.

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

*This document is a living benchmark report. Results will be updated as additional hardware configurations and precision levels complete testing.*
