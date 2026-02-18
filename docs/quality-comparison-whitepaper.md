# LLM Output Quality Comparison: Qwen2.5-7B vs Llama 3.1-8B on Intel Xe3

**Semantic Quality Analysis Across Models and Precision Levels**

*February 2026*

---

## Abstract

*To be completed after all benchmark runs are finished and response logs are analyzed.*

This paper presents a semantic quality comparison of two leading open-weight LLMs — Qwen2.5-7B-Instruct and Llama 3.1-8B-Instruct — running on Intel Xe3-LPG integrated GPU. We analyze response quality across three dimensions: (1) head-to-head model comparison at each precision level, (2) quality degradation as each model moves from FP16 to INT8 to INT4, and (3) fitness for retail kiosk conversational use cases.

---

## 1. Introduction

### 1.1 Motivation

Throughput benchmarks tell half the story. A model that generates 18 TPS but produces incoherent responses is worse than one at 14 TPS with excellent quality. For a retail kiosk deployment, response quality directly impacts customer satisfaction and brand perception.

### 1.2 Research Questions

1. **Model comparison**: At equivalent precision (INT4), does Qwen2.5-7B or Llama 3.1-8B produce higher-quality responses for retail kiosk scenarios?
2. **Quantization impact**: How much quality degrades as each model moves from FP16 → INT8 → INT4?
3. **Quality-speed tradeoff**: What is the optimal model + precision combination when balancing throughput and quality?

---

## 2. Methodology

### 2.1 Test Scenarios

All 7 kiosk conversation scenarios are evaluated:

| Scenario | Type | Complexity | Key Quality Metrics |
|----------|------|-----------|-------------------|
| greeting | single_turn | Low | Friendliness, brevity, helpfulness |
| store_hours | single_turn | Low | Accuracy, conciseness |
| product_lookup | single_turn | Medium | Detail, relevance, completeness |
| return_policy | single_turn | Medium | Policy accuracy, empathy, problem-solving |
| loyalty_program | single_turn | Medium | Explanation clarity, persuasiveness |
| multi_turn_directions | multi_turn | Medium | Spatial reasoning, step clarity |
| multi_turn_troubleshoot | multi_turn | High | Diagnostic reasoning, solution quality |

### 2.2 Evaluation Criteria

Each response is evaluated on a 1-5 scale across these dimensions:

- **Relevance**: Does the response address the user's question?
- **Accuracy**: Are facts and details correct?
- **Completeness**: Does it cover all aspects of the query?
- **Conciseness**: Is it appropriately brief for a kiosk interaction?
- **Tone**: Is it friendly, professional, and appropriate for retail?
- **Coherence**: Is the response logically structured and readable?
- **Hallucination**: Does it fabricate information not in the prompt/context?

### 2.3 Comparison Framework

#### Cross-Model Comparison (at each precision)

For each scenario, compare Qwen vs Llama side-by-side:
- Same prompt, same system prompt, same temperature
- Evaluate which model produces the better response
- Note specific strengths/weaknesses of each

#### Precision Degradation (within each model)

For each model, compare FP16 → INT8 → INT4:
- FP16 serves as the "ground truth" baseline
- Measure semantic drift: does meaning change?
- Measure quality loss: does response get worse?
- Identify precision-sensitive scenarios

### 2.4 Data Source

Responses are captured in the benchmark database (`run_metrics.response_text`) during benchmark runs. Each scenario has 10 measured runs per configuration (model × precision × temperature).

---

## 3. Qwen2.5-7B-Instruct: Precision Degradation

### 3.1 FP16 Baseline Responses

*Sample responses from each scenario at FP16 to establish quality baseline.*

### 3.2 INT8 vs FP16

*Side-by-side comparison showing any quality differences.*

### 3.3 INT4 vs FP16

*Side-by-side comparison showing any quality differences.*

### 3.4 Qwen Precision Summary

| Scenario | FP16 Quality | INT8 Δ | INT4 Δ | Notes |
|----------|-------------|--------|--------|-------|
| greeting | | | | |
| store_hours | | | | |
| product_lookup | | | | |
| return_policy | | | | |
| loyalty_program | | | | |
| multi_turn_directions | | | | |
| multi_turn_troubleshoot | | | | |

---

## 4. Llama 3.1-8B-Instruct: Precision Degradation

### 4.1 FP16 Baseline Responses

*Sample responses from each scenario at FP16 to establish quality baseline.*

### 4.2 INT8 vs FP16

*Side-by-side comparison showing any quality differences.*

### 4.3 INT4 vs FP16

*Side-by-side comparison showing any quality differences.*

### 4.4 Llama Precision Summary

| Scenario | FP16 Quality | INT8 Δ | INT4 Δ | Notes |
|----------|-------------|--------|--------|-------|
| greeting | | | | |
| store_hours | | | | |
| product_lookup | | | | |
| return_policy | | | | |
| loyalty_program | | | | |
| multi_turn_directions | | | | |
| multi_turn_troubleshoot | | | | |

---

## 5. Head-to-Head: Qwen vs Llama

### 5.1 INT4 Comparison (Primary Deployment Configuration)

*Side-by-side responses for each scenario, with analysis of which model is stronger.*

| Scenario | Winner | Qwen Strengths | Llama Strengths | Notes |
|----------|--------|---------------|-----------------|-------|
| greeting | | | | |
| store_hours | | | | |
| product_lookup | | | | |
| return_policy | | | | |
| loyalty_program | | | | |
| multi_turn_directions | | | | |
| multi_turn_troubleshoot | | | | |

### 5.2 INT8 Comparison

*Same analysis at INT8 precision.*

### 5.3 FP16 Comparison

*Same analysis at FP16 precision — establishes each model's best-case quality.*

---

## 6. Quality-Speed Tradeoff Analysis

### 6.1 Combined Scorecard

| Configuration | TPS | TTFT (ms) | Avg Quality Score | Quality/TPS Ratio |
|--------------|-----|-----------|------------------|-------------------|
| Qwen INT4 GenAI | 18.8 | 65 | | |
| Qwen INT8 GenAI | 10.9 | 102 | | |
| Qwen FP16 GenAI | | | | |
| Llama INT4 GenAI | 14.7 | 90 | | |
| Llama INT8 GenAI | 10.1 | 119 | | |
| Llama FP16 GenAI | 5.1 | 215 | | |

### 6.2 Pareto Frontier

*Which configurations are on the efficiency frontier (best quality for a given speed)?*

---

## 7. Discussion

### 7.1 Model Architecture Impact on Quality

*How do Qwen's GQA (4 KV heads) and smaller parameter count affect response quality vs Llama's larger model?*

### 7.2 Quantization Sensitivity by Scenario Type

*Are some scenario types (e.g., multi-turn, complex reasoning) more sensitive to quantization than others?*

### 7.3 Temperature Effects on Quality

*How does temp=0.7 sampling affect quality consistency vs greedy decoding?*

---

## 8. Conclusions and Recommendations

*To be completed after analysis.*

---

## Appendix A: Response Extraction

```sql
-- Extract responses from benchmark database for analysis
SELECT
    r.model_name, r.model_precision, r.temperature,
    r.scenario_name, m.response_text, m.run_number
FROM benchmark_runs r
JOIN run_metrics m ON r.id = m.run_id
WHERE m.is_warmup = 0
  AND r.target_device = 'GPU'
  AND r.experiment_name LIKE '%genai%'
ORDER BY r.model_name, r.model_precision, r.scenario_name, m.run_number
LIMIT 1;
```
