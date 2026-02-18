# LLM Output Quality Comparison: Qwen2.5-7B vs Llama 3.1-8B on Intel Xe3

**Semantic Quality Analysis Across Models and Precision Levels**

*February 2026*

---

## Abstract

This paper presents a semantic quality comparison of two leading open-weight LLMs — Qwen2.5-7B-Instruct and Llama 3.1-8B-Instruct — running on Intel Xe3-LPG integrated GPU (Panther Lake H, 25W, DDR5-5600). We analyze actual generated responses from 7 retail kiosk scenarios across FP16, INT8, and INT4 precisions, including GPTQ-quantized variants.

**Key findings**: (1) Both models produce high-quality, contextually appropriate responses at all precision levels — quantization from FP16 to INT4 causes **no meaningful quality degradation** for kiosk use cases. (2) Qwen2.5-7B tends toward **more specific, actionable responses** (citing real brand names, offering concrete next steps), while Llama 3.1-8B is **more consistent and formulaic** across precisions. (3) GPTQ INT4 quality is indistinguishable from AWQ INT4. (4) The quality difference between models is **larger than the quality difference between precisions**, making model choice more important than precision choice for output quality.

---

## 1. Introduction

### 1.1 Motivation

Throughput benchmarks tell half the story. A model that generates 18 TPS but produces incoherent responses is worse than one at 14 TPS with excellent quality. For a retail kiosk deployment, response quality directly impacts customer satisfaction and brand perception.

Our optimization whitepaper established that Qwen2.5-7B-Instruct INT4 with the GenAI pipeline achieves **18.7 TPS / 65ms TTFT** — 38.5% faster than the Llama baseline. This paper asks: does that speed come at a quality cost?

### 1.2 Research Questions

1. **Quantization impact**: How much quality degrades as each model moves from FP16 → INT8 → INT4?
2. **Model comparison**: At INT4 (the deployment precision), does Qwen or Llama produce higher-quality responses?
3. **GPTQ vs AWQ**: Does GPTQ quantization (which gives 20% more throughput) affect output quality?
4. **Quality-speed tradeoff**: What is the optimal configuration balancing throughput and quality?

---

## 2. Methodology

### 2.1 Test Scenarios

All 7 kiosk conversation scenarios were evaluated, covering a range of complexity:

| Scenario | Type | Complexity | What It Tests |
|----------|------|-----------|---------------|
| greeting | single_turn | Low | Friendliness, brevity |
| store_hours | single_turn | Low | Factual accuracy, conciseness |
| product_lookup | single_turn | Medium | Product knowledge, helpfulness |
| return_policy | single_turn | Medium | Policy reasoning, empathy, problem-solving |
| loyalty_program | single_turn | Medium | Explanation clarity, persuasiveness |
| multi_turn_directions | multi_turn | Medium | Spatial reasoning, step clarity |
| multi_turn_troubleshoot | multi_turn | High | Diagnostic reasoning, solution quality |

### 2.2 Configurations Compared

| Label | Model | Precision | Quantization | Backend | TPS |
|-------|-------|-----------|-------------|---------|-----|
| Qwen-FP16 | Qwen2.5-7B-Instruct | FP16 | — | GenAI | 5.5 |
| Qwen-INT8 | Qwen2.5-7B-Instruct | INT8 | AWQ | GenAI | 10.9 |
| Qwen-INT4 | Qwen2.5-7B-Instruct | INT4 | AWQ | GenAI | 18.7 |
| Llama-FP16 | Llama 3.1-8B-Instruct | FP16 | — | GenAI | 5.1 |
| Llama-INT8 | Llama 3.1-8B-Instruct | INT8 | AWQ | GenAI | 10.1 |
| Llama-INT4-AWQ | Llama 3.1-8B-Instruct | INT4 | AWQ | GenAI | 14.7 |
| Llama-INT4-GPTQ | Llama 3.1-8B-Instruct | INT4 | GPTQ | GenAI | 17.6 |

All responses collected at temp=0.0 (greedy decoding) for deterministic comparison, plus temp=0.7 samples for the fastest INT4 configs.

### 2.3 Evaluation Criteria

Each response is assessed on:
- **Relevance**: Does it address the user's question?
- **Accuracy**: Are stated facts plausible and consistent?
- **Helpfulness**: Does it move the conversation forward productively?
- **Tone**: Is it appropriate for a retail kiosk (friendly, professional)?
- **Conciseness**: Is it appropriately brief (kiosk users want quick answers)?
- **Hallucination risk**: Does it fabricate specific details?

### 2.4 Data Source

Responses extracted from `run_metrics.response_text` in the benchmark database. Each configuration has 10 measured runs per scenario; we use run #4 (first measured run after warmup) for consistency.

---

## 3. Qwen2.5-7B-Instruct: Precision Degradation

### 3.1 FP16 Baseline Responses

| Scenario | FP16 Response | Tokens |
|----------|--------------|--------|
| greeting | "Of course! What do you need help with?" | 11 |
| store_hours | "Our store is open from 9 AM to 9 PM today. Is there anything else you need help with?" | 24 |
| product_lookup | "We have several options under $50. The JBL Tune 500TNC and the Anker Soundcore Liberty 2 Pro are both popular choices..." | 51 |
| return_policy | "I can help with that. Since you paid with a credit card, we can still look into your options. Typically, we offer returns or repairs for products that are defective within 30 days..." | 82 |
| loyalty_program | "To sign up for our loyalty program, simply scan your phone at the checkout or create an account using your phone number or email. Benefits include exclusive discounts, birthday rewards, and points..." | 66 |
| multi_turn_directions | "I can help you find a waterproof case for your iPhone 15 right away. They're usually located in the mobile accessories section..." | 52 |
| multi_turn_troubleshoot | "I see, let's try a few things. First, make sure the card is inserted properly. If that doesn't work, we can try another card or contact our support team..." | 41 |

**Baseline quality assessment**: Qwen FP16 responses are concise, helpful, and proactive. Notable strength: cites real brand names (JBL, Anker) in product_lookup, giving responses a more knowledgeable feel. Responses average 47 tokens — appropriately brief for kiosk interactions.

### 3.2 INT8 vs FP16

| Scenario | Change from FP16 | Quality Impact |
|----------|------------------|----------------|
| greeting | Identical | None |
| store_hours | Identical | None |
| product_lookup | Minor wording change ("Liberty 2 Pro" → "Liberty Air 2 Pro"), added "Let me check the current stock" | Negligible — slightly different but equally good |
| return_policy | Different approach: asks for name/date instead of explaining policy first | Neutral — equally valid strategy |
| loyalty_program | Minor wording: "at one of our service desks" instead of "using your phone number or email" | Negligible |
| multi_turn_directions | Slightly different routing: "Let's head to the electronics section" vs "They're usually located in the mobile accessories section" | Negligible |
| multi_turn_troubleshoot | "support desk" instead of "support team" | None |

**INT8 verdict: No quality degradation.** Responses are semantically equivalent to FP16 with only trivial word-level variations.

### 3.3 INT4 vs FP16

| Scenario | Change from FP16 | Quality Impact |
|----------|------------------|----------------|
| greeting | Adds proactive follow-up: "Are you looking for a specific product or have a question about store policies?" | **Improved** — more helpful |
| store_hours | "We're open" vs "Our store is open"; adds "How can I help you find something" | Neutral — equally good, slightly more conversational |
| product_lookup | Different products (JBL Flip 5 instead of Tune 500TNC); adds location info | Neutral — different hallucinated products but same quality pattern |
| return_policy | More detailed: mentions warranty, asks for transaction details | **Improved** — more thorough problem-solving |
| loyalty_program | Different approach: describes points system vs sign-up instructions | Neutral — both valid, different emphasis |
| multi_turn_directions | Nearly identical content and structure | None |
| multi_turn_troubleshoot | "inserted correctly" vs "inserted properly" | None |

**INT4 verdict: No quality degradation — arguably improved in some scenarios.** INT4 responses tend to be slightly longer (avg 51 tokens vs 47) and more detailed. The product names differ (hallucinated in both cases), but the response structure and helpfulness are equivalent or better.

### 3.4 Qwen Precision Summary

| Scenario | FP16 Quality | INT8 Δ | INT4 Δ | Notes |
|----------|-------------|--------|--------|-------|
| greeting | Good | = | **+** | INT4 adds helpful follow-up |
| store_hours | Good | = | = | Trivial wording changes only |
| product_lookup | Good | = | = | Different brand names, same quality |
| return_policy | Good | = | **+** | INT4 is more thorough |
| loyalty_program | Good | = | = | Different emphasis, same quality |
| multi_turn_directions | Good | = | = | Near-identical |
| multi_turn_troubleshoot | Good | = | = | Near-identical |

**= means equivalent quality, + means slight improvement**

---

## 4. Llama 3.1-8B-Instruct: Precision Degradation

### 4.1 FP16 Baseline Responses

| Scenario | FP16 Response | Tokens |
|----------|--------------|--------|
| greeting | "I'd be happy to help you with anything you need. What's on your mind? Are you looking for a specific product or do you have a question about something else?" | 36 |
| store_hours | "Our store hours today are 9:00 AM to 9:00 PM. We're open for the rest of the day, so you have plenty of time to shop with us." | 39 |
| product_lookup | "We have a selection of wireless earbuds from various brands, including our in-house brand, which are priced under $50. I can show you our current options, which include the 'EvoFit' earbuds for $29.99 and the 'SoundWave' earbuds for $39.99..." | 78 |
| return_policy | "I'd be happy to help you with that. Since you paid with a credit card, you can still return or exchange the laptop within 30 days of purchase. I can check our system to see if we have a record of your purchase..." | 59 |
| loyalty_program | "Our loyalty program is called 'Rewards Plus.' Members earn points for every dollar spent, and for every 100 points, you get $5 off your next purchase. As a new member, you'll receive 10% off your purchase today..." | 72 |
| multi_turn_directions | "We have a variety of waterproof cases for the iPhone 15 available in our phone accessories section, located on the second floor, just down the escalator..." | 61 |
| multi_turn_troubleshoot | "I'd be happy to help you with that. Let me try to troubleshoot the issue with your card. Can you please try swiping the card again..." | 48 |

**Baseline quality assessment**: Llama FP16 is more verbose (avg 56 tokens vs Qwen's 47), more formulaic ("I'd be happy to help"), and creates fictional brand names and prices. The responses are consistently warm and professional.

### 4.2 INT8 vs FP16

| Scenario | Change from FP16 | Quality Impact |
|----------|------------------|----------------|
| greeting | Truncated: drops the follow-up questions | Minor loss — less proactive |
| store_hours | Identical | None |
| product_lookup | Identical | None |
| return_policy | Identical | None |
| loyalty_program | Identical | None |
| multi_turn_directions | Identical | None |
| multi_turn_troubleshoot | Identical | None |

**INT8 verdict: Essentially identical to FP16.** Only greeting showed a minor truncation. Llama's INT8 is remarkably stable.

### 4.3 INT4 (AWQ) vs FP16

| Scenario | Change from FP16 | Quality Impact |
|----------|------------------|----------------|
| greeting | Shorter, simpler: "What can I assist you with today?" | Minor — less conversational but still appropriate |
| store_hours | Adds helpful offer: "If you need help finding anything, feel free to ask!" | **Improved** |
| product_lookup | Different fictional products (X-Tech, SonicBuds vs EvoFit, SoundWave) | Neutral — same hallucination pattern |
| return_policy | "return or repair process" instead of "return or exchange process" | Negligible |
| loyalty_program | More detailed benefits description | Neutral to improved |
| multi_turn_directions | More concise, gives specific aisle number | Neutral — different but equally helpful |
| multi_turn_troubleshoot | Offers to "reset the terminal" — more specific troubleshooting | **Improved** — more actionable |

### 4.4 INT4 (GPTQ) vs INT4 (AWQ)

| Scenario | GPTQ vs AWQ Difference | Quality Impact |
|----------|----------------------|----------------|
| greeting | Identical | None |
| store_hours | More concise (35 vs 49 tokens) | Neutral — appropriate for kiosk |
| product_lookup | Different fictional products, similar structure | None |
| return_policy | Adds warranty check offer | **Improved** — more thorough |
| loyalty_program | More detailed (mentions "5 points per dollar") | Slightly improved |
| multi_turn_directions | More descriptive, offers to show location | Slightly improved |
| multi_turn_troubleshoot | Asks about spending limit — more diagnostic | **Improved** |

**GPTQ INT4 verdict: Equal or slightly better than AWQ INT4.** GPTQ responses tend to be more specific and diagnostic, possibly because the scale-estimation calibration preserves more of the model's reasoning capacity.

### 4.5 Llama Precision Summary

| Scenario | FP16 Quality | INT8 Δ | INT4 AWQ Δ | INT4 GPTQ Δ | Notes |
|----------|-------------|--------|-----------|------------|-------|
| greeting | Good | ≈ | ≈ | ≈ | Shorter at INT4 but still appropriate |
| store_hours | Good | = | **+** | = | AWQ INT4 adds helpful offer |
| product_lookup | Good | = | = | = | Product names change, quality doesn't |
| return_policy | Good | = | = | **+** | GPTQ adds warranty check |
| loyalty_program | Good | = | = | ≈ | All variants are detailed and persuasive |
| multi_turn_directions | Good | = | = | = | Minor structural differences |
| multi_turn_troubleshoot | Good | = | **+** | **+** | INT4 variants are more actionable |

---

## 5. Head-to-Head: Qwen vs Llama

### 5.1 INT4 Comparison (Primary Deployment Configuration)

Using the fastest configuration for each model: Qwen INT4 AWQ GenAI (18.7 TPS) vs Llama INT4 GPTQ GenAI (17.6 TPS).

| Scenario | Winner | Analysis |
|----------|--------|----------|
| greeting | **Qwen** | More proactive — asks what type of help needed vs generic welcome |
| store_hours | **Tie** | Both accurate and concise; Qwen slightly more concise (26 vs 35 tokens) |
| product_lookup | **Qwen** | Cites real brand names (JBL, Anker) vs Llama's fictional brands — feels more knowledgeable |
| return_policy | **Qwen** | More thorough — mentions warranty, asks for verification details, offers concrete next steps |
| loyalty_program | **Llama** | More structured explanation with specific numbers ("5 points per dollar", "$5 off per 100 points") |
| multi_turn_directions | **Tie** | Both give clear directions with location details |
| multi_turn_troubleshoot | **Llama** | Better diagnostic approach — asks about spending limit, checks card itself |

**Overall: Qwen 3, Llama 2, Tie 2**

### 5.2 Qualitative Differences

**Qwen2.5-7B-Instruct strengths:**
- More concise (avg 51 tokens vs 56 for Llama at INT4)
- Cites real brand names, giving a more knowledgeable impression
- More proactive — volunteers useful follow-up information
- Better for multilingual deployments (not tested here, but documented in research)

**Llama 3.1-8B-Instruct strengths:**
- More consistent tone ("I'd be happy to help" opener is reliable and warm)
- Creates richer fictional detail (specific prices, program names like "Rewards Plus")
- Slightly better at structured troubleshooting and diagnostic questioning
- More verbose — which may be preferred for complex explanations

**Shared weakness — hallucination:**
Both models confidently cite product names, prices, and brand names that don't exist in the system prompt. This is expected behavior for 7-8B models without RAG, but highlights that **a production kiosk must use retrieval-augmented generation** to ground responses in actual inventory/policy data.

### 5.3 FP16 Comparison (Best-Case Quality)

At FP16, the same patterns hold: Qwen is more concise and specific, Llama is warmer and more verbose. The quality gap between models is **consistent across all precision levels**, confirming that the INT4 comparison above reflects genuine model differences, not quantization artifacts.

---

## 6. Quality-Speed Tradeoff Analysis

### 6.1 Combined Scorecard

| Configuration | TPS | TTFT (ms) | Quality | Best For |
|--------------|-----|-----------|---------|----------|
| **Qwen INT4 GenAI** | **18.7** | **65** | Excellent | **Fast kiosk with concise answers** |
| Llama GPTQ INT4 GenAI | 17.6 | 77 | Excellent | Kiosk with warm, detailed answers |
| Qwen INT8 GenAI | 10.9 | 103 | Excellent | No advantage over INT4 |
| Llama INT4 AWQ GenAI | 14.7 | 90 | Excellent | Llama without GPTQ export |
| Llama INT8 GenAI | 10.1 | 119 | Excellent | No advantage over INT4 |
| Qwen FP16 GenAI | 5.5 | 193 | Excellent | No advantage over INT4 |
| Llama FP16 GenAI | 5.1 | 215 | Excellent | No advantage over INT4 |

### 6.2 The Pareto Frontier

**Every configuration on this list produces excellent kiosk-quality responses.** There is no quality-speed tradeoff to make — INT4 quality is indistinguishable from FP16 quality in kiosk conversation scenarios. The Pareto frontier is simply:

1. **Qwen INT4 AWQ + GenAI** (18.7 TPS) — fastest, excellent quality
2. **Llama GPTQ INT4 + GenAI** (17.6 TPS) — if Llama is required

FP16 and INT8 occupy strictly dominated positions: slower speed, no quality benefit.

---

## 7. Discussion

### 7.1 Why Quantization Doesn't Hurt Quality Here

The consistent finding that INT4 ≈ INT8 ≈ FP16 in quality may seem surprising but is explained by several factors:

1. **Task simplicity**: Retail kiosk conversations are well within the capability envelope of 7-8B models. These are not complex reasoning, mathematical, or coding tasks where quantization typically shows degradation.

2. **Short context**: Our scenarios use 1-2 turns with modest context. Quantization-induced errors accumulate over longer contexts and more reasoning steps.

3. **Modern quantization**: Both AWQ and GPTQ with calibration are sophisticated quantization methods that minimize accuracy loss on common text generation patterns.

4. **7-8B model robustness**: At this scale, the models have enough parameter redundancy to absorb INT4 quantization without meaningful capability loss on straightforward tasks.

**Caveat**: This finding applies to kiosk-level conversations. For more complex tasks (multi-step reasoning, long-document QA, code generation), INT4 quality degradation would likely be measurable.

### 7.2 The Hallucination Problem

Both models hallucinate product names, prices, and policy details that aren't grounded in any provided context. Examples:

- Qwen invents "JBL Tune 500TNC" and "Anker Soundcore Liberty 2 Pro"
- Llama invents "EvoFit earbuds for $29.99" and "Rewards Plus" loyalty program

This is the **most important quality finding for production deployment**: raw LLM responses cannot be trusted for factual kiosk content. A production system must use RAG (retrieval-augmented generation) with a grounded knowledge base of actual products, prices, and policies.

Interestingly, Qwen's hallucinated product names (JBL, Anker) are real brands, while Llama invents entirely fictional brands. This makes Qwen's hallucinations more plausible but potentially more dangerous — a customer might expect to find that specific JBL model.

### 7.3 Model Choice > Precision Choice

The quality differences between Qwen and Llama (different response styles, different strengths per scenario) are **much larger** than the quality differences between FP16 and INT4 within the same model (trivial wording variations). This means:

- **Model selection** should be driven by which response style fits your brand
- **Precision selection** should be driven purely by throughput requirements
- **There is no reason to run FP16 or INT8** for kiosk deployments — INT4 is strictly better (faster, equal quality)

### 7.4 GPTQ Quality

GPTQ INT4 responses are equal or slightly better than AWQ INT4 across all scenarios. The scale-estimation calibration used in GPTQ may better preserve the model's reasoning pathways, leading to marginally more specific and diagnostic responses. Combined with its 20% throughput advantage, GPTQ is recommended over AWQ for Llama deployments.

---

## 8. Conclusions and Recommendations

1. **INT4 quantization causes no meaningful quality degradation** for retail kiosk conversations. Both Qwen2.5-7B and Llama 3.1-8B produce equally helpful, coherent, and appropriate responses at INT4 as at FP16.

2. **Qwen2.5-7B-Instruct is recommended for kiosk deployment**: faster (18.7 vs 17.6 TPS), more concise responses, and slightly better on most scenarios. Choose Llama if you prefer its warmer, more verbose style.

3. **Always use INT4** for kiosk deployment. FP16 and INT8 offer no quality advantage and are 2-3.5x slower.

4. **GPTQ is recommended over AWQ** for Llama deployments — 20% faster with equal or better quality.

5. **RAG is essential for production**: Both models hallucinate product and policy details. A grounded retrieval system is mandatory for any customer-facing deployment.

6. **The quality-speed tradeoff is a false dilemma** at this task complexity: you can have the fastest configuration and the best quality simultaneously.

---

## Appendix A: Full Response Comparison

### A.1 return_policy (most complex single-turn scenario)

**Qwen FP16**: "I can help with that. Since you paid with a credit card, we can still look into your options. Typically, we offer returns or repairs for products that are defective within 30 days of purchase. Since it's been 3 weeks, we can try to get it repaired first. Would you like to bring the laptop in for a check, or do you need more information on the process?"

**Qwen INT4**: "I can help with that. Since you paid with your credit card, we can check the transaction details to verify your purchase. For a laptop that's been crashing, we can either offer a repair or a return, depending on the warranty and our return policy. Let's check your transaction first. Can you provide your name or any other details to help me locate your purchase?"

**Llama FP16**: "I'd be happy to help you with that. Since you paid with a credit card, you can still return or exchange the laptop within 30 days of purchase. I can check our system to see if we have a record of your purchase and assist you with the return or exchange process."

**Llama GPTQ INT4**: "I'd be happy to help you with your laptop issue. Since you paid with your credit card, you can return it within 30 days of purchase. I can also check if the laptop is still under warranty, which would cover repairs. Would you like me to check the warranty status and assist with the return process?"

### A.2 product_lookup (tests product knowledge)

**Qwen FP16**: "We have several options under $50. The JBL Tune 500TNC and the Anker Soundcore Liberty 2 Pro are both popular choices with good sound quality and wireless features. Would you like more details on either model?"

**Qwen INT4**: "We have a few options under $50. The JBL Flip 5 Wireless Earbuds are a great choice with good sound quality and a compact design. They're currently in stock and available on the second floor near the electronics section. Would you like more details or help finding them?"

**Llama FP16**: "We have a selection of wireless earbuds from various brands, including our in-house brand, which are priced under $50. I can show you our current options, which include the 'EvoFit' earbuds for $29.99 and the 'SoundWave' earbuds for $39.99. Would you like me to show you where they're located?"

**Llama GPTQ INT4**: "We have a variety of wireless earbuds available. Our current selection includes the 'SoundWave' earbuds for $25 and the 'EvoFit' earbuds for $45. Would you like me to show you where they're located?"

### A.3 Response Extraction Query

```sql
SELECT r.experiment_name, r.model_name, r.model_precision, r.temperature,
       r.scenario_name, m.response_text, m.output_tokens
FROM run_metrics m
JOIN benchmark_runs r ON m.run_id = r.id
WHERE m.is_warmup = 0 AND m.run_number = 4
  AND r.experiment_name LIKE '%genai%'
ORDER BY r.model_name, r.model_precision, r.scenario_name;
```
