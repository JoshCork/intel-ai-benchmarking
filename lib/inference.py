"""
Inference — OpenVINO LLM inference with per-token timing.

Wraps optimum-intel's OVModelForCausalLM for benchmarking.
Captures TTFT, per-token latency, total generation time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class InferenceResult:
    """Results from a single inference run."""

    prompt: str = ""
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0  # time to first token
    total_ms: float = 0.0  # total generation time
    tokens_per_sec: float = 0.0
    per_token_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
        }


class OpenVINOLLM:
    """OpenVINO LLM inference wrapper for benchmarking.

    Uses optimum-intel's OVModelForCausalLM under the hood.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "GPU",
        max_new_tokens: int = 256,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load the model and tokenizer. Call once before running benchmarks."""
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        print(f"Loading model from {self.model_path} on {self.device}...")
        t0 = time.perf_counter()

        self.model = OVModelForCausalLM.from_pretrained(
            str(self.model_path),
            device=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

        load_time = time.perf_counter() - t0
        print(f"  Model loaded in {load_time:.1f}s")

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> InferenceResult:
        """Run a single inference and capture timing metrics.

        For temperature=0, uses greedy decoding.
        For temperature>0, uses sampling.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        max_tokens = max_new_tokens or self.max_new_tokens

        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Tokenize using chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: concatenate messages
            parts = []
            for msg in messages:
                parts.append(f"{msg['role']}: {msg['content']}")
            parts.append("assistant:")
            input_text = "\n".join(parts)

        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        # Time the generation
        # For TTFT, we use the streamer callback
        first_token_time = None
        token_times = []

        class TimingStreamer:
            """Captures per-token timing."""

            def __init__(self):
                self.first_token_time = None
                self.token_times = []
                self.start_time = None

            def put(self, value):
                now = time.perf_counter()
                if self.first_token_time is None:
                    self.first_token_time = now
                self.token_times.append(now)

            def end(self):
                pass

        streamer = TimingStreamer()
        streamer.start_time = time.perf_counter()

        t_start = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs,
            streamer=streamer,
        )
        t_end = time.perf_counter()

        # Decode output (skip input tokens)
        new_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        output_len = len(new_tokens)

        # Compute metrics
        total_ms = (t_end - t_start) * 1000
        ttft_ms = (
            (streamer.first_token_time - t_start) * 1000
            if streamer.first_token_time
            else total_ms
        )

        # Per-token latencies
        per_token_ms = []
        times = streamer.token_times
        if times:
            per_token_ms.append((times[0] - t_start) * 1000)  # first token
            for i in range(1, len(times)):
                per_token_ms.append((times[i] - times[i - 1]) * 1000)

        tps = output_len / (total_ms / 1000) if total_ms > 0 else 0

        return InferenceResult(
            prompt=prompt,
            response=response_text,
            input_tokens=input_len,
            output_tokens=output_len,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_sec=tps,
            per_token_ms=per_token_ms,
        )
