#!/usr/bin/env python3
"""
Intel AI Benchmarking — Main CLI

Orchestrates: detect hardware → find model → warmup → measure → store results.

Usage:
  # Run all scenarios with default settings
  python benchmark.py

  # Run specific scenario at specific precision
  python benchmark.py --scenario greeting --precision INT4

  # Run with both temperatures
  python benchmark.py --temperature 0.0 0.7

  # Run on specific device
  python benchmark.py --device GPU.1

  # Dry run (show what would be benchmarked)
  python benchmark.py --dry-run

  # Show results table
  python benchmark.py --results

  # Capture PerfSpect system config before benchmarking
  python benchmark.py --perfspect

  # Just capture PerfSpect config (no benchmarking)
  python benchmark.py --perfspect-only

  # Load an existing PerfSpect JSON report
  python benchmark.py --perfspect-json ~/perfspect/perfspect_2026-02-10/LNL-GROVE.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import toml

from lib.hardware import detect_hardware
from lib.model_loader import find_model
from lib.inference import OpenVINOLLM
from lib.db import BenchmarkDB
from lib.metrics import compute_aggregates, format_comparison_table
from lib.perfspect import (
    find_perfspect, run_perfspect, find_latest_report,
    parse_perfspect_json, load_existing_report,
)
from scenarios.kiosk import ALL_SCENARIOS, get_scenario


def load_config(config_path: str = "config.toml") -> dict:
    """Load config, merging local overrides if present."""
    base = Path(config_path)
    if base.exists():
        config = toml.load(base)
    else:
        print(f"  WARNING: Config not found at {base.resolve()}, using defaults")
        config = {}

    local = base.with_suffix(".local.toml")
    if local.exists():
        local_config = toml.load(local)
        # Deep merge
        for key, value in local_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    return config


def run_benchmark(
    llm: OpenVINOLLM,
    scenario_name: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    conversation_history: list[dict] | None,
    warmup_runs: int,
    measured_runs: int,
    db: BenchmarkDB,
    run_id: int,
):
    """Execute a benchmark: warmup + measured runs."""
    total_runs = warmup_runs + measured_runs
    metrics_list = []

    for i in range(total_runs):
        is_warmup = i < warmup_runs
        run_num = i + 1
        phase = "warmup" if is_warmup else "measured"
        measured_num = 0 if is_warmup else (i - warmup_runs + 1)

        print(
            f"  [{phase} {run_num}/{total_runs}] ",
            end="", flush=True,
        )

        # For multi-turn, build full prompt from conversation history
        actual_prompt = prompt
        if conversation_history and len(conversation_history) > 1:
            # Use the last user message as the prompt
            # Previous turns are context
            actual_prompt = conversation_history[-1]["content"]

        result = llm.generate(
            prompt=actual_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        print(
            f"TTFT={result.ttft_ms:.0f}ms  "
            f"TPS={result.tokens_per_sec:.1f}  "
            f"Total={result.total_ms:.0f}ms  "
            f"Tokens={result.output_tokens}"
        )

        # Store in DB
        db.add_metric(
            run_id=run_id,
            run_number=run_num,
            is_warmup=is_warmup,
            ttft_ms=result.ttft_ms,
            total_ms=result.total_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            tokens_per_sec=result.tokens_per_sec,
            prompt_text=actual_prompt,
            response_text=result.response,
            raw_metrics={"per_token_ms": result.per_token_ms[:5]},  # first 5 only
        )

        if not is_warmup:
            metrics_list.append(result.to_dict())

    return metrics_list


def main():
    parser = argparse.ArgumentParser(description="Intel AI Benchmarking")
    parser.add_argument("--config", default="config.toml", help="Config file path")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument(
        "--precision", nargs="+",
        help="Precision(s) to test (e.g. FP16 INT4)"
    )
    parser.add_argument(
        "--scenario", nargs="+",
        help="Scenario name(s) to run (default: all)"
    )
    parser.add_argument(
        "--temperature", nargs="+", type=float,
        help="Temperature(s) to test (e.g. 0.0 0.7)"
    )
    parser.add_argument("--device", default="GPU", help="Target device (GPU, CPU, NPU)")
    parser.add_argument("--warmup", type=int, help="Override warmup runs")
    parser.add_argument("--runs", type=int, help="Override measured runs")
    parser.add_argument("--max-tokens", type=int, help="Override max new tokens")
    parser.add_argument("--codename", default="", help="CPU codename (ADL, MTL, LNL)")
    parser.add_argument("--tdp", default="", help="TDP class (e.g. 45W)")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--no-download", action="store_true", help="Don't download models")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument("--results", action="store_true", help="Show results table and exit")
    parser.add_argument("--notes", default="", help="Notes for this benchmark run")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name to group runs (default: auto-generated from codename + timestamp)"
    )
    parser.add_argument(
        "--perfspect", action="store_true",
        help="Capture PerfSpect system config before benchmarking"
    )
    parser.add_argument(
        "--perfspect-only", action="store_true",
        help="Capture PerfSpect config and exit (no benchmarking)"
    )
    parser.add_argument(
        "--perfspect-json", type=str,
        help="Load an existing PerfSpect JSON report instead of running live"
    )
    parser.add_argument(
        "--ov-config", type=str, default=None,
        help="OpenVINO runtime config flags (format: KEY=VALUE,KEY=VALUE). "
             "Example: KV_CACHE_PRECISION=u8,DYNAMIC_QUANTIZATION_GROUP_SIZE=64,PERFORMANCE_HINT=LATENCY"
    )
    parser.add_argument(
        "--model-suffix", type=str, default="",
        help="Model directory suffix (e.g. '-gptq' to use Llama-3.1-8B-Instruct-INT4-gptq)"
    )
    parser.add_argument(
        "--backend", type=str, default="optimum",
        choices=["optimum", "genai"],
        help="Inference backend: 'optimum' (default, optimum-intel) or 'genai' (openvino-genai LLMPipeline)"
    )
    args = parser.parse_args()

    # Parse ov_config string into dict
    ov_config = None
    if args.ov_config:
        ov_config = {}
        for pair in args.ov_config.split(","):
            key, value = pair.strip().split("=", 1)
            ov_config[key.strip()] = value.strip()

    # Load config
    config = load_config(args.config)
    bench_cfg = config.get("benchmark", {})
    model_cfg = config.get("model", {})
    db_cfg = config.get("database", {})

    # Resolve parameters (CLI overrides config)
    model_name = args.model or model_cfg.get("name", "meta-llama/Llama-3.1-8B-Instruct")
    precisions = args.precision or model_cfg.get("precisions", ["FP16"])
    temperatures = args.temperature or bench_cfg.get("temperatures", [0.0])
    warmup_runs = args.warmup or bench_cfg.get("warmup_runs", 3)
    measured_runs = args.runs or bench_cfg.get("measured_runs", 10)
    max_tokens = args.max_tokens or bench_cfg.get("max_new_tokens", 256)
    system_prompt = config.get("system_prompt", {}).get("text", "")
    if not system_prompt:
        print("  WARNING: No system prompt configured — model will use default behavior")
        print("           (Set [system_prompt] text in config.toml)")

    # Experiment name — auto-generate if not provided
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.experiment:
        experiment_name = args.experiment
    elif args.codename:
        experiment_name = f"{args.codename}-{timestamp}"
    else:
        experiment_name = f"run-{timestamp}"

    # Database — CLI override uses direct path, otherwise try primary with fallback
    if args.db:
        db_path = args.db
        db = BenchmarkDB(db_path)
    else:
        primary = db_cfg.get("path", "results/benchmarks.db")
        fallback = db_cfg.get("local_path", "results/benchmarks.db")
        db = BenchmarkDB.connect(primary, fallback)
        db_path = str(db.db_path)

    # Results mode
    if args.results:
        rows = db.get_comparison_table()
        print(format_comparison_table(rows))
        db.close()
        return

    # Scenarios
    if args.scenario:
        scenarios = [get_scenario(s) for s in args.scenario]
    else:
        scenarios = ALL_SCENARIOS

    # Hardware detection
    print("=== Hardware Detection ===")
    hw = detect_hardware(codename=args.codename, tdp_class=args.tdp)
    print(f"  Host: {hw.hostname}")
    print(f"  CPU: {hw.cpu_model} ({hw.cpu_codename or 'unknown'})")
    print(f"  GPU: {hw.gpu_model} ({hw.gpu_type})")
    print(f"  NPU: {hw.npu_model or 'none'}")
    print(f"  RAM: {hw.mem_total_gb}GB {hw.mem_type}")
    print(f"  OpenVINO: {hw.openvino_version}")
    print()

    machine_id = db.upsert_machine(hw)

    # PerfSpect system config capture
    system_config_id = None
    if args.perfspect or args.perfspect_only or args.perfspect_json:
        print("=== PerfSpect System Config ===")
        if args.perfspect_json:
            # Load existing report
            print(f"  Loading report: {args.perfspect_json}")
            perfspect_data = load_existing_report(args.perfspect_json)
        else:
            # Run PerfSpect live
            perfspect = find_perfspect()
            if perfspect:
                json_path = run_perfspect()
                if json_path:
                    print(f"  Report: {json_path}")
                    perfspect_data = parse_perfspect_json(json_path)
                else:
                    print("  PerfSpect run failed — continuing without config")
                    perfspect_data = None
            else:
                print("  PerfSpect not installed at ~/perfspect/ — skipping")
                perfspect_data = None

        if perfspect_data:
            system_config_id = db.save_system_config(machine_id, perfspect_data)
            print(f"  Config saved (id={system_config_id})")
            gov = perfspect_data.get("scaling_governor", "unknown")
            mem = perfspect_data.get("installed_memory", "unknown")
            print(f"  Governor: {gov}")
            print(f"  Memory: {mem}")
            if perfspect_data.get("insights"):
                print(f"  Insights: {len(perfspect_data['insights'])} recommendations")
        print()

        if args.perfspect_only:
            print("(perfspect-only mode — exiting)")
            db.close()
            return
    else:
        # Check if we have a recent config for this machine (within 24h)
        system_config_id = db.get_latest_config(machine_id)
        if system_config_id:
            print(f"  Using recent system config (id={system_config_id})")

    # Plan
    total_combos = len(precisions) * len(temperatures) * len(scenarios)
    total_runs = total_combos * (warmup_runs + measured_runs)
    print("=== Benchmark Plan ===")
    print(f"  Experiment: {experiment_name}")
    print(f"  Model: {model_name}")
    print(f"  Precisions: {precisions}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Scenarios: {[s.name for s in scenarios]}")
    print(f"  Device: {args.device}")
    print(f"  Backend: {args.backend}")
    if args.model_suffix:
        print(f"  Model suffix: {args.model_suffix}")
    if ov_config:
        print(f"  ov_config: {ov_config}")
    print(f"  Per scenario: {warmup_runs} warmup + {measured_runs} measured")
    print(f"  Total combinations: {total_combos}")
    print(f"  Total inference runs: {total_runs}")
    print()

    if args.dry_run:
        print("(dry run — exiting)")
        db.close()
        return

    # Run benchmarks
    for precision in precisions:
        print(f"\n{'='*60}")
        print(f"=== Precision: {precision} ===")
        print(f"{'='*60}")

        # Find model
        model_info = find_model(
            model_name, precision,
            usb_folder=model_cfg.get("usb_model_dir", "intel-ai-models"),
            local_dir=model_cfg.get("local_model_dir", "~/models/intel-bench"),
            auto_download=not args.no_download,
            suffix=args.model_suffix,
        )
        print(f"  Model source: {model_info.source}")
        print(f"  Model path: {model_info.path}")

        # Load model with selected backend
        if args.backend == "genai":
            from lib.inference import OpenVINOGenAILLM
            llm = OpenVINOGenAILLM(
                model_path=str(model_info.path),
                device=args.device,
                max_new_tokens=max_tokens,
                ov_config=ov_config,
            )
        else:
            llm = OpenVINOLLM(
                model_path=str(model_info.path),
                device=args.device,
                max_new_tokens=max_tokens,
                ov_config=ov_config,
            )
        llm.load()

        for temperature in temperatures:
            for scenario in scenarios:
                print(f"\n--- {scenario.name} | temp={temperature} | {precision} ---")
                print(f"  Prompt: {scenario.prompt[:80]}...")

                # Build notes with backend/ov_config info
                run_notes = args.notes
                if args.backend != "optimum":
                    run_notes = f"backend: {args.backend}" + (f"; {run_notes}" if run_notes else "")
                if ov_config:
                    config_str = ", ".join(f"{k}={v}" for k, v in ov_config.items())
                    run_notes = f"ov_config: {config_str}" + (f"; {run_notes}" if run_notes else "")

                run_id = db.create_run(
                    machine_id=machine_id,
                    model_name=model_name,
                    model_precision=precision,
                    scenario_name=scenario.name,
                    scenario_type=scenario.type,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    target_device=args.device,
                    warmup_runs=warmup_runs,
                    measured_runs=measured_runs,
                    model_source=model_info.source,
                    notes=run_notes,
                    system_config_id=system_config_id,
                    experiment_name=experiment_name,
                )

                metrics = run_benchmark(
                    llm=llm,
                    scenario_name=scenario.name,
                    prompt=scenario.prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    conversation_history=(
                        scenario.conversation_history
                        if scenario.type == "multi_turn" else None
                    ),
                    warmup_runs=warmup_runs,
                    measured_runs=measured_runs,
                    db=db,
                    run_id=run_id,
                )

                # Compute and store aggregates
                agg = compute_aggregates(metrics)
                if agg:
                    db.save_aggregates(run_id, agg)
                    print(f"\n  Summary: TPS={agg.get('tps_mean', 0):.1f} mean "
                          f"({agg.get('tps_p5', 0):.1f}-{agg.get('tps_p95', 0):.1f} p5-p95)  "
                          f"TTFT={agg.get('ttft_mean', 0):.0f}ms")

                db.finish_run(run_id)

    # Final summary
    print(f"\n{'='*60}")
    print("=== Final Results ===")
    print(f"{'='*60}\n")
    rows = db.get_comparison_table()
    print(format_comparison_table(rows))

    db.close()
    print(f"\nResults saved to: {db_path}")


if __name__ == "__main__":
    main()
