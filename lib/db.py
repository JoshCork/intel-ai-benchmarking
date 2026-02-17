"""
Database — SQLite storage for benchmark results.

Central DB lives on friday-cork (Tailscale-accessible).
Each run stores: machine fingerprint, model config, per-run metrics, aggregates.
"""

from __future__ import annotations

import sqlite3
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .hardware import HardwareFingerprint


SCHEMA = """
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hostname TEXT NOT NULL,
    cpu_model TEXT,
    cpu_codename TEXT,
    cpu_cores INTEGER,
    cpu_threads INTEGER,
    gpu_model TEXT,
    gpu_type TEXT,
    npu_model TEXT,
    mem_total_gb REAL,
    mem_type TEXT,
    mem_speed_mt INTEGER,
    mem_channels INTEGER,
    os_name TEXT,
    os_kernel TEXT,
    openvino_version TEXT,
    python_version TEXT,
    extra TEXT,  -- JSON
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(hostname, cpu_model, gpu_model)
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL REFERENCES machines(id),
    model_name TEXT NOT NULL,
    model_precision TEXT NOT NULL,
    model_source TEXT,          -- "usb", "local", "download"
    scenario_name TEXT NOT NULL,
    scenario_type TEXT,         -- "greeting", "simple_task", "complex", "multi_turn"
    system_prompt TEXT,
    temperature REAL NOT NULL DEFAULT 0.0,
    max_new_tokens INTEGER DEFAULT 256,
    target_device TEXT,         -- "GPU", "GPU.0", "GPU.1", "CPU", "NPU"
    warmup_runs INTEGER DEFAULT 3,
    measured_runs INTEGER DEFAULT 10,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    notes TEXT,
    experiment_name TEXT
);

CREATE TABLE IF NOT EXISTS run_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id),
    run_number INTEGER NOT NULL,   -- 1-based within the benchmark run
    is_warmup INTEGER NOT NULL DEFAULT 0,
    -- Timing
    ttft_ms REAL,              -- Time to first token
    total_ms REAL,             -- Total generation time
    -- Tokens
    input_tokens INTEGER,
    output_tokens INTEGER,
    -- Throughput
    tokens_per_sec REAL,       -- output_tokens / (total_ms / 1000)
    -- Snapshots
    prompt_text TEXT,
    response_text TEXT,
    -- Raw
    raw_metrics TEXT            -- JSON blob for any extra per-run data
);

CREATE TABLE IF NOT EXISTS run_aggregates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id) UNIQUE,
    -- TTFT aggregates (ms)
    ttft_mean REAL,
    ttft_median REAL,
    ttft_p5 REAL,
    ttft_p95 REAL,
    ttft_stddev REAL,
    -- Throughput aggregates (tokens/sec)
    tps_mean REAL,
    tps_median REAL,
    tps_p5 REAL,
    tps_p95 REAL,
    tps_stddev REAL,
    -- Total latency aggregates (ms)
    total_mean REAL,
    total_median REAL,
    total_p5 REAL,
    total_p95 REAL,
    total_stddev REAL,
    -- Summary
    total_measured_runs INTEGER,
    avg_output_tokens REAL
);

CREATE TABLE IF NOT EXISTS system_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL REFERENCES machines(id),
    captured_at TEXT NOT NULL,
    perfspect_version TEXT,
    -- Key fields extracted for easy querying
    scaling_governor TEXT,
    energy_perf_bias TEXT,
    turbo_boost TEXT,
    c_states TEXT,              -- comma-separated: "POLL,C1_ACPI,C2_ACPI,C3_ACPI"
    installed_memory TEXT,      -- e.g. "32GB (8x4GB LPDDR5 8533MT/s)"
    bios_version TEXT,
    kernel_version TEXT,
    -- Full data
    perfspect_json TEXT,        -- complete PerfSpect JSON for reproducibility
    insights_json TEXT          -- PerfSpect recommendations
);

CREATE INDEX IF NOT EXISTS idx_run_metrics_run_id ON run_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_machine ON benchmark_runs(machine_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_model ON benchmark_runs(model_name, model_precision);
CREATE INDEX IF NOT EXISTS idx_system_configs_machine ON system_configs(machine_id);
"""

# Migration for existing DBs — adds system_config_id to benchmark_runs
MIGRATIONS = [
    """ALTER TABLE benchmark_runs ADD COLUMN system_config_id INTEGER
       REFERENCES system_configs(id)""",
    """ALTER TABLE benchmark_runs ADD COLUMN experiment_name TEXT""",
]


class BenchmarkDB:
    """SQLite database for benchmark results."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    @staticmethod
    def connect(primary_path: str, fallback_path: str = "results/benchmarks.db") -> "BenchmarkDB":
        """Connect to DB with fallback — try primary, fall back to local on failure."""
        primary = Path(primary_path).expanduser()
        try:
            primary.parent.mkdir(parents=True, exist_ok=True)
            db = BenchmarkDB(str(primary))
            # Verify we can write (catches permission errors, read-only FS, etc.)
            db.conn.execute("SELECT 1")
            print(f"  Database: {primary} (primary)")
            return db
        except (PermissionError, OSError, sqlite3.OperationalError) as e:
            fallback = Path(fallback_path).expanduser()
            print(f"  Database: {primary} unavailable ({type(e).__name__})")
            print(f"  Falling back to: {fallback}")
            return BenchmarkDB(str(fallback))

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._run_migrations()

    def _run_migrations(self):
        """Apply pending migrations (idempotent — skips already-applied)."""
        for sql in MIGRATIONS:
            try:
                self.conn.execute(sql)
                self.conn.commit()
            except sqlite3.OperationalError:
                # Already applied (e.g. column already exists)
                pass

    def close(self):
        self.conn.close()

    # -- Machines --

    def upsert_machine(self, hw: HardwareFingerprint) -> int:
        """Insert or find existing machine, return machine_id."""
        # Check for existing
        row = self.conn.execute(
            "SELECT id FROM machines WHERE hostname=? AND cpu_model=? AND gpu_model=?",
            (hw.hostname, hw.cpu_model, hw.gpu_model),
        ).fetchone()
        if row:
            return row["id"]

        cursor = self.conn.execute(
            """INSERT INTO machines
            (hostname, cpu_model, cpu_codename, cpu_cores, cpu_threads,
             gpu_model, gpu_type, npu_model,
             mem_total_gb, mem_type, mem_speed_mt, mem_channels,
             os_name, os_kernel, openvino_version, python_version, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hw.hostname, hw.cpu_model, hw.cpu_codename,
                hw.cpu_cores, hw.cpu_threads,
                hw.gpu_model, hw.gpu_type, hw.npu_model,
                hw.mem_total_gb, hw.mem_type, hw.mem_speed_mt, hw.mem_channels,
                hw.os_name, hw.os_kernel,
                hw.openvino_version, hw.python_version,
                json.dumps(hw.extra) if hw.extra else None,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    # -- Benchmark Runs --

    def create_run(
        self,
        machine_id: int,
        model_name: str,
        model_precision: str,
        scenario_name: str,
        scenario_type: str = "",
        system_prompt: str = "",
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        target_device: str = "GPU",
        warmup_runs: int = 3,
        measured_runs: int = 10,
        model_source: str = "",
        notes: str = "",
        system_config_id: Optional[int] = None,
        experiment_name: str = "",
    ) -> int:
        """Create a new benchmark run, return run_id."""
        cursor = self.conn.execute(
            """INSERT INTO benchmark_runs
            (machine_id, model_name, model_precision, model_source,
             scenario_name, scenario_type, system_prompt,
             temperature, max_new_tokens, target_device,
             warmup_runs, measured_runs, started_at, notes,
             system_config_id, experiment_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                machine_id, model_name, model_precision, model_source,
                scenario_name, scenario_type, system_prompt,
                temperature, max_new_tokens, target_device,
                warmup_runs, measured_runs,
                datetime.now(timezone.utc).isoformat(),
                notes,
                system_config_id,
                experiment_name or None,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def finish_run(self, run_id: int):
        """Mark a run as finished."""
        self.conn.execute(
            "UPDATE benchmark_runs SET finished_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), run_id),
        )
        self.conn.commit()

    # -- Run Metrics --

    def add_metric(
        self,
        run_id: int,
        run_number: int,
        is_warmup: bool = False,
        ttft_ms: float = 0.0,
        total_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tokens_per_sec: float = 0.0,
        prompt_text: str = "",
        response_text: str = "",
        raw_metrics: Optional[dict] = None,
    ):
        """Record a single run's metrics."""
        self.conn.execute(
            """INSERT INTO run_metrics
            (run_id, run_number, is_warmup,
             ttft_ms, total_ms, input_tokens, output_tokens, tokens_per_sec,
             prompt_text, response_text, raw_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, run_number, int(is_warmup),
                ttft_ms, total_ms, input_tokens, output_tokens, tokens_per_sec,
                prompt_text, response_text,
                json.dumps(raw_metrics) if raw_metrics else None,
            ),
        )
        self.conn.commit()

    # -- Aggregates --

    def save_aggregates(self, run_id: int, agg: dict):
        """Store pre-computed aggregates for a run."""
        self.conn.execute(
            """INSERT OR REPLACE INTO run_aggregates
            (run_id,
             ttft_mean, ttft_median, ttft_p5, ttft_p95, ttft_stddev,
             tps_mean, tps_median, tps_p5, tps_p95, tps_stddev,
             total_mean, total_median, total_p5, total_p95, total_stddev,
             total_measured_runs, avg_output_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                agg.get("ttft_mean"), agg.get("ttft_median"),
                agg.get("ttft_p5"), agg.get("ttft_p95"), agg.get("ttft_stddev"),
                agg.get("tps_mean"), agg.get("tps_median"),
                agg.get("tps_p5"), agg.get("tps_p95"), agg.get("tps_stddev"),
                agg.get("total_mean"), agg.get("total_median"),
                agg.get("total_p5"), agg.get("total_p95"), agg.get("total_stddev"),
                agg.get("total_measured_runs"), agg.get("avg_output_tokens"),
            ),
        )
        self.conn.commit()

    # -- System Configs --

    def save_system_config(
        self,
        machine_id: int,
        perfspect_data: dict,
    ) -> int:
        """Store a PerfSpect system config snapshot, return config_id."""
        cursor = self.conn.execute(
            """INSERT INTO system_configs
            (machine_id, captured_at, perfspect_version,
             scaling_governor, energy_perf_bias, turbo_boost, c_states,
             installed_memory, bios_version, kernel_version,
             perfspect_json, insights_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                machine_id,
                datetime.now(timezone.utc).isoformat(),
                perfspect_data.get("perfspect_version", ""),
                perfspect_data.get("scaling_governor", ""),
                perfspect_data.get("energy_perf_bias", ""),
                perfspect_data.get("turbo_boost", ""),
                perfspect_data.get("c_states", ""),
                perfspect_data.get("installed_memory", ""),
                perfspect_data.get("bios_version", ""),
                perfspect_data.get("kernel_version", ""),
                json.dumps(perfspect_data.get("full_json")) if perfspect_data.get("full_json") else None,
                json.dumps(perfspect_data.get("insights")) if perfspect_data.get("insights") else None,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_latest_config(self, machine_id: int, max_age_hours: int = 24) -> Optional[int]:
        """Get the latest system config for a machine if captured within max_age_hours."""
        row = self.conn.execute(
            """SELECT id, captured_at FROM system_configs
               WHERE machine_id = ?
               ORDER BY captured_at DESC LIMIT 1""",
            (machine_id,),
        ).fetchone()
        if not row:
            return None

        captured = datetime.fromisoformat(row["captured_at"])
        now = datetime.now(timezone.utc)
        # Handle naive timestamps (old data without timezone)
        if captured.tzinfo is None:
            from datetime import timezone as tz
            captured = captured.replace(tzinfo=tz.utc)
        age_hours = (now - captured).total_seconds() / 3600
        if age_hours <= max_age_hours:
            return row["id"]
        return None

    # -- Queries --

    def get_runs_for_model(self, model_name: str) -> list[dict]:
        """Get all benchmark runs for a given model."""
        rows = self.conn.execute(
            """SELECT br.*, m.hostname, m.cpu_codename, m.gpu_model,
                      ra.tps_mean, ra.ttft_mean, ra.total_mean
               FROM benchmark_runs br
               JOIN machines m ON br.machine_id = m.id
               LEFT JOIN run_aggregates ra ON br.id = ra.run_id
               WHERE br.model_name = ?
               ORDER BY br.started_at DESC""",
            (model_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_comparison_table(self) -> list[dict]:
        """Get a cross-machine comparison of all benchmarks."""
        rows = self.conn.execute(
            """SELECT m.hostname, m.cpu_codename, m.gpu_model,
                      br.model_name, br.model_precision, br.temperature,
                      br.target_device, br.scenario_name,
                      br.experiment_name,
                      ra.tps_mean, ra.tps_median, ra.tps_p95,
                      ra.ttft_mean, ra.ttft_median,
                      ra.total_mean, ra.total_measured_runs
               FROM run_aggregates ra
               JOIN benchmark_runs br ON ra.run_id = br.id
               JOIN machines m ON br.machine_id = m.id
               ORDER BY br.model_name, br.model_precision, m.hostname"""
        ).fetchall()
        return [dict(r) for r in rows]
