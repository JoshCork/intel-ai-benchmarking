#!/usr/bin/env python3
"""
Database Sync — Consolidate benchmark results from remote NUCs into a central DB.

Pulls SQLite DBs from each NUC via SCP, then merges rows into the target DB
with proper ID remapping (each NUC has its own auto-increment IDs).

Deduplication: Uses (hostname, started_at) as a natural key for benchmark_runs.
Rows already present in the target are skipped.

Usage:
  # Sync from all configured machines to Friday's central DB
  python scripts/sync_db.py

  # Sync to a local consolidated DB
  python scripts/sync_db.py --target results/consolidated.db

  # Sync from specific machines only
  python scripts/sync_db.py --machines grove noyce

  # Dry run — show what would be synced
  python scripts/sync_db.py --dry-run

  # Merge local DB files (no SSH)
  python scripts/sync_db.py --local-files grove.db noyce.db --target consolidated.db
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db import BenchmarkDB, SCHEMA, MIGRATIONS


# Machine configs — SSH access info
MACHINES = {
    "friday": {
        "host": "100.91.92.97",
        "user": "friday",
        "db_path": "/home/friday/intel-bench/results/benchmarks.db",
    },
    "grove": {
        "host": "100.106.21.108",
        "user": "grove",
        "db_path": "/home/grove/intel-bench/results/benchmarks.db",
    },
    "noyce": {
        "host": "100.112.0.11",
        "user": "noyce",
        "db_path": "/home/noyce/intel-bench/results/benchmarks.db",
    },
    "koduri": {
        "host": "100.92.51.84",
        "user": "koduri",
        "db_path": "/home/koduri/intel-bench/repo/results/benchmarks.db",
    },
}


def scp_db(machine: dict, dest_path: str, timeout: int = 30) -> bool:
    """SCP a database file from a remote machine."""
    remote = f"{machine['user']}@{machine['host']}:{machine['db_path']}"
    print(f"  Fetching: {remote}")
    try:
        result = subprocess.run(
            ["scp", "-o", f"ConnectTimeout={timeout}", "-o", "BatchMode=yes",
             remote, dest_path],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        if result.returncode != 0:
            print(f"  SCP failed: {result.stderr.strip()}")
            return False
        size = os.path.getsize(dest_path)
        print(f"  Downloaded: {size:,} bytes")
        return True
    except subprocess.TimeoutExpired:
        print(f"  SCP timed out ({timeout}s)")
        return False
    except FileNotFoundError:
        print("  scp not found")
        return False


def get_source_stats(db_path: str) -> dict:
    """Get row counts from a source DB."""
    conn = sqlite3.connect(db_path)
    stats = {}
    for table in ["machines", "benchmark_runs", "run_metrics", "run_aggregates", "system_configs"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        except sqlite3.OperationalError:
            stats[table] = 0
    conn.close()
    return stats


def merge_db(source_path: str, target_conn: sqlite3.Connection, dry_run: bool = False) -> dict:
    """Merge rows from source DB into target DB with ID remapping.

    Returns stats dict with counts of inserted rows per table.
    """
    src = sqlite3.connect(source_path)
    src.row_factory = sqlite3.Row
    target_conn.row_factory = sqlite3.Row

    stats = {"machines": 0, "benchmark_runs": 0, "run_metrics": 0,
             "run_aggregates": 0, "system_configs": 0, "skipped_runs": 0}

    # Step 1: Upsert machines — build ID mapping (source_id → target_id)
    machine_map = {}
    src_machines = src.execute("SELECT * FROM machines").fetchall()

    for m in src_machines:
        m = dict(m)
        src_id = m.pop("id")
        # Check if machine exists in target
        existing = target_conn.execute(
            "SELECT id FROM machines WHERE hostname=? AND cpu_model=? AND gpu_model=?",
            (m["hostname"], m["cpu_model"], m["gpu_model"]),
        ).fetchone()

        if existing:
            machine_map[src_id] = existing["id"]
        else:
            if dry_run:
                machine_map[src_id] = -1  # placeholder
                stats["machines"] += 1
                print(f"    Would insert machine: {m['hostname']}")
                continue

            cols = [k for k in m.keys() if k != "id"]
            placeholders = ", ".join(["?"] * len(cols))
            col_names = ", ".join(cols)
            cursor = target_conn.execute(
                f"INSERT INTO machines ({col_names}) VALUES ({placeholders})",
                [m[c] for c in cols],
            )
            machine_map[src_id] = cursor.lastrowid
            stats["machines"] += 1
            print(f"    Inserted machine: {m['hostname']} (id={cursor.lastrowid})")

    # Step 2: Get existing runs in target (for dedup)
    existing_runs = set()
    for row in target_conn.execute(
        "SELECT machine_id, started_at FROM benchmark_runs"
    ).fetchall():
        existing_runs.add((row["machine_id"], row["started_at"]))

    # Step 3: Insert benchmark_runs with ID remapping
    run_map = {}  # source_run_id → target_run_id
    src_runs = src.execute("SELECT * FROM benchmark_runs ORDER BY id").fetchall()

    for r in src_runs:
        r = dict(r)
        src_run_id = r.pop("id")
        src_machine_id = r["machine_id"]

        # Remap machine_id
        target_machine_id = machine_map.get(src_machine_id)
        if target_machine_id is None:
            print(f"    WARNING: No machine mapping for source machine_id={src_machine_id}, skipping run")
            stats["skipped_runs"] += 1
            continue

        r["machine_id"] = target_machine_id

        # Remap system_config_id if present
        # (system_configs sync not yet implemented — set to NULL for now)
        r.pop("system_config_id", None)

        # Dedup check
        if (target_machine_id, r["started_at"]) in existing_runs:
            stats["skipped_runs"] += 1
            continue

        if dry_run:
            run_map[src_run_id] = -1
            stats["benchmark_runs"] += 1
            continue

        cols = [k for k in r.keys()]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        cursor = target_conn.execute(
            f"INSERT INTO benchmark_runs ({col_names}) VALUES ({placeholders})",
            [r[c] for c in cols],
        )
        run_map[src_run_id] = cursor.lastrowid
        stats["benchmark_runs"] += 1

    # Step 4: Insert run_metrics with remapped run_id
    src_metrics = src.execute("SELECT * FROM run_metrics ORDER BY id").fetchall()
    for m in src_metrics:
        m = dict(m)
        m.pop("id")
        src_run_id = m["run_id"]

        target_run_id = run_map.get(src_run_id)
        if target_run_id is None:
            continue  # Parent run was a duplicate or skipped

        m["run_id"] = target_run_id

        if dry_run:
            stats["run_metrics"] += 1
            continue

        cols = [k for k in m.keys()]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        target_conn.execute(
            f"INSERT INTO run_metrics ({col_names}) VALUES ({placeholders})",
            [m[c] for c in cols],
        )
        stats["run_metrics"] += 1

    # Step 5: Insert run_aggregates with remapped run_id
    src_aggs = src.execute("SELECT * FROM run_aggregates ORDER BY id").fetchall()
    for a in src_aggs:
        a = dict(a)
        a.pop("id")
        src_run_id = a["run_id"]

        target_run_id = run_map.get(src_run_id)
        if target_run_id is None:
            continue

        a["run_id"] = target_run_id

        if dry_run:
            stats["run_aggregates"] += 1
            continue

        cols = [k for k in a.keys()]
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        target_conn.execute(
            f"INSERT OR REPLACE INTO run_aggregates ({col_names}) VALUES ({placeholders})",
            [a[c] for c in cols],
        )
        stats["run_aggregates"] += 1

    if not dry_run:
        target_conn.commit()

    src.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Sync benchmark DBs from remote NUCs")
    parser.add_argument(
        "--target", default="results/consolidated.db",
        help="Target DB path (default: results/consolidated.db)"
    )
    parser.add_argument(
        "--machines", nargs="+", choices=list(MACHINES.keys()),
        help="Specific machines to sync (default: all)"
    )
    parser.add_argument(
        "--local-files", nargs="+",
        help="Merge local DB files instead of SCP from remotes"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--timeout", type=int, default=30, help="SSH/SCP timeout in seconds")
    args = parser.parse_args()

    target_path = Path(args.target).expanduser()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize target DB with schema
    print(f"=== Target DB: {target_path} ===")
    target_db = BenchmarkDB(str(target_path))
    target_conn = target_db.conn

    # Get initial target stats
    target_stats = get_source_stats(str(target_path))
    print(f"  Current: {target_stats['machines']} machines, "
          f"{target_stats['benchmark_runs']} runs, "
          f"{target_stats['run_metrics']} metrics")
    print()

    if args.local_files:
        # Merge local files
        source_files = [(Path(f).stem, f) for f in args.local_files]
    else:
        # SCP from remotes
        machines = args.machines or list(MACHINES.keys())
        tmpdir = tempfile.mkdtemp(prefix="bench-sync-")

        source_files = []
        for name in machines:
            machine = MACHINES[name]
            dest = os.path.join(tmpdir, f"{name}.db")
            print(f"=== Fetching {name} ({machine['host']}) ===")
            if scp_db(machine, dest, timeout=args.timeout):
                stats = get_source_stats(dest)
                print(f"  Contains: {stats['machines']} machines, "
                      f"{stats['benchmark_runs']} runs, "
                      f"{stats['run_metrics']} metrics")
                source_files.append((name, dest))
            else:
                print(f"  SKIPPED — could not fetch DB from {name}")
            print()

    # Merge each source into target
    total_stats = {"machines": 0, "benchmark_runs": 0, "run_metrics": 0,
                   "run_aggregates": 0, "skipped_runs": 0}

    for name, db_file in source_files:
        action = "Would merge" if args.dry_run else "Merging"
        print(f"=== {action}: {name} ===")
        stats = merge_db(db_file, target_conn, dry_run=args.dry_run)
        print(f"  Runs: +{stats['benchmark_runs']} new, "
              f"{stats['skipped_runs']} duplicates skipped")
        print(f"  Metrics: +{stats['run_metrics']}")
        print(f"  Aggregates: +{stats['run_aggregates']}")
        if stats['machines'] > 0:
            print(f"  Machines: +{stats['machines']} new")
        print()

        for k in total_stats:
            total_stats[k] += stats[k]

    # Final stats
    final_stats = get_source_stats(str(target_path))
    print("=" * 50)
    print("=== Sync Complete ===")
    print(f"  Machines: {final_stats['machines']}")
    print(f"  Benchmark Runs: {final_stats['benchmark_runs']}")
    print(f"  Run Metrics: {final_stats['run_metrics']}")
    print(f"  Run Aggregates: {final_stats['run_aggregates']}")
    print(f"  New runs added: {total_stats['benchmark_runs']}")
    print(f"  Duplicates skipped: {total_stats['skipped_runs']}")
    print(f"\n  Database: {target_path}")

    target_db.close()

    # Clean up temp files
    if not args.local_files and 'tmpdir' in dir():
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
