"""
PerfSpect integration — run Intel PerfSpect and parse system config.

PerfSpect collects CPU topology, power settings, BIOS config, kernel tunables,
memory config and generates detailed system reports.

Expected installation: ~/perfspect/ on each NUC.
Run with: sudo ./perfspect report
Output: ~/perfspect/perfspect_<timestamp>/<hostname>.json
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default PerfSpect location on our NUCs
DEFAULT_PERFSPECT_DIR = os.path.expanduser("~/perfspect")


def find_perfspect(perfspect_dir: str = DEFAULT_PERFSPECT_DIR) -> Optional[Path]:
    """Find the PerfSpect binary."""
    binary = Path(perfspect_dir) / "perfspect"
    if binary.exists() and os.access(str(binary), os.X_OK):
        return binary
    return None


def run_perfspect(
    perfspect_dir: str = DEFAULT_PERFSPECT_DIR,
    output_dir: Optional[str] = None,
) -> Optional[Path]:
    """Run PerfSpect report and return path to the JSON output.

    Requires sudo. Returns None on failure.
    """
    binary = find_perfspect(perfspect_dir)
    if not binary:
        print(f"  PerfSpect not found at {perfspect_dir}")
        return None

    # PerfSpect outputs to its own timestamped directory
    cmd = ["sudo", str(binary), "report"]
    if output_dir:
        cmd.extend(["-o", output_dir])

    print(f"  Running PerfSpect... (requires sudo)")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=perfspect_dir,
        )
        if result.returncode != 0:
            print(f"  PerfSpect failed: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print("  PerfSpect timed out (120s)")
        return None
    except FileNotFoundError:
        print("  sudo not available or PerfSpect not found")
        return None

    # Find the latest output JSON
    return find_latest_report(perfspect_dir)


def find_latest_report(perfspect_dir: str = DEFAULT_PERFSPECT_DIR) -> Optional[Path]:
    """Find the most recent PerfSpect JSON report."""
    base = Path(perfspect_dir)
    json_files = sorted(base.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    for f in json_files:
        # PerfSpect reports are in perfspect_<timestamp>/<hostname>.json
        if "perfspect_" in str(f.parent.name):
            return f

    return None


def parse_perfspect_json(json_path: Path) -> dict:
    """Parse PerfSpect JSON into our DB-friendly format.

    Extracts key fields for easy querying and preserves full JSON.
    """
    with open(json_path) as f:
        data = json.load(f)

    result = {
        "full_json": data,
        "perfspect_version": "",
        "scaling_governor": "",
        "energy_perf_bias": "",
        "turbo_boost": "",
        "c_states": "",
        "installed_memory": "",
        "bios_version": "",
        "kernel_version": "",
        "insights": [],
    }

    # PerfSpect JSON is a dict: { "SectionName": [list of record dicts], ... }
    # Each section contains a list of dicts with named keys.

    # PerfSpect version
    if "PerfSpect" in data and data["PerfSpect"]:
        ps = data["PerfSpect"][0]
        result["perfspect_version"] = ps.get("Version", "")

    # Power settings
    if "Power" in data and data["Power"]:
        power = data["Power"][0]
        result["scaling_governor"] = power.get("Scaling Governor", "")
        result["energy_perf_bias"] = power.get("Energy Performance Bias", "")
        result["turbo_boost"] = power.get("Turbo Boost", power.get("TDP", ""))

    # C-states
    if "C-state" in data:
        states = [cs.get("Name", "") for cs in data["C-state"]
                  if cs.get("Status", "").lower() == "enabled"]
        result["c_states"] = ",".join(states)

    # Memory — build summary from DIMM entries
    if "DIMM" in data and data["DIMM"]:
        dimms = data["DIMM"]
        # Count populated DIMMs (have a size)
        populated = [d for d in dimms if d.get("Size") and d["Size"] != "No Module Installed"]
        if populated:
            total_gb = 0
            speed = ""
            mem_type = ""
            for d in populated:
                size_str = d.get("Size", "")
                if "GB" in size_str:
                    try:
                        total_gb += int(size_str.replace("GB", "").strip())
                    except ValueError:
                        pass
                if not speed:
                    speed = d.get("Configured Speed", "")
                if not mem_type:
                    mem_type = d.get("Type", "")
            result["installed_memory"] = (
                f"{total_gb}GB ({len(populated)}x{populated[0].get('Size', '?')} "
                f"{mem_type} {speed})"
            )
    elif "Memory" in data and data["Memory"]:
        mem = data["Memory"][0]
        result["installed_memory"] = mem.get("Installed Memory", "")

    # BIOS
    if "BIOS" in data and data["BIOS"]:
        bios = data["BIOS"][0]
        result["bios_version"] = bios.get("Version", "")

    # Operating System / Kernel
    if "Operating System" in data and data["Operating System"]:
        os_info = data["Operating System"][0]
        result["kernel_version"] = os_info.get("Kernel", "")

    # Insights (recommendations)
    if "Insights" in data:
        result["insights"] = [
            {"field": i.get("Justification", ""), "value": i.get("Recommendation", "")}
            for i in data["Insights"]
        ]

    return result


def load_existing_report(json_path: str) -> dict:
    """Load and parse an existing PerfSpect JSON report file."""
    path = Path(json_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PerfSpect report not found: {path}")
    return parse_perfspect_json(path)
