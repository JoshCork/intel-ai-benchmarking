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

    # PerfSpect JSON is organized as sections, each with key-value tables
    for section in data:
        section_name = section.get("category", "")
        fields = section.get("fields", [])

        if section_name == "Software Version":
            for f in fields:
                if f.get("field_name") == "PerfSpect Version":
                    result["perfspect_version"] = f.get("field_value", "")

        elif section_name == "Power":
            for f in fields:
                name = f.get("field_name", "")
                value = f.get("field_value", "")
                if "scaling_governor" in name.lower() or "Scaling Governor" in name:
                    result["scaling_governor"] = value
                elif "energy" in name.lower() and "perf" in name.lower():
                    result["energy_perf_bias"] = value
                elif "turbo" in name.lower():
                    result["turbo_boost"] = value

        elif section_name == "C-state":
            # Collect active C-states
            states = []
            for f in fields:
                name = f.get("field_name", "")
                if name and "%" not in name:
                    states.append(name)
            if states:
                result["c_states"] = ",".join(states)

        elif section_name == "DIMM" or section_name == "Memory":
            for f in fields:
                name = f.get("field_name", "")
                value = f.get("field_value", "")
                if "installed" in name.lower() or "total" in name.lower():
                    result["installed_memory"] = value

        elif section_name == "BIOS":
            for f in fields:
                name = f.get("field_name", "")
                value = f.get("field_value", "")
                if "version" in name.lower():
                    result["bios_version"] = value
                    break

        elif section_name == "Operating System":
            for f in fields:
                name = f.get("field_name", "")
                value = f.get("field_value", "")
                if "kernel" in name.lower():
                    result["kernel_version"] = value

        elif section_name == "Insights":
            # Capture recommendations
            insights = []
            for f in fields:
                insights.append({
                    "field": f.get("field_name", ""),
                    "value": f.get("field_value", ""),
                })
            result["insights"] = insights

    return result


def load_existing_report(json_path: str) -> dict:
    """Load and parse an existing PerfSpect JSON report file."""
    path = Path(json_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PerfSpect report not found: {path}")
    return parse_perfspect_json(path)
