"""
Hardware fingerprinting — auto-detect CPU, GPU, NPU, memory, OS.

Captures everything needed to identify and compare machines.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class HardwareFingerprint:
    hostname: str = ""
    # CPU
    cpu_model: str = ""
    cpu_codename: str = ""  # ADL, MTL, LNL, PTL — set manually or from config
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_base_mhz: int = 0
    cpu_max_mhz: int = 0
    cpu_tdp_class: str = ""  # e.g. "15W", "45W" — from config
    # Memory
    mem_total_gb: float = 0.0
    mem_type: str = ""  # DDR4, DDR5, LPDDR5x
    mem_speed_mt: int = 0  # MT/s
    mem_channels: int = 0
    # GPU
    gpu_model: str = ""
    gpu_type: str = ""  # "dgpu", "igpu", "none"
    gpu_vram_gb: float = 0.0
    gpu_driver: str = ""
    # NPU
    npu_model: str = ""
    # OS
    os_name: str = ""
    os_kernel: str = ""
    # Runtime
    openvino_version: str = ""
    python_version: str = ""
    # Extras
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: str) -> str:
    """Run shell command, return stdout or empty string."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return ""


def detect_hardware(codename: str = "", tdp_class: str = "") -> HardwareFingerprint:
    """Auto-detect hardware on the current machine (Linux)."""
    hw = HardwareFingerprint()

    hw.hostname = platform.node()
    hw.python_version = platform.python_version()

    # CPU
    hw.cpu_model = _run("lscpu | grep 'Model name' | sed 's/.*: *//'")
    hw.cpu_cores = int(_run("nproc --all") or "0")
    hw.cpu_threads = int(
        _run("lscpu | grep '^CPU(s):' | awk '{print $2}'") or "0"
    )
    hw.cpu_base_mhz = int(float(
        _run("lscpu | grep 'CPU MHz' | head -1 | awk '{print $NF}'") or "0"
    ))
    hw.cpu_max_mhz = int(float(
        _run("lscpu | grep 'CPU max MHz' | awk '{print $NF}'") or "0"
    ))
    hw.cpu_codename = codename
    hw.cpu_tdp_class = tdp_class

    # Memory
    mem_kb = _run("grep MemTotal /proc/meminfo | awk '{print $2}'")
    hw.mem_total_gb = round(int(mem_kb or "0") / 1024 / 1024, 1)

    # Try to get memory type and speed from dmidecode (needs sudo)
    mem_info = _run("sudo dmidecode -t memory 2>/dev/null | grep -E 'Type:|Speed:' | head -4")
    if mem_info:
        for line in mem_info.splitlines():
            if "Type:" in line and "DDR" in line:
                hw.mem_type = line.split(":")[-1].strip()
            if "Speed:" in line and "MT/s" in line:
                speed = re.search(r"(\d+)", line)
                if speed:
                    hw.mem_speed_mt = int(speed.group(1))

    # Count memory channels from DMI slots
    slots_used = _run(
        "sudo dmidecode -t memory 2>/dev/null | grep -c 'Size:.*[0-9]' || echo 0"
    )
    hw.mem_channels = int(slots_used or "0")

    # GPU
    gpu_line = _run("lspci | grep -i 'VGA\\|Display\\|3D' | head -1")
    if gpu_line:
        hw.gpu_model = gpu_line.split(": ", 1)[-1] if ": " in gpu_line else gpu_line
        hw.gpu_type = "dgpu" if "Arc" in gpu_line and "A7" in gpu_line else "igpu"

    # GPU VRAM (try intel_gpu_top or sysfs)
    vram = _run(
        "cat /sys/class/drm/card*/device/resource0_size 2>/dev/null | head -1"
    )
    if not vram:
        # Try lspci memory regions
        vram = _run(
            "lspci -v -s $(lspci | grep -i 'VGA\\|Display' | head -1 | awk '{print $1}') 2>/dev/null"
            " | grep 'Memory.*prefetchable' | head -1 | grep -oP '\\[size=\\K[^]]+'"
        )

    # GPU driver version
    hw.gpu_driver = _run(
        "modinfo i915 2>/dev/null | grep '^version:' | awk '{print $2}'"
    ) or _run("cat /sys/module/i915/version 2>/dev/null")

    # NPU
    npu = _run("lspci | grep -i 'NPU\\|Neural\\|VPU'")
    hw.npu_model = npu.split(": ", 1)[-1] if npu and ": " in npu else ""

    # OS
    hw.os_name = _run("cat /etc/os-release | grep PRETTY_NAME | cut -d'\"' -f2") or platform.platform()
    hw.os_kernel = platform.release()

    # OpenVINO version
    try:
        import openvino as ov
        hw.openvino_version = ov.__version__
    except ImportError:
        hw.openvino_version = "not installed"

    return hw
