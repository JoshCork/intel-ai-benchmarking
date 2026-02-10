"""
Model loader — find models on USB, local disk, or download from HuggingFace.

Search order:
1. Attached USB drives (look for `intel-ai-models/` folder)
2. Local cache (~/models/intel-bench/)
3. HuggingFace Hub download (with optimum-intel export)
"""

from __future__ import annotations

import os
import glob
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Standard USB mount points on Linux
USB_MOUNT_PATHS = [
    "/media/{user}",
    "/mnt",
    "/run/media/{user}",
]


@dataclass
class ModelInfo:
    """Describes a located model."""

    name: str  # e.g. "Llama-3.1-8B-Instruct"
    precision: str  # e.g. "FP16", "INT8", "INT4"
    path: Path  # absolute path to model directory
    source: str  # "usb", "local", "download"

    @property
    def openvino_xml(self) -> Optional[Path]:
        """Path to the main OpenVINO IR .xml file."""
        xmls = list(self.path.glob("*.xml"))
        # Prefer openvino_model.xml if present
        for xml in xmls:
            if xml.name == "openvino_model.xml":
                return xml
        return xmls[0] if xmls else None

    def is_valid(self) -> bool:
        """Check the model dir has the expected OpenVINO files."""
        return self.openvino_xml is not None and self.path.exists()


def _model_dir_name(model_name: str, precision: str) -> str:
    """Convert model name + precision to expected directory name.

    e.g. "meta-llama/Llama-3.1-8B-Instruct" + "INT4"
    → "Llama-3.1-8B-Instruct-INT4"
    """
    short = model_name.split("/")[-1]
    return f"{short}-{precision}"


def _find_on_usb(
    model_name: str, precision: str, usb_folder: str = "intel-ai-models"
) -> Optional[ModelInfo]:
    """Search mounted USB drives for the model."""
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    dir_name = _model_dir_name(model_name, precision)

    for pattern in USB_MOUNT_PATHS:
        base = pattern.format(user=user)
        if not os.path.isdir(base):
            continue

        # Check each mount point under base
        for mount in os.listdir(base):
            model_path = Path(base) / mount / usb_folder / dir_name
            if model_path.is_dir():
                info = ModelInfo(
                    name=model_name,
                    precision=precision,
                    path=model_path,
                    source="usb",
                )
                if info.is_valid():
                    return info

    return None


def _find_local(
    model_name: str, precision: str, local_dir: str = "~/models/intel-bench"
) -> Optional[ModelInfo]:
    """Check local cache directory."""
    base = Path(local_dir).expanduser()
    dir_name = _model_dir_name(model_name, precision)
    model_path = base / dir_name

    if model_path.is_dir():
        info = ModelInfo(
            name=model_name,
            precision=precision,
            path=model_path,
            source="local",
        )
        if info.is_valid():
            return info

    return None


def download_and_export(
    model_name: str,
    precision: str,
    local_dir: str = "~/models/intel-bench",
) -> ModelInfo:
    """Download from HuggingFace and export to OpenVINO IR.

    Uses optimum-intel CLI:
      optimum-cli export openvino --model <name> --weight-format <fmt> <output>
    """
    base = Path(local_dir).expanduser()
    dir_name = _model_dir_name(model_name, precision)
    output_path = base / dir_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Map our precision labels to optimum-intel weight formats
    weight_fmt_map = {
        "FP16": "fp16",
        "FP32": "fp32",
        "INT8": "int8",
        "INT4": "int4",
    }
    weight_fmt = weight_fmt_map.get(precision, "fp16")

    print(f"Downloading and exporting {model_name} → {precision}...")
    print(f"  Output: {output_path}")
    print(f"  This may take a while for large models.")

    cmd = [
        "optimum-cli", "export", "openvino",
        "-m", model_name,
        "--weight-format", weight_fmt,
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Model export failed:\n{result.stderr}\n{result.stdout}"
        )

    print(f"  Export complete: {output_path}")

    return ModelInfo(
        name=model_name,
        precision=precision,
        path=output_path,
        source="download",
    )


def find_model(
    model_name: str,
    precision: str,
    usb_folder: str = "intel-ai-models",
    local_dir: str = "~/models/intel-bench",
    auto_download: bool = True,
) -> ModelInfo:
    """Find a model, searching USB → local → download.

    Args:
        model_name: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B-Instruct")
        precision: Target precision ("FP16", "INT8", "INT4")
        usb_folder: Folder name to look for on USB drives
        local_dir: Local cache directory
        auto_download: If True, download from HuggingFace when not found locally
    """
    # 1. USB
    info = _find_on_usb(model_name, precision, usb_folder)
    if info:
        print(f"Found model on USB: {info.path}")
        return info

    # 2. Local cache
    info = _find_local(model_name, precision, local_dir)
    if info:
        print(f"Found model locally: {info.path}")
        return info

    # 3. Download
    if auto_download:
        print(f"Model not found locally. Downloading {model_name} ({precision})...")
        return download_and_export(model_name, precision, local_dir)

    raise FileNotFoundError(
        f"Model {model_name} ({precision}) not found on USB or local disk. "
        f"Run with --download to fetch from HuggingFace."
    )


def copy_model_to_usb(
    model_info: ModelInfo,
    usb_folder: str = "intel-ai-models",
) -> Optional[Path]:
    """Copy a model to the first available USB drive for sneakernet transfer."""
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))

    for pattern in USB_MOUNT_PATHS:
        base = pattern.format(user=user)
        if not os.path.isdir(base):
            continue
        for mount in os.listdir(base):
            mount_path = Path(base) / mount
            if mount_path.is_mount():
                dest = mount_path / usb_folder / model_info.path.name
                print(f"Copying {model_info.path} → {dest}")
                shutil.copytree(model_info.path, dest, dirs_exist_ok=True)
                print(f"  Done. Safely eject the USB drive before removing.")
                return dest

    print("No USB drive found.")
    return None
