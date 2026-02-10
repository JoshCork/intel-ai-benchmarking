#!/usr/bin/env python3
"""
Export & quantize — HuggingFace model → OpenVINO IR at various precisions.

Usage:
  python scripts/export_model.py --model meta-llama/Llama-3.1-8B-Instruct --precision FP16
  python scripts/export_model.py --model meta-llama/Llama-3.1-8B-Instruct --precision INT4
  python scripts/export_model.py --model meta-llama/Llama-3.1-8B-Instruct --precision all

The exported models are hardware-portable — the same INT4 model runs on
Arc A770M dGPU, Lunar Lake iGPU, and Meteor Lake iGPU. OpenVINO compiles
device-specific kernels at load time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PRECISIONS = ["FP16", "INT8", "INT4"]


def export_model(model_name: str, precision: str, output_dir: str) -> Path:
    """Export a model to OpenVINO IR format using optimum-cli."""
    short_name = model_name.split("/")[-1]
    out_path = Path(output_dir).expanduser() / f"{short_name}-{precision}"

    if out_path.exists() and list(out_path.glob("*.xml")):
        print(f"  Model already exists: {out_path}")
        return out_path

    out_path.mkdir(parents=True, exist_ok=True)

    weight_fmt = {
        "FP16": "fp16",
        "FP32": "fp32",
        "INT8": "int8",
        "INT4": "int4",
    }[precision]

    cmd = [
        "optimum-cli", "export", "openvino",
        "-m", model_name,
        "--weight-format", weight_fmt,
        str(out_path),
    ]

    print(f"\n  Exporting {model_name} → {precision}...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Output:  {out_path}")

    result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout
    if result.returncode != 0:
        print(f"  ERROR: Export failed with code {result.returncode}")
        return out_path

    # Check output
    xmls = list(out_path.glob("*.xml"))
    if xmls:
        total_size = sum(f.stat().st_size for f in out_path.iterdir() if f.is_file())
        print(f"  Success: {out_path} ({total_size / 1024 / 1024:.0f} MB)")
    else:
        print(f"  WARNING: No .xml files found in {out_path}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace model to OpenVINO IR"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--precision", default="FP16",
        help="Precision: FP16, INT8, INT4, or 'all' for all three"
    )
    parser.add_argument(
        "--output-dir", default="~/models/intel-bench",
        help="Output directory for exported models"
    )
    parser.add_argument(
        "--copy-to-usb", action="store_true",
        help="After export, copy to attached USB drive"
    )
    args = parser.parse_args()

    precisions = PRECISIONS if args.precision.lower() == "all" else [args.precision.upper()]

    print(f"Model: {args.model}")
    print(f"Precisions: {precisions}")
    print(f"Output: {args.output_dir}")

    for prec in precisions:
        out_path = export_model(args.model, prec, args.output_dir)

        if args.copy_to_usb:
            # Import here to avoid circular deps
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from lib.model_loader import ModelInfo, copy_model_to_usb

            info = ModelInfo(
                name=args.model, precision=prec,
                path=out_path, source="local",
            )
            copy_model_to_usb(info)

    print("\nDone.")


if __name__ == "__main__":
    main()
