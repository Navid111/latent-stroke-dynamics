#!/usr/bin/env python3
"""Run the recovery-locked, dummy-only Phase B0 Google Colab CUDA preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from latent_stroke_dynamics.phase_b_cloud_preflight import run_colab_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run_colab_preflight(args.root)
        exit_code = 0
    except Exception as error:
        report = {
            "status": "phase_b0_colab_cuda_preflight_failed_recovery_unauthorized",
            "error_type": type(error).__name__,
            "error": str(error),
            "renderer_transitions_generated": False,
            "scientific_models_trained": False,
            "recovery_authorized": False,
        }
        exit_code = 1
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nSaved preflight report to: {args.report.resolve()}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
