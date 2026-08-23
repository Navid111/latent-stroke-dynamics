#!/usr/bin/env python3
"""Run dummy-only CUDA validation of the guarded Phase B0 recovery runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_stroke_dynamics.phase_b_recovery_validation import (
    VALIDATION_STATUS,
    run_cuda_recovery_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"Validation report already exists: {args.report}")
    report = run_cuda_recovery_validation(Path.cwd())
    if report.get("status") != VALIDATION_STATUS:
        raise RuntimeError("Recovery-runner CUDA validation did not pass.")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
