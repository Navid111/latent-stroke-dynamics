#!/usr/bin/env python3
"""Validate the frozen Phase B0 model without generating experimental data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_stroke_dynamics.phase_b_joint_embedding import (
    DEFAULT_PHASE_B_CONFIG,
    run_phase_b_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PHASE_B_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        parser.error("Only --validate-only is permitted before Phase B0 authorization.")
    return args


def main() -> None:
    args = parse_args()
    result = run_phase_b_validation(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
