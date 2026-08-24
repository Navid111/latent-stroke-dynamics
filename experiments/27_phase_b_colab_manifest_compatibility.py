#!/usr/bin/env python3
"""Run the pinned, manifest-only Phase B0 compatibility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_stroke_dynamics.phase_b_manifest_compatibility import (
    DEFAULT_COMPATIBILITY_CONFIG,
    run_manifest_compatibility_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_COMPATIBILITY_CONFIG,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_manifest_compatibility_validation(
        ROOT,
        args.output_dir,
        args.config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nSaved manifest-only evidence to: {args.output_dir.resolve()}")
    if result["hash_gate_passed"]:
        print("All four original hashes matched. Training remains unauthorized.")
    else:
        print("At least one original hash differed. Preserve the manifests; do not train.")


if __name__ == "__main__":
    main()
