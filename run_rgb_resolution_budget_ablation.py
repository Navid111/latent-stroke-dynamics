"""CLI for the guarded RGB resolution x stroke-budget ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from latent_stroke_dynamics.rgb_resolution_budget_ablation import (
    run_resolution_budget_ablation,
    validate_only_ablation_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute the fixed three-run RGB resolution x "
            "stroke-budget ablation."
        )
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate protocol, targets, baseline, and smoke tests with no output.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute conditions B, C, and D exactly once.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("local_targets/rgb-coarse-to-fine"),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/rgb-coarse-to-fine-fixed-seed73"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rgb-resolution-budget-ablation-seed73"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        report = run_resolution_budget_ablation(
            input_dir=args.input_dir,
            baseline_dir=args.baseline_dir,
            output_dir=args.output_dir,
        )
    else:
        report = validate_only_ablation_report(
            input_dir=args.input_dir,
            baseline_dir=args.baseline_dir,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
