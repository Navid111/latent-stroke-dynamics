"""CLI for the fixed RGB coarse-to-fine qualitative painter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from latent_stroke_dynamics.rgb_coarse_to_fine import (
    PainterConfig,
    run_fixed_experiment,
    validate_only_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or execute the fixed five-target RGB painter."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate targets and run a no-output synthetic smoke test.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute the complete frozen five-target qualitative run.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("local_targets/rgb-coarse-to-fine"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/rgb-coarse-to-fine-fixed-seed73"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        report = run_fixed_experiment(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            config=PainterConfig(),
        )
    else:
        report = validate_only_report(args.input_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
