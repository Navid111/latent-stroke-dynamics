"""CLI for the guarded straight-line versus quadratic-Bezier comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from latent_stroke_dynamics.quadratic_bezier_comparison import (
    FREEZE_MANIFEST_PATH,
    run_fixed_comparison,
    validate_only_comparison_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or, with a separate exact authorization, execute the fixed "
            "straight-line versus quadratic-Bezier comparison."
        )
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen inputs and tiny planners without creating outputs.",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="Execute the fixed comparison only when separately authorized.",
    )
    parser.add_argument("--freeze", type=Path, default=FREEZE_MANIFEST_PATH)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.execute:
        missing = [
            name
            for name, value in (
                ("--authorization", args.authorization),
                ("--output-dir", args.output_dir),
                ("--source-commit", args.source_commit),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Execution requires explicit values for " + ", ".join(missing) + "."
            )
        report = run_fixed_comparison(
            output_dir=args.output_dir,
            authorization_path=args.authorization,
            source_commit=args.source_commit,
            freeze_path=args.freeze,
        )
    else:
        report = validate_only_comparison_report(args.freeze)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
