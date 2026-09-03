"""CLI for the validation-only quadratic-Bezier extension gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from latent_stroke_dynamics.quadratic_bezier_extension import (
    DEFAULT_CONFIG_PATH,
    validate_only_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen straight-line versus quadratic-Bezier "
            "implementation without creating comparative outputs."
        )
    )
    parser.add_argument("--validate-only", action="store_true", default=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_only_report(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
