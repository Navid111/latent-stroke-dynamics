#!/usr/bin/env python3
"""Audit Phase B0 split fingerprints without regenerating data or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from latent_stroke_dynamics.phase_b_overlap_audit import (  # noqa: E402
    audit_transition_manifest_overlap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help=(
            "Existing directory containing train_transitions.json, "
            "validation_transitions.json, and diagnostic_test_transitions.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new JSON report path outside the source manifest directory.",
    )
    parser.add_argument(
        "--require-blank-no-op-only",
        action="store_true",
        help="Exit nonzero unless every observed cross-split overlap is the blank no-op.",
    )
    return parser.parse_args()


def _write_new_report(path: Path, manifest_dir: Path, text: str) -> None:
    destination = path.expanduser().resolve()
    source_root = manifest_dir.expanduser().resolve()
    try:
        destination.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("The audit report must not be written inside the source manifests.")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing report: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Report parent directory does not exist: {destination.parent}"
        )
    destination.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = audit_transition_manifest_overlap(args.manifest_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        _write_new_report(args.output, args.manifest_dir, text)
    print(text)
    if (
        args.require_blank_no_op_only
        and report["classification"] != "blank_no_op_only"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
