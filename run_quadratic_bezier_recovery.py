"""CLI for validation and separately authorized interrupted-run recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from latent_stroke_dynamics.quadratic_bezier_comparison import FREEZE_MANIFEST_PATH
from latent_stroke_dynamics.quadratic_bezier_recovery import (
    RECOVERY_PLAN_PATH,
    run_interrupted_comparison_recovery,
    validate_only_recovery_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or, with a separate exact authorization, recover the "
            "preserved interrupted straight-line versus quadratic-Bezier comparison."
        )
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate recovery code without accessing Drive or creating outputs.",
    )
    action.add_argument(
        "--execute-recovery",
        action="store_true",
        help="Recover only the frozen interrupted attempt when separately authorized.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=RECOVERY_PLAN_PATH)
    parser.add_argument("--freeze", type=Path, default=FREEZE_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--original-authorization", type=Path)
    parser.add_argument("--recovery-authorization", type=Path)
    parser.add_argument("--recovery-implementation-commit")
    parser.add_argument("--expected-head")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_recovery:
        report = validate_only_recovery_report(
            repo_root=args.repo_root,
            plan_path=args.plan,
            freeze_path=args.freeze,
            expected_head=args.expected_head,
        )
    else:
        missing = [
            name
            for name, value in (
                ("--output-dir", args.output_dir),
                ("--original-authorization", args.original_authorization),
                ("--recovery-authorization", args.recovery_authorization),
                (
                    "--recovery-implementation-commit",
                    args.recovery_implementation_commit,
                ),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Recovery execution requires explicit values for "
                + ", ".join(missing)
                + "."
            )
        aggregate = run_interrupted_comparison_recovery(
            output_dir=args.output_dir,
            original_authorization_path=args.original_authorization,
            recovery_authorization_path=args.recovery_authorization,
            recovery_implementation_commit=args.recovery_implementation_commit,
            repo_root=args.repo_root,
            plan_path=args.plan,
            freeze_path=args.freeze,
        )
        report = {
            "status": "quadratic_bezier_recovery_completed_blind_gate_applied",
            "completion_mode": aggregate["completion_mode"],
            "completed_run_count": aggregate["completed_run_count"],
            "completed_pair_count": aggregate["completed_pair_count"],
            "reused_completed_run_count": aggregate["reused_completed_run_count"],
            "executed_during_recovery_run_count": aggregate[
                "executed_during_recovery_run_count"
            ],
            "integrity_passed": aggregate["integrity_passed"],
            "qualitative_review_required": aggregate[
                "final_decision_requires_blinded_review"
            ],
            "quantitative_values_withheld": bool(
                aggregate["final_decision_requires_blinded_review"]
            ),
            "training_performed": False,
            "learned_model_used": False,
            "closed_experiments_changed": False,
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
