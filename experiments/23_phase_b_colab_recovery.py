#!/usr/bin/env python3
"""Validate or execute the separately guarded Phase B0 Colab recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_stroke_dynamics.phase_b_development import load_phase_b_development_config
from latent_stroke_dynamics.phase_b_recovery import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_RECOVERY_CONFIG,
    configure_recovery_determinism,
    recovery_environment_snapshot,
    require_recovery_authorized,
    validate_recovery_environment_snapshot,
    validate_recovery_runner_request,
)
from latent_stroke_dynamics.phase_b_recovery_authorization import (
    load_recovery_execution_config,
)
from latent_stroke_dynamics.phase_b_recovery_execution import (
    execute_phase_b_recovery,
    record_recovery_event,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-config", type=Path, default=DEFAULT_RECOVERY_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--recovery", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    if args.recovery and args.artifact_root is None:
        parser.error("--recovery requires --artifact-root under the mounted Google Drive.")
    if args.validate_only and args.artifact_root is not None:
        parser.error("--validate-only must not receive or create an artifact root.")
    return args


def main() -> None:
    args = parse_args()
    if args.validate_only:
        result = validate_recovery_runner_request(ROOT, args.recovery_config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    recovery_config = load_recovery_execution_config(ROOT / args.recovery_config)
    base_config = load_phase_b_development_config(ROOT / DEFAULT_BASE_CONFIG)
    paths = require_recovery_authorized(recovery_config, args.artifact_root)
    configure_recovery_determinism()
    environment = recovery_environment_snapshot()
    validate_recovery_environment_snapshot(recovery_config, environment)
    try:
        decision = execute_phase_b_recovery(
            base_config,
            recovery_config,
            paths,
            repository_root=ROOT,
            environment_snapshot=environment,
        )
    except BaseException as error:
        if paths.incomplete.exists():
            record_recovery_event(
                paths.journal,
                "recovery_execution",
                "interrupted_or_failed",
                {"error_type": type(error).__name__, "message": str(error)},
            )
        raise
    print("\nPhase B0 Colab recovery decision\n")
    print(json.dumps(decision, indent=2))
    print(f"\nSaved artifacts to: {paths.final}")
    print("Do not rerun or tune against this result.")


if __name__ == "__main__":
    main()
