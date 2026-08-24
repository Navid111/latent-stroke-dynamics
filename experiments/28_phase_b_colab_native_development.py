#!/usr/bin/env python3
"""Validate or execute one new cloud-native Phase B0 development run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_stroke_dynamics.phase_b_cloud_native import (
    DEFAULT_CLOUD_NATIVE_CONFIG,
    load_cloud_native_execution_config,
    require_cloud_native_authorized,
    validate_cloud_native_runner_request,
)
from latent_stroke_dynamics.phase_b_development import (
    DEFAULT_PHASE_B_CONFIG,
    load_phase_b_development_config,
)
from latent_stroke_dynamics.phase_b_recovery import (
    configure_recovery_determinism,
    recovery_environment_snapshot,
    validate_recovery_environment_snapshot,
)
from latent_stroke_dynamics.phase_b_recovery_execution import (
    execute_phase_b_recovery,
    record_recovery_event,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CLOUD_NATIVE_CONFIG
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--development", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    if args.development and args.artifact_root is None:
        parser.error("--development requires the authorized Google Drive artifact root.")
    if args.validate_only and args.artifact_root is not None:
        parser.error("--validate-only must not receive an artifact root.")
    return args


def main() -> None:
    args = parse_args()
    if args.validate_only:
        print(
            json.dumps(
                validate_cloud_native_runner_request(ROOT, args.config),
                indent=2,
                sort_keys=True,
            )
        )
        return

    cloud = load_cloud_native_execution_config(ROOT / args.config)
    base = load_phase_b_development_config(ROOT / DEFAULT_PHASE_B_CONFIG)
    paths = require_cloud_native_authorized(cloud, args.artifact_root)
    configure_recovery_determinism()
    environment = recovery_environment_snapshot()
    validate_recovery_environment_snapshot(cloud, environment)
    execution_environment = {
        **environment,
        "execution_mode": "new_cloud_native_development_not_mac_recovery",
        "cloud_native_experiment_id": cloud["experiment_id"],
    }
    try:
        # Reuse the previously validated CUDA execution engine. Its historical
        # helper name says recovery, but this config, output root, manifests,
        # authorization, and evidential role define a separate new experiment.
        engine_decision = execute_phase_b_recovery(
            base,
            cloud,
            paths,
            repository_root=ROOT,
            environment_snapshot=execution_environment,
        )
    except BaseException as error:
        if paths.incomplete.exists():
            record_recovery_event(
                paths.journal,
                "cloud_native_development_execution",
                "interrupted_or_failed",
                {"error_type": type(error).__name__, "message": str(error)},
            )
        raise
    decision = {
        **engine_decision,
        "execution_mode": "new_cloud_native_development_not_mac_recovery",
        "cloud_native_development_completed": True,
        "historical_mac_attempt_recovered": False,
    }
    print("\nPhase B0 cloud-native development decision\n")
    print(json.dumps(decision, indent=2))
    print(f"\nSaved persistent artifacts to: {paths.final}")
    print("Do not rerun or tune against this development result.")


if __name__ == "__main__":
    main()
