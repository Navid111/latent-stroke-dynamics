#!/usr/bin/env python3
"""Fail-closed readiness check for an already authorized Phase B0 recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from latent_stroke_dynamics.phase_b_cloud_preflight import (
    verify_loaded_model_states,
    verify_raw_resources,
)
from latent_stroke_dynamics.phase_b_development import load_phase_b_development_config
from latent_stroke_dynamics.phase_b_recovery import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_RECOVERY_CONFIG,
    configure_recovery_determinism,
    recovery_environment_snapshot,
    require_recovery_authorized,
    validate_recovery_environment_snapshot,
)
from latent_stroke_dynamics.phase_b_recovery_authorization import (
    load_recovery_execution_config,
)


ROOT = Path(__file__).resolve().parents[1]
READY_STATUS = "phase_b0_colab_recovery_execution_ready_authorized_once"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report.exists():
        raise FileExistsError(f"Readiness report already exists: {args.report}")
    config = load_recovery_execution_config(ROOT / DEFAULT_RECOVERY_CONFIG)
    base = load_phase_b_development_config(ROOT / DEFAULT_BASE_CONFIG)
    paths = require_recovery_authorized(config, args.artifact_root)
    configure_recovery_determinism()
    environment = recovery_environment_snapshot()
    validate_recovery_environment_snapshot(config, environment)
    raw = verify_raw_resources(ROOT)
    states = verify_loaded_model_states(ROOT)
    if states.get("ranking_aware_models_loaded") is not False:
        raise RuntimeError("Recovery readiness loaded a ranking-aware model.")
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    report = {
        "status": READY_STATUS,
        "git_head": head,
        "base_status": base["status"],
        "recovery_status": config["status"],
        "authorization": config["recovery_authorization"],
        "environment": environment,
        "raw_resource_sha256": raw,
        "loaded_model_states": states,
        "artifact_root": str(args.artifact_root),
        "final_output_path": str(paths.final),
        "incomplete_output_path": str(paths.incomplete),
        "final_output_absent": not paths.final.exists(),
        "incomplete_output_absent": not paths.incomplete.exists(),
        "renderer_transitions_generated": False,
        "targets_generated": False,
        "state_banks_generated": False,
        "candidate_sets_generated": False,
        "models_trained": False,
        "recovery_output_created": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
