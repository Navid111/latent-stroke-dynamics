"""Lifecycle guards for the single frozen Phase B0 development execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .phase_b_joint_embedding import validate_phase_b_config


DEFAULT_PHASE_B_CONFIG = Path("configs/phase-b-saliency-latent-2026-08-23.json")
AUTHORIZATION_FILENAME = "phase-b0-development-authorization-2026-08-23.json"
INITIAL_STATUS = "frozen_before_implementation_and_data"
AUTHORIZED_STATUS = "development_authorized_once"
VALIDATION_MANIFEST = Path("docs/phase-b0-implementation-manifest.md")
DEVELOPMENT_VARIANTS = ("joint_prediction_only", "joint_prediction_progress")
LONG_HORIZON_METHODS = (
    "exact_pixel",
    "learned_pixel",
    "archived_mse_l1_forced",
    "joint_prediction_only_forced",
    "joint_prediction_progress_forced",
    "joint_prediction_progress_no_op",
)
EXPECTED_DEVELOPMENT_AUTHORIZATION = {
    "experiment_id": "phase-b0-action-conditioned-joint-embedding-2026-08-23",
    "status": AUTHORIZED_STATUS,
    "authorized_phase": "development",
    "authorized": True,
    "authorization_date": "2026-08-23",
    "validated_test_count": 116,
    "validated_runner_status": "phase_b0_development_runner_valid_unauthorized",
    "core_development_commit": "0aa58cad24b7a8ccc1e91c5855581883c2ae5d01",
    "validated_runner_commit": "323df8328e99c26a63fc05194edc43a4ca781efe",
    "runner_instructions_commit": "797fb2d3fe59fb0f9325384b0e0404870a93a925",
    "maximum_completed_executions": 1,
    "authorization_consumed": False,
    "formal_authorized": False,
    "phase_b1_authorized": False,
    "phase_b2_authorized": False,
}


@dataclass(frozen=True)
class PhaseBOutputPaths:
    final: Path
    incomplete: Path


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _static_protocol_copy(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(config))
    normalized["status"] = INITIAL_STATUS
    development = dict(_mapping(normalized.get("development"), "development"))
    development["authorized"] = False
    normalized["development"] = development
    return normalized


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def load_phase_b_development_config(
    path: str | Path = DEFAULT_PHASE_B_CONFIG,
) -> dict[str, Any]:
    """Load the immutable protocol and apply the separately committed authorization."""

    config_path = Path(path)
    config = _load_json(config_path)
    authorization_path = config_path.with_name(AUTHORIZATION_FILENAME)
    if authorization_path.exists():
        authorization = _load_json(authorization_path)
        if authorization != EXPECTED_DEVELOPMENT_AUTHORIZATION:
            raise ValueError("Phase B0 development authorization record changed.")
        development = _mapping(config.get("development"), "development")
        if config.get("status") != INITIAL_STATUS or development.get("authorized") is not False:
            raise ValueError("Authorization must overlay the immutable initial protocol.")
        config["protocol_status"] = config["status"]
        config["status"] = AUTHORIZED_STATUS
        updated_development = dict(development)
        updated_development["authorized"] = True
        config["development"] = updated_development
        config["development_authorization"] = authorization

    status = config.get("status")
    if status not in {INITIAL_STATUS, AUTHORIZED_STATUS}:
        raise ValueError("Unexpected Phase B0 development lifecycle status.")
    validate_phase_b_config(_static_protocol_copy(config))
    development = _mapping(config.get("development"), "development")
    if (status == AUTHORIZED_STATUS) != (development.get("authorized") is True):
        raise ValueError("Phase B0 status and development authorization disagree.")
    if status == AUTHORIZED_STATUS:
        if dict(_mapping(config.get("development_authorization"), "authorization")) != (
            EXPECTED_DEVELOPMENT_AUTHORIZATION
        ):
            raise ValueError("Authorized Phase B0 config lacks the validated one-time record.")
    elif config.get("development_authorization") is not None:
        raise ValueError("Unauthorized Phase B0 config cannot carry an authorization record.")
    for name in ("formal_reserved", "region_scheduler_reserved", "rgb_high_resolution_reserved"):
        if _mapping(config.get(name), name).get("authorized") is not False:
            raise ValueError(f"{name} must remain unauthorized.")
    return config


def phase_b_output_paths(config: Mapping[str, Any]) -> PhaseBOutputPaths:
    development = _mapping(config.get("development"), "development")
    final = Path(str(development.get("output_dir", "")))
    if not final.name:
        raise ValueError("Phase B0 development output directory is invalid.")
    return PhaseBOutputPaths(
        final=final,
        incomplete=final.with_name(final.name + ".incomplete"),
    )


def require_phase_b_outputs_absent(paths: PhaseBOutputPaths) -> None:
    if paths.final.exists():
        raise FileExistsError(f"Phase B0 development output exists: {paths.final}")
    if paths.incomplete.exists():
        raise FileExistsError(
            f"Incomplete Phase B0 output exists: {paths.incomplete}. Preserve and review it before any retry."
        )


def validate_phase_b_development_runner_request(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete runner before model loading or renderer-data generation."""

    if config.get("status") != INITIAL_STATUS:
        raise ValueError("Runner validation requires the initial unauthorized status.")
    if _mapping(config.get("development"), "development").get("authorized") is not False:
        raise ValueError("Development must be unauthorized during runner validation.")
    if not VALIDATION_MANIFEST.exists():
        raise FileNotFoundError("The passing Phase B0 implementation manifest is missing.")
    paths = phase_b_output_paths(config)
    require_phase_b_outputs_absent(paths)
    development = _mapping(config["development"], "development")
    transition_splits = _mapping(development["transition_splits"], "transition_splits")
    planner_train = _mapping(development["planner_supervision_train"], "planner train")
    planner_validation = _mapping(
        development["planner_supervision_validation"], "planner validation"
    )
    long_horizon = _mapping(development["long_horizon"], "long_horizon")
    train_sets = len(planner_train["target_seeds"]) * len(planner_train["states"])
    validation_sets = len(planner_validation["target_seeds"]) * len(
        planner_validation["states"]
    )
    return {
        "status": "phase_b0_development_runner_valid_unauthorized",
        "config_status": config["status"],
        "variants": list(DEVELOPMENT_VARIANTS),
        "transition_samples": {
            name: int(payload["samples"]) for name, payload in transition_splits.items()
        },
        "planner_training_candidate_sets": train_sets,
        "planner_validation_candidate_sets": validation_sets,
        "planner_candidates_per_set": int(planner_train["candidates_per_state"]),
        "long_horizon_methods": list(long_horizon["methods"]),
        "long_horizon_targets": len(long_horizon["target_seeds"]),
        "long_horizon_steps": int(long_horizon["maximum_steps"]),
        "long_horizon_candidates_per_step": int(long_horizon["candidates_per_step"]),
        "development_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "output_dir_available": True,
        "incomplete_dir_available": True,
        "historical_models_loaded": False,
        "renderer_transitions_generated": False,
        "targets_generated": False,
        "state_banks_generated": False,
        "candidate_sets_generated": False,
        "models_trained": False,
        "output_directories_created": False,
        "historical_results_unchanged": True,
    }


def require_phase_b_development_authorized(config: Mapping[str, Any]) -> None:
    """Fail before loading checkpoints, generating data, or creating outputs."""

    development = _mapping(config.get("development"), "development")
    authorization = _mapping(
        config.get("development_authorization"), "development_authorization"
    )
    if (
        config.get("status") != AUTHORIZED_STATUS
        or development.get("authorized") is not True
        or dict(authorization) != EXPECTED_DEVELOPMENT_AUTHORIZATION
        or authorization.get("authorization_consumed") is not False
        or authorization.get("maximum_completed_executions") != 1
    ):
        raise PermissionError(
            "Phase B0 development is not authorized. No model was loaded, no renderer data were generated, and no output directory was created."
        )
    for name in ("formal_reserved", "region_scheduler_reserved", "rgb_high_resolution_reserved"):
        if _mapping(config.get(name), name).get("authorized") is not False:
            raise PermissionError(f"{name} must remain unauthorized during Phase B0 development.")
    require_phase_b_outputs_absent(phase_b_output_paths(config))
