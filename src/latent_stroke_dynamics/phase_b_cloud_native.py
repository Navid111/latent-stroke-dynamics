"""Fail-closed lifecycle for a new, Colab-native Phase B0 development run."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from .phase_b_development import (
    ABORTED_STATUS,
    DEFAULT_PHASE_B_CONFIG,
    load_phase_b_development_config,
)
from .phase_b_recovery import (
    EXPECTED_MANIFEST_HASHES as HISTORICAL_MAC_MANIFEST_HASHES,
    RecoveryOutputPaths,
    require_recovery_outputs_absent,
)


DEFAULT_CLOUD_NATIVE_CONFIG = Path(
    "configs/phase-b0-colab-native-development-2026-08-24.json"
)
AUTHORIZATION_FILENAME = (
    "phase-b0-colab-native-development-authorization-2026-08-24.json"
)
FROZEN_STATUS = "cloud_native_development_frozen_unauthorized"
AUTHORIZED_STATUS = "cloud_native_development_authorized_once"
VALIDATION_STATUS = "phase_b0_colab_native_runner_valid_unauthorized"
EXPECTED_CLOUD_MANIFEST_HASHES = {
    "diagnostic_test_transitions.json": "97d7e6527b27ade5671732fd025e069cb4497c85e64ad6c853c0a3cf0cbfee0b",
    "planner_supervision.json": "d5d0355c87f5f108b12a414c5e83c5a0bab733bcdfe41ddce9b5fc68e2feae62",
    "train_transitions.json": "18551716942c747ee3daf8728bf1a8d1d21b9b075f85d71fe1365bcfd6a6e6e8",
    "validation_transitions.json": "2b4fe2b782699538b91d3d13b453051fdb7e957d55fd371aba1cfdf56b44600a",
}
EXPECTED_ARTIFACT_ROOT = (
    "/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-cloud-native"
)
EXPECTED_ENVIRONMENT = {
    "provider": "google_colab_free",
    "device": "cuda:0",
    "gpu_name": "Tesla T4",
    "compute_capability": [7, 5],
    "gpu_memory_bytes": 15637086208,
    "platform": "Linux-6.6.122+-x86_64-with-glibc2.35",
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "cudnn": 91900,
    "float_precision": "float32",
    "automatic_mixed_precision": False,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "matmul_tf32": False,
    "cudnn_tf32": False,
}
# This remains None until Navid runs the complete local suite on the handoff
# commit. Authorization, when issued, must be a separate direct-child commit.
EXPECTED_CLOUD_NATIVE_AUTHORIZATION: dict[str, Any] | None = None
AUTHORIZATION_KEYS = {
    "experiment_id",
    "status",
    "authorized_phase",
    "authorized",
    "authorization_date",
    "validated_handoff_commit",
    "validated_local_test_count",
    "external_artifact_root",
    "maximum_completed_executions",
    "completed_executions",
    "authorization_consumed",
    "formal_authorized",
    "phase_b1_authorized",
    "phase_b2_authorized",
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def authorization_path(
    config_path: str | Path = DEFAULT_CLOUD_NATIVE_CONFIG,
) -> Path:
    return Path(config_path).with_name(AUTHORIZATION_FILENAME)


def validate_cloud_native_config(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "experiment_id",
        "status",
        "evidential_role",
        "branch",
        "base_config",
        "historical_results_unchanged",
        "recovery_of_mac_attempt",
        "prior_attempts",
        "manifest_basis",
        "environment",
        "device_allocation",
        "data_manifest_sha256_required_before_training",
        "resource_policy",
        "scientific_settings",
        "persistence",
        "development",
        "formal_reserved",
        "phase_b1_reserved",
        "phase_b2_reserved",
        "validation_boundary",
    }
    if set(config) != expected_keys:
        raise ValueError("Cloud-native Phase B0 config fields changed.")
    if config.get("experiment_id") != "phase-b0-colab-native-development-2026-08-24":
        raise ValueError("Unexpected cloud-native experiment id.")
    if config.get("status") != FROZEN_STATUS:
        raise ValueError("Cloud-native development must remain frozen and unauthorized.")
    if config.get("evidential_role") != (
        "new_cloud_native_development_after_zero_completion_local_attempt"
    ):
        raise ValueError("Cloud-native evidential role changed.")
    if config.get("branch") != "phase-b/saliency-latent":
        raise ValueError("Cloud-native branch changed.")
    if config.get("base_config") != str(DEFAULT_PHASE_B_CONFIG):
        raise ValueError("Cloud-native base config changed.")
    if config.get("historical_results_unchanged") is not True:
        raise ValueError("Historical results must remain unchanged.")
    if config.get("recovery_of_mac_attempt") is not False:
        raise ValueError("This must be a new experiment, not a Mac recovery.")
    manifests = dict(
        _mapping(
            config.get("data_manifest_sha256_required_before_training"),
            "cloud manifest hashes",
        )
    )
    if manifests != EXPECTED_CLOUD_MANIFEST_HASHES:
        raise ValueError("Cloud-native manifest hashes changed.")
    if any(
        manifests[name] == HISTORICAL_MAC_MANIFEST_HASHES[name]
        for name in manifests
    ):
        raise ValueError("Cloud-native hashes must not rewrite the historical Mac protocol.")
    basis = _mapping(config.get("manifest_basis"), "manifest_basis")
    if dict(basis) != {
        "canonical_platform": "linux_x86_64_google_colab",
        "independent_cloud_reproductions": 2,
        "training_results_observed_before_freeze": False,
        "mac_hashes_replaced_in_old_protocol": False,
        "new_experiment": True,
    }:
        raise ValueError("Cloud-native manifest basis changed.")
    if dict(_mapping(config.get("environment"), "environment")) != EXPECTED_ENVIRONMENT:
        raise ValueError("Cloud-native environment changed.")
    persistence = _mapping(config.get("persistence"), "persistence")
    if persistence.get("external_artifact_root") != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("Cloud-native artifact root changed.")
    if persistence.get("output_leaf") != "phase-b0-cloud-native-development-2026-08-24":
        raise ValueError("Cloud-native output leaf changed.")
    if persistence.get("incomplete_suffix") != ".incomplete":
        raise ValueError("Cloud-native incomplete suffix changed.")
    if persistence.get("automatic_resume_authorized") is not False:
        raise ValueError("Automatic resume must remain unauthorized.")
    science = _mapping(config.get("scientific_settings"), "scientific_settings")
    if science.get("hyperparameter_grid_allowed") is not False:
        raise ValueError("A cloud-native hyperparameter grid is not allowed.")
    if science.get("maximum_completed_executions") != 1:
        raise ValueError("Cloud-native development must remain a single execution.")
    resources = _mapping(config.get("resource_policy"), "resource_policy")
    if resources.get("required_resource_count") != 6:
        raise ValueError("Cloud-native resource count changed.")
    if resources.get("load_ranking_aware_predictors") is not False:
        raise ValueError("Unused ranking-aware predictors must not be loaded.")
    development = _mapping(config.get("development"), "development")
    if dict(development) != {"authorized": False, "single_run": True}:
        raise ValueError("Cloud-native development lifecycle changed.")
    for name in ("formal_reserved", "phase_b1_reserved", "phase_b2_reserved"):
        if _mapping(config.get(name), name).get("authorized") is not False:
            raise ValueError(f"{name} must remain unauthorized.")
    boundary = _mapping(config.get("validation_boundary"), "validation_boundary")
    forbidden = (
        "may_generate_renderer_data",
        "may_load_model_resources",
        "may_create_scientific_output",
        "may_train_models",
        "may_authorize_development_or_later_phases",
    )
    if any(boundary.get(name) is not False for name in forbidden):
        raise ValueError("Validation boundary permits a scientific side effect.")


def load_cloud_native_config(
    path: str | Path = DEFAULT_CLOUD_NATIVE_CONFIG,
) -> dict[str, Any]:
    config = _load_json(Path(path))
    validate_cloud_native_config(config)
    return config


def cloud_native_output_paths(
    config: Mapping[str, Any], artifact_root: str | Path
) -> RecoveryOutputPaths:
    persistence = _mapping(config.get("persistence"), "persistence")
    root = Path(artifact_root)
    leaf = str(persistence.get("output_leaf", ""))
    if not leaf or persistence.get("incomplete_suffix") != ".incomplete":
        raise ValueError("Cloud-native output configuration is invalid.")
    final = root / leaf
    incomplete = root / f"{leaf}.incomplete"
    # The validated execution engine retains this historical internal filename.
    return RecoveryOutputPaths(
        final=final,
        incomplete=incomplete,
        journal=incomplete / "recovery_stage_journal.json",
    )


def require_cloud_native_outputs_absent(paths: RecoveryOutputPaths) -> None:
    require_recovery_outputs_absent(paths)


def validate_cloud_native_runner_request(
    root: str | Path,
    config_path: str | Path = DEFAULT_CLOUD_NATIVE_CONFIG,
) -> dict[str, Any]:
    repository = Path(root)
    config = load_cloud_native_config(repository / config_path)
    base = load_phase_b_development_config(repository / DEFAULT_PHASE_B_CONFIG)
    if base.get("status") != ABORTED_STATUS:
        raise ValueError("The local zero-completion attempt is not archived.")
    if base["development"]["authorized"] is not False:
        raise ValueError("Historical local development must remain unauthorized.")
    return {
        "status": VALIDATION_STATUS,
        "experiment_id": config["experiment_id"],
        "config_status": config["status"],
        "new_experiment": True,
        "recovery_of_mac_attempt": False,
        "canonical_platform": "linux_x86_64_google_colab",
        "expected_cloud_manifest_hashes": dict(EXPECTED_CLOUD_MANIFEST_HASHES),
        "expected_resource_count": 6,
        "required_device": "cuda:0",
        "required_gpu": "Tesla T4",
        "cloud_native_development_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "renderer_data_generated": False,
        "model_resources_loaded": False,
        "scientific_models_trained": False,
        "scientific_output_created": False,
        "historical_local_incomplete_touched": False,
    }


def _validate_authorization(payload: Mapping[str, Any]) -> None:
    if set(payload) != AUTHORIZATION_KEYS:
        raise ValueError("Cloud-native authorization fields changed.")
    if payload.get("experiment_id") != "phase-b0-colab-native-development-2026-08-24":
        raise ValueError("Cloud-native authorization experiment changed.")
    if payload.get("status") != AUTHORIZED_STATUS:
        raise ValueError("Cloud-native authorization status changed.")
    if payload.get("authorized_phase") != "phase_b0_cloud_native_development":
        raise ValueError("Cloud-native authorization phase changed.")
    if payload.get("authorized") is not True:
        raise ValueError("Cloud-native authorization must be true.")
    commit = payload.get("validated_handoff_commit")
    if not (
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Validated handoff commit is invalid.")
    if payload.get("validated_local_test_count") != 160:
        raise ValueError("Cloud-native authorization requires the 160-test suite.")
    if payload.get("external_artifact_root") != EXPECTED_ARTIFACT_ROOT:
        raise ValueError("Cloud-native authorization artifact root changed.")
    if payload.get("maximum_completed_executions") != 1:
        raise ValueError("Cloud-native authorization must permit exactly one execution.")
    if payload.get("completed_executions") != 0:
        raise ValueError("Cloud-native authorization must begin with zero completed executions.")
    if payload.get("authorization_consumed") is not False:
        raise ValueError("Cloud-native authorization must begin unconsumed.")
    for name in ("formal_authorized", "phase_b1_authorized", "phase_b2_authorized"):
        if payload.get(name) is not False:
            raise ValueError(f"{name} must remain false.")


def load_cloud_native_execution_config(
    config_path: str | Path = DEFAULT_CLOUD_NATIVE_CONFIG,
) -> dict[str, Any]:
    path = Path(config_path)
    config = load_cloud_native_config(path)
    auth_path = authorization_path(path)
    if EXPECTED_CLOUD_NATIVE_AUTHORIZATION is None:
        raise PermissionError(
            "Cloud-native Phase B0 development is not yet authorized; run the complete local suite first."
        )
    if not auth_path.is_file():
        raise PermissionError("Cloud-native authorization record is absent.")
    payload = _load_json(auth_path)
    _validate_authorization(payload)
    if payload != EXPECTED_CLOUD_NATIVE_AUTHORIZATION:
        raise ValueError("Cloud-native authorization does not match the frozen expectation.")
    overlaid = deepcopy(config)
    overlaid["protocol_status"] = overlaid["status"]
    overlaid["status"] = AUTHORIZED_STATUS
    overlaid["development"] = {"authorized": True, "single_run": True}
    overlaid["cloud_native_authorization"] = payload
    return overlaid


def require_cloud_native_authorized(
    config: Mapping[str, Any], artifact_root: str | Path
) -> RecoveryOutputPaths:
    development = _mapping(config.get("development"), "development")
    if config.get("status") != AUTHORIZED_STATUS or development.get("authorized") is not True:
        raise PermissionError("Cloud-native Phase B0 development is not authorized.")
    authorization = _mapping(
        config.get("cloud_native_authorization"), "cloud_native_authorization"
    )
    if dict(authorization) != EXPECTED_CLOUD_NATIVE_AUTHORIZATION:
        raise PermissionError("Cloud-native authorization is invalid.")
    if str(Path(artifact_root).resolve()) != str(Path(EXPECTED_ARTIFACT_ROOT).resolve()):
        raise PermissionError("Cloud-native artifacts must use the authorized Drive root.")
    paths = cloud_native_output_paths(config, artifact_root)
    require_cloud_native_outputs_absent(paths)
    return paths
