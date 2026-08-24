"""Separate one-time authorization overlay for the validated Phase B0 recovery."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from .phase_b_recovery import (
    AUTHORIZED_RECOVERY_STATUS,
    DEFAULT_RECOVERY_CONFIG,
    FROZEN_RECOVERY_STATUS,
    load_recovery_config,
)


RECOVERY_AUTHORIZATION_FILENAME = (
    "phase-b0-colab-recovery-authorization-2026-08-24.json"
)
EXPECTED_RECOVERY_AUTHORIZATION: dict[str, Any] | None = {
    "experiment_id": "phase-b0-colab-cuda-recovery-2026-08-24",
    "status": "recovery_authorized_once",
    "authorized_phase": "phase_b0_recovery",
    "authorized": True,
    "authorization_date": "2026-08-24",
    "validated_runner_source_commit": "2c38ffeee6a1182153cfed65fbcd1ece9f357781",
    "validated_execution_handoff_commit": "3191e3e6b382bea96bf48569f3ac5af3eec61b24",
    "validated_local_test_count": 145,
    "validated_colab_test_count": 138,
    "validated_colab_validation_status": "phase_b0_colab_recovery_implementation_valid_unauthorized",
    "validated_colab_bundle_sha256": "2d5d8ab7c15d33f72d4d4db7b69e7b96a903fa33881f712a1cf6433969bd7138",
    "validated_colab_report": "docs/artifacts/phase-b0-colab-recovery-validation-report-2026-08-24.json",
    "external_artifact_root": "/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-recovery",
    "maximum_completed_executions": 1,
    "completed_executions": 0,
    "authorization_consumed": False,
    "formal_authorized": False,
    "phase_b1_authorized": False,
    "phase_b2_authorized": False,
}
AUTHORIZATION_KEYS = {
    "experiment_id",
    "status",
    "authorized_phase",
    "authorized",
    "authorization_date",
    "validated_runner_source_commit",
    "validated_execution_handoff_commit",
    "validated_local_test_count",
    "validated_colab_test_count",
    "validated_colab_validation_status",
    "validated_colab_bundle_sha256",
    "validated_colab_report",
    "external_artifact_root",
    "maximum_completed_executions",
    "completed_executions",
    "authorization_consumed",
    "formal_authorized",
    "phase_b1_authorized",
    "phase_b2_authorized",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _is_commit(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_authorization_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != AUTHORIZATION_KEYS:
        raise ValueError("Recovery authorization fields changed.")
    if payload.get("experiment_id") != "phase-b0-colab-cuda-recovery-2026-08-24":
        raise ValueError("Recovery authorization experiment changed.")
    if payload.get("status") != AUTHORIZED_RECOVERY_STATUS:
        raise ValueError("Recovery authorization status changed.")
    if payload.get("authorized_phase") != "phase_b0_recovery":
        raise ValueError("Recovery authorization phase changed.")
    if payload.get("authorized") is not True:
        raise ValueError("Recovery authorization must be true.")
    if not _is_commit(payload.get("validated_runner_source_commit")):
        raise ValueError("Validated runner commit is invalid.")
    if payload.get("validated_runner_source_commit") != (
        "2c38ffeee6a1182153cfed65fbcd1ece9f357781"
    ):
        raise ValueError("Validated runner source commit changed.")
    if not _is_commit(payload.get("validated_execution_handoff_commit")):
        raise ValueError("Validated execution-handoff commit is invalid.")
    if payload.get("validated_local_test_count") != 145:
        raise ValueError("Recovery authorization requires the final 145-test suite.")
    if payload.get("validated_colab_test_count") != 138:
        raise ValueError("Recovery authorization Colab test count changed.")
    if payload.get("validated_colab_validation_status") != (
        "phase_b0_colab_recovery_implementation_valid_unauthorized"
    ):
        raise ValueError("Recovery authorization references the wrong Colab result.")
    if payload.get("validated_colab_bundle_sha256") != (
        "2d5d8ab7c15d33f72d4d4db7b69e7b96a903fa33881f712a1cf6433969bd7138"
    ):
        raise ValueError("Recovery authorization references the wrong Colab bundle.")
    if payload.get("validated_colab_report") != (
        "docs/artifacts/phase-b0-colab-recovery-validation-report-2026-08-24.json"
    ):
        raise ValueError("Recovery authorization report path changed.")
    if payload.get("external_artifact_root") != (
        "/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-recovery"
    ):
        raise ValueError("Recovery authorization artifact root changed.")
    if payload.get("maximum_completed_executions") != 1:
        raise ValueError("Recovery must permit exactly one completed execution.")
    if payload.get("completed_executions") != 0:
        raise ValueError("Recovery authorization must begin with zero completed executions.")
    if payload.get("authorization_consumed") is not False:
        raise ValueError("Fresh recovery authorization must be unconsumed.")
    for name in ("formal_authorized", "phase_b1_authorized", "phase_b2_authorized"):
        if payload.get(name) is not False:
            raise ValueError(f"{name} must remain false.")


def authorization_path(
    config_path: str | Path = DEFAULT_RECOVERY_CONFIG,
) -> Path:
    return Path(config_path).with_name(RECOVERY_AUTHORIZATION_FILENAME)


def load_recovery_execution_config(
    config_path: str | Path = DEFAULT_RECOVERY_CONFIG,
) -> dict[str, Any]:
    """Load the immutable recovery config plus an exact separately committed authorization."""

    path = Path(config_path)
    config = load_recovery_config(path)
    if config.get("status") != FROZEN_RECOVERY_STATUS:
        raise ValueError("Recovery authorization must overlay the immutable frozen config.")
    auth_path = authorization_path(path)
    if not auth_path.is_file():
        raise PermissionError(
            "Phase B0 Colab recovery is not authorized; the separate authorization record is absent."
        )
    if EXPECTED_RECOVERY_AUTHORIZATION is None:
        raise PermissionError(
            "Phase B0 Colab recovery is not authorized; the expected authorization has not been frozen in code."
        )
    authorization = _load_json(auth_path)
    _validate_authorization_payload(authorization)
    if authorization != EXPECTED_RECOVERY_AUTHORIZATION:
        raise ValueError("Recovery authorization record does not match the frozen expectation.")
    overlaid = deepcopy(config)
    overlaid["protocol_status"] = overlaid["status"]
    overlaid["status"] = AUTHORIZED_RECOVERY_STATUS
    overlaid["recovery"] = {"authorized": True, "single_run": True}
    overlaid["recovery_authorization"] = authorization
    return overlaid
