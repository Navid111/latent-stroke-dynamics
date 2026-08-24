"""Historical authorization record and permanent completed-run lock."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import phase_b_cloud_native as core


AUTHORIZATION_FILENAME = (
    "phase-b0-colab-native-development-authorization-2026-08-24.json"
)
COMPLETED_ATTEMPT_FILENAME = (
    "phase-b0-colab-native-completed-attempt-2026-08-24.json"
)
ISSUED_CLOUD_NATIVE_AUTHORIZATION: dict[str, Any] = {
    "experiment_id": "phase-b0-colab-native-development-2026-08-24",
    "status": "cloud_native_development_authorized_once",
    "authorized_phase": "phase_b0_cloud_native_development",
    "authorized": True,
    "authorization_date": "2026-08-24",
    "validated_handoff_commit": "b57ab921abfd51f4382b0436c8e10f49247402c7",
    "validated_local_test_count": 160,
    "external_artifact_root": core.EXPECTED_ARTIFACT_ROOT,
    "maximum_completed_executions": 1,
    "completed_executions": 0,
    "authorization_consumed": False,
    "formal_authorized": False,
    "phase_b1_authorized": False,
    "phase_b2_authorized": False,
}
# The authorization remains immutable historical evidence, but the completed
# handoff proves that its one permitted execution finished. It is not reusable.
EXPECTED_CLOUD_NATIVE_AUTHORIZATION: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _validate_completed_attempt(payload: Mapping[str, Any]) -> None:
    expected = {
        "experiment_id": "phase-b0-colab-native-development-2026-08-24",
        "status": "phase_b0_colab_native_development_complete_not_eligible",
        "source_commit": "4f0b70dab03f1700a0fbbe5dc9598a1d019b8cc0",
        "execution_attempt_consumed": True,
        "rerun_authorized": False,
        "completed_executions": 1,
        "models_trained": True,
        "development_completed": True,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "do_not_rerun": True,
    }
    changed = [name for name, value in expected.items() if payload.get(name) != value]
    if changed:
        raise ValueError(
            "Completed cloud-native attempt record changed: " + ", ".join(changed)
        )


def load_cloud_native_execution_config(
    config_path: str | Path = core.DEFAULT_CLOUD_NATIVE_CONFIG,
) -> dict[str, Any]:
    """Always reject reuse after the single completed development execution."""

    path = Path(config_path)
    completed_path = path.with_name(COMPLETED_ATTEMPT_FILENAME)
    if not completed_path.is_file():
        raise PermissionError(
            "Cloud-native Phase B0 authorization cannot be reused; the completed-attempt lock is missing."
        )
    _validate_completed_attempt(_load_json(completed_path))
    core.EXPECTED_CLOUD_NATIVE_AUTHORIZATION = None
    raise PermissionError(
        "Cloud-native Phase B0 authorization was consumed by the completed GPU development run; no rerun is authorized."
    )
