"""Separate one-time authorization overlay for cloud-native Phase B0."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import phase_b_cloud_native as core


EXPECTED_CLOUD_NATIVE_AUTHORIZATION: dict[str, Any] = {
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


def load_cloud_native_execution_config(
    config_path: str | Path = core.DEFAULT_CLOUD_NATIVE_CONFIG,
) -> dict[str, Any]:
    """Apply only the exact direct-child authorization issued after 160 tests."""

    existing = core.EXPECTED_CLOUD_NATIVE_AUTHORIZATION
    if existing is not None and existing != EXPECTED_CLOUD_NATIVE_AUTHORIZATION:
        raise ValueError("A conflicting cloud-native authorization is already configured.")
    core.EXPECTED_CLOUD_NATIVE_AUTHORIZATION = EXPECTED_CLOUD_NATIVE_AUTHORIZATION
    return core.load_cloud_native_execution_config(config_path)
