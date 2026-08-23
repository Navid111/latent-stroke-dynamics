import json
from pathlib import Path

import pytest

from latent_stroke_dynamics.phase_b_cloud_preflight import (
    ABORTED_ATTEMPT_RECORD,
    FROZEN_RESOURCE_RAW_HASHES,
    dummy_optimizer_smoke,
    required_resource_paths,
    validate_cloud_preflight_boundary,
)
from latent_stroke_dynamics.phase_b_development import ABORTED_STATUS


ROOT = Path(__file__).resolve().parents[1]


def test_phase_b_cloud_preflight_boundary_is_recovery_locked() -> None:
    result = validate_cloud_preflight_boundary(ROOT)
    assert result["status"] == (
        "phase_b0_cloud_preflight_boundary_valid_recovery_unauthorized"
    )
    assert result["config_status"] == ABORTED_STATUS
    assert result["development_authorized"] is False
    assert result["scientific_training_allowed"] is False
    assert result["dummy_tensors_only"] is True


def test_phase_b_cloud_resource_hashes_match_interruption_audit() -> None:
    record = json.loads((ROOT / ABORTED_ATTEMPT_RECORD).read_text(encoding="utf-8"))
    audited = {
        item["path"]: item["raw_file_sha256"]
        for item in record["frozen_resource_inventory"]
    }
    assert audited == FROZEN_RESOURCE_RAW_HASHES


def test_phase_b_cloud_bundle_requires_only_six_used_resources() -> None:
    paths = required_resource_paths()
    assert len(paths) == 6
    assert not any("ranking_aware_seed" in path for path in paths)
    assert all(".incomplete" not in path for path in paths)


def test_phase_b_dummy_optimizer_smoke_is_finite_on_cpu() -> None:
    result = dummy_optimizer_smoke("cpu")
    assert result["device"] == "cpu"
    assert result["finite"] is True
    assert result["joint_prediction_only_dummy_loss"] >= 0.0
    assert result["joint_prediction_progress_dummy_loss"] >= 0.0
