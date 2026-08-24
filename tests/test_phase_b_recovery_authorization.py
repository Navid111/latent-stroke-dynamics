import json
from pathlib import Path

import pytest

import latent_stroke_dynamics.phase_b_recovery_authorization as authorization
from latent_stroke_dynamics.phase_b_recovery import AUTHORIZED_RECOVERY_STATUS


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "phase-b0-colab-recovery-2026-08-24.json"


def _authorization_payload() -> dict:
    return {
        "experiment_id": "phase-b0-colab-cuda-recovery-2026-08-24",
        "status": AUTHORIZED_RECOVERY_STATUS,
        "authorized_phase": "phase_b0_recovery",
        "authorized": True,
        "authorization_date": "2026-08-24",
        "validated_runner_source_commit": "2c38ffeee6a1182153cfed65fbcd1ece9f357781",
        "validated_execution_handoff_commit": "a" * 40,
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


def test_recovery_execution_config_requires_separate_authorization() -> None:
    with pytest.raises(PermissionError, match="not authorized"):
        authorization.load_recovery_execution_config(BASE)


def test_unexpected_authorization_file_is_rejected_before_overlay(tmp_path: Path) -> None:
    config = tmp_path / BASE.name
    config.write_text(BASE.read_text(encoding="utf-8"), encoding="utf-8")
    authorization.authorization_path(config).write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="not authorized"):
        authorization.load_recovery_execution_config(config)


def test_exact_separate_authorization_can_overlay_without_mutating_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / BASE.name
    original = json.loads(BASE.read_text(encoding="utf-8"))
    config.write_text(json.dumps(original), encoding="utf-8")
    payload = _authorization_payload()
    authorization.authorization_path(config).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(authorization, "EXPECTED_RECOVERY_AUTHORIZATION", payload)
    loaded = authorization.load_recovery_execution_config(config)
    assert loaded["protocol_status"] == original["status"]
    assert loaded["status"] == AUTHORIZED_RECOVERY_STATUS
    assert loaded["recovery"] == {"authorized": True, "single_run": True}
    assert loaded["recovery_authorization"] == payload
    assert json.loads(config.read_text(encoding="utf-8")) == original
