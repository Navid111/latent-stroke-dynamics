import json
from pathlib import Path

import pytest

import latent_stroke_dynamics.phase_b_recovery_authorization as authorization
from latent_stroke_dynamics.phase_b_recovery import AUTHORIZED_RECOVERY_STATUS


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "phase-b0-colab-recovery-2026-08-24.json"


def _authorization_payload() -> dict:
    assert authorization.EXPECTED_RECOVERY_AUTHORIZATION is not None
    return dict(authorization.EXPECTED_RECOVERY_AUTHORIZATION)


def test_recovery_execution_config_loads_exact_separate_authorization() -> None:
    loaded = authorization.load_recovery_execution_config(BASE)
    assert loaded["protocol_status"] == "frozen_before_recovery_implementation"
    assert loaded["status"] == AUTHORIZED_RECOVERY_STATUS
    assert loaded["recovery"] == {"authorized": True, "single_run": True}
    assert loaded["recovery_authorization"] == _authorization_payload()
    assert loaded["formal_reserved"]["authorized"] is False
    assert loaded["phase_b1_reserved"]["authorized"] is False
    assert loaded["phase_b2_reserved"]["authorized"] is False


def test_unexpected_authorization_file_is_rejected_before_overlay(tmp_path: Path) -> None:
    config = tmp_path / BASE.name
    config.write_text(BASE.read_text(encoding="utf-8"), encoding="utf-8")
    authorization.authorization_path(config).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        authorization.load_recovery_execution_config(config)


def test_exact_separate_authorization_can_overlay_without_mutating_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / BASE.name
    original = json.loads(BASE.read_text(encoding="utf-8"))
    config.write_text(json.dumps(original), encoding="utf-8")
    payload = _authorization_payload()
    payload["validated_execution_handoff_commit"] = "a" * 40
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
