import json
from pathlib import Path

import pytest

import latent_stroke_dynamics.phase_b_recovery_authorization as authorization
from latent_stroke_dynamics.phase_b_recovery import AUTHORIZED_RECOVERY_STATUS


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "phase-b0-colab-recovery-2026-08-24.json"
ISSUANCE = ROOT / "configs" / authorization.RECOVERY_AUTHORIZATION_FILENAME
CONSUMED = ROOT / "configs" / authorization.CONSUMED_RECOVERY_ATTEMPT_FILENAME


def _issued_payload() -> dict:
    return json.loads(ISSUANCE.read_text(encoding="utf-8"))


def test_recovery_execution_config_rejects_consumed_authorization() -> None:
    record = json.loads(CONSUMED.read_text(encoding="utf-8"))
    assert record["execution_attempt_consumed"] is True
    assert record["rerun_authorized"] is False
    assert record["training_started"] is False
    assert authorization.EXPECTED_RECOVERY_AUTHORIZATION is None
    with pytest.raises(PermissionError, match="consumed"):
        authorization.load_recovery_execution_config(BASE)


def test_unexpected_authorization_file_is_rejected_before_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / BASE.name
    config.write_text(BASE.read_text(encoding="utf-8"), encoding="utf-8")
    authorization.authorization_path(config).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        authorization, "EXPECTED_RECOVERY_AUTHORIZATION", _issued_payload()
    )
    with pytest.raises(ValueError, match="fields changed"):
        authorization.load_recovery_execution_config(config)


def test_historical_authorization_path_can_only_be_exercised_when_explicitly_patched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / BASE.name
    original = json.loads(BASE.read_text(encoding="utf-8"))
    config.write_text(json.dumps(original), encoding="utf-8")
    payload = _issued_payload()
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
