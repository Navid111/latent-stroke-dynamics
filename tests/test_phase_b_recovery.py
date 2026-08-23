import json
from pathlib import Path

import pytest

import latent_stroke_dynamics.phase_b_recovery as recovery
from latent_stroke_dynamics.phase_b_recovery import (
    EXPECTED_MANIFEST_HASHES,
    FROZEN_RECOVERY_STATUS,
    RecoveryOutputPaths,
    load_recovery_config,
    recovery_output_paths,
    require_recovery_authorized,
    require_recovery_outputs_absent,
    validate_expected_data_manifests,
    validate_recovery_environment_snapshot,
    validate_recovery_runner_request,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase-b0-colab-recovery-2026-08-24.json"


def test_recovery_config_is_frozen_and_unauthorized() -> None:
    config = load_recovery_config(CONFIG)
    assert config["status"] == FROZEN_RECOVERY_STATUS
    assert config["recovery"] == {"authorized": False, "single_run": True}
    assert config["formal_reserved"]["authorized"] is False
    assert config["phase_b1_reserved"]["authorized"] is False
    assert config["phase_b2_reserved"]["authorized"] is False


def test_recovery_runner_validation_has_no_scientific_side_effects() -> None:
    result = validate_recovery_runner_request(ROOT)
    assert result["status"] == "phase_b0_colab_recovery_runner_valid_unauthorized"
    assert result["recovery_authorized"] is False
    assert result["ranking_aware_models_allowed"] is False
    assert result["renderer_transitions_generated"] is False
    assert result["targets_generated"] is False
    assert result["candidate_sets_generated"] is False
    assert result["models_trained"] is False
    assert result["recovery_output_created"] is False
    assert result["local_incomplete_directory_touched"] is False


def test_recovery_authorization_guard_fails_before_output_creation(tmp_path: Path) -> None:
    config = load_recovery_config(CONFIG)
    with pytest.raises(PermissionError, match="not authorized"):
        require_recovery_authorized(config, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_recovery_output_paths_and_existing_output_guard(tmp_path: Path) -> None:
    config = load_recovery_config(CONFIG)
    paths = recovery_output_paths(config, tmp_path)
    assert paths.final.parent == tmp_path
    assert paths.incomplete.name.endswith(".incomplete")
    assert paths.journal.parent == paths.incomplete
    require_recovery_outputs_absent(paths)
    paths.incomplete.mkdir()
    with pytest.raises(FileExistsError, match="Preserve and audit"):
        require_recovery_outputs_absent(paths)


def test_recovery_environment_snapshot_requires_exact_frozen_values() -> None:
    config = load_recovery_config(CONFIG)
    snapshot = dict(config["environment"])
    validate_recovery_environment_snapshot(config, snapshot)
    snapshot["gpu_name"] = "different"
    with pytest.raises(RuntimeError, match="gpu_name"):
        validate_recovery_environment_snapshot(config, snapshot)


def test_recovery_data_manifest_continuity_is_fail_closed(tmp_path: Path) -> None:
    config = load_recovery_config(CONFIG)
    data_root = tmp_path / "data_manifests"
    data_root.mkdir()
    for name in EXPECTED_MANIFEST_HASHES:
        (data_root / name).write_text(name, encoding="utf-8")
    rewritten = json.loads(json.dumps(config))
    rewritten["data_manifest_sha256_required_before_training"] = {
        name: recovery.file_sha256(data_root / name)
        for name in EXPECTED_MANIFEST_HASHES
    }
    assert validate_expected_data_manifests(rewritten, data_root) == (
        rewritten["data_manifest_sha256_required_before_training"]
    )
    (data_root / "train_transitions.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_expected_data_manifests(rewritten, data_root)


def test_recovery_loads_only_three_mse_predictors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Loaded:
        def __init__(self, seed: int) -> None:
            self.seed = seed

    def fake_loader(path, *, expected_method, expected_seed, expected_state_sha256):
        calls.append((path, expected_method, expected_seed, expected_state_sha256))
        return Loaded(expected_seed)

    monkeypatch.setattr(recovery, "load_formal_latent_predictor", fake_loader)
    config = {
        "latent_predictors": {
            "mse_only": [
                {"path": f"mse_{seed}.pt", "seed": seed, "state_sha256": str(seed)}
                for seed in (11, 22, 33)
            ],
            "ranking_aware": [
                {"path": f"ranking_{seed}.pt", "seed": seed, "state_sha256": "unused"}
                for seed in (11, 22, 33)
            ],
        }
    }
    loaded = recovery.load_recovery_mse_only_predictors(config)
    assert tuple(item.seed for item in loaded) == (11, 22, 33)
    assert [item[1] for item in calls] == ["mse_only"] * 3
    assert not any("ranking" in item[0] for item in calls)
