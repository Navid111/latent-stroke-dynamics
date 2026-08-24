import json
from pathlib import Path
import subprocess
import sys

import pytest

import latent_stroke_dynamics.phase_b_cloud_native as cloud
import latent_stroke_dynamics.phase_b_cloud_native_authorization as authorization
from latent_stroke_dynamics.phase_b_cloud_native import (
    EXPECTED_CLOUD_MANIFEST_HASHES,
    FROZEN_STATUS,
    cloud_native_output_paths,
    load_cloud_native_config,
    require_cloud_native_outputs_absent,
    validate_cloud_native_runner_request,
)
from latent_stroke_dynamics.phase_b_recovery import EXPECTED_MANIFEST_HASHES


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase-b0-colab-native-development-2026-08-24.json"
COMPLETED = ROOT / "configs" / authorization.COMPLETED_ATTEMPT_FILENAME
RUNNER = ROOT / "experiments" / "28_phase_b_colab_native_development.py"
BUILDER = ROOT / "scripts" / "build_phase_b_colab_native_execution_bundle.py"
NOTEBOOK = ROOT / "notebooks" / "phase_b0_colab_native_development.ipynb"


def test_cloud_native_config_freezes_a_new_linux_experiment() -> None:
    config = load_cloud_native_config(CONFIG)
    assert config["status"] == FROZEN_STATUS
    assert config["recovery_of_mac_attempt"] is False
    assert config["manifest_basis"]["new_experiment"] is True
    assert config["manifest_basis"]["independent_cloud_reproductions"] == 2
    assert config["data_manifest_sha256_required_before_training"] == (
        EXPECTED_CLOUD_MANIFEST_HASHES
    )
    assert all(
        EXPECTED_CLOUD_MANIFEST_HASHES[name] != EXPECTED_MANIFEST_HASHES[name]
        for name in EXPECTED_CLOUD_MANIFEST_HASHES
    )
    assert config["development"] == {"authorized": False, "single_run": True}


def test_cloud_native_validation_is_scientifically_side_effect_free() -> None:
    result = validate_cloud_native_runner_request(ROOT)
    assert result["status"] == "phase_b0_colab_native_runner_valid_unauthorized"
    assert result["new_experiment"] is True
    assert result["recovery_of_mac_attempt"] is False
    assert result["cloud_native_development_authorized"] is False
    assert result["renderer_data_generated"] is False
    assert result["model_resources_loaded"] is False
    assert result["scientific_models_trained"] is False
    assert result["scientific_output_created"] is False


def test_cloud_native_authorization_is_locked_after_completed_run() -> None:
    record = json.loads(COMPLETED.read_text(encoding="utf-8"))
    assert record["execution_attempt_consumed"] is True
    assert record["rerun_authorized"] is False
    assert record["models_trained"] is True
    assert record["decision"]["status"] == "not_eligible"
    assert authorization.EXPECTED_CLOUD_NATIVE_AUTHORIZATION is None
    with pytest.raises(PermissionError, match="consumed"):
        authorization.load_cloud_native_execution_config(CONFIG)
    assert cloud.EXPECTED_CLOUD_NATIVE_AUTHORIZATION is None


def test_cloud_native_output_guard_preserves_existing_attempt(tmp_path: Path) -> None:
    config = load_cloud_native_config(CONFIG)
    paths = cloud_native_output_paths(config, tmp_path)
    assert paths.final.name == "phase-b0-cloud-native-development-2026-08-24"
    assert paths.incomplete.name.endswith(".incomplete")
    require_cloud_native_outputs_absent(paths)
    paths.incomplete.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Preserve and audit"):
        require_cloud_native_outputs_absent(paths)


def test_cloud_native_cli_validation_runs_without_artifacts(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--validate-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "phase_b0_colab_native_runner_valid_unauthorized"
    assert payload["scientific_models_trained"] is False
    assert list(tmp_path.iterdir()) == []


def test_cloud_native_runner_reuses_validated_cuda_engine_without_mac_recovery() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "execute_phase_b_recovery" in source
    assert "new_cloud_native_development_not_mac_recovery" in source
    assert "load_cloud_native_execution_config" in source
    assert "load_recovery_execution_config" not in source
    assert "historical_mac_attempt_recovered" in source


def test_cloud_native_handoff_is_one_bundle_one_notebook_one_training_switch() -> None:
    builder = BUILDER.read_text(encoding="utf-8")
    assert "load_cloud_native_execution_config" in builder
    assert "FROZEN_RESOURCE_RAW_HASHES" in builder
    assert "EXPECTED_TEST_COUNT = 160" in builder
    assert "phase-b0-colab-native-execution-" in builder
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "160 passed" in source
    assert "drive.mount('/content/drive')" in source
    assert "RUN_CLOUD_NATIVE_TRAINING = False" in source
    assert "--development" in source
    assert "phase-b0-cloud-native-completion-handoff.json" in source
    assert "manifest compatibility" not in source.lower()
