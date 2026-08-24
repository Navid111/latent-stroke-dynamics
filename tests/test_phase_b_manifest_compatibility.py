import json
from pathlib import Path

import pytest

from latent_stroke_dynamics.phase_b_manifest_compatibility import (
    EXPECTED_ORIGINAL_MANIFEST_HASHES,
    FROZEN_COMPATIBILITY_STATUS,
    compare_manifest_hashes,
    file_sha256,
    guard_output_directory,
    load_consumed_attempt_record,
    load_manifest_compatibility_config,
    validate_required_environment,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase-b0-colab-manifest-compatibility-2026-08-24.json"
ENTRYPOINT = ROOT / "experiments" / "27_phase_b_colab_manifest_compatibility.py"
BUILDER = ROOT / "scripts" / "build_phase_b_colab_manifest_compatibility_bundle.py"
NOTEBOOK = ROOT / "notebooks" / "phase_b0_colab_manifest_compatibility.ipynb"


def test_manifest_compatibility_config_is_frozen_and_non_training() -> None:
    config = load_manifest_compatibility_config(CONFIG)
    assert config["status"] == FROZEN_COMPATIBILITY_STATUS
    assert config["expected_original_manifest_sha256"] == (
        EXPECTED_ORIGINAL_MANIFEST_HASHES
    )
    boundary = config["validation_boundary"]
    assert boundary["renderer_manifest_generation_allowed"] is True
    assert boundary["model_resource_loading_allowed"] is False
    assert boundary["scientific_training_allowed"] is False
    assert boundary["recovery_execution_allowed"] is False
    assert config["post_validation"]["old_authorization_reusable"] is False


def test_dependency_gate_accepts_exact_versions_and_torch_build_suffix() -> None:
    config = load_manifest_compatibility_config(CONFIG)
    snapshot = {
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "torch": "2.11.0+cu128",
        "torch_base": "2.11.0",
        "generation_device": "cpu",
        "renderer_boundary": "PIL.ImageDraw.line",
    }
    validate_required_environment(config, snapshot)


def test_dependency_gate_rejects_mismatch() -> None:
    config = load_manifest_compatibility_config(CONFIG)
    snapshot = {
        "numpy": "2.5.1",
        "pillow": "12.3.0",
        "torch_base": "2.11.0",
        "generation_device": "cpu",
        "renderer_boundary": "PIL.ImageDraw.line",
    }
    with pytest.raises(RuntimeError, match="numpy"):
        validate_required_environment(config, snapshot)


def test_manifest_hash_comparison_is_exact_and_requires_filename_set(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    first = data_root / "first.json"
    second = data_root / "second.json"
    first.write_text('{"a": 1}', encoding="utf-8")
    second.write_text('{"b": 2}', encoding="utf-8")
    expected = {
        first.name: file_sha256(first),
        second.name: file_sha256(second),
    }
    passing = compare_manifest_hashes(data_root, expected)
    assert passing["all_hashes_match"] is True
    second.write_text('{"b": 3}', encoding="utf-8")
    assert compare_manifest_hashes(data_root, expected)["all_hashes_match"] is False
    (data_root / "unexpected.json").write_text("{}", encoding="utf-8")
    result = compare_manifest_hashes(data_root, expected)
    assert result["filename_set_matches"] is False
    assert result["all_hashes_match"] is False


def test_output_guard_rejects_repository_drive_and_existing_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    with pytest.raises(PermissionError, match="inside the repository"):
        guard_output_directory(repository / "generated", repository)
    with pytest.raises(PermissionError, match="Google Drive"):
        guard_output_directory(
            "/content/drive/MyDrive/forbidden-manifest-output", repository
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        guard_output_directory(existing, repository)
    fresh = tmp_path / "fresh"
    assert guard_output_directory(fresh, repository) == fresh.resolve()


def test_consumed_attempt_record_is_fail_closed() -> None:
    config = load_manifest_compatibility_config(CONFIG)
    record = load_consumed_attempt_record(ROOT, config)
    assert record["execution_attempt_consumed"] is True
    assert record["rerun_authorized"] is False
    assert record["training_started"] is False
    assert record["next_allowed_action"] == (
        "new_unauthorized_manifest_only_compatibility_protocol"
    )


def test_entrypoint_and_bundle_builder_cannot_train_or_reuse_authorization() -> None:
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    builder = BUILDER.read_text(encoding="utf-8")
    assert "run_manifest_compatibility_validation" in entrypoint
    assert "train_phase_b_variant" not in entrypoint
    assert "execute_phase_b_recovery" not in entrypoint
    assert "--recovery" not in entrypoint
    assert "load_recovery_execution_config" not in entrypoint
    assert '"recovery_authorized": False' in builder
    assert '"scientific_training_allowed": False' in builder
    assert "FROZEN_RESOURCE_RAW_HASHES" not in builder
    assert "--local-report" in builder


def test_colab_notebook_pins_dependencies_and_stays_off_drive() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "numpy==2.5.2" in source
    assert "pillow==12.3.0" in source
    assert "8 passed" in source
    assert "27_phase_b_colab_manifest_compatibility.py" in source
    assert "drive.mount" not in source
    assert "/content/drive" not in source
    assert "--recovery" not in source
    assert "train_phase_b_variant" not in source
