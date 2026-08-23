import json
from pathlib import Path

from latent_stroke_dynamics.phase_b_cloud_preflight import required_resource_paths
from latent_stroke_dynamics.phase_b_recovery_validation import (
    dummy_recovery_execution_smoke,
    validate_cuda_recovery_boundary,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "phase_b0_colab_recovery_validation.ipynb"
BUILDER = ROOT / "scripts" / "build_phase_b_colab_recovery_validation_bundle.py"
VALIDATOR = ROOT / "experiments" / "24_phase_b_colab_recovery_validation.py"


def test_cuda_recovery_validation_boundary_is_unauthorized_and_dummy_only() -> None:
    result = validate_cuda_recovery_boundary(ROOT)
    assert result["status"] == (
        "phase_b0_colab_recovery_cuda_boundary_valid_unauthorized"
    )
    assert result["recovery_authorized"] is False
    assert result["dummy_tensors_only"] is True
    assert result["scientific_training_allowed"] is False
    assert result["recovery_output_allowed"] is False


def test_dummy_recovery_execution_paths_are_finite_on_cpu() -> None:
    result = dummy_recovery_execution_smoke("cpu")
    assert result["device"] == "cpu"
    assert result["all_values_finite"] is True
    assert result["temporary_dummy_checkpoints_removed"] is True
    assert result["planner_candidate_metric_rows"] == 2
    assert result["prediction_score_count"] == 4
    assert result["progress_score_count"] == 4
    assert result["scientific_evidence"] is False


def test_recovery_validation_uses_only_six_required_resources() -> None:
    paths = required_resource_paths()
    assert len(paths) == 6
    assert not any("ranking_aware_seed" in path for path in paths)


def test_recovery_validation_entrypoint_only_writes_requested_report() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    assert "--report" in source
    assert "--recovery" not in source
    assert "run_cuda_recovery_validation" in source


def test_recovery_validation_bundle_builder_stays_unauthorized() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert 'BRANCH = "phase-b/saliency-latent"' in source
    assert '"recovery_authorized": False' in source
    assert '"scientific_training_allowed": False' in source
    assert "FROZEN_RESOURCE_RAW_HASHES" in source


def test_recovery_validation_notebook_is_fail_closed() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    assert "subprocess.run" in source
    assert "check=True" in source
    assert "138 passed" in source
    assert "phase_b0_colab_recovery_implementation_valid_unauthorized" in source
    assert "--recovery" not in source
    assert "drive.mount" not in source
