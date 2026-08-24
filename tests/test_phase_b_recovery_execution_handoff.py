import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_phase_b_colab_recovery_execution_bundle.py"
NOTEBOOK = ROOT / "notebooks" / "phase_b0_colab_recovery_execution.ipynb"
READINESS = ROOT / "experiments" / "26_phase_b_colab_recovery_execution_check.py"


def test_execution_bundle_builder_requires_exact_authorization() -> None:
    source = BUILDER.read_text(encoding="utf-8")
    assert "load_recovery_execution_config" in source
    assert "validated_execution_handoff_commit" in source
    assert "HEAD^" in source
    assert "FROZEN_RESOURCE_RAW_HASHES" in source
    assert "EXPECTED_TEST_COUNT = 145" in source


def test_execution_notebook_requires_drive_readiness_and_explicit_switch() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "145 passed" in source
    assert "drive.mount('/content/drive')" in source
    assert "/content/drive/MyDrive/latent-stroke-dynamics-phase-b0-recovery" in source
    assert "phase_b0_colab_recovery_execution_ready_authorized_once" in source
    assert "RUN_AUTHORIZED_RECOVERY = False" in source
    assert "--recovery" in source
    assert "subprocess.run" in source and "check=True" in source
    assert "Do not start a second run" in source


def test_execution_readiness_check_cannot_train_or_create_recovery_output() -> None:
    source = READINESS.read_text(encoding="utf-8")
    assert "load_recovery_execution_config" in source
    assert "require_recovery_authorized" in source
    assert "verify_raw_resources" in source
    assert "verify_loaded_model_states" in source
    assert "execute_phase_b_recovery" not in source
    assert '"models_trained": False' in source
    assert '"recovery_output_created": False' in source


def test_execution_bundle_builder_currently_refuses_without_authorization() -> None:
    before = set((ROOT / "dist").glob("phase-b0-colab-recovery-execution-*.tar.gz"))
    completed = subprocess.run(
        [sys.executable, str(BUILDER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    after = set((ROOT / "dist").glob("phase-b0-colab-recovery-execution-*.tar.gz"))
    assert completed.returncode != 0
    assert "not authorized" in completed.stderr
    assert after == before
