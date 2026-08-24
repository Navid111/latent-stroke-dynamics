import json
from pathlib import Path
import subprocess
import sys

import pytest

from latent_stroke_dynamics.phase_b_recovery_execution import (
    _artifact_hash_manifest,
    _write_json_atomic,
    record_recovery_event,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "23_phase_b_colab_recovery.py"


def test_recovery_json_writer_preserves_manifest_byte_convention(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_json_atomic(path, {"a": 1, "b": [2, 3]})
    assert path.read_text(encoding="utf-8") == json.dumps(
        {"a": 1, "b": [2, 3]}, indent=2
    )
    assert not path.read_bytes().endswith(b"\n")
    assert not path.with_suffix(".json.tmp").exists()


def test_recovery_journal_is_atomic_and_append_only(tmp_path: Path) -> None:
    journal = tmp_path / "journal.json"
    record_recovery_event(journal, "first", "completed", {"value": 1})
    record_recovery_event(journal, "second", "completed")
    payload = json.loads(journal.read_text(encoding="utf-8"))
    assert payload["automatic_resume_authorized"] is False
    assert [item["stage"] for item in payload["events"]] == ["first", "second"]
    assert not journal.with_suffix(".json.tmp").exists()


def test_recovery_integrity_manifest_excludes_only_lifecycle_files(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("data", encoding="utf-8")
    (tmp_path / "recovery_stage_journal.json").write_text("journal", encoding="utf-8")
    (tmp_path / "integrity_manifest.json").write_text("manifest", encoding="utf-8")
    (tmp_path / "leftover.tmp").write_text("temporary", encoding="utf-8")
    result = _artifact_hash_manifest(tmp_path)
    assert set(result) == {"data.txt"}
    assert len(result["data.txt"]) == 64


def test_recovery_cli_validation_is_side_effect_free(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--validate-only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "phase_b0_colab_recovery_runner_valid_unauthorized"
    assert payload["recovery_authorized"] is False
    assert payload["models_trained"] is False
    assert payload["recovery_output_created"] is False
    assert list(tmp_path.iterdir()) == []


def test_recovery_cli_rejects_non_drive_root_before_output_when_authorized(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "drive"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--recovery",
            "--artifact-root",
            str(artifact_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "must be written under the frozen Google Drive root" in completed.stderr
    assert not artifact_root.exists()
