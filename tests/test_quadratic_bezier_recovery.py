from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest

from latent_stroke_dynamics.quadratic_bezier_comparison import (
    FREEZE_MANIFEST_PATH,
    FROZEN_TARGET_HASHES,
)
from latent_stroke_dynamics.quadratic_bezier_extension import (
    PROTOCOL_ID,
    TARGET_IDS,
    extension_painter_config,
)
from latent_stroke_dynamics.quadratic_bezier_recovery import (
    FROZEN_RUNNER_COMMIT,
    RECOVERY_PLAN_PATH,
    _quarantine_partial_unit,
    _strict_hash_tree,
    expected_unit_schedule,
    load_recovery_plan,
    run_interrupted_comparison_recovery,
    validate_completed_run_directory,
    validate_only_recovery_report,
    validate_recovery_source_continuity,
)
from latent_stroke_dynamics.rgb_coarse_to_fine import file_sha256


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_AUTHORIZATION = (
    ROOT / "configs" / "quadratic-bezier-execution-authorization-2026-09-04.json"
)


def _write_valid_tiny_completed_unit(run_dir: Path, unit: str) -> None:
    target_id, seed_text, primitive = unit.split("/")
    seed = int(seed_text.removeprefix("seed_"))
    run_dir.mkdir(parents=True)
    artifact = run_dir / "artifact.txt"
    artifact.write_text("preserved", encoding="utf-8")
    actions = run_dir / "actions.json"
    actions.write_text(
        json.dumps([{"primitive": primitive}]), encoding="utf-8"
    )
    progress = run_dir / "progress.csv"
    progress.write_text(
        "step,primitive,mse_before,mse_after,improvement\n"
        f"1,{primitive},0.2,0.1,0.1\n",
        encoding="utf-8",
    )
    run_config = {
        "protocol_id": PROTOCOL_ID,
        "target_id": target_id,
        "target_stream": TARGET_IDS.index(target_id),
        "target_pixel_sha256": FROZEN_TARGET_HASHES[target_id],
        "primitive": primitive,
        "seed": seed,
        "painter_config": json.loads(
            json.dumps(asdict(extension_painter_config(seed)))
        ),
        "selection": "exact_rendered_rgb_target_pixel_mse",
        "training_performed": False,
        "learned_model_used": False,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config), encoding="utf-8"
    )
    artifacts = _strict_hash_tree(run_dir)
    summary = {
        "status": "quadratic_bezier_comparison_run_complete",
        "protocol_id": PROTOCOL_ID,
        "target_id": target_id,
        "target_stream": TARGET_IDS.index(target_id),
        "target_pixel_sha256": FROZEN_TARGET_HASHES[target_id],
        "primitive": primitive,
        "seed": seed,
        "planning_size": 128,
        "replay_size": 512,
        "initial_mse": 0.2,
        "best_mse": 0.1,
        "final_mse": 0.1,
        "final_mae": 0.1,
        "best_step": 1,
        "executed_strokes": 1,
        "maximum_strokes": 420,
        "candidate_pools": 1,
        "candidates_per_pool": 64,
        "candidate_renders": 64,
        "stage_stats": [
            {"maximum_steps": 80, "executed_steps": 1, "candidate_pools": 1},
            {"maximum_steps": 140, "executed_steps": 0, "candidate_pools": 0},
            {"maximum_steps": 200, "executed_steps": 0, "candidate_pools": 0},
        ],
        "every_executed_stroke_improved": True,
        "best_not_worse_than_final": True,
        "high_resolution_best_mse": 0.1,
        "high_resolution_final_mse": 0.1,
        "high_resolution_final_mae": 0.1,
        "runtime_seconds": 1.0,
        "training_performed": False,
        "learned_model_used": False,
        "closed_experiments_changed": False,
        "artifact_sha256": artifacts,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "summary.sha256").write_text(
        file_sha256(summary_path) + "\n", encoding="utf-8"
    )


def test_recovery_plan_matches_stable_post_termination_diagnostic() -> None:
    plan = load_recovery_plan(ROOT / RECOVERY_PLAN_PATH)
    assert plan["expected_completed_run_count"] == 17
    assert plan["expected_partial_units"] == [
        "03_organic_silhouette/seed_211/quadratic_bezier"
    ]
    assert plan["expected_not_started_run_count"] == 18
    assert plan["expected_missing_run_count"] == 19
    assert plan["stable_diagnostic_repetitions"] == 2
    assert plan["recovery_execution_authorized"] is False


def test_recovery_missing_units_follow_original_frozen_schedule() -> None:
    plan = load_recovery_plan(ROOT / RECOVERY_PLAN_PATH)
    schedule = expected_unit_schedule()
    assert len(schedule) == 36
    assert len(set(schedule)) == 36
    expected_missing = tuple(
        unit for unit in schedule if unit not in set(plan["expected_completed_units"])
    )
    assert tuple(plan["expected_missing_units"]) == expected_missing
    assert expected_missing[0] == "03_organic_silhouette/seed_211/quadratic_bezier"


def test_frozen_runner_dependencies_are_unchanged() -> None:
    result = validate_recovery_source_continuity(ROOT)
    assert result["frozen_runner_commit"] == FROZEN_RUNNER_COMMIT
    assert result["verified_unchanged_path_count"] >= 7


def test_recovery_validation_is_output_free_and_unauthorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    report = validate_only_recovery_report(
        repo_root=ROOT,
        plan_path=ROOT / RECOVERY_PLAN_PATH,
        freeze_path=ROOT / FREEZE_MANIFEST_PATH,
    )
    assert list(tmp_path.iterdir()) == []
    assert report["status"] == (
        "quadratic_bezier_recovery_implementation_valid_no_outputs_unauthorized"
    )
    assert report["preserved_completed_run_count"] == 17
    assert report["missing_run_count"] == 19
    assert report["recovery_execution_authorized"] is False
    assert report["interrupted_output_accessed"] is False
    assert report["recovery_output_created"] is False
    assert report["comparative_metrics_revealed"] is False


def test_completed_unit_validator_detects_artifact_tampering(tmp_path: Path) -> None:
    unit = "01_ring_symbol/seed_73/straight"
    run_dir = tmp_path / "runs" / unit
    _write_valid_tiny_completed_unit(run_dir, unit)
    result = validate_completed_run_directory(run_dir, unit=unit)
    assert result["status"] == "quadratic_bezier_comparison_run_complete"
    (run_dir / "artifact.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hashes"):
        validate_completed_run_directory(run_dir, unit=unit)


def test_partial_quarantine_preserves_every_byte(tmp_path: Path) -> None:
    unit = "03_organic_silhouette/seed_211/quadratic_bezier"
    partial = tmp_path / "runs" / unit
    partial.mkdir(parents=True)
    (partial / "one.bin").write_bytes(b"one")
    (partial / "nested").mkdir()
    (partial / "nested" / "two.bin").write_bytes(b"two")
    manifest = _strict_hash_tree(partial)
    destination = _quarantine_partial_unit(
        tmp_path,
        unit=unit,
        expected_manifest=manifest,
    )
    assert not partial.exists()
    assert destination.is_dir()
    assert _strict_hash_tree(destination) == manifest


def test_recovery_refuses_without_separate_authorization_before_mutation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "quadratic-bezier-fixed-comparison-v1"
    missing_authorization = tmp_path / "missing-recovery-authorization.json"
    with pytest.raises(FileNotFoundError):
        run_interrupted_comparison_recovery(
            output_dir=output_dir,
            original_authorization_path=ORIGINAL_AUTHORIZATION,
            recovery_authorization_path=missing_authorization,
            recovery_implementation_commit=subprocess_head(),
            repo_root=ROOT,
            plan_path=ROOT / RECOVERY_PLAN_PATH,
            freeze_path=ROOT / FREEZE_MANIFEST_PATH,
        )
    assert not output_dir.exists()
    assert not output_dir.with_name(output_dir.name + ".incomplete").exists()


def subprocess_head() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
