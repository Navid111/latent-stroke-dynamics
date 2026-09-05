from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from latent_stroke_dynamics.quadratic_bezier_comparison import (
    EXPECTED_PAIR_COUNT,
    EXPECTED_RUN_COUNT,
    FREEZE_MANIFEST_PATH,
    FROZEN_TARGET_HASHES,
    FROZEN_TARGET_SET_SHA256,
    _blinded_primitives,
    evaluate_quantitative_decision,
    run_fixed_comparison,
    run_target_seed_condition,
    validate_only_comparison_report,
    validate_target_freeze,
)
from latent_stroke_dynamics.quadratic_bezier_extension import (
    SEEDS,
    TARGET_IDS,
    GeneratedTarget,
)
from latent_stroke_dynamics.rgb_coarse_to_fine import PainterConfig, StageConfig


def _pair_rows(curve_by_target: list[float], straight: float = 1.0) -> list[dict]:
    rows = []
    for target_id, curve in zip(TARGET_IDS, curve_by_target, strict=True):
        for seed in SEEDS:
            rows.append(
                {
                    "target_id": target_id,
                    "seed": seed,
                    "straight_mse_512": straight,
                    "quadratic_bezier_mse_512": curve,
                }
            )
    return rows


def test_target_freeze_matches_generated_pixels_and_schedule() -> None:
    freeze = validate_target_freeze(FREEZE_MANIFEST_PATH)
    assert freeze["target_hashes_frozen"] is True
    assert freeze["target_set_sha256"] == FROZEN_TARGET_SET_SHA256
    assert freeze["target_sha256"] == FROZEN_TARGET_HASHES
    assert freeze["target_order"] == list(TARGET_IDS)
    assert freeze["seed_order"] == list(SEEDS)
    assert freeze["execution_authorized"] is False


def test_target_freeze_rejects_one_changed_hash(tmp_path: Path) -> None:
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    freeze["target_sha256"][TARGET_IDS[0]] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="target_sha256"):
        validate_target_freeze(path)


def test_runner_validation_is_output_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = FREEZE_MANIFEST_PATH.resolve()
    monkeypatch.chdir(tmp_path)
    report = validate_only_comparison_report(freeze)
    assert list(tmp_path.iterdir()) == []
    assert report["status"] == "quadratic_bezier_comparison_runner_valid_no_outputs"
    assert report["expected_pair_count"] == EXPECTED_PAIR_COUNT == 18
    assert report["expected_run_count"] == EXPECTED_RUN_COUNT == 36
    assert report["output_side_effects"] is False
    assert report["execution_authorized"] is False
    assert report["comparative_outputs_viewed"] is False
    assert all(item["deterministic"] for item in report["synthetic_smoke"].values())
    assert all(item["monotonic"] for item in report["synthetic_smoke"].values())


def test_execution_refuses_unauthorized_manifest_without_output(tmp_path: Path) -> None:
    output = tmp_path / "should-not-exist"
    with pytest.raises(PermissionError):
        run_fixed_comparison(
            output_dir=output,
            authorization_path=FREEZE_MANIFEST_PATH,
            source_commit="f" * 40,
            freeze_path=FREEZE_MANIFEST_PATH,
        )
    assert not output.exists()
    assert not output.with_name(output.name + ".incomplete").exists()


def test_material_gain_remains_pending_blinded_review() -> None:
    decision = evaluate_quantitative_decision(
        _pair_rows([0.90] * 6), integrity_passed=True
    )
    assert decision["improved_target_count"] == 6
    assert decision["quantitatively_materially_eligible"] is True
    assert decision["provisional_decision"] == "material_improvement_pending_blinded_review"
    assert decision["qualitative_review_required"] is True
    assert decision["final_decision"] is None


def test_positive_gain_below_threshold_is_minor() -> None:
    decision = evaluate_quantitative_decision(
        _pair_rows([0.97] * 6), integrity_passed=True
    )
    assert decision["quantitatively_materially_eligible"] is False
    assert decision["provisional_decision"] == "minor_improvement"
    assert decision["final_decision"] == "minor_improvement"


def test_no_aggregate_gain_is_no_material_improvement() -> None:
    decision = evaluate_quantitative_decision(
        _pair_rows([1.01] * 6), integrity_passed=True
    )
    assert decision["provisional_decision"] == "no_material_improvement"
    assert decision["final_decision"] == "no_material_improvement"


def test_per_target_worsening_guard_blocks_material_classification() -> None:
    decision = evaluate_quantitative_decision(
        _pair_rows([0.80, 0.80, 0.80, 0.80, 0.80, 1.06]),
        integrity_passed=True,
    )
    assert decision["curve_improvement_fraction"] > 0.05
    assert decision["checks"]["per_target_worsening_threshold_passed"] is False
    assert decision["provisional_decision"] == "minor_improvement"


def test_integrity_failure_forces_no_material_improvement() -> None:
    decision = evaluate_quantitative_decision(
        _pair_rows([0.80] * 6), integrity_passed=False
    )
    assert decision["provisional_decision"] == "no_material_improvement"
    assert decision["final_decision"] == "no_material_improvement"


def test_blinded_assignment_is_deterministic() -> None:
    first = [_blinded_primitives(target_id, seed) for target_id in TARGET_IDS for seed in SEEDS]
    second = [_blinded_primitives(target_id, seed) for target_id in TARGET_IDS for seed in SEEDS]
    assert first == second
    assert all(set(item) == {"straight", "quadratic_bezier"} for item in first)


def test_one_tiny_condition_writes_complete_hashed_evidence(tmp_path: Path) -> None:
    image = Image.new("RGB", (32, 32), color=(80, 110, 140))
    target = GeneratedTarget(
        TARGET_IDS[0],
        "synthetic_test",
        "Original synthetic unit-test target.",
        image,
    )
    config = PainterConfig(
        planning_size=32,
        replay_size=32,
        supersample=1,
        candidates_per_pool=8,
        error_guided_fraction=0.75,
        patience=3,
        min_improvement=1e-9,
        seed=73,
        gif_stride=1,
        max_attempts_per_candidate=100,
        stages=(StageConfig("smoke", 2, 0.20, 0.80, 0.12, 0.30),),
    )
    summary = run_target_seed_condition(
        target,
        primitive="straight",
        seed=73,
        target_stream=0,
        output_dir=tmp_path / "run",
        config=config,
    )
    assert summary["status"] == "quadratic_bezier_comparison_run_complete"
    assert summary["every_executed_stroke_improved"] is True
    assert summary["candidate_renders"] >= 16
    assert (tmp_path / "run" / "summary.json").is_file()
    assert (tmp_path / "run" / "summary.sha256").is_file()
    assert (tmp_path / "run" / "final_512.png").is_file()
