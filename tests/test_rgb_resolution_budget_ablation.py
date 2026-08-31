from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from latent_stroke_dynamics.rgb_coarse_to_fine import TARGET_SPECS
from latent_stroke_dynamics.rgb_resolution_budget_ablation import (
    BASELINE_AGGREGATE_SHA256,
    BASELINE_MEAN_512_MSE,
    CONDITIONS,
    NEW_CONDITIONS,
    config_for_condition,
    evaluate_quantitative_decision,
    marginal_gain_from_progress,
    protocol_manifest,
    validate_baseline_result,
    validate_protocol,
)


def _aggregate(
    condition_id: str,
    values: list[float],
    *,
    integrity: bool = True,
) -> dict:
    return {
        "condition": {"condition_id": condition_id},
        "targets": [
            {
                "target_id": spec.target_id,
                "high_resolution_final_mse": value,
            }
            for spec, value in zip(TARGET_SPECS, values, strict=True)
        ],
        "acceptance_checks": {
            "all_five_targets_completed": integrity,
            "all_executed_strokes_improved": integrity,
            "all_best_frames_not_worse_than_final": integrity,
            "all_frozen_decisions_preserved": integrity,
        },
        "training_performed": False,
        "learned_model_used": False,
        "frozen_phase_b0_decision_changed": False,
    }


def _baseline(values: list[float]) -> dict:
    return {
        "targets": [
            {
                "target_id": spec.target_id,
                "high_resolution_final_mse": value,
            }
            for spec, value in zip(TARGET_SPECS, values, strict=True)
        ]
    }


def test_protocol_has_only_the_locked_three_new_runs() -> None:
    manifest = validate_protocol()
    assert [condition.condition_id for condition in CONDITIONS] == ["A", "B", "C", "D"]
    assert [condition.condition_id for condition in NEW_CONDITIONS] == ["B", "C", "D"]
    expected = {
        "A": (96, 210, False),
        "B": (96, 420, True),
        "C": (128, 210, True),
        "D": (128, 420, True),
    }
    for condition in CONDITIONS:
        assert (
            condition.planning_size,
            condition.total_strokes,
            condition.execute,
        ) == expected[condition.condition_id]
    assert manifest == protocol_manifest()
    assert manifest["baseline_mean_512_mse"] == BASELINE_MEAN_512_MSE


def test_only_factor_settings_change_across_conditions() -> None:
    configs = {condition.condition_id: config_for_condition(condition) for condition in CONDITIONS}
    for config in configs.values():
        assert config.replay_size == 512
        assert config.supersample == 2
        assert config.candidates_per_pool == 64
        assert config.error_guided_fraction == 0.80
        assert config.patience == 12
        assert config.min_improvement == 1e-9
        assert config.seed == 73
        assert config.gif_stride == 3
    assert [stage.max_steps for stage in configs["A"].stages] == [40, 70, 100]
    assert [stage.max_steps for stage in configs["B"].stages] == [80, 140, 200]
    assert [stage.max_steps for stage in configs["C"].stages] == [40, 70, 100]
    assert [stage.max_steps for stage in configs["D"].stages] == [80, 140, 200]


def test_decision_requires_mean_gain_and_per_target_guard() -> None:
    baseline = _baseline([1.0] * 5)
    summaries = {
        "B": _aggregate("B", [0.89] * 5),
        "C": _aggregate("C", [0.95] * 5),
        "D": _aggregate("D", [0.85, 0.85, 0.85, 0.85, 1.06]),
    }
    decision = evaluate_quantitative_decision(baseline, summaries)
    records = {item["condition_id"]: item for item in decision["conditions"]}
    assert records["B"]["quantitatively_eligible"] is True
    assert records["C"]["quantitatively_eligible"] is False
    assert records["D"]["mean_improvement_threshold_passed"] is True
    assert records["D"]["per_target_worsening_threshold_passed"] is False
    assert decision["selected_condition_pending_visual_review"] == "B"
    assert decision["final_decision"] is None


def test_least_expensive_near_best_condition_is_selected() -> None:
    baseline = _baseline([1.0] * 5)
    summaries = {
        "B": _aggregate("B", [0.890] * 5),
        "C": _aggregate("C", [0.895] * 5),
        "D": _aggregate("D", [0.950] * 5),
    }
    decision = evaluate_quantitative_decision(baseline, summaries)
    # C lies within one percent of B and has a smaller pixel-step compute proxy.
    assert decision["selected_condition_pending_visual_review"] == "C"


def test_integrity_failure_blocks_eligibility() -> None:
    baseline = _baseline([1.0] * 5)
    summaries = {
        "B": _aggregate("B", [0.80] * 5, integrity=False),
        "C": _aggregate("C", [0.95] * 5),
        "D": _aggregate("D", [0.95] * 5),
    }
    decision = evaluate_quantitative_decision(baseline, summaries)
    assert decision["quantitatively_eligible_conditions"] == []
    assert decision["provisional_decision"] == "retain_archived_baseline"
    assert decision["final_decision"] == "retain_archived_baseline"


def test_marginal_gain_uses_requested_tail(tmp_path: Path) -> None:
    path = tmp_path / "progress.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mse_before", "mse_after"])
        writer.writeheader()
        writer.writerows(
            [
                {"mse_before": 1.0, "mse_after": 0.8},
                {"mse_before": 0.8, "mse_after": 0.6},
                {"mse_before": 0.6, "mse_after": 0.5},
            ]
        )
    result = marginal_gain_from_progress(path, tail_size=2)
    assert result["used_tail_size"] == 2
    assert result["start_mse"] == pytest.approx(0.8)
    assert result["end_mse"] == pytest.approx(0.5)
    assert result["absolute_gain"] == pytest.approx(0.3)


def test_baseline_validation_fails_before_accepting_substitute(tmp_path: Path) -> None:
    (tmp_path / "aggregate_summary.json").write_text(
        json.dumps({"status": "substitute"}),
        encoding="utf-8",
    )
    (tmp_path / "aggregate_summary.sha256").write_text(
        BASELINE_AGGREGATE_SHA256 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_baseline_result(tmp_path)
