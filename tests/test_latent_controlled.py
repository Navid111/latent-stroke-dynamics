from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from latent_stroke_dynamics.latent_controlled import (
    CONTROLLED_METHODS,
    CONTROLLED_PLANNER_SEEDS,
    CONTROLLED_TARGET_SEEDS,
    aggregate_controlled_summary,
    controlled_output_paths,
    make_controlled_decision,
    require_controlled_authorized,
    require_controlled_outputs_absent,
    validate_controlled_runner_request,
    validate_controlled_summary,
)
from latent_stroke_dynamics.latent_planner import load_latent_planner_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "latent-planner-2026-08-23.json"


def unauthorized_config(path: Path) -> dict:
    config = deepcopy(load_latent_planner_config(CONFIG))
    config["status"] = "smoke_complete_controlled_unauthorized"
    config["controlled"]["authorized"] = False
    config["controlled"]["output_dir"] = str(path)
    return config


def synthetic_summary(config: dict, *, ranking_final: float = 0.11) -> pd.DataFrame:
    final_values = {
        "random": 0.15,
        "exact_pixel": 0.08,
        "learned_pixel": 0.09,
        "latent_mse": 0.10,
        "latent_ranking": ranking_final,
    }
    rows = []
    for index, (target_seed, planner_seed) in enumerate(
        zip(CONTROLLED_TARGET_SEEDS, CONTROLLED_PLANNER_SEEDS, strict=True),
        start=1,
    ):
        for method in CONTROLLED_METHODS:
            learned = method in ("learned_pixel", "latent_mse", "latent_ranking")
            latent = method in ("latent_mse", "latent_ranking")
            final = final_values[method]
            rows.append(
                {
                    "target_id": f"target_{index:02d}",
                    "target_seed": target_seed,
                    "planner_seed": planner_seed,
                    "method": method,
                    "steps": config["planner"]["steps"],
                    "candidates_per_step": config["planner"]["candidates_per_step"],
                    "initial_mse": 0.20,
                    "final_mse": final,
                    "best_mse": final,
                    "best_step": 100,
                    "final_mae": 0.10,
                    "relative_final_mse_improvement": (0.20 - final) / 0.20,
                    "relative_best_mse_improvement": (0.20 - final) / 0.20,
                    "improved_steps": 80,
                    "elapsed_seconds": 1.0,
                    "exact_top1_rate": 0.25 if learned else None,
                    "exact_top5_rate": 0.75 if learned else None,
                    "mean_exact_rank": 4.0 if learned else None,
                    "mean_exact_regret": 0.001 if learned else None,
                    "max_exact_regret": 0.01 if learned else None,
                    "mean_score_exact_spearman": 0.5 if latent else None,
                }
            )
    return pd.DataFrame(rows)


def test_controlled_validation_is_unauthorized_and_side_effect_free(tmp_path: Path) -> None:
    output = tmp_path / "controlled"
    result = validate_controlled_runner_request(unauthorized_config(output))
    assert result["status"] == "latent_planner_controlled_runner_valid_unauthorized"
    assert result["target_count"] == 6
    assert result["controlled_authorized"] is False
    assert result["models_loaded"] is False
    assert result["controlled_targets_generated"] is False
    assert result["controlled_planner_data_generated"] is False
    assert not output.exists()
    assert not (tmp_path / "controlled.incomplete").exists()


def test_controlled_is_closed_after_completed_execution() -> None:
    config = load_latent_planner_config(CONFIG)
    assert config["controlled"]["authorized"] is False
    assert config["smoke"]["authorized"] is False
    with pytest.raises(PermissionError, match="not authorized"):
        require_controlled_authorized(config)


def test_controlled_output_guard_preserves_incomplete_evidence(tmp_path: Path) -> None:
    config = unauthorized_config(tmp_path / "controlled")
    paths = controlled_output_paths(config)
    paths.incomplete.mkdir()
    marker = paths.incomplete / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incomplete"):
        require_controlled_outputs_absent(paths)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_controlled_aggregation_and_frozen_decision_pass() -> None:
    config = load_latent_planner_config(CONFIG)
    summary = synthetic_summary(config)
    validate_controlled_summary(summary, config)
    aggregate = aggregate_controlled_summary(summary)
    assert tuple(aggregate["method"]) == CONTROLLED_METHODS
    assert set(aggregate["targets"]) == {6}
    decision = make_controlled_decision(
        summary,
        config,
        implementation_integrity_passed=True,
    )
    assert decision["status"] == "success"
    assert decision["latent_ranking_improved_every_target"] is True
    assert decision["criteria_passed"]["minimum_mean_reduction_vs_random"] is True
    assert decision["criteria_passed"]["maximum_mean_ratio_to_exact_pixel"] is True


def test_controlled_decision_fails_without_changing_criteria() -> None:
    config = load_latent_planner_config(CONFIG)
    summary = synthetic_summary(config, ranking_final=0.16)
    decision = make_controlled_decision(
        summary,
        config,
        implementation_integrity_passed=False,
    )
    assert decision["status"] == "fail"
    assert decision["criteria_passed"]["minimum_mean_reduction_vs_random"] is False
    assert decision["criteria_passed"]["maximum_mean_ratio_to_exact_pixel"] is False
    assert decision["criteria_passed"]["implementation_integrity"] is False
