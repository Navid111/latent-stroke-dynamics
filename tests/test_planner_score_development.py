from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from latent_stroke_dynamics.extension_training import create_patch_predictor
from latent_stroke_dynamics.latent_planner import load_latent_planner_config
from latent_stroke_dynamics.planner_score_alignment import load_score_alignment_config
from latent_stroke_dynamics.planner_score_development import (
    PLANNER_DEVELOPMENT_METHODS,
    PLANNER_DEVELOPMENT_SEEDS,
    PLANNER_DEVELOPMENT_TARGET_SEEDS,
    aggregate_planner_development_summary,
    load_frozen_development_selection,
    make_planner_development_decision,
    normalized_latent_l1_candidate_scores,
    planner_development_output_paths,
    require_planner_development_authorized,
    require_planner_development_outputs_absent,
    run_selected_score_planner,
    should_take_no_op,
    validate_planner_development_resources,
    validate_planner_development_runner_request,
    validate_planner_development_summary,
)
from latent_stroke_dynamics.planning import ProposalConfig
from latent_stroke_dynamics.representation_extension import (
    LatentChannelStatistics,
    StrokeAutoencoder,
)
from latent_stroke_dynamics.renderer import Stroke, random_base_canvas


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "planner-score-alignment-2026-08-23.json"
CLOSED_CONFIG = ROOT / "configs" / "latent-planner-2026-08-23.json"
SELECTION = ROOT / "results" / "planner-score-alignment" / "development-selection.json"


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def frozen_predictors() -> tuple[torch.nn.Module, ...]:
    models = []
    for seed in (11, 22, 33):
        torch.manual_seed(seed)
        models.append(freeze(create_patch_predictor("mlp", 32, (16, 16), 256)))
    return tuple(models)


def test_archived_selection_and_all_frozen_resources_match() -> None:
    config = load_score_alignment_config(CONFIG)
    closed = load_latent_planner_config(CLOSED_CONFIG)
    selection = load_frozen_development_selection(SELECTION)
    assert selection["selection"]["predictor_family"] == "mse_only"
    assert selection["selection"]["score_name"] == "normalized_latent_l1"
    checks = validate_planner_development_resources(config, closed, selection)
    assert all(checks.values())


def test_planner_development_validation_is_unauthorized_and_side_effect_free(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_score_alignment_config(CONFIG))
    config["status"] = "development_score_audit_complete_planner_unauthorized"
    config["planner_development"]["authorized"] = False
    config["planner_development"]["output_dir"] = str(tmp_path / "planner-development")
    selection = load_frozen_development_selection(SELECTION)
    result = validate_planner_development_runner_request(config, selection)
    assert result["status"] == "planner_score_planner_development_runner_valid_unauthorized"
    assert result["target_count"] == 3
    assert result["methods"] == list(PLANNER_DEVELOPMENT_METHODS)
    assert result["selected_predictor_family"] == "mse_only"
    assert result["selected_score_name"] == "normalized_latent_l1"
    assert result["models_loaded"] is False
    assert result["targets_generated"] is False
    assert result["planner_data_generated"] is False
    assert result["models_trained_or_finetuned"] is False
    assert not (tmp_path / "planner-development").exists()


def test_planner_development_guards_authorization_and_incomplete_output(
    tmp_path: Path,
) -> None:
    config = deepcopy(load_score_alignment_config(CONFIG))
    config["planner_development"]["output_dir"] = str(tmp_path / "planner-development")
    require_planner_development_authorized(config)
    unauthorized = deepcopy(config)
    unauthorized["status"] = "development_score_audit_complete_planner_unauthorized"
    unauthorized["planner_development"]["authorized"] = False
    with pytest.raises(PermissionError, match="not authorized"):
        require_planner_development_authorized(unauthorized)
    paths = planner_development_output_paths(config)
    paths.incomplete.mkdir()
    marker = paths.incomplete / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incomplete"):
        require_planner_development_outputs_absent(paths)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_normalized_latent_l1_scores_match_manual_ensemble_mean() -> None:
    predictors = frozen_predictors()[:2]
    generator = torch.Generator().manual_seed(303)
    current = torch.randn(1, 256, 32, generator=generator)
    target = torch.randn(1, 256, 32, generator=generator)
    candidates = (
        Stroke(0.1, 0.2, 0.9, 0.8, width=2, value=32),
        Stroke(0.2, 0.8, 0.8, 0.2, width=4, value=96),
    )
    aggregate, per_model = normalized_latent_l1_candidate_scores(
        predictors,
        current,
        target,
        candidates,
        batch_size=1,
    )
    from latent_stroke_dynamics.planner_score_alignment import predict_candidate_latents

    predicted = predict_candidate_latents(
        predictors,
        current,
        candidates,
        batch_size=1,
    )
    manual = (
        F.normalize(predicted, dim=-1)
        - F.normalize(target, dim=-1)[None, :, :, :]
    ).abs().mean(dim=(2, 3)).numpy()
    assert np.array_equal(per_model, manual.astype(np.float64))
    assert np.allclose(aggregate, per_model.mean(axis=0), rtol=0.0, atol=0.0)


def test_zero_margin_no_op_rule_is_exact_and_untuned() -> None:
    scores = np.asarray([0.2, 0.3, 0.4])
    assert should_take_no_op(0.2, scores, margin=0.0) is True
    assert should_take_no_op(0.199, scores, margin=0.0) is True
    assert should_take_no_op(0.201, scores, margin=0.0) is False
    with pytest.raises(ValueError, match="margin"):
        should_take_no_op(0.2, scores, margin=0.1)


def test_selected_score_forced_planner_is_deterministic_and_observes_exactly() -> None:
    autoencoder = freeze(StrokeAutoencoder())
    statistics = LatentChannelStatistics(mean=torch.zeros(32), std=torch.ones(32))
    predictors = frozen_predictors()
    target = random_base_canvas(
        size=64,
        prior_strokes=3,
        rng=np.random.default_rng(707),
    )
    proposal = ProposalConfig(count=4)
    first = run_selected_score_planner(
        target,
        autoencoder,
        statistics,
        predictors,
        maximum_steps=2,
        seed=909,
        proposal_config=proposal,
        prediction_batch_size=2,
        allow_no_op=False,
        capture_frames=True,
    )
    second = run_selected_score_planner(
        target,
        autoencoder,
        statistics,
        predictors,
        maximum_steps=2,
        seed=909,
        proposal_config=proposal,
        prediction_batch_size=2,
        allow_no_op=False,
        capture_frames=False,
    )
    assert first.method == "development_selected_score_forced"
    assert len(first.steps) == 2
    assert first.steps == second.steps
    assert np.array_equal(np.asarray(first.final_canvas), np.asarray(second.final_canvas))
    assert first.target_encoding_count == 1
    assert first.observed_canvas_encoding_count == 2
    assert first.proposal_rounds_evaluated == 2
    assert first.stop_decision is None


def synthetic_summary(config: dict) -> pd.DataFrame:
    final_values = {
        "exact_pixel": 0.06,
        "learned_pixel": 0.08,
        "current_latent_mse_forced": 0.10,
        "development_selected_score_forced": 0.095,
        "development_selected_score_no_op": 0.09,
    }
    rows = []
    for index, (target_seed, planner_seed) in enumerate(
        zip(
            PLANNER_DEVELOPMENT_TARGET_SEEDS,
            PLANNER_DEVELOPMENT_SEEDS,
            strict=True,
        ),
        start=1,
    ):
        for method in PLANNER_DEVELOPMENT_METHODS:
            learned = method != "exact_pixel"
            latent = method in (
                "current_latent_mse_forced",
                "development_selected_score_forced",
                "development_selected_score_no_op",
            )
            no_op = method == "development_selected_score_no_op"
            executed = 60 if no_op else 100
            final = final_values[method]
            rows.append(
                {
                    "target_id": f"target_{index:02d}",
                    "target_seed": target_seed,
                    "planner_seed": planner_seed,
                    "method": method,
                    "maximum_steps": 100,
                    "executed_steps": executed,
                    "candidates_per_step": 128,
                    "initial_mse": 0.20,
                    "final_mse": final,
                    "best_mse": final,
                    "best_step": executed,
                    "final_mae": 0.10,
                    "relative_final_mse_improvement": (0.20 - final) / 0.20,
                    "relative_best_mse_improvement": (0.20 - final) / 0.20,
                    "improved_steps": 50,
                    "elapsed_seconds": 1.0,
                    "stopped_early": no_op,
                    "stop_round": 61 if no_op else None,
                    "current_score_at_stop": 0.01 if no_op else None,
                    "best_candidate_score_at_stop": 0.011 if no_op else None,
                    "exact_top1_rate": 0.2 if learned else None,
                    "exact_top5_rate": 0.6 if learned else None,
                    "mean_exact_rank": 5.0 if learned else None,
                    "mean_exact_regret": 0.001 if learned else None,
                    "max_exact_regret": 0.01 if learned else None,
                    "mean_score_exact_spearman": 0.5 if latent else None,
                }
            )
    return pd.DataFrame(rows)


def test_planner_development_aggregation_and_eligibility_decision() -> None:
    config = load_score_alignment_config(CONFIG)
    summary = synthetic_summary(config)
    validate_planner_development_summary(summary, config)
    aggregate = aggregate_planner_development_summary(summary)
    assert tuple(aggregate["method"]) == PLANNER_DEVELOPMENT_METHODS
    assert set(aggregate["targets"]) == {3}
    decision = make_planner_development_decision(
        summary,
        config,
        implementation_integrity_passed=True,
        selected_pair_matches=True,
    )
    assert decision["status"] == "eligible_for_confirmatory"
    assert all(decision["criteria_passed"].values())
    assert np.isclose(
        decision[
            "selected_no_op_mean_final_mse_reduction_vs_current_latent_mse_forced"
        ],
        0.1,
    )
