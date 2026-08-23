from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from latent_stroke_dynamics.extension_training import create_patch_predictor
from latent_stroke_dynamics.latent_planner import load_latent_planner_config
from latent_stroke_dynamics.planner_score_alignment import (
    PREDICTOR_FAMILIES,
    SCORE_NAMES,
    candidate_score_variants,
    exact_candidate_metrics,
    load_score_alignment_config,
    pixel_error_patch_weights,
    predict_candidate_latents,
    require_score_audit_authorized,
    require_score_audit_outputs_absent,
    score_audit_output_paths,
    select_score_pair,
    validate_closed_resource_references,
    validate_score_alignment_config,
    validate_score_audit_runner_request,
)
from latent_stroke_dynamics.representation_extension import (
    LatentChannelStatistics,
    StrokeAutoencoder,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "planner-score-alignment-2026-08-23.json"
CLOSED_CONFIG = ROOT / "configs" / "latent-planner-2026-08-23.json"


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_score_alignment_audit_is_closed_and_resources_match() -> None:
    config = load_score_alignment_config(CONFIG)
    closed = load_latent_planner_config(CLOSED_CONFIG)
    assert config["status"] == "planner_development_complete_confirmatory_unauthorized"
    assert config["development_score_audit"]["authorized"] is False
    assert config["planner_development"]["authorized"] is False
    assert config["confirmatory_reserved"]["authorized"] is False
    assert tuple(config["development_score_audit"]["scores"]) == SCORE_NAMES
    assert tuple(config["development_score_audit"]["predictor_families"]) == PREDICTOR_FAMILIES
    with pytest.raises(PermissionError, match="not authorized"):
        require_score_audit_authorized(config)
    authorized = deepcopy(config)
    authorized["status"] = "development_score_audit_authorized_once"
    authorized["development_score_audit"]["authorized"] = True
    authorized["planner_development"]["authorized"] = False
    require_score_audit_authorized(authorized)
    result = validate_closed_resource_references(config, closed)
    assert all(result.values())


def test_score_alignment_config_rejects_score_drift() -> None:
    config = load_score_alignment_config(CONFIG)
    broken = deepcopy(config)
    broken["development_score_audit"]["sobel_edge_weight"] = 1.0
    with pytest.raises(ValueError, match="score-audit settings"):
        validate_score_alignment_config(broken)


def test_score_audit_validation_is_unauthorized_and_side_effect_free(tmp_path: Path) -> None:
    config = deepcopy(load_score_alignment_config(CONFIG))
    config["status"] = "frozen_before_implementation_and_data"
    config["development_score_audit"]["authorized"] = False
    config["planner_development"]["authorized"] = False
    output = tmp_path / "score-audit"
    config["development_score_audit"]["output_dir"] = str(output)
    result = validate_score_audit_runner_request(config)
    assert result["status"] == "planner_score_audit_runner_valid_unauthorized"
    assert result["candidate_sets"] == 72
    assert result["predictor_score_pairs"] == 10
    assert result["models_loaded"] is False
    assert result["targets_generated"] is False
    assert result["state_trajectories_generated"] is False
    assert result["candidate_sets_generated"] is False
    assert result["models_trained_or_finetuned"] is False
    assert not output.exists()
    assert not (tmp_path / "score-audit.incomplete").exists()
    with pytest.raises(PermissionError, match="not authorized"):
        require_score_audit_authorized(config)


def test_score_audit_output_guard_preserves_incomplete_evidence(tmp_path: Path) -> None:
    config = deepcopy(load_score_alignment_config(CONFIG))
    config["development_score_audit"]["output_dir"] = str(tmp_path / "score-audit")
    paths = score_audit_output_paths(config)
    paths.incomplete.mkdir()
    marker = paths.incomplete / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incomplete"):
        require_score_audit_outputs_absent(paths)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_candidate_latent_predictions_are_finite_deterministic_and_frozen() -> None:
    torch.manual_seed(11)
    first = freeze(create_patch_predictor("mlp", 32, (16, 16), 256))
    torch.manual_seed(22)
    second = freeze(create_patch_predictor("mlp", 32, (16, 16), 256))
    generator = torch.Generator().manual_seed(919)
    current = torch.randn(1, 256, 32, generator=generator)
    candidates = (
        Stroke(0.1, 0.2, 0.9, 0.8, width=2, value=32),
        Stroke(0.2, 0.8, 0.8, 0.2, width=4, value=96),
    )
    first_result = predict_candidate_latents(
        (first, second), current, candidates, batch_size=1
    )
    second_result = predict_candidate_latents(
        (first, second), current, candidates, batch_size=1
    )
    assert first_result.shape == (2, 2, 256, 32)
    assert torch.equal(first_result, second_result)
    assert bool(torch.isfinite(first_result).all())


def test_all_five_frozen_scores_are_finite_and_have_expected_shapes() -> None:
    generator = torch.Generator().manual_seed(808)
    predicted = torch.randn(2, 3, 256, 32, generator=generator)
    target_tokens = torch.randn(1, 256, 32, generator=generator)
    target_pixels = torch.rand(1, 1, 64, 64, generator=generator)
    autoencoder = freeze(StrokeAutoencoder())
    statistics = LatentChannelStatistics(
        mean=torch.zeros(32),
        std=torch.ones(32),
    )
    target = render_stroke(
        blank_canvas(64),
        Stroke(0.1, 0.1, 0.9, 0.9, width=3, value=32),
    )
    weights = pixel_error_patch_weights(blank_canvas(64), target)
    results = candidate_score_variants(
        predicted,
        target_tokens,
        target_pixels,
        weights,
        autoencoder,
        statistics,
        batch_size=2,
    )
    assert tuple(results) == SCORE_NAMES
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1e-6, rtol=0.0)
    for result in results.values():
        assert result.aggregate.shape == (3,)
        assert result.per_model.shape == (2, 3)
        assert np.isfinite(result.aggregate).all()
        assert np.isfinite(result.per_model).all()
        assert np.allclose(
            result.aggregate,
            result.per_model.mean(axis=0),
            rtol=1e-7,
            atol=1e-9,
        )


def test_exact_metrics_and_frozen_pair_selection() -> None:
    metrics = exact_candidate_metrics(
        np.asarray([0.3, 0.2, 0.1]),
        np.asarray([0.4, 0.1, 0.2]),
    )
    assert metrics["selected_index"] == 2
    assert metrics["exact_selected_rank"] == 2
    assert metrics["exact_top1"] is False
    assert metrics["exact_top5"] is True
    assert np.isclose(metrics["exact_regret"], 0.1)

    aggregate = pd.DataFrame(
        [
            {
                "predictor_family": "mse_only",
                "score_name": "normalized_latent_mse",
                "mean_exact_regret": 0.02,
                "exact_top5_rate": 0.5,
                "mean_score_exact_spearman": 0.4,
            },
            {
                "predictor_family": "ranking_aware",
                "score_name": "decoded_pixel_l1",
                "mean_exact_regret": 0.01,
                "exact_top5_rate": 0.4,
                "mean_score_exact_spearman": 0.3,
            },
        ]
    )
    winner = select_score_pair(aggregate)
    assert winner["predictor_family"] == "ranking_aware"
    assert winner["score_name"] == "decoded_pixel_l1"
