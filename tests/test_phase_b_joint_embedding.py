from copy import deepcopy
from pathlib import Path

import pytest
import torch

from latent_stroke_dynamics.phase_b_joint_embedding import (
    EXPECTED_TRAINABLE_PARAMETERS,
    MultiScaleActionJointEmbeddingModel,
    candidate_ranking_loss,
    load_phase_b_config,
    no_op_action_raster,
    no_op_consistency_loss,
    phase_b_objective,
    run_phase_b_validation,
    trainable_parameter_count,
    validate_phase_b_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase-b-saliency-latent-2026-08-23.json"


def test_phase_b_protocol_is_frozen_and_every_data_phase_is_unauthorized() -> None:
    config = load_phase_b_config(CONFIG)
    assert config["status"] == "frozen_before_implementation_and_data"
    assert config["development"]["authorized"] is False
    assert config["formal_reserved"]["authorized"] is False
    assert config["region_scheduler_reserved"]["authorized"] is False
    assert config["rgb_high_resolution_reserved"]["authorized"] is False
    assert config["closed_targets_may_be_reused"] is False


def test_phase_b_config_rejects_architecture_and_authorization_drift() -> None:
    config = load_phase_b_config(CONFIG)
    broken_architecture = deepcopy(config)
    broken_architecture["model"]["latent_scales"]["16"] = [32, 16, 16]
    with pytest.raises(ValueError, match="latent scales"):
        validate_phase_b_config(broken_architecture)
    broken_authorization = deepcopy(config)
    broken_authorization["development"]["authorized"] = True
    with pytest.raises(ValueError, match="unauthorized"):
        validate_phase_b_config(broken_authorization)


def test_phase_b_config_rejects_overlapping_data_seeds() -> None:
    config = load_phase_b_config(CONFIG)
    broken = deepcopy(config)
    broken["formal_reserved"]["transition_splits"]["test"]["seed"] = 20270401
    with pytest.raises(ValueError, match="disjoint"):
        validate_phase_b_config(broken)


def test_phase_b_model_shapes_parameter_count_and_target_freeze() -> None:
    torch.manual_seed(73)
    model = MultiScaleActionJointEmbeddingModel()
    assert trainable_parameter_count(model) == EXPECTED_TRAINABLE_PARAMETERS
    assert EXPECTED_TRAINABLE_PARAMETERS < 500_000
    current = torch.rand(2, 1, 64, 64)
    actions = torch.rand(2, 2, 64, 64)
    goal = torch.rand(2, 1, 64, 64)
    output = model(current, actions, goal)
    assert output["current"]["32"].shape == (2, 32, 32, 32)
    assert output["current"]["16"].shape == (2, 64, 16, 16)
    assert output["action"]["32"].shape == (2, 16, 32, 32)
    assert output["action"]["16"].shape == (2, 32, 16, 16)
    assert output["residual"]["32"].shape == (2, 32, 32, 32)
    assert output["residual"]["16"].shape == (2, 64, 16, 16)
    assert output["predicted_progress"].shape == (2,)
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_phase_b_target_encoder_ema_matches_manual_update() -> None:
    torch.manual_seed(73)
    model = MultiScaleActionJointEmbeddingModel()
    online = next(model.online_encoder.parameters())
    target = next(model.target_encoder.parameters())
    target_before = target.detach().clone()
    with torch.no_grad():
        online.add_(0.25)
        expected = 0.99 * target_before + 0.01 * online.detach()
    model.update_target_encoder(0.99)
    assert torch.equal(target, expected)
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_phase_b_combined_objective_has_finite_model_gradients() -> None:
    torch.manual_seed(73)
    model = MultiScaleActionJointEmbeddingModel().train()
    batch = 4
    current = torch.rand(batch, 1, 64, 64)
    actual_next = torch.rand(batch, 1, 64, 64)
    goal = torch.rand(batch, 1, 64, 64)
    actions = torch.rand(batch, 2, 64, 64)
    actions[0].zero_()
    output = model(current, actions, goal)
    target_next = model.encode_target(actual_next)
    losses = phase_b_objective(
        variant="joint_prediction_progress",
        online_features=output["current"],
        predicted_next=output["predicted_next"],
        target_next=target_next,
        residuals=output["residual"],
        action_rasters=actions,
        no_op_examples=torch.tensor([True, False, False, False]),
        predicted_progress=output["predicted_progress"].reshape(1, batch),
        exact_progress=torch.tensor([[0.0, 0.02, -0.01, 0.01]]),
    )
    losses["total"].backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert bool(torch.isfinite(losses["total"]))
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in trainable)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_phase_b_no_op_and_candidate_tie_rules_are_exact() -> None:
    raster = no_op_action_raster(3)
    assert raster.shape == (3, 2, 64, 64)
    assert torch.count_nonzero(raster) == 0
    residuals = {
        "32": torch.zeros(3, 32, 32, 32),
        "16": torch.zeros(3, 64, 16, 16),
    }
    assert no_op_consistency_loss(
        residuals,
        torch.tensor([True, False, True]),
    ).item() == 0.0
    predicted = torch.tensor([[0.1, 0.3, -0.2]], requires_grad=True)
    exact = torch.tensor([[0.5, 0.5, 0.1]])
    ranking, targets = candidate_ranking_loss(predicted, exact)
    assert targets.tolist() == [0]
    ranking.backward()
    assert predicted.grad is not None
    assert bool(torch.isfinite(predicted.grad).all())


def test_phase_b_validation_uses_dummy_tensors_and_has_no_data_side_effects() -> None:
    config = load_phase_b_config(CONFIG)
    development_output = ROOT / config["development"]["output_dir"]
    formal_output = ROOT / config["formal_reserved"]["output_dir"]
    assert not development_output.exists()
    assert not formal_output.exists()
    result = run_phase_b_validation(CONFIG)
    assert result["status"] == "phase_b0_architecture_and_objectives_valid_unauthorized"
    assert result["trainable_parameter_count"] == EXPECTED_TRAINABLE_PARAMETERS
    assert result["all_trainable_gradients_present"] is True
    assert result["all_trainable_gradients_finite"] is True
    assert result["target_encoder_frozen"] is True
    assert result["ema_maximum_error"] == 0.0
    assert result["dummy_tensors_only"] is True
    assert result["historical_checkpoints_loaded"] is False
    assert result["renderer_transitions_generated"] is False
    assert result["targets_generated"] is False
    assert result["candidate_sets_generated"] is False
    assert result["output_directories_created"] is False
    assert result["models_trained_on_renderer_data"] is False
    assert not development_output.exists()
    assert not formal_output.exists()
