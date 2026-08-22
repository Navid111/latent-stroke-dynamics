from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from latent_stroke_dynamics.extension_training import create_patch_predictor
from latent_stroke_dynamics.gate2 import parameter_count
from latent_stroke_dynamics.ranking_latent import (
    counterfactual_ranking_loss,
    load_latent_channel_statistics,
    load_ranking_config,
    ranking_aware_objective,
    validate_ranking_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "ranking-aware-latent-2026-08-22.json"
STATISTICS_SHA256 = "c2a3d781dab19a4714189d580dafb5ea95231af06021d3980beb495a3b85d903"


def example_tensors() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(411)
    current = torch.randn(3, 4, 5, generator=generator)
    true_delta = 0.1 * torch.randn(3, 4, 5, generator=generator)
    true_next = current + true_delta
    candidates = torch.stack(
        (
            true_next,
            true_next + 0.25,
            true_next - 0.50,
            true_next + 0.75,
        ),
        dim=1,
    )
    masks = torch.ones(3, 4)
    return current, true_delta, true_next, candidates, masks


def test_frozen_ranking_config_authorizes_development_not_formal() -> None:
    config = load_ranking_config(CONFIG)
    assert config["development"]["authorized"] is True
    assert config["formal_reserved"]["authorized"] is False
    assert config["frozen_representation"]["latent_statistics_sha256"] == (
        STATISTICS_SHA256
    )
    assert config["ranking_grid"]["lambda"] == [0.1, 0.3, 1.0]
    assert config["ranking_grid"]["temperature"] == [0.05, 0.1]


def test_config_rejects_development_authorization_without_statistics_hash() -> None:
    config = load_ranking_config(CONFIG)
    broken = deepcopy(config)
    broken["frozen_representation"]["latent_statistics_sha256"] = None
    broken["development"]["authorized"] = True
    with pytest.raises(ValueError, match="statistics hash"):
        validate_ranking_config(broken)


def test_config_rejects_formal_authorization() -> None:
    config = load_ranking_config(CONFIG)
    broken = deepcopy(config)
    broken["formal_reserved"]["authorized"] = True
    with pytest.raises(ValueError, match="Formal data remain unauthorized"):
        validate_ranking_config(broken)


def test_latent_statistics_loader_validates_saved_shape_and_mean(tmp_path: Path) -> None:
    path = tmp_path / "statistics.json"
    values = [float(index + 1) / 100.0 for index in range(32)]
    path.write_text(
        json.dumps(
            {
                "source": "full_train_current_and_next_canvases_only",
                "mean": [0.0] * 32,
                "std": values,
                "mean_channel_std": sum(values) / len(values),
            }
        ),
        encoding="utf-8",
    )
    statistics = load_latent_channel_statistics(path)
    assert statistics.mean.shape == (32,)
    assert statistics.std.shape == (32,)
    assert float(statistics.std.mean()) == pytest.approx(sum(values) / len(values))


def test_ranking_loss_is_lower_at_true_candidate_than_wrong_candidate() -> None:
    _, _, true_next, candidates, masks = example_tensors()
    true_loss = counterfactual_ranking_loss(
        true_next,
        candidates,
        masks,
        temperature=0.1,
    )
    wrong_loss = counterfactual_ranking_loss(
        candidates[:, 1],
        candidates,
        masks,
        temperature=0.1,
    )
    assert true_loss < wrong_loss


def test_ranking_aware_objective_has_finite_gradient() -> None:
    current, true_delta, _, candidates, masks = example_tensors()
    predicted_delta = torch.zeros_like(true_delta, requires_grad=True)
    losses = ranking_aware_objective(
        current,
        predicted_delta,
        true_delta,
        masks,
        candidates,
        masks,
        ranking_weight=0.3,
        temperature=0.1,
    )
    losses["total"].backward()
    assert bool(torch.isfinite(losses["total"]))
    assert predicted_delta.grad is not None
    assert bool(torch.isfinite(predicted_delta.grad).all())


def test_invalid_ranking_hyperparameters_are_rejected() -> None:
    current, true_delta, true_next, candidates, masks = example_tensors()
    with pytest.raises(ValueError, match="temperature"):
        counterfactual_ranking_loss(
            true_next,
            candidates,
            masks,
            temperature=0.0,
        )
    with pytest.raises(ValueError, match="ranking_weight"):
        ranking_aware_objective(
            current,
            true_delta,
            true_delta,
            masks,
            candidates,
            masks,
            ranking_weight=-0.1,
            temperature=0.1,
        )


def test_ranking_predictor_matches_frozen_parameter_count() -> None:
    model = create_patch_predictor("mlp", 32, (16, 16), 256)
    assert parameter_count(model) == 19_232
