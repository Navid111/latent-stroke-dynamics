from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from latent_stroke_dynamics.extension_training import create_patch_predictor
from latent_stroke_dynamics.latent_planner import (
    latent_candidate_scores,
    load_formal_latent_predictor,
    load_latent_planner_config,
    validate_latent_planner_config,
)
from latent_stroke_dynamics.renderer import Stroke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "latent-planner-2026-08-23.json"
EXPECTED_LATENT_HASHES = {
    "mse_only": {
        11: "5023e63d268ea37c17bf328a8fc4ef5f66219ed7e1127e0d4bf109330f833431",
        22: "8a8ea0e7e6dbfc9c5f64ae5212a48a5fc12ee5bb5ff1f23c0394c8891b5a89dc",
        33: "3a58be00b601a08cf9faa55e1643bf24adfd08c0a0648cbe851e9f9e49388c5e",
    },
    "ranking_aware": {
        11: "1833f9a4f68aa402587f2842e3143ee018423ff12c36afdbc61f64c0120d9588",
        22: "cbd1aae6c83fe7d06a1226ea58cc5953e3745f81912ad93b9a1328ffdf267ac2",
        33: "f4543610be2dab8adf7a14ca4bf9b862bacde49c3a3c048fb6e46d3f50097675",
    },
}


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_latent_planner_hashes_are_frozen_and_runs_unauthorized() -> None:
    config = load_latent_planner_config(CONFIG)
    assert config["status"] == "hashes_frozen_before_smoke"
    assert config["smoke"]["authorized"] is False
    assert config["controlled"]["authorized"] is False
    assert config["latent_predictors"]["model_seeds"] == [11, 22, 33]
    observed = {
        method: {
            int(entry["seed"]): entry["state_sha256"]
            for entry in config["latent_predictors"][method]
        }
        for method in ("mse_only", "ranking_aware")
    }
    assert observed == EXPECTED_LATENT_HASHES


def test_latent_planner_config_rejects_scoring_drift() -> None:
    config = load_latent_planner_config(CONFIG)
    broken = deepcopy(config)
    broken["planner"]["latent_score"] = "different_score"
    with pytest.raises(ValueError, match="mechanism"):
        validate_latent_planner_config(broken)


def test_latent_planner_config_rejects_missing_frozen_hash() -> None:
    config = load_latent_planner_config(CONFIG)
    broken = deepcopy(config)
    broken["latent_predictors"]["mse_only"][0]["state_sha256"] = None
    with pytest.raises(ValueError, match="must be frozen"):
        validate_latent_planner_config(broken)


def test_formal_latent_checkpoint_round_trip_and_freeze(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = create_patch_predictor("mlp", 32, (16, 16), 256)
    path = tmp_path / "mse.pt"
    torch.save(
        {"method": "mse_only", "seed": 11, "state_dict": model.state_dict()},
        path,
    )
    loaded = load_formal_latent_predictor(
        path,
        expected_method="mse_only",
        expected_seed=11,
    )
    assert loaded.method == "mse_only"
    assert loaded.seed == 11
    assert len(loaded.state_sha256) == 64
    assert not any(parameter.requires_grad for parameter in loaded.model.parameters())
    current = torch.randn(1, 256, 32)
    actions = torch.randn(1, 7)
    masks = torch.rand(1, 256)
    with torch.inference_mode():
        expected = model(current, actions, masks)
        actual = loaded.model(current, actions, masks)
    assert torch.equal(expected, actual)


def test_ranking_checkpoint_rejects_hyperparameter_drift(tmp_path: Path) -> None:
    model = create_patch_predictor("mlp", 32, (16, 16), 256)
    path = tmp_path / "ranking.pt"
    torch.save(
        {
            "method": "ranking_aware",
            "seed": 22,
            "ranking_weight": 0.3,
            "temperature": 0.05,
            "state_dict": model.state_dict(),
        },
        path,
    )
    with pytest.raises(ValueError, match="weight"):
        load_formal_latent_predictor(
            path,
            expected_method="ranking_aware",
            expected_seed=22,
        )


def test_latent_candidate_scores_are_finite_deterministic_and_ensemble_mean() -> None:
    first = freeze(create_patch_predictor("mlp", 32, (16, 16), 256))
    torch.manual_seed(22)
    second = freeze(create_patch_predictor("mlp", 32, (16, 16), 256))
    generator = torch.Generator().manual_seed(515)
    current = torch.randn(1, 256, 32, generator=generator)
    target = torch.randn(1, 256, 32, generator=generator)
    candidates = (
        Stroke(0.1, 0.2, 0.9, 0.8, width=2, value=32),
        Stroke(0.2, 0.8, 0.8, 0.2, width=4, value=96),
    )
    aggregate, per_model = latent_candidate_scores(
        (first, second), current, target, candidates, batch_size=1
    )
    repeated, repeated_per_model = latent_candidate_scores(
        (first, second), current, target, candidates, batch_size=2
    )
    assert aggregate.shape == (2,)
    assert per_model.shape == (2, 2)
    assert np.isfinite(aggregate).all()
    assert np.array_equal(aggregate, repeated)
    assert np.array_equal(per_model, repeated_per_model)
    assert np.allclose(
        aggregate,
        per_model.mean(axis=0),
        rtol=1e-7,
        atol=1e-9,
    )
