import json
from pathlib import Path

import torch

from latent_stroke_dynamics.extension_training import (
    PatchCounterfactualPayload,
    PatchFeaturePayload,
)
from latent_stroke_dynamics.ranking_training import (
    protocol_oracle_retrieval,
    select_ranking_setting,
)


ROOT = Path(__file__).resolve().parents[1]
COMMAND = (
    ROOT
    / "configs"
    / "ranking-aware-latent-development-command-2026-08-22.json"
)


def tiny_payloads(drift: float = 0.0) -> tuple[PatchFeaturePayload, PatchCounterfactualPayload]:
    current = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    next_features = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]])
    candidates = torch.stack(
        (
            next_features + drift,
            next_features + torch.tensor([[[0.3, -0.2], [0.3, -0.2]]]),
            next_features + torch.tensor([[[-0.2, 0.3], [-0.2, 0.3]]]),
            next_features + torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]),
        ),
        dim=1,
    )
    payload = PatchFeaturePayload(
        current=current,
        next_features=next_features,
        actions=torch.zeros(1, 7),
        action_masks=torch.ones(1, 2),
        patch_grid=(1, 2),
        crowding=torch.zeros(1, dtype=torch.int64),
        width=torch.ones(1, dtype=torch.int64),
        value=torch.zeros(1, dtype=torch.int64),
        length=torch.ones(1),
        fingerprints=("tiny",),
    )
    counterfactuals = PatchCounterfactualPayload(
        candidate_next=candidates,
        union_masks=torch.ones(1, 2),
        all_encoded_candidates_unique=True,
    )
    return payload, counterfactuals


def test_protocol_oracle_requires_retrieval_not_bit_equality() -> None:
    payload, counterfactuals = tiny_payloads(drift=1e-6)
    result = protocol_oracle_retrieval(payload, counterfactuals)
    assert result["top1_accuracy"] == 1.0
    assert result["maximum_candidate_zero_difference"] > 0.0
    assert result["candidate_zero_bit_equality_required"] is False
    assert result["passed"] is True


def test_ranking_setting_selection_uses_frozen_tie_break_order() -> None:
    common = {
        "mean_validation_top1": 0.5,
        "mean_validation_true_margin": 0.02,
        "mean_validation_action_region_mse": 0.3,
    }
    rows = [
        {
            "model": "higher_weight",
            "ranking_weight": 0.3,
            "temperature": 0.1,
            **common,
        },
        {
            "model": "selected",
            "ranking_weight": 0.1,
            "temperature": 0.1,
            **common,
        },
        {
            "model": "lower_temperature",
            "ranking_weight": 0.1,
            "temperature": 0.05,
            **common,
        },
    ]
    selected = select_ranking_setting(rows)
    assert selected["model"] == "selected"


def test_ranking_setting_prioritizes_retrieval_before_mse() -> None:
    rows = [
        {
            "model": "better_mse",
            "ranking_weight": 0.1,
            "temperature": 0.1,
            "mean_validation_top1": 0.4,
            "mean_validation_true_margin": 0.1,
            "mean_validation_action_region_mse": 0.01,
        },
        {
            "model": "better_retrieval",
            "ranking_weight": 0.3,
            "temperature": 0.05,
            "mean_validation_top1": 0.5,
            "mean_validation_true_margin": 0.0,
            "mean_validation_action_region_mse": 1.0,
        },
    ]
    selected = select_ranking_setting(rows)
    assert selected["model"] == "better_retrieval"


def test_development_command_is_frozen_but_unauthorized() -> None:
    command = json.loads(COMMAND.read_text(encoding="utf-8"))
    assert command["status"] == "implemented_before_development_authorization"
    assert command["authorized"] is False
    assert command["formal_data_generation_allowed"] is False
    assert command["development_seeds"] == [20261101, 20261102, 20261103]
