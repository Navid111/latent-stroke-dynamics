"""Validation and loss utilities for the ranking-aware latent follow-up."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .gate2 import balanced_patch_mse, counterfactual_retrieval
from .representation_extension import LatentChannelStatistics


DEFAULT_RANKING_CONFIG = Path(
    "configs/ranking-aware-latent-2026-08-22.json"
)
PROTECTED_EXTENSION_SEEDS = set(range(20261020, 20261031))


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_ranking_config(
    path: str | Path = DEFAULT_RANKING_CONFIG,
) -> dict[str, Any]:
    """Load and validate the frozen ranking-aware follow-up config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_ranking_config(config)
    return config


def validate_ranking_config(config: Mapping[str, Any]) -> None:
    """Reject scientific drift before follow-up data generation."""

    if config.get("experiment_id") != "ranking-aware-latent-followup-2026-08-22":
        raise ValueError("Unexpected ranking follow-up experiment_id.")
    if config.get("status") != "frozen_before_implementation_and_data":
        raise ValueError("Ranking follow-up config is not in its frozen state.")
    if config.get("historical_decisions_unchanged") is not True:
        raise ValueError("Historical decisions must remain unchanged.")
    if config.get("canvas_size") != 64 or config.get("device") != "cpu":
        raise ValueError("The canvas size and CPU device are frozen.")

    representation = _mapping(
        config.get("frozen_representation"),
        "frozen_representation",
    )
    expected_representation = {
        "architecture": "StrokeAutoencoder",
        "latent_grid": [16, 16],
        "feature_dim": 32,
        "total_parameter_count": 49_569,
        "selected_seed": 101,
        "selected_epoch": 50,
        "checkpoint_path": (
            "outputs/representation-extension-2026-08-22/"
            "task_autoencoder/checkpoints/task_autoencoder.pt"
        ),
        "checkpoint_sha256": (
            "95de3ecef8eeb7a350e862fa21185a168"
            "d9304870cb0c8391cbd008e88d93900"
        ),
        "latent_statistics_path": (
            "outputs/representation-extension-2026-08-22/"
            "task_autoencoder/latent_channel_statistics.json"
        ),
        "latent_statistics_hash_must_be_frozen_before_development": True,
        "encoder_frozen": True,
    }
    for key, expected in expected_representation.items():
        if representation.get(key) != expected:
            raise ValueError(f"Frozen representation field {key!r} changed.")
    statistics_hash = representation.get("latent_statistics_sha256")
    if statistics_hash is not None and not _is_sha256(statistics_hash):
        raise ValueError("latent_statistics_sha256 must be null or a lowercase SHA-256.")

    distribution = _mapping(
        config.get("transition_distribution"),
        "transition_distribution",
    )
    if distribution.get("primary_crowding") != [0, 5, 15]:
        raise ValueError("Primary crowding changed.")
    if distribution.get("widths") != [1, 2, 3, 4]:
        raise ValueError("Stroke widths changed.")
    if distribution.get("values") != [0, 32, 64, 96, 128]:
        raise ValueError("Stroke values changed.")
    if float(distribution.get("minimum_length", -1)) != 0.2:
        raise ValueError("Minimum stroke length changed.")
    if distribution.get("counterfactual_order") != [
        "true",
        "shift_position",
        "change_width",
        "change_intensity",
    ]:
        raise ValueError("Counterfactual order changed.")

    development = _mapping(config.get("development"), "development")
    formal = _mapping(config.get("formal_reserved"), "formal_reserved")
    development_authorized = development.get("authorized")
    formal_authorized = formal.get("authorized")
    if development_authorized is not False and development_authorized is not True:
        raise ValueError("Development authorization must be boolean.")
    if formal_authorized is not False and formal_authorized is not True:
        raise ValueError("Formal authorization must be boolean.")
    if development_authorized is True and statistics_hash is None:
        raise ValueError("Development cannot be authorized before the statistics hash.")
    if formal_authorized is True:
        if development_authorized is not True:
            raise ValueError("Formal authorization requires completed development authorization.")
        if statistics_hash is None:
            raise ValueError("Formal authorization requires the frozen statistics hash.")
    expected_development = {
        "train": (128, 20261101),
        "validation": (64, 20261102),
        "diagnostic_test": (64, 20261103),
    }
    expected_formal = {
        "train": (1000, 20261104),
        "validation": (200, 20261105),
        "test": (300, 20261106),
        "unseen_width_5": (100, 20261107),
        "unseen_intensities": (100, 20261108),
        "crowding_30": (100, 20261109),
        "crowding_60": (100, 20261110),
    }
    seeds: list[int] = []
    for name, (samples, seed) in expected_development.items():
        split = _mapping(development.get(name), f"development.{name}")
        if int(split.get("samples", -1)) != samples or int(
            split.get("seed", -1)
        ) != seed:
            raise ValueError(f"Development split {name!r} changed.")
        seeds.append(seed)
    for name, (samples, seed) in expected_formal.items():
        split = _mapping(formal.get(name), f"formal_reserved.{name}")
        if int(split.get("samples", -1)) != samples or int(
            split.get("seed", -1)
        ) != seed:
            raise ValueError(f"Formal split {name!r} changed.")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Follow-up split seeds must be disjoint.")
    if set(seeds).intersection(PROTECTED_EXTENSION_SEEDS):
        raise ValueError("Follow-up seeds overlap the completed extension.")
    unseen_intensities = _mapping(
        formal.get("unseen_intensities"),
        "formal_reserved.unseen_intensities",
    )
    if unseen_intensities.get("values") != [16, 80, 176]:
        raise ValueError("Reserved unseen intensities changed.")

    predictor = _mapping(config.get("predictor"), "predictor")
    expected_predictor = {
        "family": "mlp",
        "hidden_dim": 256,
        "parameter_count": 19_232,
        "model_seeds": [11, 22, 33],
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 16,
        "max_epochs": 30,
        "patience": 6,
        "baseline_objective": "balanced_patch_residual_mse",
        "ranking_objective": (
            "balanced_patch_residual_mse_plus_counterfactual_cross_entropy"
        ),
    }
    if dict(predictor) != expected_predictor:
        raise ValueError("Frozen predictor settings changed.")

    ranking_grid = _mapping(config.get("ranking_grid"), "ranking_grid")
    if ranking_grid.get("lambda") != [0.1, 0.3, 1.0]:
        raise ValueError("Ranking lambda grid changed.")
    if ranking_grid.get("temperature") != [0.05, 0.1]:
        raise ValueError("Ranking temperature grid changed.")
    if ranking_grid.get("selection_order") != [
        "highest_mean_validation_top1",
        "highest_mean_validation_true_margin",
        "lowest_mean_validation_action_region_mse",
        "lower_lambda",
        "higher_temperature",
    ]:
        raise ValueError("Ranking selection order changed.")

    classification = _mapping(config.get("classification"), "classification")
    expected_classification = {
        "minimum_formal_retrieval": 0.5,
        "minimum_absolute_retrieval_gain_over_mse": 0.1,
        "minimum_improvement_vs_identity": 0.3,
        "minimum_improvement_vs_mean_delta": 0.3,
        "minimum_primary_crowding_improvement": 0.0,
        "required_oracle_retrieval": 1.0,
        "chance_retrieval": 0.25,
    }
    if dict(classification) != expected_classification:
        raise ValueError("Frozen classification thresholds changed.")
    if config.get("development_output_dir") != (
        "outputs/ranking-aware-latent-development-2026-08-22"
    ):
        raise ValueError("Development output directory changed.")
    if config.get("formal_output_dir") != (
        "outputs/ranking-aware-latent-formal-2026-08-22"
    ):
        raise ValueError("Formal output directory changed.")


def load_latent_channel_statistics(
    path: str | Path,
    *,
    expected_channels: int = 32,
) -> LatentChannelStatistics:
    """Load the immutable saved full-run latent standardization statistics."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("source") != "full_train_current_and_next_canvases_only":
        raise ValueError("Latent statistics have an unexpected source.")
    mean = torch.tensor(payload.get("mean"), dtype=torch.float32)
    std = torch.tensor(payload.get("std"), dtype=torch.float32)
    if mean.shape != (expected_channels,) or std.shape != (expected_channels,):
        raise ValueError("Latent statistics do not have the frozen channel shape.")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("Latent statistics contain a non-finite value.")
    if not bool((std > 0).all()):
        raise ValueError("Every frozen latent standard deviation must be positive.")
    reported_mean_std = payload.get("mean_channel_std")
    if not isinstance(reported_mean_std, (int, float)):
        raise ValueError("Latent statistics lack mean_channel_std.")
    if not torch.isclose(
        std.mean(),
        torch.tensor(float(reported_mean_std)),
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError("Reported latent mean-channel std is inconsistent.")
    return LatentChannelStatistics(mean=mean, std=std)


def counterfactual_ranking_loss(
    predicted_next: torch.Tensor,
    candidate_next: torch.Tensor,
    union_masks: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Cross-entropy that makes candidate zero closest to predicted next tokens."""

    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    result = counterfactual_retrieval(
        predicted_next,
        candidate_next,
        union_masks,
    )
    scores = result["scores"]
    targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    loss = F.cross_entropy(-scores / temperature, targets)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Counterfactual ranking loss is non-finite.")
    return loss


def ranking_aware_objective(
    current: torch.Tensor,
    predicted_delta: torch.Tensor,
    true_delta: torch.Tensor,
    action_masks: torch.Tensor,
    candidate_next: torch.Tensor,
    union_masks: torch.Tensor,
    *,
    ranking_weight: float,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Return frozen balanced-MSE, ranking, and combined losses."""

    if ranking_weight < 0:
        raise ValueError("ranking_weight cannot be negative.")
    mse = balanced_patch_mse(predicted_delta, true_delta, action_masks)
    ranking = counterfactual_ranking_loss(
        current + predicted_delta,
        candidate_next,
        union_masks,
        temperature=temperature,
    )
    total = mse + ranking_weight * ranking
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Ranking-aware total objective is non-finite.")
    return {
        "total": total,
        "balanced_mse": mse,
        "ranking_cross_entropy": ranking,
    }
