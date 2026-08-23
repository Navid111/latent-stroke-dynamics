"""Frozen-resource loading and candidate scoring for latent-space planning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .extension_training import (
    create_patch_predictor,
    encode_autoencoder_maps,
    load_autoencoder_checkpoint,
    model_state_sha256,
    total_parameter_count,
)
from .gate2 import parameter_count, stroke_action_vector, stroke_patch_coverage
from .representation_extension import (
    LatentChannelStatistics,
    StrokeAutoencoder,
    images_to_grayscale_tensor,
    standardize_latent_tokens,
)
from .ranking_latent import file_sha256, load_latent_channel_statistics
from .renderer import Stroke


DEFAULT_LATENT_PLANNER_CONFIG = Path("configs/latent-planner-2026-08-23.json")
VALID_CONFIG_STATUSES = {
    "frozen_before_implementation_and_planner_data",
    "hashes_frozen_before_smoke",
    "smoke_authorized_once",
    "smoke_complete_controlled_unauthorized",
    "controlled_authorized_once",
}


@dataclass(frozen=True)
class LoadedLatentPredictor:
    """One strictly loaded and frozen formal dynamics checkpoint."""

    method: str
    seed: int
    path: Path
    state_sha256: str
    model: nn.Module


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


def load_latent_planner_config(
    path: str | Path = DEFAULT_LATENT_PLANNER_CONFIG,
) -> dict[str, Any]:
    """Load and strictly validate the frozen latent-planner protocol."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_latent_planner_config(config)
    return config


def validate_latent_planner_config(config: Mapping[str, Any]) -> None:
    """Reject planner, scoring, model, seed, and authorization drift."""

    status = config.get("status")
    if config.get("experiment_id") != "latent-space-planner-2026-08-23":
        raise ValueError("Unexpected latent-planner experiment_id.")
    if status not in VALID_CONFIG_STATUSES:
        raise ValueError("Unexpected latent-planner config status.")
    if config.get("evidential_role") != "post_formal_deployment_study":
        raise ValueError("Latent-planner evidential role changed.")
    if config.get("historical_results_unchanged") is not True:
        raise ValueError("Historical results must remain unchanged.")
    if config.get("device") != "cpu" or config.get("canvas_size") != 64:
        raise ValueError("CPU device and 64x64 canvas are frozen.")

    representation = _mapping(config.get("representation"), "representation")
    expected_representation = {
        "autoencoder_checkpoint": (
            "outputs/representation-extension-2026-08-22/"
            "task_autoencoder/checkpoints/task_autoencoder.pt"
        ),
        "autoencoder_state_sha256": (
            "95de3ecef8eeb7a350e862fa21185a168"
            "d9304870cb0c8391cbd008e88d93900"
        ),
        "latent_statistics": (
            "outputs/representation-extension-2026-08-22/"
            "task_autoencoder/latent_channel_statistics.json"
        ),
        "latent_statistics_sha256": (
            "c2a3d781dab19a4714189d580dafb5ea"
            "95231af06021d3980beb495a3b85d903"
        ),
        "latent_grid": [16, 16],
        "feature_dim": 32,
        "encoder_frozen": True,
    }
    if dict(representation) != expected_representation:
        raise ValueError("Frozen task representation changed.")

    predictors = _mapping(config.get("latent_predictors"), "latent_predictors")
    fixed_predictors = {
        "architecture": "MLPPatchDeltaPredictor",
        "parameter_count": 19_232,
        "model_seeds": [11, 22, 33],
        "score_aggregation": "mean_across_three_seed_scores",
        "ranking_weight": 1.0,
        "temperature": 0.05,
        "all_predictors_frozen": True,
    }
    for name, expected in fixed_predictors.items():
        if predictors.get(name) != expected:
            raise ValueError(f"Frozen latent predictor field {name!r} changed.")

    all_hashes: list[object] = []
    for method, prefix in (
        ("mse_only", "mse_only"),
        ("ranking_aware", "ranking_aware"),
    ):
        entries = predictors.get(method)
        if not isinstance(entries, list) or len(entries) != 3:
            raise ValueError(f"{method} must contain exactly three checkpoints.")
        for entry, seed in zip(entries, (11, 22, 33), strict=True):
            checkpoint = _mapping(entry, f"{method} checkpoint")
            expected_path = (
                "outputs/ranking-aware-latent-formal-2026-08-22/"
                f"checkpoints/{prefix}_seed{seed}.pt"
            )
            if checkpoint.get("seed") != seed or checkpoint.get("path") != expected_path:
                raise ValueError(f"Frozen {method} checkpoint path or seed changed.")
            digest = checkpoint.get("state_sha256")
            if digest is not None and not _is_sha256(digest):
                raise ValueError(f"Invalid {method} checkpoint SHA-256.")
            all_hashes.append(digest)
    if status == "frozen_before_implementation_and_planner_data":
        if any(value is not None for value in all_hashes):
            raise ValueError("Planner hashes cannot be populated before hash freeze status.")
    elif not all(_is_sha256(value) for value in all_hashes):
        raise ValueError("Every latent checkpoint hash must be frozen before smoke.")

    pixel = _mapping(config.get("pixel_predictor"), "pixel_predictor")
    if pixel.get("path") != "checkpoints/stage3-pixel-mlp-seed11.pt":
        raise ValueError("Frozen pixel predictor path changed.")
    if not _is_sha256(pixel.get("state_sha256")):
        raise ValueError("Frozen pixel predictor hash must be a lowercase SHA-256.")
    if set(pixel) != {"path", "state_sha256"}:
        raise ValueError("Frozen pixel predictor fields changed.")

    planner = _mapping(config.get("planner"), "planner")
    expected_planner = {
        "methods": [
            "random",
            "exact_pixel",
            "learned_pixel",
            "latent_mse",
            "latent_ranking",
        ],
        "target_strokes": 20,
        "steps": 100,
        "candidates_per_step": 128,
        "prediction_batch_size": 32,
        "latent_score": (
            "mean_full_grid_mse_of_l2_normalized_patch_features_to_target"
        ),
        "execute_with_exact_renderer": True,
        "reencode_observed_canvas_each_step": True,
        "use_predicted_latent_as_next_state": False,
        "preserve_best_and_final_frames": True,
        "proposal": {
            "error_guided_fraction": 0.8,
            "min_length": 0.1,
            "max_length": 0.6,
            "width_choices": [1, 2, 3, 4],
            "value_choices": [0, 32, 64, 96, 128],
        },
    }
    if dict(planner) != expected_planner:
        raise ValueError("Frozen latent-planner mechanism changed.")

    smoke = _mapping(config.get("smoke"), "smoke")
    expected_smoke_authorization = status == "smoke_authorized_once"
    if dict(smoke) != {
        "authorized": expected_smoke_authorization,
        "target_seed": 20261201,
        "planner_seed": 20261202,
        "steps": 20,
        "candidates_per_step": 32,
        "output_dir": "outputs/latent-planner-smoke-2026-08-23",
    }:
        raise ValueError("Smoke settings or authorization changed.")
    controlled = _mapping(config.get("controlled"), "controlled")
    expected_controlled_authorization = status == "controlled_authorized_once"
    if dict(controlled) != {
        "authorized": expected_controlled_authorization,
        "target_seeds": [
            20261211,
            20261212,
            20261213,
            20261214,
            20261215,
            20261216,
        ],
        "planner_seeds": [
            20261221,
            20261222,
            20261223,
            20261224,
            20261225,
            20261226,
        ],
        "output_dir": "outputs/latent-planner-controlled-2026-08-23",
        "single_run": True,
    }:
        raise ValueError("Controlled settings or authorization changed.")
    criteria = _mapping(config.get("success_criteria"), "success_criteria")
    if dict(criteria) != {
        "latent_ranking_improves_every_target_from_initial": True,
        "minimum_mean_final_mse_reduction_vs_random": 0.2,
        "maximum_mean_final_mse_ratio_to_exact_pixel": 1.5,
        "implementation_integrity_required": True,
        "outperform_learned_pixel_required": False,
    }:
        raise ValueError("Latent-planner success criteria changed.")
    foundation = _mapping(
        config.get("foundation_validation"), "foundation_validation"
    )
    if dict(foundation) != {
        "may_load_existing_models": True,
        "may_run_in_memory_synthetic_checks": True,
        "may_generate_smoke_or_controlled_targets": False,
        "may_train_models": False,
        "latent_checkpoint_hash_freeze_required_before_smoke": True,
    }:
        raise ValueError("Foundation-validation boundary changed.")


def load_formal_latent_predictor(
    path: str | Path,
    *,
    expected_method: str,
    expected_seed: int,
    expected_state_sha256: str | None = None,
) -> LoadedLatentPredictor:
    """Load one formal checkpoint, validate metadata, hash it, and freeze it."""

    if expected_method not in {"mse_only", "ranking_aware"}:
        raise ValueError("Unexpected formal latent predictor method.")
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Formal latent checkpoint must contain a mapping.")
    if payload.get("method") != expected_method:
        raise ValueError("Formal latent checkpoint method mismatch.")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError("Formal latent checkpoint seed mismatch.")
    if expected_method == "ranking_aware":
        if float(payload.get("ranking_weight", -1)) != 1.0:
            raise ValueError("Ranking checkpoint weight mismatch.")
        if float(payload.get("temperature", -1)) != 0.05:
            raise ValueError("Ranking checkpoint temperature mismatch.")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Formal latent checkpoint state_dict is missing.")

    model = create_patch_predictor("mlp", 32, (16, 16), 256).cpu()
    model.load_state_dict(state, strict=True)
    if parameter_count(model) != 19_232:
        raise ValueError("Formal latent predictor parameter count changed.")
    digest = model_state_sha256(model)
    if expected_state_sha256 is not None and digest != expected_state_sha256:
        raise ValueError("Formal latent predictor state SHA-256 mismatch.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedLatentPredictor(
        method=expected_method,
        seed=expected_seed,
        path=checkpoint_path,
        state_sha256=digest,
        model=model,
    )


def load_latent_predictor_ensembles(
    config: Mapping[str, Any],
) -> dict[str, tuple[LoadedLatentPredictor, ...]]:
    """Load the frozen three-seed MSE and ranking ensembles."""

    predictors = _mapping(config.get("latent_predictors"), "latent_predictors")
    result: dict[str, tuple[LoadedLatentPredictor, ...]] = {}
    for method in ("mse_only", "ranking_aware"):
        entries = predictors[method]
        loaded = [
            load_formal_latent_predictor(
                entry["path"],
                expected_method=method,
                expected_seed=int(entry["seed"]),
                expected_state_sha256=entry["state_sha256"],
            )
            for entry in entries
        ]
        result[method] = tuple(loaded)
    return result


def load_task_latent_resources(
    config: Mapping[str, Any],
) -> tuple[StrokeAutoencoder, LatentChannelStatistics]:
    """Load and verify the frozen task autoencoder and statistics."""

    representation = _mapping(config.get("representation"), "representation")
    checkpoint = Path(representation["autoencoder_checkpoint"])
    statistics_path = Path(representation["latent_statistics"])
    model, _ = load_autoencoder_checkpoint(checkpoint)
    if model_state_sha256(model) != representation["autoencoder_state_sha256"]:
        raise ValueError("Task autoencoder state SHA-256 mismatch.")
    if file_sha256(statistics_path) != representation["latent_statistics_sha256"]:
        raise ValueError("Task latent statistics SHA-256 mismatch.")
    if total_parameter_count(model) != 49_569:
        raise ValueError("Task autoencoder parameter count changed.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Task autoencoder is not frozen.")
    return model, load_latent_channel_statistics(statistics_path)


@torch.inference_mode()
def encode_task_latents(
    model: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    images: Sequence[Image.Image],
    *,
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode exact canvases to standardized `[N, 256, 32]` task latents."""

    tensors = images_to_grayscale_tensor(images)
    maps = encode_autoencoder_maps(model, tensors, batch_size)
    tokens = standardize_latent_tokens(maps, statistics)
    if tokens.shape != (len(images), 256, 32):
        raise RuntimeError("Unexpected task-latent token shape.")
    if not bool(torch.isfinite(tokens).all()):
        raise RuntimeError("Task-latent encoding is non-finite.")
    return tokens.cpu()


@torch.inference_mode()
def latent_candidate_scores(
    predictors: Sequence[nn.Module],
    current_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    candidates: Sequence[Stroke],
    *,
    canvas_size: int = 64,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Score candidates by mean ensemble distance to the target latent."""

    if not predictors:
        raise ValueError("At least one frozen predictor is required.")
    if not candidates:
        raise ValueError("At least one candidate stroke is required.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if current_tokens.shape != (1, 256, 32) or target_tokens.shape != (1, 256, 32):
        raise ValueError("Current and target tokens must have shape [1, 256, 32].")
    if not bool(torch.isfinite(current_tokens).all() and torch.isfinite(target_tokens).all()):
        raise ValueError("Current and target tokens must be finite.")
    if any(parameter.requires_grad for model in predictors for parameter in model.parameters()):
        raise ValueError("Latent planner received a trainable predictor.")

    target_normalized = F.normalize(target_tokens.float(), dim=-1)
    per_model: list[torch.Tensor] = []
    for model in predictors:
        model.eval()
        score_parts: list[torch.Tensor] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            actions = torch.stack([stroke_action_vector(stroke) for stroke in batch])
            masks = torch.stack(
                [
                    stroke_patch_coverage(stroke, canvas_size, (16, 16))
                    for stroke in batch
                ]
            )
            repeated_current = current_tokens.float().expand(len(batch), -1, -1)
            predicted_next = repeated_current + model(
                repeated_current,
                actions.float(),
                masks.float(),
            )
            predicted_normalized = F.normalize(predicted_next, dim=-1)
            scores = (
                predicted_normalized - target_normalized.expand(len(batch), -1, -1)
            ).square().mean(dim=(1, 2))
            score_parts.append(scores.cpu())
        per_model.append(torch.cat(score_parts))
    stacked = torch.stack(per_model, dim=0)
    aggregate = stacked.mean(dim=0)
    aggregate_array = aggregate.numpy().astype(np.float64, copy=False)
    per_model_array = stacked.numpy().astype(np.float64, copy=False)
    if aggregate_array.shape != (len(candidates),):
        raise RuntimeError("Aggregate latent score shape is invalid.")
    if per_model_array.shape != (len(predictors), len(candidates)):
        raise RuntimeError("Per-model latent score shape is invalid.")
    if not bool(np.isfinite(aggregate_array).all() and np.isfinite(per_model_array).all()):
        raise RuntimeError("Latent candidate scoring produced non-finite values.")
    return aggregate_array, per_model_array
