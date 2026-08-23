"""Frozen configuration, scoring, and guards for planner-score alignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .gate2 import stroke_action_vector, stroke_patch_coverage
from .latent_smoke import spearman_rank_correlation
from .planning import pixel_mse, render_candidate_canvases
from .representation_extension import (
    LatentChannelStatistics,
    StrokeAutoencoder,
    images_to_grayscale_tensor,
)
from .renderer import Stroke


DEFAULT_SCORE_ALIGNMENT_CONFIG = Path(
    "configs/planner-score-alignment-2026-08-23.json"
)
VALID_STATUSES = {
    "frozen_before_implementation_and_data",
    "development_score_audit_authorized_once",
    "development_score_audit_complete_planner_unauthorized",
    "planner_development_authorized_once",
    "planner_development_complete_confirmatory_unauthorized",
    "confirmatory_authorized_once",
    "confirmatory_complete_closed",
}
PREDICTOR_FAMILIES = ("mse_only", "ranking_aware")
SCORE_NAMES = (
    "normalized_latent_mse",
    "normalized_latent_l1",
    "pixel_error_weighted_normalized_latent_mse",
    "decoded_pixel_l1",
    "decoded_pixel_l1_plus_quarter_sobel_l1",
)
DEVELOPMENT_TARGET_SEEDS = tuple(range(20270101, 20270109))
DEVELOPMENT_STATE_SEEDS = tuple(range(20270111, 20270119))
DEVELOPMENT_CANDIDATE_SEEDS = tuple(range(20270121, 20270129))
PLANNER_DEVELOPMENT_TARGET_SEEDS = tuple(range(20270201, 20270204))
PLANNER_DEVELOPMENT_SEEDS = tuple(range(20270211, 20270214))
CONFIRMATORY_TARGET_SEEDS = tuple(range(20270301, 20270307))
CONFIRMATORY_PLANNER_SEEDS = tuple(range(20270311, 20270317))
PROTECTED_PREVIOUS_SEEDS = set(range(20261020, 20261227))


@dataclass(frozen=True)
class ScoreAuditOutputPaths:
    final: Path
    incomplete: Path


@dataclass(frozen=True)
class CandidateScoreResult:
    aggregate: np.ndarray
    per_model: np.ndarray


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


def load_score_alignment_config(
    path: str | Path = DEFAULT_SCORE_ALIGNMENT_CONFIG,
) -> dict[str, Any]:
    """Load and strictly validate the frozen Stage A protocol."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_score_alignment_config(config)
    return config


def validate_score_alignment_config(config: Mapping[str, Any]) -> None:
    """Reject score, seed, resource, phase, or authorization drift."""

    status = config.get("status")
    if config.get("experiment_id") != "planner-score-alignment-2026-08-23":
        raise ValueError("Unexpected planner-score experiment_id.")
    if status not in VALID_STATUSES:
        raise ValueError("Unexpected planner-score status.")
    if config.get("evidential_role") != "post_controlled_exploratory_extension":
        raise ValueError("Planner-score evidential role changed.")
    if config.get("historical_results_unchanged") is not True:
        raise ValueError("Historical results must remain unchanged.")
    if config.get("closed_controlled_targets_may_be_reused") is not False:
        raise ValueError("Closed controlled targets must remain prohibited.")
    if config.get("device") != "cpu" or config.get("canvas_size") != 64:
        raise ValueError("Stage A requires CPU and a 64x64 canvas.")

    resources = _mapping(config.get("frozen_resources"), "frozen_resources")
    expected_resources = {
        "latent_planner_config": "configs/latent-planner-2026-08-23.json",
        "required_latent_planner_status": "controlled_complete_closed",
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
        "predictor_families": list(PREDICTOR_FAMILIES),
        "model_seeds": [11, 22, 33],
        "predictor_state_sha256": {
            "mse_only": {
                "11": "5023e63d268ea37c17bf328a8fc4ef5f66219ed7e1127e0d4bf109330f833431",
                "22": "8a8ea0e7e6dbfc9c5f64ae5212a48a5fc12ee5bb5ff1f23c0394c8891b5a89dc",
                "33": "3a58be00b601a08cf9faa55e1643bf24adfd08c0a0648cbe851e9f9e49388c5e",
            },
            "ranking_aware": {
                "11": "1833f9a4f68aa402587f2842e3143ee018423ff12c36afdbc61f64c0120d9588",
                "22": "cbd1aae6c83fe7d06a1226ea58cc5953e3745f81912ad93b9a1328ffdf267ac2",
                "33": "f4543610be2dab8adf7a14ca4bf9b862bacde49c3a3c048fb6e46d3f50097675",
            },
        },
        "all_models_frozen": True,
        "training_or_finetuning_allowed": False,
    }
    if dict(resources) != expected_resources:
        raise ValueError("Frozen Stage A resources changed.")

    proposal = _mapping(config.get("proposal"), "proposal")
    if dict(proposal) != {
        "error_guided_fraction": 0.8,
        "min_length": 0.1,
        "max_length": 0.6,
        "width_choices": [1, 2, 3, 4],
        "value_choices": [0, 32, 64, 96, 128],
    }:
        raise ValueError("Stage A proposal settings changed.")

    development = _mapping(
        config.get("development_score_audit"),
        "development_score_audit",
    )
    expected_development_authorized = status == "development_score_audit_authorized_once"
    if dict(development) != {
        "authorized": expected_development_authorized,
        "target_seeds": list(DEVELOPMENT_TARGET_SEEDS),
        "state_planner_seeds": list(DEVELOPMENT_STATE_SEEDS),
        "candidate_seeds": list(DEVELOPMENT_CANDIDATE_SEEDS),
        "target_strokes": 20,
        "state_trajectory_steps": 80,
        "state_sources": {
            "include_blank": True,
            "exact_pixel_steps": [20, 40, 60, 80],
            "random_steps": [20, 40, 60, 80],
        },
        "states_per_target": 9,
        "candidates_per_state": 128,
        "prediction_batch_size": 32,
        "predictor_families": list(PREDICTOR_FAMILIES),
        "scores": list(SCORE_NAMES),
        "score_definitions": {
            "normalized_latent_mse": (
                "mean full-grid squared error after per-patch L2 normalization"
            ),
            "normalized_latent_l1": (
                "mean full-grid absolute error after per-patch L2 normalization"
            ),
            "pixel_error_weighted_normalized_latent_mse": (
                "normalized-latent patch MSE weighted by pooled exact current-to-target "
                "absolute pixel error plus its patch mean, then normalized to unit mean weight"
            ),
            "decoded_pixel_l1": (
                "mean absolute error between the exact target pixels and the frozen decoder "
                "output of the predicted standardized next latent after inverse standardization"
            ),
            "decoded_pixel_l1_plus_quarter_sobel_l1": (
                "decoded pixel L1 plus 0.25 times mean absolute Sobel-x and Sobel-y response error"
            ),
        },
        "sobel_edge_weight": 0.25,
        "exact_label": "pixel_mse_of_exactly_rendered_candidate_to_target",
        "exact_rank_tolerance": 1e-12,
        "selection_order": [
            "lowest_mean_exact_regret",
            "highest_exact_top5_rate",
            "highest_mean_score_exact_spearman",
            "fixed_score_simplicity_order",
            "mse_only_before_ranking_aware",
        ],
        "score_simplicity_order": list(SCORE_NAMES),
        "output_dir": "outputs/planner-score-audit-development-2026-08-23",
        "single_run": True,
    }:
        raise ValueError("Development score-audit settings changed.")

    planner_development = _mapping(
        config.get("planner_development"),
        "planner_development",
    )
    expected_planner_authorized = status == "planner_development_authorized_once"
    if dict(planner_development) != {
        "authorized": expected_planner_authorized,
        "target_seeds": list(PLANNER_DEVELOPMENT_TARGET_SEEDS),
        "planner_seeds": list(PLANNER_DEVELOPMENT_SEEDS),
        "target_strokes": 20,
        "maximum_steps": 100,
        "candidates_per_step": 128,
        "prediction_batch_size": 32,
        "methods": [
            "exact_pixel",
            "learned_pixel",
            "current_latent_mse_forced",
            "development_selected_score_forced",
            "development_selected_score_no_op",
        ],
        "no_op_rule": (
            "stop when the selected score of the exactly observed current canvas is less "
            "than or equal to the minimum predicted candidate score"
        ),
        "no_op_margin": 0.0,
        "eligibility_for_confirmatory": {
            "implementation_integrity_required": True,
            "selected_pair_must_match_score_audit": True,
            "improve_every_target_from_blank": True,
            "minimum_mean_final_mse_reduction_vs_current_latent_mse_forced": 0.0,
        },
        "output_dir": (
            "outputs/planner-score-audit-planner-development-2026-08-23"
        ),
        "single_run": True,
    }:
        raise ValueError("Planner-development settings changed.")

    confirmatory = _mapping(
        config.get("confirmatory_reserved"),
        "confirmatory_reserved",
    )
    expected_confirmatory_authorized = status == "confirmatory_authorized_once"
    if dict(confirmatory) != {
        "authorized": expected_confirmatory_authorized,
        "target_seeds": list(CONFIRMATORY_TARGET_SEEDS),
        "planner_seeds": list(CONFIRMATORY_PLANNER_SEEDS),
        "target_strokes": 20,
        "maximum_steps": 100,
        "candidates_per_step": 128,
        "methods": [
            "exact_pixel",
            "learned_pixel",
            "current_latent_mse_forced",
            "development_selected_score_forced",
            "development_selected_score_no_op",
        ],
        "success_criteria": {
            "selected_no_op_improves_every_target_from_blank": True,
            "minimum_mean_final_mse_reduction_vs_current_latent_mse_forced": 0.05,
            "maximum_mean_final_mse_ratio_to_exact_pixel": 1.5,
            "implementation_integrity_required": True,
            "outperform_learned_pixel_required": False,
        },
        "output_dir": "outputs/planner-score-audit-confirmatory-2026-08-23",
        "single_run": True,
    }:
        raise ValueError("Confirmatory settings changed.")

    boundary = _mapping(config.get("validation_boundary"), "validation_boundary")
    if dict(boundary) != {
        "may_load_models": False,
        "may_generate_targets": False,
        "may_generate_state_trajectories": False,
        "may_generate_candidate_sets": False,
        "may_train_or_finetune": False,
        "may_create_output_directories": False,
    }:
        raise ValueError("Validation-only boundary changed.")

    all_new_seeds = (
        DEVELOPMENT_TARGET_SEEDS
        + DEVELOPMENT_STATE_SEEDS
        + DEVELOPMENT_CANDIDATE_SEEDS
        + PLANNER_DEVELOPMENT_TARGET_SEEDS
        + PLANNER_DEVELOPMENT_SEEDS
        + CONFIRMATORY_TARGET_SEEDS
        + CONFIRMATORY_PLANNER_SEEDS
    )
    if len(all_new_seeds) != len(set(all_new_seeds)):
        raise ValueError("Stage A seed groups must be disjoint.")
    if set(all_new_seeds).intersection(PROTECTED_PREVIOUS_SEEDS):
        raise ValueError("Stage A seeds overlap prior protected experiments.")


def validate_closed_resource_references(
    config: Mapping[str, Any],
    closed_config: Mapping[str, Any],
) -> dict[str, bool]:
    """Verify that Stage A references only the immutable closed resources."""

    resources = _mapping(config.get("frozen_resources"), "frozen_resources")
    if closed_config.get("status") != resources["required_latent_planner_status"]:
        raise ValueError("The source latent-planner experiment is not closed.")
    if _mapping(closed_config.get("smoke"), "closed smoke").get("authorized") is not False:
        raise ValueError("The closed smoke became authorized.")
    if _mapping(closed_config.get("controlled"), "closed controlled").get("authorized") is not False:
        raise ValueError("The closed controlled comparison became authorized.")
    representation = _mapping(closed_config.get("representation"), "representation")
    for source_name, resource_name in (
        ("autoencoder_checkpoint", "autoencoder_checkpoint"),
        ("autoencoder_state_sha256", "autoencoder_state_sha256"),
        ("latent_statistics", "latent_statistics"),
        ("latent_statistics_sha256", "latent_statistics_sha256"),
    ):
        if representation.get(source_name) != resources.get(resource_name):
            raise ValueError(f"Frozen resource {resource_name!r} changed.")
    predictors = _mapping(closed_config.get("latent_predictors"), "latent_predictors")
    observed = {
        family: {
            str(entry["seed"]): entry["state_sha256"]
            for entry in predictors[family]
        }
        for family in PREDICTOR_FAMILIES
    }
    if observed != resources["predictor_state_sha256"]:
        raise ValueError("Frozen predictor hashes changed.")
    return {
        "closed_latent_planner_verified": True,
        "closed_smoke_unauthorized": True,
        "closed_controlled_unauthorized": True,
        "autoencoder_hash_reference_verified": True,
        "latent_statistics_hash_reference_verified": True,
        "six_predictor_hash_references_verified": True,
    }


def score_audit_output_paths(config: Mapping[str, Any]) -> ScoreAuditOutputPaths:
    development = _mapping(
        config.get("development_score_audit"),
        "development_score_audit",
    )
    final = Path(str(development.get("output_dir", "")))
    if not final.name:
        raise ValueError("Development score-audit output directory is invalid.")
    return ScoreAuditOutputPaths(
        final=final,
        incomplete=final.with_name(final.name + ".incomplete"),
    )


def require_score_audit_outputs_absent(paths: ScoreAuditOutputPaths) -> None:
    """Refuse overwrites and preserve any partial development evidence."""

    if paths.final.exists():
        raise FileExistsError(f"Development score-audit output exists: {paths.final}")
    if paths.incomplete.exists():
        raise FileExistsError(
            f"Incomplete score-audit output exists: {paths.incomplete}. "
            "Preserve and review it before any retry."
        )


def require_score_audit_authorized(config: Mapping[str, Any]) -> None:
    """Stop before model loading or development target generation."""

    development = _mapping(
        config.get("development_score_audit"),
        "development_score_audit",
    )
    if (
        config.get("status") != "development_score_audit_authorized_once"
        or development.get("authorized") is not True
    ):
        raise PermissionError(
            "The development score audit is not authorized. No models were loaded, "
            "and no target, state trajectory, or candidate set was generated."
        )


def validate_score_audit_runner_request(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the unauthorized development runner without side effects."""

    if config.get("status") != "frozen_before_implementation_and_data":
        raise ValueError("Score-audit validation requires the initial frozen status.")
    development = _mapping(
        config.get("development_score_audit"),
        "development_score_audit",
    )
    planner_development = _mapping(
        config.get("planner_development"),
        "planner_development",
    )
    confirmatory = _mapping(
        config.get("confirmatory_reserved"),
        "confirmatory_reserved",
    )
    if any(
        phase.get("authorized") is not False
        for phase in (development, planner_development, confirmatory)
    ):
        raise ValueError("Every Stage A phase must be unauthorized during validation.")
    paths = score_audit_output_paths(config)
    require_score_audit_outputs_absent(paths)
    candidate_sets = len(DEVELOPMENT_TARGET_SEEDS) * int(
        development["states_per_target"]
    )
    return {
        "status": "planner_score_audit_runner_valid_unauthorized",
        "config_status": config["status"],
        "target_seeds_reserved": list(DEVELOPMENT_TARGET_SEEDS),
        "state_planner_seeds_reserved": list(DEVELOPMENT_STATE_SEEDS),
        "candidate_seeds_reserved": list(DEVELOPMENT_CANDIDATE_SEEDS),
        "target_count": len(DEVELOPMENT_TARGET_SEEDS),
        "states_per_target": development["states_per_target"],
        "candidate_sets": candidate_sets,
        "candidates_per_state": development["candidates_per_state"],
        "predictor_families": list(PREDICTOR_FAMILIES),
        "score_names": list(SCORE_NAMES),
        "predictor_score_pairs": len(PREDICTOR_FAMILIES) * len(SCORE_NAMES),
        "development_authorized": False,
        "planner_development_authorized": False,
        "confirmatory_authorized": False,
        "output_dir_available": True,
        "incomplete_dir_available": True,
        "models_loaded": False,
        "targets_generated": False,
        "state_trajectories_generated": False,
        "candidate_sets_generated": False,
        "models_trained_or_finetuned": False,
        "closed_targets_reused": False,
        "historical_results_unchanged": True,
    }


@torch.inference_mode()
def predict_candidate_latents(
    predictors: Sequence[nn.Module],
    current_tokens: torch.Tensor,
    candidates: Sequence[Stroke],
    *,
    canvas_size: int = 64,
    batch_size: int = 32,
) -> torch.Tensor:
    """Return frozen-model predicted next states as `[models, candidates, 256, 32]`."""

    if not predictors or not candidates:
        raise ValueError("Predictors and candidates cannot be empty.")
    if current_tokens.shape != (1, 256, 32):
        raise ValueError("current_tokens must have shape [1, 256, 32].")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if any(parameter.requires_grad for model in predictors for parameter in model.parameters()):
        raise ValueError("Score audit received a trainable predictor.")

    model_outputs: list[torch.Tensor] = []
    for model in predictors:
        model.eval()
        parts: list[torch.Tensor] = []
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            actions = torch.stack([stroke_action_vector(stroke) for stroke in batch])
            masks = torch.stack(
                [
                    stroke_patch_coverage(stroke, canvas_size, (16, 16))
                    for stroke in batch
                ]
            )
            repeated = current_tokens.float().expand(len(batch), -1, -1)
            predicted = repeated + model(
                repeated,
                actions.float(),
                masks.float(),
            )
            parts.append(predicted.cpu())
        model_outputs.append(torch.cat(parts, dim=0))
    result = torch.stack(model_outputs, dim=0)
    expected = (len(predictors), len(candidates), 256, 32)
    if result.shape != expected:
        raise RuntimeError(f"Unexpected predicted-latent shape: {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Predicted candidate latents are non-finite.")
    return result


def pixel_error_patch_weights(
    current: Image.Image,
    target: Image.Image,
) -> torch.Tensor:
    """Return exact current-target error weights on the frozen 16x16 patch grid."""

    tensors = images_to_grayscale_tensor((current, target)).float()
    error = (tensors[0:1] - tensors[1:2]).abs()
    pooled = F.avg_pool2d(error, kernel_size=4, stride=4).flatten(start_dim=1)
    mean_error = pooled.mean()
    if float(mean_error.item()) <= 1e-12:
        weights = torch.ones_like(pooled)
    else:
        weights = pooled + mean_error
        weights = weights / weights.mean()
    if weights.shape != (1, 256) or not bool(torch.isfinite(weights).all()):
        raise RuntimeError("Pixel-error patch weights are invalid.")
    return weights.cpu()


@torch.inference_mode()
def decode_standardized_tokens(
    autoencoder: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    tokens: torch.Tensor,
    *,
    batch_size: int = 32,
) -> torch.Tensor:
    """Inverse-standardize `[N, 256, 32]` tokens and decode frozen canvases."""

    if tokens.ndim != 3 or tokens.shape[1:] != (256, 32):
        raise ValueError("tokens must have shape [N, 256, 32].")
    if tokens.shape[0] < 1 or batch_size < 1:
        raise ValueError("Token count and batch_size must be positive.")
    if any(parameter.requires_grad for parameter in autoencoder.parameters()):
        raise ValueError("Score audit requires a frozen autoencoder.")
    if statistics.mean.shape != (32,) or statistics.std.shape != (32,):
        raise ValueError("Frozen latent statistics have the wrong shape.")

    safe_std = statistics.std.float().clamp_min(1e-6)
    raw_tokens = (
        tokens.float() * safe_std[None, None, :]
        + statistics.mean.float()[None, None, :]
    )
    maps = raw_tokens.reshape(-1, 16, 16, 32).permute(0, 3, 1, 2)
    autoencoder.eval()
    parts = [
        autoencoder.decode_map(maps[start : start + batch_size]).cpu()
        for start in range(0, len(maps), batch_size)
    ]
    decoded = torch.cat(parts, dim=0)
    if decoded.shape != (len(tokens), 1, 64, 64):
        raise RuntimeError("Decoded score-audit canvas shape is invalid.")
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("Decoded score-audit canvases are non-finite.")
    return decoded


def _sobel_responses(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1:] != (1, 64, 64):
        raise ValueError("Sobel inputs must have shape [N, 1, 64, 64].")
    kernels = torch.tensor(
        [
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        ],
        dtype=images.dtype,
        device=images.device,
    )
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernels)


@torch.inference_mode()
def candidate_score_variants(
    predicted_next: torch.Tensor,
    target_tokens: torch.Tensor,
    target_pixels: torch.Tensor,
    patch_weights: torch.Tensor,
    autoencoder: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    *,
    batch_size: int = 32,
    sobel_edge_weight: float = 0.25,
) -> dict[str, CandidateScoreResult]:
    """Compute all five frozen scores without selecting or tuning one."""

    if predicted_next.ndim != 4 or predicted_next.shape[2:] != (256, 32):
        raise ValueError("predicted_next must have shape [M, C, 256, 32].")
    if target_tokens.shape != (1, 256, 32):
        raise ValueError("target_tokens must have shape [1, 256, 32].")
    if target_pixels.shape != (1, 1, 64, 64):
        raise ValueError("target_pixels must have shape [1, 1, 64, 64].")
    if patch_weights.shape != (1, 256):
        raise ValueError("patch_weights must have shape [1, 256].")
    if sobel_edge_weight != 0.25:
        raise ValueError("The frozen Sobel edge weight must remain 0.25.")

    model_count, candidate_count = predicted_next.shape[:2]
    predicted_normalized = F.normalize(predicted_next.float(), dim=-1)
    target_normalized = F.normalize(target_tokens.float(), dim=-1)
    difference = predicted_normalized - target_normalized[None, :, :, :]
    latent_mse = difference.square().mean(dim=(2, 3))
    latent_l1 = difference.abs().mean(dim=(2, 3))
    per_patch_mse = difference.square().mean(dim=3)
    weighted_mse = (
        per_patch_mse * patch_weights.float()[None, :, :]
    ).mean(dim=2)

    flattened = predicted_next.reshape(model_count * candidate_count, 256, 32)
    decoded = decode_standardized_tokens(
        autoencoder,
        statistics,
        flattened,
        batch_size=batch_size,
    )
    repeated_target = target_pixels.float().expand(len(decoded), -1, -1, -1)
    pixel_l1_flat = (decoded - repeated_target).abs().mean(dim=(1, 2, 3))
    decoded_edges = _sobel_responses(decoded)
    target_edges = _sobel_responses(target_pixels.float())
    edge_l1_flat = (
        decoded_edges - target_edges.expand(len(decoded), -1, -1, -1)
    ).abs().mean(dim=(1, 2, 3))
    pixel_l1 = pixel_l1_flat.reshape(model_count, candidate_count)
    pixel_edge = (
        pixel_l1_flat + sobel_edge_weight * edge_l1_flat
    ).reshape(model_count, candidate_count)

    tensors = {
        "normalized_latent_mse": latent_mse,
        "normalized_latent_l1": latent_l1,
        "pixel_error_weighted_normalized_latent_mse": weighted_mse,
        "decoded_pixel_l1": pixel_l1,
        "decoded_pixel_l1_plus_quarter_sobel_l1": pixel_edge,
    }
    results: dict[str, CandidateScoreResult] = {}
    for name in SCORE_NAMES:
        per_model = tensors[name].cpu().numpy().astype(np.float64, copy=False)
        aggregate = per_model.mean(axis=0)
        if per_model.shape != (model_count, candidate_count):
            raise RuntimeError(f"Per-model score shape is invalid for {name}.")
        if aggregate.shape != (candidate_count,):
            raise RuntimeError(f"Aggregate score shape is invalid for {name}.")
        if not bool(np.isfinite(per_model).all() and np.isfinite(aggregate).all()):
            raise RuntimeError(f"Score {name} produced a non-finite value.")
        results[name] = CandidateScoreResult(
            aggregate=aggregate,
            per_model=per_model,
        )
    return results


def exact_candidate_metrics(
    predicted_scores: np.ndarray,
    exact_scores: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> dict[str, int | float | bool]:
    """Evaluate one score vector against exact target-pixel candidate outcomes."""

    predicted = np.asarray(predicted_scores, dtype=np.float64)
    exact = np.asarray(exact_scores, dtype=np.float64)
    if predicted.ndim != 1 or exact.shape != predicted.shape or len(predicted) < 2:
        raise ValueError("Predicted and exact scores must be matching 1D candidate arrays.")
    if tolerance < 0 or not bool(np.isfinite(predicted).all() and np.isfinite(exact).all()):
        raise ValueError("Candidate scores and tolerance must be finite and valid.")
    selected_index = int(np.argmin(predicted))
    exact_selected = float(exact[selected_index])
    exact_best = float(exact.min())
    exact_rank = 1 + int(np.sum(exact < exact_selected - tolerance))
    return {
        "selected_index": selected_index,
        "candidate_count": len(predicted),
        "predicted_selected_score": float(predicted[selected_index]),
        "predicted_score_range": float(np.ptp(predicted)),
        "exact_selected_mse": exact_selected,
        "exact_best_candidate_mse": exact_best,
        "exact_selected_rank": exact_rank,
        "exact_top1": exact_rank == 1,
        "exact_top5": exact_rank <= 5,
        "exact_regret": max(0.0, exact_selected - exact_best),
        "score_exact_spearman": spearman_rank_correlation(predicted, exact),
    }


def validate_score_audit_summary(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Validate all 720 development pair/state records before selection."""

    development = _mapping(
        config.get("development_score_audit"),
        "development_score_audit",
    )
    expected_rows = (
        len(DEVELOPMENT_TARGET_SEEDS)
        * int(development["states_per_target"])
        * len(PREDICTOR_FAMILIES)
        * len(SCORE_NAMES)
    )
    if len(summary) != expected_rows:
        raise RuntimeError("Score-audit summary row count is incorrect.")
    keys = ["target_id", "state_id", "predictor_family", "score_name"]
    if bool(summary.duplicated(keys).any()):
        raise RuntimeError("Score-audit summary contains duplicate pair/state rows.")
    for index, target_seed in enumerate(DEVELOPMENT_TARGET_SEEDS, start=1):
        target_id = f"target_{index:02d}"
        subset = summary.loc[summary["target_id"] == target_id]
        expected_target_rows = (
            int(development["states_per_target"])
            * len(PREDICTOR_FAMILIES)
            * len(SCORE_NAMES)
        )
        if len(subset) != expected_target_rows or set(subset["target_seed"]) != {target_seed}:
            raise RuntimeError(f"Score-audit target rows changed for {target_id}.")
        if subset["state_id"].nunique() != int(development["states_per_target"]):
            raise RuntimeError(f"Score-audit state count changed for {target_id}.")
    if set(summary["predictor_family"]) != set(PREDICTOR_FAMILIES):
        raise RuntimeError("Score-audit predictor families changed.")
    if set(summary["score_name"]) != set(SCORE_NAMES):
        raise RuntimeError("Score-audit score set changed.")
    if set(summary["candidate_count"]) != {development["candidates_per_state"]}:
        raise RuntimeError("Score-audit candidate count changed.")
    numeric = [
        "predicted_selected_score",
        "predicted_score_range",
        "exact_selected_mse",
        "exact_best_candidate_mse",
        "exact_selected_rank",
        "exact_regret",
        "score_exact_spearman",
        "elapsed_seconds",
    ]
    if not bool(np.isfinite(summary[numeric].to_numpy(dtype=np.float64)).all()):
        raise RuntimeError("Score-audit summary contains a non-finite value.")
    if not bool(
        (
            (summary["exact_selected_rank"] >= 1)
            & (summary["exact_selected_rank"] <= development["candidates_per_state"])
        ).all()
    ):
        raise RuntimeError("Score-audit exact rank is out of range.")


def aggregate_score_audit(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every frozen predictor/score pair over equal state weight."""

    rows: list[dict[str, Any]] = []
    for family in PREDICTOR_FAMILIES:
        for score_name in SCORE_NAMES:
            subset = summary.loc[
                (summary["predictor_family"] == family)
                & (summary["score_name"] == score_name)
            ]
            rows.append(
                {
                    "predictor_family": family,
                    "score_name": score_name,
                    "candidate_sets": int(len(subset)),
                    "exact_top1_rate": float(subset["exact_top1"].mean()),
                    "exact_top5_rate": float(subset["exact_top5"].mean()),
                    "mean_exact_rank": float(subset["exact_selected_rank"].mean()),
                    "mean_exact_regret": float(subset["exact_regret"].mean()),
                    "max_exact_regret": float(subset["exact_regret"].max()),
                    "mean_score_exact_spearman": float(
                        subset["score_exact_spearman"].mean()
                    ),
                    "mean_predicted_score_range": float(
                        subset["predicted_score_range"].mean()
                    ),
                    "mean_elapsed_seconds": float(subset["elapsed_seconds"].mean()),
                }
            )
    aggregate = pd.DataFrame(rows)
    if not bool(
        np.isfinite(
            aggregate.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
    ):
        raise RuntimeError("Aggregated score-audit metrics are non-finite.")
    return aggregate


def select_score_pair(aggregate: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen lexicographic development selection rule."""

    required = {
        "predictor_family",
        "score_name",
        "mean_exact_regret",
        "exact_top5_rate",
        "mean_score_exact_spearman",
    }
    if not required.issubset(aggregate.columns) or aggregate.empty:
        raise ValueError("Aggregate score-audit table is incomplete.")
    score_order = {name: index for index, name in enumerate(SCORE_NAMES)}
    family_order = {name: index for index, name in enumerate(PREDICTOR_FAMILIES)}
    working = aggregate.copy()
    working["_score_order"] = working["score_name"].map(score_order)
    working["_family_order"] = working["predictor_family"].map(family_order)
    if bool(working[["_score_order", "_family_order"]].isna().any().any()):
        raise ValueError("Aggregate table contains an unknown predictor or score.")
    ordered = working.sort_values(
        [
            "mean_exact_regret",
            "exact_top5_rate",
            "mean_score_exact_spearman",
            "_score_order",
            "_family_order",
        ],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    winner = ordered.iloc[0]
    return {
        "predictor_family": str(winner["predictor_family"]),
        "score_name": str(winner["score_name"]),
        "mean_exact_regret": float(winner["mean_exact_regret"]),
        "exact_top5_rate": float(winner["exact_top5_rate"]),
        "mean_score_exact_spearman": float(
            winner["mean_score_exact_spearman"]
        ),
        "selection_order": [
            "lowest_mean_exact_regret",
            "highest_exact_top5_rate",
            "highest_mean_score_exact_spearman",
            "fixed_score_simplicity_order",
            "mse_only_before_ranking_aware",
        ],
    }


def exact_candidate_scores(
    current: Image.Image,
    target: Image.Image,
    candidates: Sequence[Stroke],
) -> tuple[tuple[Image.Image, ...], np.ndarray]:
    """Render a candidate set once and return exact target-pixel MSE labels."""

    canvases = render_candidate_canvases(current, candidates)
    scores = np.asarray(
        [pixel_mse(canvas, target) for canvas in canvases],
        dtype=np.float64,
    )
    if not bool(np.isfinite(scores).all()):
        raise RuntimeError("Exact candidate labels are non-finite.")
    return canvases, scores
