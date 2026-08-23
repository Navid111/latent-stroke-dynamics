"""Validation-only architecture and objectives for the frozen Phase B0 protocol."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PHASE_B_CONFIG = Path(
    "configs/phase-b-saliency-latent-2026-08-23.json"
)
EXPECTED_TRAINABLE_PARAMETERS = 392_345


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


def load_phase_b_config(
    path: str | Path = DEFAULT_PHASE_B_CONFIG,
) -> dict[str, Any]:
    """Load and strictly validate the Phase B0 frozen protocol."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_phase_b_config(config)
    return config


def _data_seeds(config: Mapping[str, Any]) -> list[int]:
    seeds: list[int] = []
    development = _mapping(config.get("development"), "development")
    for split in _mapping(
        development.get("transition_splits"),
        "development.transition_splits",
    ).values():
        seeds.append(int(_mapping(split, "development transition split")["seed"]))
    for section_name in (
        "planner_supervision_train",
        "planner_supervision_validation",
    ):
        section = _mapping(development.get(section_name), f"development.{section_name}")
        for key in ("target_seeds", "trajectory_seeds", "candidate_seeds"):
            seeds.extend(int(value) for value in section[key])
    long_horizon = _mapping(development.get("long_horizon"), "development.long_horizon")
    seeds.extend(int(value) for value in long_horizon["target_seeds"])
    seeds.extend(int(value) for value in long_horizon["planner_seeds"])

    formal = _mapping(config.get("formal_reserved"), "formal_reserved")
    for split in _mapping(
        formal.get("transition_splits"),
        "formal_reserved.transition_splits",
    ).values():
        seeds.append(int(_mapping(split, "formal transition split")["seed"]))
    for section_name in (
        "planner_supervision_train",
        "planner_supervision_validation",
    ):
        section = _mapping(formal.get(section_name), f"formal_reserved.{section_name}")
        for key in ("target_seeds", "trajectory_seeds", "candidate_seeds"):
            seeds.extend(int(value) for value in section[key])
    formal_horizon = _mapping(formal.get("long_horizon"), "formal_reserved.long_horizon")
    seeds.extend(int(value) for value in formal_horizon["target_seeds"])
    seeds.extend(int(value) for value in formal_horizon["planner_seeds"])
    return seeds


def validate_phase_b_config(config: Mapping[str, Any]) -> None:
    """Reject architecture, objective, seed, and authorization drift."""

    expected_scalars = {
        "experiment_id": "phase-b0-action-conditioned-joint-embedding-2026-08-23",
        "status": "frozen_before_implementation_and_data",
        "evidential_role": "post_stage_a_exploratory_training_extension",
        "branch": "phase-b/saliency-latent",
        "base_commit": "c211c3ab3a37b9c37eda5ba3c07c01173fd4c7f7",
        "historical_results_unchanged": True,
        "closed_targets_may_be_reused": False,
        "device": "cpu",
        "canvas_size": 64,
    }
    for key, expected in expected_scalars.items():
        if config.get(key) != expected:
            raise ValueError(f"Frozen Phase B0 field {key!r} changed.")

    renderer = _mapping(config.get("renderer"), "renderer")
    if dict(renderer) != {
        "mode": "L",
        "stroke_primitive": "normalized_straight_line",
        "widths": [1, 2, 3, 4],
        "values": [0, 32, 64, 96, 128],
        "minimum_transition_length": 0.2,
        "transition_crowding": [0, 5, 15, 30],
        "target_strokes": 20,
        "execute_with_exact_renderer": True,
        "renderer_changes_allowed": False,
    }:
        raise ValueError("Frozen Phase B0 renderer settings changed.")

    comparators = _mapping(config.get("closed_comparators"), "closed_comparators")
    if comparators.get("all_closed_comparators_frozen") is not True:
        raise ValueError("Every historical comparator must remain frozen.")
    task_autoencoder = _mapping(comparators.get("task_autoencoder"), "task_autoencoder")
    latent_statistics = _mapping(comparators.get("latent_statistics"), "latent_statistics")
    pixel = _mapping(comparators.get("learned_pixel_predictor"), "learned_pixel_predictor")
    if task_autoencoder.get("path") != (
        "outputs/representation-extension-2026-08-22/task_autoencoder/"
        "checkpoints/task_autoencoder.pt"
    ) or task_autoencoder.get("state_sha256") != (
        "95de3ecef8eeb7a350e862fa21185a168d9304870cb0c8391cbd008e88d93900"
    ):
        raise ValueError("Frozen task-autoencoder reference changed.")
    if latent_statistics.get("path") != (
        "outputs/representation-extension-2026-08-22/task_autoencoder/"
        "latent_channel_statistics.json"
    ) or latent_statistics.get("sha256") != (
        "c2a3d781dab19a4714189d580dafb5ea95231af06021d3980beb495a3b85d903"
    ):
        raise ValueError("Frozen latent-statistics reference changed.")
    if pixel.get("path") != "checkpoints/stage3-pixel-mlp-seed11.pt" or pixel.get(
        "state_sha256"
    ) != "e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472":
        raise ValueError("Frozen learned-pixel reference changed.")
    mse_predictors = _mapping(
        comparators.get("mse_only_predictors"),
        "mse_only_predictors",
    )
    if mse_predictors.get("score") != "normalized_latent_l1" or mse_predictors.get(
        "seeds"
    ) != [11, 22, 33]:
        raise ValueError("Archived latent comparator changed.")
    hashes = _mapping(mse_predictors.get("state_sha256"), "mse predictor hashes")
    if not all(_is_sha256(hashes.get(str(seed))) for seed in (11, 22, 33)):
        raise ValueError("Archived MSE-only hashes must remain frozen.")

    model = _mapping(config.get("model"), "model")
    if model.get("name") != "MultiScaleActionJointEmbeddingModel":
        raise ValueError("Frozen Phase B0 model name changed.")
    if model.get("input_shape") != [1, 64, 64]:
        raise ValueError("Frozen Phase B0 input shape changed.")
    if model.get("latent_scales") != {
        "32": [32, 32, 32],
        "16": [64, 16, 16],
    }:
        raise ValueError("Frozen Phase B0 latent scales changed.")
    if len(model.get("online_encoder", [])) != 8:
        raise ValueError("Frozen online encoder layer count changed.")
    if len(model.get("action_encoder", [])) != 2:
        raise ValueError("Frozen action encoder layer count changed.")
    if len(model.get("predictor_32", [])) != 2:
        raise ValueError("Frozen 32-scale predictor changed.")
    predictor_16 = _mapping(model.get("predictor_16"), "predictor_16")
    if predictor_16.get("pooled_residual_32_channels") != 32 or len(
        predictor_16.get("layers", [])
    ) != 2:
        raise ValueError("Frozen 16-scale predictor changed.")
    target_encoder = _mapping(model.get("target_encoder"), "target_encoder")
    if dict(target_encoder) != {
        "architecture": "exact_online_encoder_copy",
        "gradient": False,
        "update": "exponential_moving_average_after_optimizer_step",
        "momentum": 0.99,
        "stop_gradient_targets": True,
    }:
        raise ValueError("Frozen target-encoder mechanism changed.")
    action_raster = _mapping(model.get("action_raster"), "action_raster")
    if action_raster.get("shape") != [2, 64, 64] or action_raster.get(
        "no_op"
    ) != "all_zeros":
        raise ValueError("Frozen action-raster representation changed.")
    progress_head = _mapping(model.get("progress_head"), "progress_head")
    if progress_head.get("input_dim") != 224 or progress_head.get("layers") != [
        224,
        128,
        64,
        1,
    ]:
        raise ValueError("Frozen progress head changed.")
    if model.get("decoder_used") is not False:
        raise ValueError("Phase B0 must not use a decoder.")
    if int(model.get("maximum_trainable_parameters", 0)) != 500_000:
        raise ValueError("Frozen trainable parameter cap changed.")
    if model.get("exact_parameter_count_manifest_required_before_development") is not True:
        raise ValueError("Phase B0 implementation manifest requirement changed.")

    objectives = _mapping(config.get("objectives"), "objectives")
    joint = _mapping(objectives.get("joint_prediction"), "joint_prediction")
    if joint.get("type") != "balanced_smooth_l1_on_predicted_next_latents" or joint.get(
        "scale_weights"
    ) != {"32": 0.5, "16": 0.5}:
        raise ValueError("Frozen joint-prediction objective changed.")
    collapse = _mapping(objectives.get("anti_collapse"), "anti_collapse")
    if dict(collapse) != {
        "variance_floor": 1.0,
        "variance_weight": 0.1,
        "covariance_weight": 0.01,
        "axes": "batch_and_spatial_per_scale",
    }:
        raise ValueError("Frozen anti-collapse objective changed.")
    no_op = _mapping(objectives.get("no_op_consistency"), "no_op_consistency")
    if dict(no_op) != {
        "transition_fraction": 0.1,
        "residual_weight": 0.25,
        "exact_next_equals_current": True,
    }:
        raise ValueError("Frozen no-op objective changed.")
    planner = _mapping(objectives.get("planner_progress"), "planner_progress")
    expected_planner = {
        "regression": "smooth_l1_on_train_standardized_exact_progress",
        "regression_weight": 1.0,
        "regression_beta": 1.0,
        "ranking": "cross_entropy_over_predicted_standardized_progress",
        "ranking_weight": 0.3,
        "ranking_temperature": 0.1,
        "candidates_per_state": 32,
        "no_op_index": 0,
        "target_guided_strokes": 31,
        "exact_tie_tolerance": 1e-12,
        "exact_tie_break": "lowest_candidate_index",
    }
    if dict(planner) != expected_planner:
        raise ValueError("Frozen planner-progress objective changed.")
    if objectives.get("development_variants") != [
        "joint_prediction_only",
        "joint_prediction_progress",
    ] or objectives.get("hyperparameter_grid_allowed") is not False:
        raise ValueError("Frozen Phase B0 development variants changed.")

    training = _mapping(config.get("training"), "training")
    if dict(training) != {
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "batch_size": 16,
        "maximum_epochs": 40,
        "patience": 8,
        "gradient_clip_norm": 5.0,
        "development_model_seed": 73,
        "maximum_completed_development_executions": 1,
        "development_wall_clock_cap_hours": 6.0,
        "device": "cpu",
    }:
        raise ValueError("Frozen Phase B0 training settings changed.")

    development = _mapping(config.get("development"), "development")
    formal = _mapping(config.get("formal_reserved"), "formal_reserved")
    b1 = _mapping(config.get("region_scheduler_reserved"), "region_scheduler_reserved")
    b2 = _mapping(config.get("rgb_high_resolution_reserved"), "rgb_high_resolution_reserved")
    if development.get("authorized") is not False:
        raise ValueError("Phase B0 development must remain unauthorized.")
    if formal.get("authorized") is not False:
        raise ValueError("Formal Phase B0 must remain unauthorized.")
    if b1.get("authorized") is not False or b1.get(
        "requires_separate_frozen_protocol"
    ) is not True:
        raise ValueError("Phase B1 must remain separately frozen and unauthorized.")
    if b2.get("authorized") is not False or b2.get(
        "requires_separate_frozen_protocol"
    ) is not True:
        raise ValueError("Phase B2 must remain separately frozen and unauthorized.")

    seeds = _data_seeds(config)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Every Phase B0 data seed must be disjoint.")
    if not seeds or min(seeds) < 20270401 or max(seeds) > 20270586:
        raise ValueError("Phase B0 data seeds left the frozen reserved range.")

    eligibility = _mapping(
        config.get("development_eligibility"),
        "development_eligibility",
    )
    if dict(eligibility) != {
        "implementation_integrity_required": True,
        "historical_artifacts_unchanged_required": True,
        "minimum_mean_channel_std_each_scale": 0.5,
        "minimum_diagnostic_four_way_retrieval": 0.5,
        "minimum_mean_128_way_regret_reduction_vs_archived_mse_l1": 0.1,
        "no_op_improves_every_target_from_blank": True,
        "minimum_mean_final_mse_reduction_vs_archived_mse_l1": 0.05,
        "no_op_no_worse_than_joint_prediction_only_forced": True,
        "maximum_mean_final_mse_ratio_to_exact_pixel": 1.5,
        "maximum_premature_stop_rate": 0.1,
        "compute_cap_required": True,
    }:
        raise ValueError("Frozen Phase B0 eligibility criteria changed.")

    boundary = _mapping(config.get("validation_boundary"), "validation_boundary")
    allowed_true = {
        "may_load_config",
        "may_instantiate_random_models",
        "may_use_deterministic_random_dummy_tensors",
        "may_compute_shapes_gradients_ema_losses_and_parameter_counts",
        "may_run_dummy_overfit_check",
    }
    for key, value in boundary.items():
        if key in allowed_true and value is not True:
            raise ValueError(f"Validation permission {key!r} changed.")
        if key not in allowed_true and value is not False:
            raise ValueError(f"Validation prohibition {key!r} changed.")


class ConvGNGeLU(nn.Module):
    """One frozen convolution, GroupNorm, and GELU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
    ) -> None:
        super().__init__()
        if out_channels % groups:
            raise ValueError("GroupNorm groups must divide output channels.")
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(value)))


class MultiScaleJointEncoder(nn.Module):
    """The frozen 32×32 and 16×16 Phase B0 image encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvGNGeLU(1, 24, kernel_size=3, stride=1, padding=1, groups=6)
        self.stage_64 = ConvGNGeLU(
            24, 24, kernel_size=3, stride=1, padding=1, groups=6
        )
        self.down_32 = ConvGNGeLU(
            24, 48, kernel_size=4, stride=2, padding=1, groups=8
        )
        self.stage_32 = ConvGNGeLU(
            48, 48, kernel_size=3, stride=1, padding=1, groups=8
        )
        self.down_16 = ConvGNGeLU(
            48, 64, kernel_size=4, stride=2, padding=1, groups=8
        )
        self.stage_16 = ConvGNGeLU(
            64, 64, kernel_size=3, stride=1, padding=1, groups=8
        )
        self.projection_32 = nn.Conv2d(48, 32, kernel_size=1, bias=True)
        self.projection_16 = nn.Conv2d(64, 64, kernel_size=1, bias=True)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4 or images.shape[1:] != (1, 64, 64):
            raise ValueError("Phase B0 encoder expects [batch, 1, 64, 64].")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("Phase B0 encoder input must be finite.")
        value = self.stage_64(self.stem(images))
        value_32 = self.stage_32(self.down_32(value))
        value_16 = self.stage_16(self.down_16(value_32))
        result = {
            "32": self.projection_32(value_32),
            "16": self.projection_16(value_16),
        }
        if result["32"].shape[1:] != (32, 32, 32):
            raise RuntimeError("Unexpected Phase B0 32-scale shape.")
        if result["16"].shape[1:] != (64, 16, 16):
            raise RuntimeError("Unexpected Phase B0 16-scale shape.")
        return result


class SpatialActionEncoder(nn.Module):
    """Encode the two-channel pre-rendered stroke action at both scales."""

    def __init__(self) -> None:
        super().__init__()
        self.action_32 = ConvGNGeLU(
            2, 16, kernel_size=5, stride=2, padding=2, groups=4
        )
        self.action_16 = ConvGNGeLU(
            16, 32, kernel_size=3, stride=2, padding=1, groups=8
        )

    def forward(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if actions.ndim != 4 or actions.shape[1:] != (2, 64, 64):
            raise ValueError("Phase B0 action raster expects [batch, 2, 64, 64].")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("Phase B0 action raster must be finite.")
        value_32 = self.action_32(actions)
        value_16 = self.action_16(value_32)
        return {"32": value_32, "16": value_16}


class MultiScaleLatentPredictor(nn.Module):
    """Predict one latent residual at each frozen spatial scale."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_32 = ConvGNGeLU(
            48, 64, kernel_size=3, stride=1, padding=1, groups=8
        )
        self.output_32 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True)
        self.hidden_16 = ConvGNGeLU(
            128, 96, kernel_size=3, stride=1, padding=1, groups=8
        )
        self.output_16 = nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=True)

    def forward(
        self,
        current: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        residual_32 = self.output_32(
            self.hidden_32(torch.cat((current["32"], actions["32"]), dim=1))
        )
        pooled_32 = F.avg_pool2d(residual_32, kernel_size=2, stride=2)
        residual_16 = self.output_16(
            self.hidden_16(
                torch.cat(
                    (current["16"], actions["16"], pooled_32),
                    dim=1,
                )
            )
        )
        return {"32": residual_32, "16": residual_16}


class ExactProgressHead(nn.Module):
    """Predict standardized exact pixel-MSE reduction for one candidate."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(224, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        current_16: torch.Tensor,
        target_16: torch.Tensor,
        predicted_next_16: torch.Tensor,
        action_16: torch.Tensor,
    ) -> torch.Tensor:
        tensors = (current_16, target_16, predicted_next_16, action_16)
        if any(value.ndim != 4 for value in tensors):
            raise ValueError("Progress-head inputs must be spatial tensors.")
        pooled = torch.cat(
            (
                current_16.mean(dim=(2, 3)),
                target_16.mean(dim=(2, 3)),
                (predicted_next_16 - target_16).mean(dim=(2, 3)),
                action_16.mean(dim=(2, 3)),
            ),
            dim=1,
        )
        if pooled.shape[1] != 224:
            raise RuntimeError("Unexpected progress-head input dimension.")
        return self.network(pooled).squeeze(-1)


class MultiScaleActionJointEmbeddingModel(nn.Module):
    """Frozen Phase B0 architecture; no renderer data are used here."""

    def __init__(self) -> None:
        super().__init__()
        self.online_encoder = MultiScaleJointEncoder()
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.eval()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.action_encoder = SpatialActionEncoder()
        self.predictor = MultiScaleLatentPredictor()
        self.progress_head = ExactProgressHead()

    def train(self, mode: bool = True) -> "MultiScaleActionJointEmbeddingModel":
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def encode_target(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.target_encoder.eval()
        return {
            scale: value.detach()
            for scale, value in self.target_encoder(images).items()
        }

    def forward(
        self,
        current_images: torch.Tensor,
        action_rasters: torch.Tensor,
        goal_images: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        current = self.online_encoder(current_images)
        action = self.action_encoder(action_rasters)
        residual = self.predictor(current, action)
        predicted_next = {
            scale: current[scale] + residual[scale] for scale in ("32", "16")
        }
        result: dict[str, Any] = {
            "current": current,
            "action": action,
            "residual": residual,
            "predicted_next": predicted_next,
        }
        if goal_images is not None:
            target = self.encode_target(goal_images)
            result["goal"] = target
            result["predicted_progress"] = self.progress_head(
                current["16"],
                target["16"],
                predicted_next["16"],
                action["16"],
            )
        return result

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.99) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must lie in [0, 1).")
        for target, online in zip(
            self.target_encoder.parameters(),
            self.online_encoder.parameters(),
            strict=True,
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(
            self.target_encoder.buffers(),
            self.online_encoder.buffers(),
            strict=True,
        ):
            target.copy_(online)
        self.target_encoder.eval()


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def no_op_action_raster(
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    return torch.zeros(batch_size, 2, 64, 64, device=device, dtype=dtype)


def _masked_spatial_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    numerator = (values * weights).flatten(start_dim=1).sum(dim=1)
    denominator = weights.flatten(start_dim=1).sum(dim=1)
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )


def balanced_spatial_smooth_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    action_coverage: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    """Balance action-covered and uncovered positions equally per example."""

    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("Predicted and target features must share a 4D shape.")
    if action_coverage.ndim != 4 or action_coverage.shape[:2] != (
        predicted.shape[0],
        1,
    ):
        raise ValueError("Action coverage must have shape [batch, 1, H, W].")
    if beta <= 0:
        raise ValueError("Smooth L1 beta must be positive.")
    coverage = F.interpolate(
        action_coverage,
        size=predicted.shape[-2:],
        mode="area",
    )[:, 0]
    per_position = F.smooth_l1_loss(
        predicted,
        target,
        reduction="none",
        beta=beta,
    ).mean(dim=1)
    inside = _masked_spatial_mean(per_position, coverage > 0)
    outside = _masked_spatial_mean(per_position, coverage <= 0)
    return (0.5 * (inside + outside)).mean()


def joint_prediction_loss(
    predicted_next: Mapping[str, torch.Tensor],
    target_next: Mapping[str, torch.Tensor],
    action_rasters: torch.Tensor,
) -> dict[str, torch.Tensor]:
    coverage = action_rasters[:, 0:1]
    loss_32 = balanced_spatial_smooth_l1(
        predicted_next["32"],
        target_next["32"],
        coverage,
    )
    loss_16 = balanced_spatial_smooth_l1(
        predicted_next["16"],
        target_next["16"],
        coverage,
    )
    return {
        "joint_prediction_32": loss_32,
        "joint_prediction_16": loss_16,
        "joint_prediction": 0.5 * loss_32 + 0.5 * loss_16,
    }


def representation_regularization(
    features: Mapping[str, torch.Tensor],
    *,
    variance_floor: float = 1.0,
) -> dict[str, torch.Tensor]:
    """VICReg-style variance and covariance penalties over both scales."""

    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive.")
    variance_losses: list[torch.Tensor] = []
    covariance_losses: list[torch.Tensor] = []
    for scale in ("32", "16"):
        value = features[scale].permute(0, 2, 3, 1).reshape(-1, features[scale].shape[1])
        if value.shape[0] < 2:
            raise ValueError("Representation regularization needs two positions.")
        std = torch.sqrt(value.var(dim=0, unbiased=False) + 1e-4)
        variance_losses.append(F.relu(variance_floor - std).mean())
        centered = value - value.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (value.shape[0] - 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        covariance_losses.append(off_diagonal.square().sum() / value.shape[1])
    return {
        "variance": torch.stack(variance_losses).mean(),
        "covariance": torch.stack(covariance_losses).mean(),
    }


def no_op_consistency_loss(
    residuals: Mapping[str, torch.Tensor],
    no_op_examples: torch.Tensor,
) -> torch.Tensor:
    if no_op_examples.ndim != 1 or no_op_examples.shape[0] != residuals["32"].shape[0]:
        raise ValueError("no_op_examples must identify the residual batch.")
    if no_op_examples.dtype != torch.bool:
        no_op_examples = no_op_examples.bool()
    if not bool(no_op_examples.any()):
        return sum(value.sum() * 0.0 for value in residuals.values())
    losses = [
        F.smooth_l1_loss(value[no_op_examples], torch.zeros_like(value[no_op_examples]))
        for value in residuals.values()
    ]
    return torch.stack(losses).mean()


def progress_regression_loss(
    predicted_standardized_progress: torch.Tensor,
    exact_progress: torch.Tensor,
    *,
    training_mean: float,
    training_std: float,
    beta: float = 1.0,
) -> torch.Tensor:
    if predicted_standardized_progress.shape != exact_progress.shape:
        raise ValueError("Predicted and exact progress shapes must match.")
    if training_std <= 0 or beta <= 0:
        raise ValueError("Progress standard deviation and beta must be positive.")
    standardized = (exact_progress - training_mean) / training_std
    return F.smooth_l1_loss(
        predicted_standardized_progress,
        standardized,
        beta=beta,
    )


def candidate_ranking_loss(
    predicted_standardized_progress: torch.Tensor,
    exact_progress: torch.Tensor,
    *,
    temperature: float = 0.1,
    tie_tolerance: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if predicted_standardized_progress.shape != exact_progress.shape:
        raise ValueError("Predicted and exact candidate progress shapes must match.")
    if predicted_standardized_progress.ndim != 2 or predicted_standardized_progress.shape[1] < 2:
        raise ValueError("Candidate progress must have shape [batch, candidates].")
    if temperature <= 0 or tie_tolerance < 0:
        raise ValueError("Invalid ranking temperature or tie tolerance.")
    maximum = exact_progress.max(dim=1, keepdim=True).values
    tied = (maximum - exact_progress) <= tie_tolerance
    targets = tied.to(torch.int64).argmax(dim=1)
    loss = F.cross_entropy(
        predicted_standardized_progress / temperature,
        targets,
    )
    return loss, targets


def phase_b_objective(
    *,
    variant: str,
    online_features: Mapping[str, torch.Tensor],
    predicted_next: Mapping[str, torch.Tensor],
    target_next: Mapping[str, torch.Tensor],
    residuals: Mapping[str, torch.Tensor],
    action_rasters: torch.Tensor,
    no_op_examples: torch.Tensor,
    predicted_progress: torch.Tensor | None = None,
    exact_progress: torch.Tensor | None = None,
    progress_training_mean: float = 0.0,
    progress_training_std: float = 1.0,
) -> dict[str, torch.Tensor]:
    if variant not in {"joint_prediction_only", "joint_prediction_progress"}:
        raise ValueError("Unexpected Phase B0 objective variant.")
    losses = joint_prediction_loss(predicted_next, target_next, action_rasters)
    regularization = representation_regularization(online_features)
    no_op = no_op_consistency_loss(residuals, no_op_examples)
    total = (
        losses["joint_prediction"]
        + 0.10 * regularization["variance"]
        + 0.01 * regularization["covariance"]
        + 0.25 * no_op
    )
    result = {
        **losses,
        "variance": regularization["variance"],
        "covariance": regularization["covariance"],
        "no_op_consistency": no_op,
    }
    if variant == "joint_prediction_progress":
        if predicted_progress is None or exact_progress is None:
            raise ValueError("Planner-aligned objective requires progress tensors.")
        progress = progress_regression_loss(
            predicted_progress,
            exact_progress,
            training_mean=progress_training_mean,
            training_std=progress_training_std,
        )
        ranking, _ = candidate_ranking_loss(
            predicted_progress,
            exact_progress,
        )
        total = total + progress + 0.30 * ranking
        result["progress_regression"] = progress
        result["candidate_ranking"] = ranking
    result["total"] = total
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Phase B0 objective is non-finite.")
    return result


def validate_phase_b_runner_request(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validation-only boundary without model or data side effects."""

    validate_phase_b_config(config)
    if config["development"]["authorized"] is not False:
        raise PermissionError("Phase B0 development is unexpectedly authorized.")
    if config["formal_reserved"]["authorized"] is not False:
        raise PermissionError("Formal Phase B0 is unexpectedly authorized.")
    return {
        "status": "phase_b0_validation_scaffold_valid_unauthorized",
        "config_status": config["status"],
        "development_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "historical_checkpoints_loaded": False,
        "renderer_transitions_generated": False,
        "targets_generated": False,
        "state_banks_generated": False,
        "candidate_sets_generated": False,
        "output_directories_created": False,
        "models_trained_on_renderer_data": False,
    }


def run_phase_b_validation(
    path: str | Path = DEFAULT_PHASE_B_CONFIG,
) -> dict[str, Any]:
    """Run deterministic dummy-only architecture and gradient checks."""

    config = load_phase_b_config(path)
    result = validate_phase_b_runner_request(config)
    torch.manual_seed(int(config["training"]["development_model_seed"]))
    model = MultiScaleActionJointEmbeddingModel().cpu().train()
    count = trainable_parameter_count(model)
    if count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"Phase B0 parameter count changed: {count} != "
            f"{EXPECTED_TRAINABLE_PARAMETERS}."
        )
    if count > int(config["model"]["maximum_trainable_parameters"]):
        raise RuntimeError("Phase B0 trainable parameter cap exceeded.")

    generator = torch.Generator().manual_seed(20260823)
    batch = 4
    current = torch.rand(batch, 1, 64, 64, generator=generator)
    actual_next = torch.rand(batch, 1, 64, 64, generator=generator)
    goal = torch.rand(batch, 1, 64, 64, generator=generator)
    actions = torch.rand(batch, 2, 64, 64, generator=generator)
    actions[0].zero_()
    outputs = model(current, actions, goal)
    target_next = model.encode_target(actual_next)
    predicted_progress = outputs["predicted_progress"].reshape(1, batch)
    exact_progress = torch.tensor([[0.0, 0.02, -0.01, 0.01]])
    losses = phase_b_objective(
        variant="joint_prediction_progress",
        online_features=outputs["current"],
        predicted_next=outputs["predicted_next"],
        target_next=target_next,
        residuals=outputs["residual"],
        action_rasters=actions,
        no_op_examples=torch.tensor([True, False, False, False]),
        predicted_progress=predicted_progress,
        exact_progress=exact_progress,
    )
    model.zero_grad(set_to_none=True)
    losses["total"].backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients_present = all(parameter.grad is not None for parameter in trainable)
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in trainable
    )
    if not gradients_present or not gradients_finite:
        raise RuntimeError("Phase B0 dummy gradient validation failed.")
    target_frozen = not any(
        parameter.requires_grad for parameter in model.target_encoder.parameters()
    ) and all(parameter.grad is None for parameter in model.target_encoder.parameters())
    if not target_frozen:
        raise RuntimeError("Phase B0 target encoder is not frozen.")

    online_parameter = next(model.online_encoder.parameters())
    target_parameter = next(model.target_encoder.parameters())
    target_before = target_parameter.detach().clone()
    with torch.no_grad():
        online_parameter.add_(0.01)
        expected_target = 0.99 * target_before + 0.01 * online_parameter.detach()
    model.update_target_encoder(momentum=0.99)
    ema_error = float((target_parameter - expected_target).abs().max().item())
    if ema_error != 0.0:
        raise RuntimeError("Phase B0 target-encoder EMA update is incorrect.")

    scalar_losses = {
        name: float(value.detach().item())
        for name, value in losses.items()
        if value.ndim == 0
    }
    result.update(
        {
            "status": "phase_b0_architecture_and_objectives_valid_unauthorized",
            "trainable_parameter_count": count,
            "parameter_cap": int(config["model"]["maximum_trainable_parameters"]),
            "online_shapes": {
                scale: list(value.shape) for scale, value in outputs["current"].items()
            },
            "action_shapes": {
                scale: list(value.shape) for scale, value in outputs["action"].items()
            },
            "residual_shapes": {
                scale: list(value.shape) for scale, value in outputs["residual"].items()
            },
            "progress_shape": list(outputs["predicted_progress"].shape),
            "losses": scalar_losses,
            "all_trainable_gradients_present": gradients_present,
            "all_trainable_gradients_finite": gradients_finite,
            "target_encoder_frozen": target_frozen,
            "ema_maximum_error": ema_error,
            "dummy_tensors_only": True,
            "implementation_manifest_required_before_development": True,
        }
    )
    return result
