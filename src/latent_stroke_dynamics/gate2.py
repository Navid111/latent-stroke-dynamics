"""Deterministic transitions, action encodings, and predictors for Gate 2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import atan2, cos, hypot, sin, sqrt
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .renderer import (
    Stroke,
    blank_canvas,
    random_base_canvas,
    render_stroke,
    sample_stroke,
)


ACTION_DIM = 7
COUNTERFACTUAL_ORDER: tuple[str, ...] = (
    "true",
    "shift_position",
    "change_width",
    "change_intensity",
)
PRIMARY_CROWDING: tuple[int, ...] = (0, 5, 15)
PRIMARY_WIDTHS: tuple[int, ...] = (1, 2, 3, 4)
PRIMARY_VALUES: tuple[int, ...] = (0, 32, 64, 96, 128)


@dataclass(frozen=True)
class TransitionExample:
    """One exact deterministic canvas transition."""

    current: Image.Image
    next_canvas: Image.Image
    stroke: Stroke
    crowding: int
    sample_id: int


def _images_differ(left: Image.Image, right: Image.Image) -> bool:
    return bool(np.any(np.asarray(left) != np.asarray(right)))


def build_transition_split(
    samples: int,
    canvas_size: int,
    crowding_levels: Sequence[int],
    seed: int,
    width_choices: Sequence[int] = PRIMARY_WIDTHS,
    value_choices: Sequence[int] = PRIMARY_VALUES,
    min_length: float = 0.20,
) -> list[TransitionExample]:
    """Generate an independent split of exact one-stroke transitions."""

    if samples < 1:
        raise ValueError("samples must be positive.")
    levels = tuple(sorted(set(int(level) for level in crowding_levels)))
    if not levels or levels[0] < 0:
        raise ValueError("crowding_levels must be non-empty and non-negative.")
    widths = tuple(int(width) for width in width_choices)
    values = tuple(int(value) for value in value_choices)
    if not widths or not values:
        raise ValueError("width_choices and value_choices must be non-empty.")

    rng = np.random.default_rng(seed)
    examples: list[TransitionExample] = []

    for sample_id in range(samples):
        crowding = int(rng.choice(levels))
        current = random_base_canvas(canvas_size, crowding, rng)

        for _ in range(100):
            stroke = sample_stroke(
                rng,
                width_choices=widths,
                value_choices=values,
                min_length=min_length,
            )
            next_canvas = render_stroke(current, stroke)
            if _images_differ(current, next_canvas):
                examples.append(
                    TransitionExample(
                        current=current,
                        next_canvas=next_canvas,
                        stroke=stroke,
                        crowding=crowding,
                        sample_id=sample_id,
                    )
                )
                break
        else:
            raise RuntimeError("Could not sample a stroke that changes the canvas.")

    return examples


def transition_fingerprint(example: TransitionExample) -> str:
    """Return a stable digest used to check split separation."""

    digest = sha256()
    digest.update(np.asarray(example.current, dtype=np.uint8).tobytes())
    digest.update(np.asarray(example.next_canvas, dtype=np.uint8).tobytes())
    digest.update(
        np.asarray(
            [
                example.stroke.x0,
                example.stroke.y0,
                example.stroke.x1,
                example.stroke.y1,
            ],
            dtype=np.float64,
        ).tobytes()
    )
    digest.update(
        np.asarray(
            [example.stroke.width, example.stroke.value, example.crowding],
            dtype=np.int32,
        ).tobytes()
    )
    return digest.hexdigest()


def stroke_action_vector(stroke: Stroke, max_width: int = 5) -> torch.Tensor:
    """Encode an undirected line action as seven normalized values.

    The vector contains midpoint x/y, normalized length, cos(2 theta),
    sin(2 theta), normalized width, and darkness. Doubling the angle makes
    reversing the two endpoints produce the same orientation encoding.
    """

    if max_width < 2:
        raise ValueError("max_width must be at least 2.")

    dx = stroke.x1 - stroke.x0
    dy = stroke.y1 - stroke.y0
    theta = atan2(dy, dx)
    width = (stroke.width - 1) / (max_width - 1)
    values = [
        (stroke.x0 + stroke.x1) / 2.0,
        (stroke.y0 + stroke.y1) / 2.0,
        hypot(dx, dy) / sqrt(2.0),
        cos(2.0 * theta),
        sin(2.0 * theta),
        max(0.0, min(1.0, width)),
        1.0 - stroke.value / 255.0,
    ]
    return torch.tensor(values, dtype=torch.float32)


def stroke_patch_coverage(
    stroke: Stroke,
    canvas_size: int,
    patch_grid: tuple[int, int],
) -> torch.Tensor:
    """Rasterize a proposed action into fractional patch coverage in [0, 1]."""

    rows, columns = patch_grid
    if rows < 1 or columns < 1:
        raise ValueError("patch_grid dimensions must be positive.")

    geometry_stroke = replace(stroke, value=0)
    rendered = render_stroke(blank_canvas(canvas_size), geometry_stroke)
    changed = (np.asarray(rendered) != 255).astype(np.uint8) * 255
    resized = Image.fromarray(changed).resize(
        (columns, rows),
        resample=Image.Resampling.BOX,
    )
    coverage = np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0
    return torch.from_numpy(coverage)


def patch_coordinates(patch_grid: tuple[int, int]) -> torch.Tensor:
    """Return normalized patch-center coordinates in row-major order."""

    rows, columns = patch_grid
    if rows < 1 or columns < 1:
        raise ValueError("patch_grid dimensions must be positive.")
    y = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    x = (torch.arange(columns, dtype=torch.float32) + 0.5) / columns
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)


def build_action_tensors(
    examples: Sequence[TransitionExample],
    canvas_size: int,
    patch_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized action vectors and spatial masks for a split."""

    if not examples:
        raise ValueError("At least one transition is required.")
    actions = torch.stack([stroke_action_vector(item.stroke) for item in examples])
    masks = torch.stack(
        [
            stroke_patch_coverage(item.stroke, canvas_size, patch_grid)
            for item in examples
        ]
    )
    return actions, masks


def _translated_counterfactual(stroke: Stroke) -> Stroke:
    candidates = (
        (0.08, 0.00),
        (-0.08, 0.00),
        (0.00, 0.08),
        (0.00, -0.08),
        (0.06, 0.06),
        (-0.06, -0.06),
    )
    for dx, dy in candidates:
        coordinates = (
            stroke.x0 + dx,
            stroke.y0 + dy,
            stroke.x1 + dx,
            stroke.y1 + dy,
        )
        if all(0.0 <= coordinate <= 1.0 for coordinate in coordinates):
            return replace(
                stroke,
                x0=coordinates[0],
                y0=coordinates[1],
                x1=coordinates[2],
                y1=coordinates[3],
            )

    reflected = replace(
        stroke,
        x0=1.0 - stroke.x0,
        y0=1.0 - stroke.y0,
        x1=1.0 - stroke.x1,
        y1=1.0 - stroke.y1,
    )
    if reflected != stroke:
        return reflected
    return stroke.shifted(dx=0.02, dy=0.0)


def counterfactual_strokes(stroke: Stroke) -> tuple[Stroke, Stroke, Stroke, Stroke]:
    """Return true, shifted, width-changed, and intensity-changed actions."""

    shifted = _translated_counterfactual(stroke)
    changed_width = replace(
        stroke,
        width=stroke.width + 1 if stroke.width < 5 else stroke.width - 1,
    )
    changed_value = replace(
        stroke,
        value=224 if stroke.value <= 96 else 16,
    )
    return stroke, shifted, changed_width, changed_value


def counterfactual_canvases(example: TransitionExample) -> tuple[Image.Image, ...]:
    """Render the four exact outcomes used by the retrieval diagnostic."""

    return tuple(
        render_stroke(example.current, stroke)
        for stroke in counterfactual_strokes(example.stroke)
    )


def counterfactual_union_mask(
    stroke: Stroke,
    canvas_size: int,
    patch_grid: tuple[int, int],
) -> torch.Tensor:
    """Union of all action-covered patches in the retrieval candidate set."""

    masks = torch.stack(
        [
            stroke_patch_coverage(item, canvas_size, patch_grid)
            for item in counterfactual_strokes(stroke)
        ]
    )
    return masks.max(dim=0).values


def make_patch_inputs(
    current_features: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
    patch_grid: tuple[int, int],
    normalize_current: bool = False,
) -> torch.Tensor:
    """Concatenate patch state, action, mask, and patch position."""

    if current_features.ndim != 3:
        raise ValueError("current_features must have shape [batch, patches, features].")
    batch, patches, _ = current_features.shape
    if actions.shape != (batch, ACTION_DIM):
        raise ValueError(
            f"actions must have shape {(batch, ACTION_DIM)}, received {actions.shape}."
        )
    if action_masks.shape != (batch, patches):
        raise ValueError(
            f"action_masks must have shape {(batch, patches)}, "
            f"received {action_masks.shape}."
        )
    expected_patches = patch_grid[0] * patch_grid[1]
    if patches != expected_patches:
        raise ValueError(
            f"patch_grid implies {expected_patches} patches, received {patches}."
        )

    current = F.layer_norm(current_features, current_features.shape[-1:]) if normalize_current else current_features
    repeated_actions = actions[:, None, :].expand(-1, patches, -1)
    coordinates = patch_coordinates(patch_grid).to(
        device=current_features.device,
        dtype=current_features.dtype,
    )
    coordinates = coordinates[None, :, :].expand(batch, -1, -1)
    return torch.cat(
        (
            current,
            repeated_actions.to(current_features.dtype),
            action_masks[:, :, None].to(current_features.dtype),
            coordinates,
        ),
        dim=-1,
    )


class IdentityPatchDeltaPredictor(nn.Module):
    """The no-change baseline."""

    def forward(
        self,
        current_features: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        del actions, action_masks
        return torch.zeros_like(current_features)


class MeanPatchDeltaPredictor(nn.Module):
    """Repeat a training-set mean residual for every example."""

    def __init__(self, mean_delta: torch.Tensor) -> None:
        super().__init__()
        if mean_delta.ndim != 2:
            raise ValueError("mean_delta must have shape [patches, features].")
        self.register_buffer("mean_delta", mean_delta.detach().float())

    def forward(
        self,
        current_features: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        del actions, action_masks
        return self.mean_delta[None, :, :].expand(current_features.shape[0], -1, -1)


class LinearPatchDeltaPredictor(nn.Module):
    """A shared affine action-conditioned mapping applied to every patch."""

    def __init__(
        self,
        feature_dim: int,
        patch_grid: tuple[int, int],
    ) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        input_dim = feature_dim + ACTION_DIM + 1 + 2
        self.projection = nn.Linear(input_dim, feature_dim)

    def forward(
        self,
        current_features: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        inputs = make_patch_inputs(
            current_features,
            actions,
            action_masks,
            self.patch_grid,
            normalize_current=False,
        )
        return self.projection(inputs)


class MLPPatchDeltaPredictor(nn.Module):
    """A small shared nonlinear residual predictor."""

    def __init__(
        self,
        feature_dim: int,
        patch_grid: tuple[int, int],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        self.patch_grid = patch_grid
        input_dim = feature_dim + ACTION_DIM + 1 + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        current_features: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        inputs = make_patch_inputs(
            current_features,
            actions,
            action_masks,
            self.patch_grid,
            normalize_current=True,
        )
        return self.network(inputs)


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _masked_patch_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = (mask > 0).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def balanced_patch_mse(
    predicted_delta: torch.Tensor,
    true_delta: torch.Tensor,
    action_masks: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Balance changed-action and outside regions equally per example."""

    if predicted_delta.shape != true_delta.shape:
        raise ValueError("predicted_delta and true_delta must have matching shapes.")
    if predicted_delta.ndim != 3:
        raise ValueError("delta tensors must have shape [batch, patches, features].")
    if action_masks.shape != predicted_delta.shape[:2]:
        raise ValueError("action_masks must match the batch and patch dimensions.")

    per_patch = (predicted_delta - true_delta).square().mean(dim=-1)
    inside = _masked_patch_mean(per_patch, action_masks)
    outside = _masked_patch_mean(per_patch, action_masks <= 0)
    per_example = 0.5 * (inside + outside)

    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    raise ValueError("reduction must be 'none' or 'mean'.")


def residual_error_metrics(
    current_features: torch.Tensor,
    predicted_delta: torch.Tensor,
    true_delta: torch.Tensor,
    action_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-example spatial prediction errors."""

    if current_features.shape != true_delta.shape:
        raise ValueError("current_features and true_delta must have matching shapes.")
    if predicted_delta.shape != true_delta.shape:
        raise ValueError("predicted_delta and true_delta must have matching shapes.")

    per_patch = (predicted_delta - true_delta).square().mean(dim=-1)
    action_mse = _masked_patch_mean(per_patch, action_masks)
    outside_mse = _masked_patch_mean(per_patch, action_masks <= 0)

    predicted_next = F.normalize(current_features + predicted_delta, dim=-1)
    true_next = F.normalize(current_features + true_delta, dim=-1)
    cosine = (1.0 - (predicted_next * true_next).sum(dim=-1)).clamp(0.0, 2.0)

    return {
        "full_patch_mse": per_patch.mean(dim=1),
        "action_region_mse": action_mse,
        "outside_region_mse": outside_mse,
        "action_region_next_cosine_distance": _masked_patch_mean(
            cosine,
            action_masks,
        ),
    }


def counterfactual_retrieval(
    predicted_next: torch.Tensor,
    candidate_next: torch.Tensor,
    union_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Retrieve the true outcome at candidate index zero.

    ``candidate_next`` has shape [batch, candidates, patches, features].
    Scores are normalized-feature MSE over the fixed union of candidate action
    regions for each example.
    """

    if predicted_next.ndim != 3 or candidate_next.ndim != 4:
        raise ValueError("Unexpected prediction or candidate tensor rank.")
    if candidate_next.shape[0] != predicted_next.shape[0]:
        raise ValueError("Batch dimensions must match.")
    if candidate_next.shape[2:] != predicted_next.shape[1:]:
        raise ValueError("Candidate patch and feature dimensions must match.")
    if union_masks.shape != predicted_next.shape[:2]:
        raise ValueError("union_masks must match batch and patch dimensions.")
    if candidate_next.shape[1] < 2:
        raise ValueError("At least two retrieval candidates are required.")

    predicted = F.normalize(predicted_next, dim=-1)
    candidates = F.normalize(candidate_next, dim=-1)
    per_patch = (predicted[:, None, :, :] - candidates).square().mean(dim=-1)
    weights = (union_masks > 0).to(per_patch.dtype)[:, None, :]
    scores = (per_patch * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    predicted_index = scores.argmin(dim=1)
    true_score = scores[:, 0]
    best_counterfactual_score = scores[:, 1:].min(dim=1).values

    return {
        "scores": scores,
        "predicted_index": predicted_index,
        "top1_correct": predicted_index.eq(0),
        "true_margin": best_counterfactual_score - true_score,
    }
