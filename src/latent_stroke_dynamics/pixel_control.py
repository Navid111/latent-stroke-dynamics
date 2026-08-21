"""Minimal action-conditioned pixel-space control for one-stroke dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .gate2 import (
    ACTION_DIM,
    COUNTERFACTUAL_ORDER,
    TransitionExample,
    build_counterfactual_set,
    stroke_action_vector,
)
from .renderer import Stroke, blank_canvas, render_stroke


PIXEL_INPUT_DIM = 1 + ACTION_DIM + 1 + 2


@dataclass(frozen=True)
class PixelTensors:
    """Normalized current/next canvases and action-conditioned inputs."""

    current: torch.Tensor
    next_canvas: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor


@dataclass(frozen=True)
class PixelCounterfactualTensors:
    """Four exact candidate canvases and their union action mask."""

    candidate_next: torch.Tensor
    union_masks: torch.Tensor
    all_candidates_unique: bool


def image_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a square grayscale image to a float tensor in [0, 1]."""

    if image.mode != "L" or image.width != image.height:
        raise ValueError("Expected a square grayscale ('L') image.")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy())


def stroke_pixel_mask(stroke: Stroke, canvas_size: int) -> torch.Tensor:
    """Return the exact binary pixels covered by a proposed stroke."""

    geometry = Stroke(
        stroke.x0,
        stroke.y0,
        stroke.x1,
        stroke.y1,
        width=stroke.width,
        value=0,
    )
    rendered = render_stroke(blank_canvas(canvas_size), geometry)
    return torch.from_numpy((np.asarray(rendered) != 255).astype(np.float32))


def pixel_coordinates(canvas_size: int) -> torch.Tensor:
    """Return normalized pixel-center coordinates with shape [H, W, 2]."""

    if canvas_size < 1:
        raise ValueError("canvas_size must be positive.")
    values = (torch.arange(canvas_size, dtype=torch.float32) + 0.5) / canvas_size
    y, x = torch.meshgrid(values, values, indexing="ij")
    return torch.stack((x, y), dim=-1)


def build_pixel_tensors(
    examples: Sequence[TransitionExample],
    canvas_size: int,
) -> PixelTensors:
    """Build all normalized pixel and action tensors for one split."""

    if not examples:
        raise ValueError("At least one transition is required.")
    for example in examples:
        if example.current.size != (canvas_size, canvas_size):
            raise ValueError("A transition canvas does not match canvas_size.")
        if example.next_canvas.size != (canvas_size, canvas_size):
            raise ValueError("A next canvas does not match canvas_size.")

    return PixelTensors(
        current=torch.stack(
            [image_to_normalized_tensor(example.current) for example in examples]
        ),
        next_canvas=torch.stack(
            [image_to_normalized_tensor(example.next_canvas) for example in examples]
        ),
        actions=torch.stack(
            [stroke_action_vector(example.stroke) for example in examples]
        ),
        action_masks=torch.stack(
            [stroke_pixel_mask(example.stroke, canvas_size) for example in examples]
        ),
    )


def build_pixel_counterfactual_tensors(
    examples: Sequence[TransitionExample],
    canvas_size: int,
) -> PixelCounterfactualTensors:
    """Build exact pixel candidates and union masks for retrieval."""

    if not examples:
        raise ValueError("At least one transition is required.")

    candidate_batches: list[torch.Tensor] = []
    union_masks: list[torch.Tensor] = []
    all_unique = True
    for example in examples:
        candidate_set = build_counterfactual_set(example)
        candidates = torch.stack(
            [image_to_normalized_tensor(canvas) for canvas in candidate_set.canvases]
        )
        if candidates.shape[0] != len(COUNTERFACTUAL_ORDER):
            raise RuntimeError("Unexpected counterfactual candidate count.")
        for left in range(candidates.shape[0]):
            for right in range(left + 1, candidates.shape[0]):
                all_unique = all_unique and not torch.equal(
                    candidates[left], candidates[right]
                )
        candidate_batches.append(candidates)
        masks = torch.stack(
            [stroke_pixel_mask(stroke, canvas_size) for stroke in candidate_set.strokes]
        )
        union_masks.append(masks.max(dim=0).values)

    return PixelCounterfactualTensors(
        candidate_next=torch.stack(candidate_batches),
        union_masks=torch.stack(union_masks),
        all_candidates_unique=bool(all_unique),
    )


def make_pixel_inputs(
    current: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
) -> torch.Tensor:
    """Concatenate current pixel, global action, mask, and pixel coordinates."""

    if current.ndim != 3:
        raise ValueError("current must have shape [batch, height, width].")
    batch, height, width = current.shape
    if height != width:
        raise ValueError("Pixel control expects square canvases.")
    if actions.shape != (batch, ACTION_DIM):
        raise ValueError(
            f"actions must have shape {(batch, ACTION_DIM)}, received {actions.shape}."
        )
    if action_masks.shape != current.shape:
        raise ValueError("action_masks must match current canvas shape.")

    repeated_actions = actions[:, None, None, :].expand(-1, height, width, -1)
    coordinates = pixel_coordinates(height).to(
        device=current.device,
        dtype=current.dtype,
    )
    coordinates = coordinates[None, :, :, :].expand(batch, -1, -1, -1)
    inputs = torch.cat(
        (
            current[:, :, :, None],
            repeated_actions.to(current.dtype),
            action_masks[:, :, :, None].to(current.dtype),
            coordinates,
        ),
        dim=-1,
    )
    if inputs.shape[-1] != PIXEL_INPUT_DIM:
        raise RuntimeError("Unexpected pixel input dimension.")
    return inputs


class IdentityPixelDeltaPredictor(nn.Module):
    """No-change pixel baseline."""

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        del actions, action_masks
        return torch.zeros_like(current)


class MeanPixelDeltaPredictor(nn.Module):
    """Repeat the training-set mean residual image."""

    def __init__(self, mean_delta: torch.Tensor) -> None:
        super().__init__()
        if mean_delta.ndim != 2 or mean_delta.shape[0] != mean_delta.shape[1]:
            raise ValueError("mean_delta must be a square [height, width] tensor.")
        self.register_buffer("mean_delta", mean_delta.detach().float())

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        del actions, action_masks
        if current.shape[1:] != self.mean_delta.shape:
            raise ValueError("Current canvas shape does not match mean_delta.")
        return self.mean_delta[None, :, :].expand(current.shape[0], -1, -1)


class LinearPixelDeltaPredictor(nn.Module):
    """Shared affine residual predictor applied independently to every pixel."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(PIXEL_INPUT_DIM, 1)

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.projection(
            make_pixel_inputs(current, actions, action_masks)
        ).squeeze(-1)


class MLPPixelDeltaPredictor(nn.Module):
    """Tiny shared nonlinear per-pixel residual predictor."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        self.network = nn.Sequential(
            nn.Linear(PIXEL_INPUT_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            make_pixel_inputs(current, actions, action_masks)
        ).squeeze(-1)


def exact_compositor_delta(
    current: torch.Tensor,
    actions: torch.Tensor,
    action_masks: torch.Tensor,
) -> torch.Tensor:
    """Return the renderer-equivalent hard-mask compositing residual."""

    if current.ndim != 3 or action_masks.shape != current.shape:
        raise ValueError("current and action_masks must have shape [batch, H, W].")
    if actions.shape != (current.shape[0], ACTION_DIM):
        raise ValueError("actions have an unexpected shape.")
    stroke_value = (1.0 - actions[:, 6]).to(current.dtype)[:, None, None]
    mask = (action_masks > 0).to(current.dtype)
    exact_next = current * (1.0 - mask) + stroke_value * mask
    return exact_next - current


def _masked_pixel_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = (mask > 0).to(values.dtype)
    flat_values = values.flatten(start_dim=1)
    flat_weights = weights.flatten(start_dim=1)
    return (flat_values * flat_weights).sum(dim=1) / flat_weights.sum(
        dim=1
    ).clamp_min(1.0)


def balanced_pixel_mse(
    predicted_delta: torch.Tensor,
    true_delta: torch.Tensor,
    action_masks: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Balance proposed-action and outside pixels equally per example."""

    if predicted_delta.shape != true_delta.shape:
        raise ValueError("predicted_delta and true_delta must have matching shapes.")
    if predicted_delta.ndim != 3:
        raise ValueError("Pixel deltas must have shape [batch, height, width].")
    if action_masks.shape != predicted_delta.shape:
        raise ValueError("action_masks must match pixel delta shape.")

    squared = (predicted_delta - true_delta).square()
    inside = _masked_pixel_mean(squared, action_masks)
    outside = _masked_pixel_mean(squared, action_masks <= 0)
    per_example = 0.5 * (inside + outside)
    if reduction == "none":
        return per_example
    if reduction == "mean":
        return per_example.mean()
    raise ValueError("reduction must be 'none' or 'mean'.")


def pixel_error_metrics(
    current: torch.Tensor,
    predicted_delta: torch.Tensor,
    true_delta: torch.Tensor,
    action_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-example errors on clamped predicted next canvases."""

    if current.shape != predicted_delta.shape or current.shape != true_delta.shape:
        raise ValueError("current and delta tensors must have matching shapes.")
    if action_masks.shape != current.shape:
        raise ValueError("action_masks must match current canvas shape.")

    predicted_next = (current + predicted_delta).clamp(0.0, 1.0)
    true_next = (current + true_delta).clamp(0.0, 1.0)
    difference = predicted_next - true_next
    squared = difference.square()
    absolute = difference.abs()
    return {
        "full_pixel_mse": squared.flatten(start_dim=1).mean(dim=1),
        "action_region_mse": _masked_pixel_mean(squared, action_masks),
        "outside_region_mse": _masked_pixel_mean(squared, action_masks <= 0),
        "action_region_mae": _masked_pixel_mean(absolute, action_masks),
    }


def pixel_counterfactual_retrieval(
    predicted_next: torch.Tensor,
    candidate_next: torch.Tensor,
    union_masks: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Retrieve the true pixel outcome at candidate index zero."""

    if predicted_next.ndim != 3 or candidate_next.ndim != 4:
        raise ValueError("Unexpected pixel prediction or candidate tensor rank.")
    if candidate_next.shape[0] != predicted_next.shape[0]:
        raise ValueError("Batch dimensions must match.")
    if candidate_next.shape[2:] != predicted_next.shape[1:]:
        raise ValueError("Candidate and prediction canvas dimensions must match.")
    if union_masks.shape != predicted_next.shape:
        raise ValueError("union_masks must match predicted_next shape.")
    if candidate_next.shape[1] < 2:
        raise ValueError("At least two candidates are required.")

    squared = (predicted_next[:, None, :, :] - candidate_next).square()
    weights = (union_masks > 0).to(squared.dtype)[:, None, :, :]
    scores = (squared * weights).flatten(start_dim=2).sum(dim=2) / weights.flatten(
        start_dim=2
    ).sum(dim=2).clamp_min(1.0)
    predicted_index = scores.argmin(dim=1)
    true_score = scores[:, 0]
    best_counterfactual_score = scores[:, 1:].min(dim=1).values
    return {
        "scores": scores,
        "predicted_index": predicted_index,
        "top1_correct": predicted_index.eq(0),
        "true_margin": best_counterfactual_score - true_score,
    }
