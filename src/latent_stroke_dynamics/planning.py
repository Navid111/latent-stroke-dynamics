"""Deterministic target preprocessing and pixel-space stroke planning."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image, ImageOps

from .renderer import Stroke, blank_canvas, render_stroke


PlannerMethod = Literal["random", "exact"]
DEFAULT_WIDTH_CHOICES = (1, 2, 3, 4)
DEFAULT_VALUE_CHOICES = (0, 32, 64, 96, 128)


@dataclass(frozen=True)
class ProposalConfig:
    """Frozen candidate-proposal settings for one planning run."""

    count: int = 128
    error_guided_fraction: float = 0.80
    min_length: float = 0.10
    max_length: float = 0.60
    width_choices: tuple[int, ...] = DEFAULT_WIDTH_CHOICES
    value_choices: tuple[int, ...] = DEFAULT_VALUE_CHOICES
    max_attempts_per_candidate: int = 100

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be positive.")
        if not 0.0 <= self.error_guided_fraction <= 1.0:
            raise ValueError("error_guided_fraction must lie in [0, 1].")
        if not 0.0 < self.min_length <= self.max_length:
            raise ValueError("Require 0 < min_length <= max_length.")
        if self.max_length > 1.5:
            raise ValueError("max_length is unexpectedly large for normalized coordinates.")
        if not self.width_choices or any(width < 1 for width in self.width_choices):
            raise ValueError("width_choices must contain positive integers.")
        if not self.value_choices or any(
            value < 0 or value > 255 for value in self.value_choices
        ):
            raise ValueError("value_choices must lie in [0, 255].")
        if self.max_attempts_per_candidate < 1:
            raise ValueError("max_attempts_per_candidate must be positive.")


@dataclass(frozen=True)
class PlanningStep:
    """One exactly executed decision in a sequential planning run."""

    step: int
    selected_index: int
    stroke: Stroke
    candidate_count: int
    mse_before: float
    mse_after: float
    mae_after: float
    best_candidate_mse: float | None
    improved: bool


@dataclass(frozen=True)
class PlanningRun:
    """A complete deterministic random or exact-greedy planning trajectory."""

    method: PlannerMethod
    seed: int
    target: Image.Image
    initial_canvas: Image.Image
    final_canvas: Image.Image
    steps: tuple[PlanningStep, ...]
    frames: tuple[Image.Image, ...]


def _white_composite(image: Image.Image) -> Image.Image:
    """Composite transparent inputs over white before grayscale conversion."""

    if "A" not in image.getbands():
        return image
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background


def preprocess_target(image: Image.Image, size: int = 64) -> Image.Image:
    """EXIF-correct, center-crop, resize, and convert an image to grayscale."""

    if size < 8:
        raise ValueError("Target size must be at least 8 pixels.")
    working = ImageOps.exif_transpose(image)
    working = _white_composite(working).convert("L")
    side = min(working.width, working.height)
    if side < 1:
        raise ValueError("Input image must have positive dimensions.")
    left = (working.width - side) // 2
    top = (working.height - side) // 2
    square = working.crop((left, top, left + side, top + side))
    return square.resize((size, size), resample=Image.Resampling.LANCZOS)


def load_target(path: str | Path, size: int = 64) -> Image.Image:
    """Load and preprocess one target image without retaining an open file handle."""

    with Image.open(Path(path)) as image:
        return preprocess_target(image, size=size)


def _normalized_array(image: Image.Image) -> np.ndarray:
    if image.mode != "L" or image.width != image.height:
        raise ValueError("Expected a square grayscale ('L') image.")
    return np.asarray(image, dtype=np.float32) / 255.0


def _validate_image_pair(left: Image.Image, right: Image.Image) -> None:
    if left.mode != "L" or right.mode != "L":
        raise ValueError("Pixel metrics require grayscale ('L') images.")
    if left.size != right.size or left.width != left.height:
        raise ValueError("Pixel metrics require same-sized square images.")


def pixel_mse(left: Image.Image, right: Image.Image) -> float:
    """Mean squared error in normalized [0, 1] pixel units."""

    _validate_image_pair(left, right)
    difference = _normalized_array(left) - _normalized_array(right)
    return float(np.mean(np.square(difference), dtype=np.float64))


def pixel_mae(left: Image.Image, right: Image.Image) -> float:
    """Mean absolute error in normalized [0, 1] pixel units."""

    _validate_image_pair(left, right)
    difference = np.abs(_normalized_array(left) - _normalized_array(right))
    return float(np.mean(difference, dtype=np.float64))


def _sample_midpoint(
    error: np.ndarray,
    guided: bool,
    rng: np.random.Generator,
) -> tuple[float, float]:
    height, width = error.shape
    if guided:
        weights = error.astype(np.float64, copy=False).ravel()
        total = float(weights.sum())
        if total > 0.0:
            index = int(rng.choice(weights.size, p=weights / total))
            y, x = divmod(index, width)
            return (x + 0.5) / width, (y + 0.5) / height
    x, y = rng.uniform(0.0, 1.0, size=2)
    return float(x), float(y)


def _sample_geometry(
    midpoint: tuple[float, float],
    config: ProposalConfig,
    rng: np.random.Generator,
) -> Stroke:
    center_x, center_y = midpoint
    angle = float(rng.uniform(0.0, pi))
    length = float(rng.uniform(config.min_length, config.max_length))
    half_dx = 0.5 * length * cos(angle)
    half_dy = 0.5 * length * sin(angle)
    return Stroke(
        x0=float(np.clip(center_x - half_dx, 0.0, 1.0)),
        y0=float(np.clip(center_y - half_dy, 0.0, 1.0)),
        x1=float(np.clip(center_x + half_dx, 0.0, 1.0)),
        y1=float(np.clip(center_y + half_dy, 0.0, 1.0)),
        width=int(rng.choice(config.width_choices)),
        value=0,
    )


def _target_matched_value(
    target_values: np.ndarray,
    geometry: Stroke,
    value_choices: Sequence[int],
) -> int:
    size = target_values.shape[0]
    mask_image = render_stroke(blank_canvas(size), geometry)
    mask = np.asarray(mask_image) != 255
    if not bool(mask.any()):
        raise RuntimeError("A proposed stroke covered no pixels.")
    mean_value = float(target_values[mask].mean())
    return int(
        min(
            value_choices,
            key=lambda value: (abs(float(value) - mean_value), int(value)),
        )
    )


def propose_strokes(
    current: Image.Image,
    target: Image.Image,
    rng: np.random.Generator,
    config: ProposalConfig | None = None,
) -> tuple[Stroke, ...]:
    """Generate a deterministic mixture of error-guided and uniform strokes."""

    _validate_image_pair(current, target)
    config = config or ProposalConfig()
    current_values = np.asarray(current)
    target_values = np.asarray(target)
    error = np.abs(
        current_values.astype(np.float32) - target_values.astype(np.float32)
    )
    guided_count = int(round(config.count * config.error_guided_fraction))

    accepted: list[Stroke] = []
    seen_outcomes: set[bytes] = set()
    max_attempts = config.count * config.max_attempts_per_candidate
    attempts = 0
    while len(accepted) < config.count and attempts < max_attempts:
        attempts += 1
        guided = len(accepted) < guided_count
        midpoint = _sample_midpoint(error, guided=guided, rng=rng)
        geometry = _sample_geometry(midpoint, config=config, rng=rng)
        value = _target_matched_value(
            target_values,
            geometry,
            config.value_choices,
        )
        stroke = Stroke(
            geometry.x0,
            geometry.y0,
            geometry.x1,
            geometry.y1,
            width=geometry.width,
            value=value,
        )
        rendered = render_stroke(current, stroke)
        rendered_values = np.asarray(rendered)
        if np.array_equal(rendered_values, current_values):
            continue
        outcome_signature = rendered_values.tobytes()
        if outcome_signature in seen_outcomes:
            continue
        seen_outcomes.add(outcome_signature)
        accepted.append(stroke)

    if len(accepted) != config.count:
        raise RuntimeError(
            f"Generated {len(accepted)} unique changing candidates; "
            f"required {config.count}."
        )
    return tuple(accepted)


def render_candidate_canvases(
    current: Image.Image,
    candidates: Sequence[Stroke],
) -> tuple[Image.Image, ...]:
    """Render all proposed one-stroke outcomes from the same current canvas."""

    if not candidates:
        raise ValueError("At least one candidate is required.")
    return tuple(render_stroke(current, stroke) for stroke in candidates)


def select_exact_greedy(
    current: Image.Image,
    target: Image.Image,
    candidates: Sequence[Stroke],
) -> tuple[int, Image.Image, np.ndarray]:
    """Choose the exactly rendered candidate with minimum target pixel MSE."""

    canvases = render_candidate_canvases(current, candidates)
    scores = np.asarray(
        [pixel_mse(canvas, target) for canvas in canvases],
        dtype=np.float64,
    )
    selected_index = int(np.argmin(scores))
    return selected_index, canvases[selected_index], scores


def _step_rng(seed: int, step: int, stream: int) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return np.random.default_rng(np.random.SeedSequence([seed, step, stream]))


def run_planner(
    target: Image.Image,
    method: PlannerMethod,
    steps: int = 100,
    seed: int = 0,
    proposal_config: ProposalConfig | None = None,
    capture_frames: bool = False,
) -> PlanningRun:
    """Run random or exact-greedy one-step replanning against a fixed target."""

    if target.mode != "L" or target.width != target.height:
        raise ValueError("target must be a square grayscale ('L') image.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    if method not in ("random", "exact"):
        raise ValueError("method must be 'random' or 'exact'.")
    config = proposal_config or ProposalConfig()

    initial_canvas = blank_canvas(target.width)
    current = initial_canvas.copy()
    records: list[PlanningStep] = []
    frames: list[Image.Image] = [current.copy()] if capture_frames else []

    for step in range(1, steps + 1):
        candidate_rng = _step_rng(seed, step, stream=0)
        candidates = propose_strokes(
            current,
            target,
            rng=candidate_rng,
            config=config,
        )
        mse_before = pixel_mse(current, target)
        best_candidate_mse: float | None = None

        if method == "exact":
            selected_index, next_canvas, scores = select_exact_greedy(
                current,
                target,
                candidates,
            )
            best_candidate_mse = float(scores[selected_index])
        else:
            selection_rng = _step_rng(seed, step, stream=1)
            selected_index = int(selection_rng.integers(0, len(candidates)))
            next_canvas = render_stroke(current, candidates[selected_index])

        mse_after = pixel_mse(next_canvas, target)
        records.append(
            PlanningStep(
                step=step,
                selected_index=selected_index,
                stroke=candidates[selected_index],
                candidate_count=len(candidates),
                mse_before=mse_before,
                mse_after=mse_after,
                mae_after=pixel_mae(next_canvas, target),
                best_candidate_mse=best_candidate_mse,
                improved=mse_after < mse_before,
            )
        )
        current = next_canvas
        if capture_frames:
            frames.append(current.copy())

    return PlanningRun(
        method=method,
        seed=seed,
        target=target.copy(),
        initial_canvas=initial_canvas,
        final_canvas=current,
        steps=tuple(records),
        frames=tuple(frames),
    )
