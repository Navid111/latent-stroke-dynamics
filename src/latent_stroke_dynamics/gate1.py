"""Controlled pair generation and spatial diagnostics for Gate 1."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, hypot
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .renderer import Stroke, blank_canvas, render_stroke, sample_stroke


COMPARISON_ORDER: tuple[str, ...] = (
    "no_change",
    "tiny_pixel_noise",
    "pixel_matched_noise",
    "add_stroke",
    "shift_position",
    "change_width",
    "change_intensity",
)

STRUCTURAL_COMPARISONS: tuple[str, ...] = (
    "add_stroke",
    "shift_position",
    "change_width",
    "change_intensity",
)


@dataclass(frozen=True)
class ComparisonPair:
    """Two canvases differing by one controlled intervention."""

    before: Image.Image
    after: Image.Image
    comparison: str
    crowding: int
    sample_id: int
    stroke: Stroke


def pixel_distance(before: Image.Image, after: Image.Image) -> float:
    """Mean absolute pixel difference normalized to [0, 1]."""

    left = np.asarray(before, dtype=np.float32)
    right = np.asarray(after, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError("Images must have the same shape.")
    return float(np.mean(np.abs(left - right)) / 255.0)


def changed_pixel_count(before: Image.Image, after: Image.Image) -> int:
    """Number of pixels whose integer grayscale value changed."""

    left = np.asarray(before)
    right = np.asarray(after)
    if left.shape != right.shape:
        raise ValueError("Images must have the same shape.")
    return int(np.count_nonzero(left != right))


def stroke_length_normalized(stroke: Stroke) -> float:
    return float(hypot(stroke.x1 - stroke.x0, stroke.y1 - stroke.y0))


def tiny_pixel_noise(
    image: Image.Image,
    rng: np.random.Generator,
    sigma: float = 1.25,
) -> Image.Image:
    """Add low-amplitude Gaussian noise across the whole canvas."""

    values = np.asarray(image, dtype=np.float32)
    noisy = np.clip(values + rng.normal(0.0, sigma, size=values.shape), 0, 255)
    return Image.fromarray(np.rint(noisy).astype(np.uint8))


def pixel_matched_noise(
    image: Image.Image,
    target_mae: float,
    rng: np.random.Generator,
    max_iterations: int = 8,
) -> Image.Image:
    """Create distributed noise with approximately the requested normalized MAE.

    This control matches the amount of pixel change caused by the added stroke
    while destroying its coherent line structure.
    """

    if target_mae < 0:
        raise ValueError("target_mae cannot be negative.")
    if target_mae == 0:
        return image.copy()

    base = np.asarray(image, dtype=np.float32)
    raw = rng.normal(0.0, 1.0, size=base.shape).astype(np.float32)
    raw -= float(raw.mean())
    raw_mae = float(np.mean(np.abs(raw)))
    if raw_mae == 0:
        raise RuntimeError("Could not generate a non-zero noise pattern.")

    scale = target_mae * 255.0 / raw_mae
    candidate = base.copy()
    for _ in range(max_iterations):
        candidate = np.clip(base + raw * scale, 0, 255)
        observed = float(np.mean(np.abs(candidate - base)) / 255.0)
        if observed == 0:
            scale *= 2.0
            continue
        relative_error = abs(observed - target_mae) / target_mae
        if relative_error <= 0.01:
            break
        scale *= target_mae / observed

    return Image.fromarray(np.rint(candidate).astype(np.uint8))


def _shift_without_clipping(stroke: Stroke) -> Stroke:
    """Translate a stroke while preserving its shape inside the canvas."""

    if max(stroke.x0, stroke.x1) <= 0.92:
        dx = 0.08
    elif min(stroke.x0, stroke.x1) >= 0.08:
        dx = -0.08
    else:
        dx = 0.0

    if min(stroke.y0, stroke.y1) >= 0.06:
        dy = -0.06
    elif max(stroke.y0, stroke.y1) <= 0.94:
        dy = 0.06
    else:
        dy = 0.0

    return stroke.shifted(dx=dx, dy=dy)


def _nested_base_canvases(
    size: int,
    crowding_levels: Sequence[int],
    rng: np.random.Generator,
) -> dict[int, Image.Image]:
    """Build nested canvases so crowding is the only changed context variable."""

    levels = sorted(set(int(level) for level in crowding_levels))
    if not levels:
        raise ValueError("At least one crowding level is required.")
    if levels[0] < 0:
        raise ValueError("Crowding levels cannot be negative.")

    maximum = levels[-1]
    prior_strokes = [sample_stroke(rng) for _ in range(maximum)]
    requested = set(levels)
    canvases: dict[int, Image.Image] = {}
    canvas = blank_canvas(size=size)

    for count in range(maximum + 1):
        if count in requested:
            canvases[count] = canvas.copy()
        if count < maximum:
            canvas = render_stroke(canvas, prior_strokes[count])

    return canvases


def build_pairs(
    samples: int,
    canvas_size: int,
    crowding_levels: Sequence[int],
    seed: int,
) -> list[ComparisonPair]:
    """Build paired interventions using one fixed test stroke per sample.

    For a given ``sample_id``, the same proposed stroke is evaluated on nested
    canvases at every crowding level. The default added stroke is black and two
    pixels wide so crowding is not confounded with random action intensity or
    width.
    """

    if samples < 1:
        raise ValueError("samples must be positive.")
    levels = sorted(set(int(level) for level in crowding_levels))
    if not levels or levels[0] < 0:
        raise ValueError("Crowding levels must be non-empty and non-negative.")

    rng = np.random.default_rng(seed)
    pairs: list[ComparisonPair] = []

    for sample_id in range(samples):
        stroke = sample_stroke(
            rng,
            width_choices=(2,),
            value_choices=(0,),
            min_length=0.35,
        )
        shifted_stroke = _shift_without_clipping(stroke)
        bases = _nested_base_canvases(canvas_size, levels, rng)

        for crowding in levels:
            base = bases[crowding]
            added = render_stroke(base, stroke)
            shifted = render_stroke(base, shifted_stroke)
            thin = render_stroke(base, replace(stroke, width=1))
            thick = render_stroke(base, replace(stroke, width=5))
            dark = render_stroke(base, replace(stroke, value=16))
            light = render_stroke(base, replace(stroke, value=176))
            target_mae = pixel_distance(base, added)

            pairs.extend(
                [
                    ComparisonPair(
                        base,
                        base.copy(),
                        "no_change",
                        crowding,
                        sample_id,
                        stroke,
                    ),
                    ComparisonPair(
                        base,
                        tiny_pixel_noise(base, rng),
                        "tiny_pixel_noise",
                        crowding,
                        sample_id,
                        stroke,
                    ),
                    ComparisonPair(
                        base,
                        pixel_matched_noise(base, target_mae, rng),
                        "pixel_matched_noise",
                        crowding,
                        sample_id,
                        stroke,
                    ),
                    ComparisonPair(
                        base,
                        added,
                        "add_stroke",
                        crowding,
                        sample_id,
                        stroke,
                    ),
                    ComparisonPair(
                        added,
                        shifted,
                        "shift_position",
                        crowding,
                        sample_id,
                        stroke,
                    ),
                    ComparisonPair(
                        thin,
                        thick,
                        "change_width",
                        crowding,
                        sample_id,
                        stroke,
                    ),
                    ComparisonPair(
                        dark,
                        light,
                        "change_intensity",
                        crowding,
                        sample_id,
                        stroke,
                    ),
                ]
            )

    return pairs


def patch_change_mask(
    before: Image.Image,
    after: Image.Image,
    patch_grid: tuple[int, int],
) -> torch.Tensor:
    """Downsample the exact pixel-change mask to the encoder patch grid."""

    left = np.asarray(before)
    right = np.asarray(after)
    if left.shape != right.shape:
        raise ValueError("Images must have the same shape.")

    changed = (left != right).astype(np.uint8) * 255
    rows, columns = patch_grid
    resized = Image.fromarray(changed).resize(
        (columns, rows),
        resample=Image.Resampling.BOX,
    )
    coverage = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(coverage.reshape(-1) > 0.0)


def patch_summary_metrics(
    patch_distances: torch.Tensor,
    before: Image.Image,
    after: Image.Image,
    patch_grid: tuple[int, int],
) -> dict[str, float]:
    """Summarize average, concentrated, and spatially localized patch change."""

    values = patch_distances.detach().float().cpu().reshape(-1)
    expected = patch_grid[0] * patch_grid[1]
    if values.numel() != expected:
        raise ValueError(
            f"Expected {expected} patch distances for grid {patch_grid}, "
            f"received {values.numel()}."
        )

    top_count = max(1, ceil(values.numel() * 0.10))
    top_values = torch.topk(values, k=top_count).values
    mask = patch_change_mask(before, after, patch_grid)

    metrics: dict[str, float] = {
        "patch_mean_cosine_distance": float(values.mean()),
        "patch_max_cosine_distance": float(values.max()),
        "patch_top10pct_mean_cosine_distance": float(top_values.mean()),
        "changed_patch_fraction": float(mask.float().mean()),
    }

    if not bool(mask.any()):
        metrics.update(
            {
                "patch_changed_region_mean_cosine_distance": float("nan"),
                "patch_unchanged_region_mean_cosine_distance": float(values.mean()),
                "localization_enrichment": float("nan"),
                "localization_topk_recall": float("nan"),
                "localization_topk_lift": float("nan"),
            }
        )
        return metrics

    changed_values = values[mask]
    unchanged_values = values[~mask]
    changed_mean = float(changed_values.mean())
    unchanged_mean = (
        float(unchanged_values.mean()) if unchanged_values.numel() else float("nan")
    )

    changed_count = int(mask.sum())
    top_indices = torch.topk(values, k=changed_count).indices
    topk_recall = float(mask[top_indices].float().mean())
    random_recall = changed_count / values.numel()

    metrics.update(
        {
            "patch_changed_region_mean_cosine_distance": changed_mean,
            "patch_unchanged_region_mean_cosine_distance": unchanged_mean,
            "localization_enrichment": (
                changed_mean / max(unchanged_mean, 1e-8)
                if np.isfinite(unchanged_mean)
                else float("nan")
            ),
            "localization_topk_recall": topk_recall,
            "localization_topk_lift": (
                topk_recall / random_recall if random_recall > 0 else float("nan")
            ),
        }
    )
    return metrics
