"""Deterministic high-resolution replay for extension stroke sequences."""

from __future__ import annotations

from typing import Sequence

from PIL import Image

from .quadratic_bezier_extension import (
    QuadraticBezierStroke,
    Stroke,
    render_extension_stroke,
)
from .rgb_coarse_to_fine import RGBStroke, blank_rgb_canvas


def scale_extension_stroke(stroke: Stroke, width_scale: float) -> Stroke:
    """Scale only the pixel-width component of a normalized stroke."""

    if width_scale <= 0.0:
        raise ValueError("width_scale must be positive.")
    scaled_width = max(1, int(round(stroke.width * width_scale)))
    if isinstance(stroke, RGBStroke):
        return RGBStroke(
            stroke.x0,
            stroke.y0,
            stroke.x1,
            stroke.y1,
            scaled_width,
            stroke.color,
        )
    if isinstance(stroke, QuadraticBezierStroke):
        return QuadraticBezierStroke(
            stroke.x0,
            stroke.y0,
            stroke.cx,
            stroke.cy,
            stroke.x1,
            stroke.y1,
            scaled_width,
            stroke.color,
        )
    raise TypeError(f"Unsupported stroke type: {type(stroke).__name__}")


def replay_extension_strokes_high_resolution(
    strokes: Sequence[Stroke],
    *,
    planning_size: int,
    output_size: int,
    supersample: int,
    best_step: int,
    capture_steps: set[int],
    gif_stride: int,
) -> tuple[Image.Image, Image.Image, tuple[Image.Image, ...]]:
    """Replay straight or quadratic strokes at one common output resolution."""

    if planning_size < 8 or output_size < 8:
        raise ValueError("Planning and output sizes must be at least 8.")
    if not 1 <= supersample <= 4:
        raise ValueError("supersample must lie between 1 and 4.")
    if gif_stride < 1:
        raise ValueError("gif_stride must be positive.")
    if not 0 <= best_step <= len(strokes):
        raise ValueError("best_step lies outside the stroke sequence.")
    render_size = output_size * supersample
    if render_size > 4096:
        raise ValueError("Replay render size must not exceed 4096.")

    working = blank_rgb_canvas(render_size)
    width_scale = render_size / planning_size

    def presentation_frame(image: Image.Image) -> Image.Image:
        if render_size == output_size:
            return image.copy()
        return image.resize((output_size, output_size), Image.Resampling.LANCZOS)

    initial = presentation_frame(working)
    frames: list[Image.Image] = [initial]
    best = initial.copy()
    for step, stroke in enumerate(strokes, start=1):
        working = render_extension_stroke(
            working,
            scale_extension_stroke(stroke, width_scale),
        )
        if step == best_step:
            best = presentation_frame(working)
        if step % gif_stride == 0 or step in capture_steps or step == len(strokes):
            frames.append(presentation_frame(working))
    return best, presentation_frame(working), tuple(frames)
