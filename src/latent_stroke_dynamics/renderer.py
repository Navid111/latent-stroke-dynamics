"""A deliberately small deterministic stroke renderer for the first thesis gate."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Stroke:
    """A grayscale straight-line stroke with normalized endpoint coordinates."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: int = 2
    value: int = 0

    def __post_init__(self) -> None:
        coordinates = (self.x0, self.y0, self.x1, self.y1)
        if any(not 0.0 <= coordinate <= 1.0 for coordinate in coordinates):
            raise ValueError("Stroke coordinates must lie in [0, 1].")
        if self.width < 1:
            raise ValueError("Stroke width must be at least 1 pixel.")
        if not 0 <= self.value <= 255:
            raise ValueError("Grayscale value must lie in [0, 255].")

    def shifted(self, dx: float, dy: float) -> "Stroke":
        """Return a translated stroke, clamped to the normalized canvas."""

        return replace(
            self,
            x0=_clamp01(self.x0 + dx),
            y0=_clamp01(self.y0 + dy),
            x1=_clamp01(self.x1 + dx),
            y1=_clamp01(self.y1 + dy),
        )


def blank_canvas(size: int = 64, background: int = 255) -> Image.Image:
    """Create a square grayscale canvas."""

    if size < 8:
        raise ValueError("Canvas size must be at least 8 pixels.")
    if not 0 <= background <= 255:
        raise ValueError("Background must lie in [0, 255].")
    return Image.new("L", (size, size), color=int(background))


def _to_pixel(value: float, size: int) -> int:
    return int(round(_clamp01(value) * (size - 1)))


def render_stroke(canvas: Image.Image, stroke: Stroke) -> Image.Image:
    """Render one stroke onto a copy of ``canvas`` without mutating the input."""

    if canvas.mode != "L":
        raise ValueError("Gate 1 expects a grayscale ('L') canvas.")
    if canvas.width != canvas.height:
        raise ValueError("Gate 1 expects a square canvas.")

    result = canvas.copy()
    size = result.width
    draw = ImageDraw.Draw(result)
    draw.line(
        (
            _to_pixel(stroke.x0, size),
            _to_pixel(stroke.y0, size),
            _to_pixel(stroke.x1, size),
            _to_pixel(stroke.y1, size),
        ),
        fill=int(stroke.value),
        width=int(stroke.width),
    )
    return result


def render_strokes(canvas: Image.Image, strokes: Iterable[Stroke]) -> Image.Image:
    """Render a sequence of strokes in order."""

    result = canvas.copy()
    for stroke in strokes:
        result = render_stroke(result, stroke)
    return result


def sample_stroke(
    rng: np.random.Generator,
    width_choices: Sequence[int] = (1, 2, 3, 4),
    value_choices: Sequence[int] = (0, 32, 64, 96, 128),
    min_length: float = 0.20,
) -> Stroke:
    """Sample a non-trivial random stroke away from the extreme canvas border."""

    for _ in range(100):
        x0, y0, x1, y1 = rng.uniform(0.05, 0.95, size=4)
        if hypot(x1 - x0, y1 - y0) >= min_length:
            return Stroke(
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                width=int(rng.choice(width_choices)),
                value=int(rng.choice(value_choices)),
            )
    raise RuntimeError("Could not sample a sufficiently long stroke.")


def random_base_canvas(
    size: int,
    prior_strokes: int,
    rng: np.random.Generator,
) -> Image.Image:
    """Create a canvas containing ``prior_strokes`` random strokes."""

    if prior_strokes < 0:
        raise ValueError("prior_strokes cannot be negative.")
    canvas = blank_canvas(size=size)
    for _ in range(prior_strokes):
        canvas = render_stroke(canvas, sample_stroke(rng))
    return canvas
