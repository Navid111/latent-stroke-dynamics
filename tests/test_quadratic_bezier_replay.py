from __future__ import annotations

import numpy as np
import pytest

from latent_stroke_dynamics.quadratic_bezier_extension import QuadraticBezierStroke
from latent_stroke_dynamics.quadratic_bezier_replay import (
    replay_extension_strokes_high_resolution,
    scale_extension_stroke,
)
from latent_stroke_dynamics.rgb_coarse_to_fine import RGBStroke


def test_scale_extension_stroke_preserves_normalized_geometry() -> None:
    curve = QuadraticBezierStroke(0.1, 0.8, 0.5, 0.1, 0.9, 0.8, 2, (4, 5, 6))
    scaled = scale_extension_stroke(curve, 4.0)
    assert isinstance(scaled, QuadraticBezierStroke)
    assert scaled.width == 8
    assert scaled.x0 == curve.x0
    assert scaled.cx == curve.cx
    assert scaled.x1 == curve.x1
    assert scaled.color == curve.color


def test_high_resolution_replay_is_deterministic_for_mixed_test_sequence() -> None:
    strokes = (
        RGBStroke(0.1, 0.1, 0.9, 0.9, 2, (10, 20, 30)),
        QuadraticBezierStroke(0.1, 0.9, 0.5, 0.1, 0.9, 0.9, 2, (180, 80, 45)),
    )
    kwargs = {
        "planning_size": 32,
        "output_size": 128,
        "supersample": 2,
        "best_step": 1,
        "capture_steps": {1, 2},
        "gif_stride": 1,
    }
    first = replay_extension_strokes_high_resolution(strokes, **kwargs)
    second = replay_extension_strokes_high_resolution(strokes, **kwargs)
    assert first[0].size == (128, 128)
    assert first[1].size == (128, 128)
    assert len(first[2]) == 3
    assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
    assert np.array_equal(np.asarray(first[1]), np.asarray(second[1]))
    assert not np.array_equal(np.asarray(first[0]), np.asarray(first[1]))


def test_high_resolution_replay_rejects_invalid_best_step() -> None:
    with pytest.raises(ValueError, match="outside"):
        replay_extension_strokes_high_resolution(
            (),
            planning_size=32,
            output_size=128,
            supersample=2,
            best_step=1,
            capture_steps=set(),
            gif_stride=1,
        )
