import numpy as np

from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def test_blank_canvas_is_white_and_square() -> None:
    canvas = blank_canvas(size=32)
    values = np.asarray(canvas)
    assert canvas.mode == "L"
    assert values.shape == (32, 32)
    assert np.all(values == 255)


def test_rendering_changes_pixels_without_mutating_input() -> None:
    canvas = blank_canvas(size=32)
    original = np.asarray(canvas).copy()
    stroke = Stroke(0.10, 0.20, 0.90, 0.80, width=3, value=0)

    rendered = render_stroke(canvas, stroke)

    assert np.array_equal(np.asarray(canvas), original)
    assert np.any(np.asarray(rendered) != original)


def test_shifted_stroke_stays_inside_canvas_coordinates() -> None:
    stroke = Stroke(0.90, 0.90, 1.00, 1.00, width=2, value=32)
    shifted = stroke.shifted(dx=0.50, dy=-2.00)
    assert shifted.x0 == 1.0
    assert shifted.x1 == 1.0
    assert shifted.y0 == 0.0
    assert shifted.y1 == 0.0
