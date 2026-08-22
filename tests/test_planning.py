import numpy as np
import pytest
from PIL import Image

from latent_stroke_dynamics.planning import (
    ProposalConfig,
    pixel_mse,
    preprocess_target,
    propose_strokes,
    run_planner,
    select_exact_greedy,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def test_preprocess_target_center_crops_and_converts_to_grayscale() -> None:
    values = np.zeros((20, 40, 3), dtype=np.uint8)
    values[:, 10:30, :] = 200
    processed = preprocess_target(Image.fromarray(values, mode="RGB"), size=20)
    assert processed.mode == "L"
    assert processed.size == (20, 20)
    assert np.all(np.asarray(processed) == 200)


def test_candidate_proposal_is_deterministic_unique_and_changing() -> None:
    current = blank_canvas(32)
    target = render_stroke(
        current,
        Stroke(0.1, 0.5, 0.9, 0.5, width=3, value=32),
    )
    config = ProposalConfig(count=12, min_length=0.1, max_length=0.3)
    first = propose_strokes(current, target, np.random.default_rng(123), config)
    second = propose_strokes(current, target, np.random.default_rng(123), config)
    assert first == second
    assert len(first) == config.count
    outcomes = [np.asarray(render_stroke(current, stroke)) for stroke in first]
    assert len({outcome.tobytes() for outcome in outcomes}) == config.count
    assert all(not np.array_equal(outcome, np.asarray(current)) for outcome in outcomes)
    assert all(stroke.width in config.width_choices for stroke in first)
    assert all(stroke.value in config.value_choices for stroke in first)


def test_fully_error_guided_candidates_center_on_changed_target_region() -> None:
    current = blank_canvas(32)
    target = render_stroke(
        current,
        Stroke(0.1, 0.5, 0.9, 0.5, width=3, value=0),
    )
    config = ProposalConfig(
        count=8,
        error_guided_fraction=1.0,
        min_length=0.1,
        max_length=0.1,
        width_choices=(1,),
        value_choices=(0,),
    )
    candidates = propose_strokes(
        current,
        target,
        np.random.default_rng(7),
        config,
    )
    midpoint_y = [(stroke.y0 + stroke.y1) / 2.0 for stroke in candidates]
    assert all(abs(value - 0.5) < 0.08 for value in midpoint_y)


def test_exact_greedy_selects_the_true_target_stroke() -> None:
    current = blank_canvas(32)
    true_stroke = Stroke(0.1, 0.2, 0.9, 0.8, width=3, value=32)
    wrong_stroke = Stroke(0.1, 0.8, 0.9, 0.2, width=1, value=128)
    target = render_stroke(current, true_stroke)
    selected, next_canvas, scores = select_exact_greedy(
        current,
        target,
        (wrong_stroke, true_stroke),
    )
    assert selected == 1
    assert scores[1] == 0.0
    assert np.array_equal(np.asarray(next_canvas), np.asarray(target))


def test_planner_is_deterministic_and_exact_is_no_worse_than_random() -> None:
    target = render_stroke(
        blank_canvas(32),
        Stroke(0.05, 0.2, 0.95, 0.8, width=4, value=0),
    )
    config = ProposalConfig(count=16, min_length=0.1, max_length=0.4)
    exact_first = run_planner(
        target,
        "exact",
        steps=2,
        seed=91,
        proposal_config=config,
        capture_frames=True,
    )
    exact_second = run_planner(
        target,
        "exact",
        steps=2,
        seed=91,
        proposal_config=config,
        capture_frames=True,
    )
    random_run = run_planner(
        target,
        "random",
        steps=1,
        seed=91,
        proposal_config=config,
    )
    assert exact_first.steps == exact_second.steps
    assert np.array_equal(
        np.asarray(exact_first.final_canvas),
        np.asarray(exact_second.final_canvas),
    )
    assert len(exact_first.frames) == 3
    assert exact_first.steps[0].mse_after <= random_run.steps[0].mse_after


def test_pixel_metric_rejects_non_grayscale_images() -> None:
    with pytest.raises(ValueError, match="grayscale"):
        pixel_mse(Image.new("RGB", (16, 16)), Image.new("RGB", (16, 16)))
