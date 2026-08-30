from __future__ import annotations

import numpy as np
from PIL import Image

from latent_stroke_dynamics.rgb_coarse_to_fine import (
    DEFAULT_STAGES,
    PainterConfig,
    RGBStroke,
    StageConfig,
    TARGET_SET_SHA256,
    TARGET_SPECS,
    blank_rgb_canvas,
    plan_rgb_target,
    propose_rgb_strokes,
    render_rgb_stroke,
    replay_rgb_strokes_high_resolution,
    resize_with_padding,
)


def test_rgb_renderer_is_deterministic_and_non_mutating() -> None:
    canvas = blank_rgb_canvas(32)
    before = np.asarray(canvas).copy()
    stroke = RGBStroke(0.1, 0.2, 0.9, 0.8, 4, (12, 34, 56))
    first = render_rgb_stroke(canvas, stroke)
    second = render_rgb_stroke(canvas, stroke)
    assert np.array_equal(np.asarray(canvas), before)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), before)


def test_resize_with_padding_preserves_aspect_ratio() -> None:
    source = Image.new("RGB", (200, 100), color=(255, 0, 0))
    resized, metadata = resize_with_padding(source, 96)
    assert resized.size == (96, 96)
    assert metadata["resized_size"] == [96, 48]
    assert metadata["offset"] == [0, 24]
    assert resized.getpixel((48, 0)) == (255, 255, 255)
    assert resized.getpixel((48, 48)) == (255, 0, 0)


def test_proposals_are_unique_and_change_canvas() -> None:
    current = blank_rgb_canvas(32)
    target = Image.new("RGB", (32, 32), color=(40, 90, 150))
    stage = StageConfig("test", 2, 0.20, 0.80, 0.10, 0.25)
    candidates = propose_rgb_strokes(
        current,
        target,
        stage,
        np.random.default_rng(9),
        count=8,
        error_guided_fraction=0.75,
        max_attempts_per_candidate=100,
    )
    outcomes = [
        np.asarray(render_rgb_stroke(current, stroke)).tobytes()
        for stroke in candidates
    ]
    assert len(candidates) == 8
    assert len(set(outcomes)) == 8
    assert all(
        not np.array_equal(
            np.asarray(render_rgb_stroke(current, stroke)),
            np.asarray(current),
        )
        for stroke in candidates
    )


def test_planner_is_deterministic_and_monotonic() -> None:
    config = PainterConfig(
        planning_size=24,
        replay_size=32,
        supersample=1,
        candidates_per_pool=8,
        error_guided_fraction=0.75,
        patience=3,
        seed=73,
        gif_stride=1,
        max_attempts_per_candidate=100,
        stages=(StageConfig("test", 3, 0.20, 0.80, 0.15, 0.35),),
    )
    target = Image.new("RGB", (24, 24), color=(50, 100, 150))
    first = plan_rgb_target(target, config, target_stream=0)
    second = plan_rgb_target(target, config, target_stream=0)
    assert first.progress
    assert first.progress == second.progress
    assert first.strokes == second.strokes
    assert np.array_equal(
        np.asarray(first.final_canvas),
        np.asarray(second.final_canvas),
    )
    assert all(
        row["mse_after"] < row["mse_before"]
        and row["improvement"] > config.min_improvement
        for row in first.progress
    )
    assert first.best_mse <= first.final_mse


def test_high_resolution_replay_scales_rgb_strokes() -> None:
    strokes = (
        RGBStroke(0.1, 0.1, 0.9, 0.9, 2, (10, 20, 30)),
        RGBStroke(0.1, 0.9, 0.9, 0.1, 1, (200, 100, 50)),
    )
    best, final, frames = replay_rgb_strokes_high_resolution(
        strokes,
        planning_size=32,
        output_size=128,
        supersample=2,
        best_step=1,
        capture_steps={1, 2},
        gif_stride=1,
    )
    assert best.size == (128, 128)
    assert final.size == (128, 128)
    assert len(frames) == 3
    assert not np.array_equal(np.asarray(best), np.asarray(final))


def test_default_configuration_matches_locked_protocol() -> None:
    config = PainterConfig()
    assert config.planning_size == 96
    assert config.replay_size == 512
    assert config.candidates_per_pool == 64
    assert config.error_guided_fraction == 0.80
    assert config.patience == 12
    assert config.seed == 73
    assert config.stages == DEFAULT_STAGES
    assert sum(stage.max_steps for stage in config.stages) == 210
    assert len(TARGET_SPECS) == 5
    assert len({spec.filename for spec in TARGET_SPECS}) == 5
    assert TARGET_SET_SHA256 == (
        "31e1fcc2bf344f8b72d3f04dfbc9109c61c39fc8cbc10668c1b78d575a673b42"
    )
