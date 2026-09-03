from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from latent_stroke_dynamics.quadratic_bezier_extension import (
    BASE_COMMIT,
    DEFAULT_CONFIG_PATH,
    PRIMITIVES,
    PROTOCOL_ID,
    SEEDS,
    QuadraticBezierStroke,
    extension_painter_config,
    generate_rights_safe_targets,
    generated_target_manifest,
    image_pixel_sha256,
    load_protocol_config,
    plan_primitive_target,
    propose_quadratic_bezier_strokes,
    quadratic_bezier_mask,
    render_quadratic_bezier,
    stroke_from_record,
    stroke_to_record,
    validate_only_report,
    validate_protocol_config,
)
from latent_stroke_dynamics.rgb_coarse_to_fine import (
    RGBStroke,
    PainterConfig,
    StageConfig,
    blank_rgb_canvas,
    render_rgb_stroke,
)


def test_protocol_is_locked_to_validation_only() -> None:
    config = validate_protocol_config(load_protocol_config(DEFAULT_CONFIG_PATH))
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["base_commit"] == BASE_COMMIT
    assert config["conditions"] == list(PRIMITIVES)
    assert config["seeds"] == list(SEEDS)
    assert config["execution_authorized"] is False
    assert config["target_hashes_frozen"] is False
    assert config["completed_executions"] == 0


def test_full_config_matches_matched_budget() -> None:
    for seed in SEEDS:
        config = extension_painter_config(seed)
        assert config.planning_size == 128
        assert config.replay_size == 512
        assert config.candidates_per_pool == 64
        assert config.error_guided_fraction == 0.80
        assert config.patience == 12
        assert [stage.max_steps for stage in config.stages] == [80, 140, 200]
        assert sum(stage.max_steps for stage in config.stages) == 420


def test_bezier_dataclass_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        QuadraticBezierStroke(-0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 2, (0, 0, 0))
    with pytest.raises(ValueError, match="at least one"):
        QuadraticBezierStroke(0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0, (0, 0, 0))
    with pytest.raises(ValueError, match="three values"):
        QuadraticBezierStroke(0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 2, (0, 0, 999))


def test_bezier_renderer_is_deterministic_and_non_mutating() -> None:
    canvas = blank_rgb_canvas(64)
    before = np.asarray(canvas).copy()
    stroke = QuadraticBezierStroke(0.1, 0.8, 0.5, 0.05, 0.9, 0.8, 5, (20, 80, 160))
    first = render_quadratic_bezier(canvas, stroke)
    second = render_quadratic_bezier(canvas, stroke)
    assert np.array_equal(np.asarray(canvas), before)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), before)
    assert quadratic_bezier_mask(stroke, 64).any()


def test_collinear_bezier_is_near_equivalent_to_straight_raster() -> None:
    canvas = blank_rgb_canvas(64)
    straight = RGBStroke(0.1, 0.2, 0.9, 0.8, 3, (15, 25, 35))
    curved = QuadraticBezierStroke(0.1, 0.2, 0.5, 0.5, 0.9, 0.8, 3, (15, 25, 35))
    straight_values = np.asarray(render_rgb_stroke(canvas, straight))
    curved_values = np.asarray(render_quadratic_bezier(canvas, curved))
    differing_pixels = np.any(straight_values != curved_values, axis=2)
    assert float(differing_pixels.mean()) <= 0.03


def test_degenerate_curve_is_finite_and_deterministic() -> None:
    canvas = blank_rgb_canvas(32)
    stroke = QuadraticBezierStroke(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 3, (7, 8, 9))
    first = render_quadratic_bezier(canvas, stroke)
    second = render_quadratic_bezier(canvas, stroke)
    values = np.asarray(first)
    assert np.isfinite(values).all()
    assert np.array_equal(values, np.asarray(second))
    assert quadratic_bezier_mask(stroke, 32).any()


def test_serialization_round_trip_for_both_primitives() -> None:
    strokes = (
        RGBStroke(0.1, 0.2, 0.8, 0.9, 3, (1, 2, 3)),
        QuadraticBezierStroke(0.1, 0.8, 0.5, 0.1, 0.9, 0.8, 4, (4, 5, 6)),
    )
    for stroke in strokes:
        assert stroke_from_record(stroke_to_record(stroke)) == stroke


def test_curve_proposals_are_unique_and_change_canvas() -> None:
    current = blank_rgb_canvas(32)
    target = Image.new("RGB", (32, 32), color=(50, 100, 150))
    stage = StageConfig("test", 2, 0.20, 0.80, 0.10, 0.25)
    candidates = propose_quadratic_bezier_strokes(
        current,
        target,
        stage,
        np.random.default_rng(17),
        count=8,
        error_guided_fraction=0.75,
        max_attempts_per_candidate=100,
    )
    outcomes = [np.asarray(render_quadratic_bezier(current, stroke)).tobytes() for stroke in candidates]
    assert len(candidates) == 8
    assert len(set(outcomes)) == 8
    assert all(outcome != np.asarray(current).tobytes() for outcome in outcomes)


def test_procedural_targets_are_deterministic_unique_and_rights_safe() -> None:
    first = generate_rights_safe_targets(256)
    second = generate_rights_safe_targets(256)
    assert len(first) == 6
    assert [item.target_id for item in first] == [item.target_id for item in second]
    first_hashes = [image_pixel_sha256(item.image) for item in first]
    second_hashes = [image_pixel_sha256(item.image) for item in second]
    assert first_hashes == second_hashes
    assert len(set(first_hashes)) == 6
    assert all("Original" in item.provenance for item in first)
    manifest = generated_target_manifest(256)
    assert manifest["target_count"] == 6
    assert len(manifest["target_set_sha256"]) == 64


@pytest.mark.parametrize("primitive", PRIMITIVES)
def test_smoke_planner_is_deterministic_and_monotonic(primitive: str) -> None:
    config = PainterConfig(
        planning_size=32,
        replay_size=32,
        supersample=1,
        candidates_per_pool=8,
        error_guided_fraction=0.75,
        patience=3,
        min_improvement=1e-9,
        seed=73,
        gif_stride=1,
        max_attempts_per_candidate=100,
        stages=(StageConfig("smoke", 3, 0.20, 0.80, 0.12, 0.30),),
    )
    target = generate_rights_safe_targets(128)[3].image.resize((32, 32), Image.Resampling.LANCZOS)
    first = plan_primitive_target(target, primitive, config, target_stream=0)
    second = plan_primitive_target(target, primitive, config, target_stream=0)
    assert first.progress
    assert first.progress == second.progress
    assert first.strokes == second.strokes
    assert np.array_equal(np.asarray(first.final_canvas), np.asarray(second.final_canvas))
    assert all(row["mse_after"] < row["mse_before"] for row in first.progress)
    assert first.best_mse <= first.final_mse


def test_validation_report_has_no_output_or_execution_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = DEFAULT_CONFIG_PATH.resolve()
    monkeypatch.chdir(tmp_path)
    report = validate_only_report(config_path)
    assert list(tmp_path.iterdir()) == []
    assert report["status"] == "quadratic_bezier_extension_valid_no_outputs"
    assert report["proposed_target_manifest"]["target_count"] == 6
    assert report["output_side_effects"] is False
    assert report["comparative_outputs_viewed"] is False
    assert report["execution_authorized"] is False
    assert report["training_performed"] is False
    assert report["closed_experiments_changed"] is False
