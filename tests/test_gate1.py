import numpy as np
import torch

from latent_stroke_dynamics.gate1 import (
    COMPARISON_ORDER,
    build_pairs,
    changed_pixel_count,
    patch_change_mask,
    patch_summary_metrics,
    pixel_change_mask_image,
    pixel_distance,
    pixel_matched_noise,
    sparse_pixel_matched_noise,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def test_build_pairs_reuses_action_across_crowding() -> None:
    pairs = build_pairs(samples=1, canvas_size=32, crowding_levels=[0, 5], seed=7)

    assert len(pairs) == len(COMPARISON_ORDER) * 2
    add_pairs = [pair for pair in pairs if pair.comparison == "add_stroke"]
    assert len(add_pairs) == 2
    assert add_pairs[0].stroke == add_pairs[1].stroke

    blank_pair = next(
        pair
        for pair in pairs
        if pair.comparison == "no_change" and pair.crowding == 0
    )
    crowded_pair = next(
        pair
        for pair in pairs
        if pair.comparison == "no_change" and pair.crowding == 5
    )
    assert np.all(np.asarray(blank_pair.before) == 255)
    assert np.any(np.asarray(crowded_pair.before) != 255)


def test_dense_pixel_matched_noise_matches_target_mae() -> None:
    rng = np.random.default_rng(11)
    before = blank_canvas(size=32)
    after = render_stroke(
        before,
        Stroke(0.1, 0.2, 0.9, 0.8, width=2, value=0),
    )
    target = pixel_distance(before, after)
    matched = pixel_matched_noise(before, target, rng)
    observed = pixel_distance(before, matched)

    assert target > 0
    assert abs(observed - target) / target < 0.15


def test_sparse_noise_matches_support_and_pixel_budget() -> None:
    rng = np.random.default_rng(13)
    before = blank_canvas(size=32)
    after = render_stroke(
        before,
        Stroke(0.1, 0.2, 0.9, 0.8, width=2, value=0),
    )
    matched = sparse_pixel_matched_noise(before, after, rng)

    assert changed_pixel_count(before, matched) == changed_pixel_count(before, after)
    assert abs(pixel_distance(before, matched) - pixel_distance(before, after)) < 1e-12
    assert not np.array_equal(np.asarray(matched), np.asarray(after))


def test_localization_metrics_reward_matching_feature_changes() -> None:
    before = blank_canvas(size=32)
    after = render_stroke(
        before,
        Stroke(0.1, 0.1, 0.9, 0.9, width=2, value=0),
    )
    patch_grid = (4, 4)
    mask = patch_change_mask(before, after, patch_grid)
    distances = torch.full((16,), 0.1)
    distances[mask] = 1.0
    reference_mask = pixel_change_mask_image(before, after)

    metrics = patch_summary_metrics(
        distances,
        before,
        after,
        patch_grid,
        reference_mask=reference_mask,
    )

    assert metrics["localization_topk_recall"] == 1.0
    assert metrics["localization_topk_lift"] > 1.0
    assert metrics["localization_enrichment"] > 5.0
    assert metrics["patch_reference_region_mean_cosine_distance"] == 1.0
