import numpy as np
import torch

from latent_stroke_dynamics.gate2 import ACTION_DIM, build_transition_split
from latent_stroke_dynamics.pixel_control import (
    PIXEL_INPUT_DIM,
    ExactCompositorPixelDeltaPredictor,
    LinearPixelDeltaPredictor,
    MLPPixelDeltaPredictor,
    balanced_pixel_mse,
    build_pixel_counterfactual_tensors,
    build_pixel_tensors,
    exact_compositor_delta,
    make_pixel_inputs,
    pixel_counterfactual_retrieval,
    stroke_pixel_mask,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def test_pixel_inputs_have_frozen_dimension_and_exact_mask() -> None:
    stroke = Stroke(0.1, 0.2, 0.9, 0.8, width=3, value=64)
    current_image = blank_canvas(32)
    next_image = render_stroke(current_image, stroke)
    example = build_transition_split(1, 32, [0], seed=17)[0]
    tensors = build_pixel_tensors([example], canvas_size=32)
    inputs = make_pixel_inputs(
        tensors.current,
        tensors.actions,
        tensors.action_masks,
    )
    expected_mask = np.asarray(next_image) != np.asarray(current_image)
    actual_mask = stroke_pixel_mask(stroke, 32).numpy().astype(bool)
    assert inputs.shape == (1, 32, 32, PIXEL_INPUT_DIM)
    assert tensors.actions.shape == (1, ACTION_DIM)
    assert np.array_equal(actual_mask, expected_mask)


def test_pixel_predictors_return_one_residual_per_pixel() -> None:
    current = torch.rand(2, 8, 8)
    actions = torch.rand(2, ACTION_DIM)
    masks = torch.zeros(2, 8, 8)
    masks[:, 2:5, 1:7] = 1.0
    linear = LinearPixelDeltaPredictor()
    mlp = MLPPixelDeltaPredictor(hidden_dim=16)
    oracle = ExactCompositorPixelDeltaPredictor()
    assert linear(current, actions, masks).shape == current.shape
    assert mlp(current, actions, masks).shape == current.shape
    assert oracle(current, actions, masks).shape == current.shape


def test_balanced_pixel_loss_preserves_small_action_region() -> None:
    true_delta = torch.zeros(1, 4, 4)
    true_delta[:, 0, 0] = -1.0
    identity = torch.zeros_like(true_delta)
    mask = torch.zeros_like(true_delta)
    mask[:, 0, 0] = 1.0
    assert float(balanced_pixel_mse(identity, true_delta, mask)) == 0.5
    assert float(balanced_pixel_mse(true_delta, true_delta, mask)) == 0.0


def test_exact_compositor_reproduces_renderer() -> None:
    examples = build_transition_split(8, 64, [0, 5, 15], seed=20260830)
    tensors = build_pixel_tensors(examples, canvas_size=64)
    exact_delta = exact_compositor_delta(
        tensors.current,
        tensors.actions,
        tensors.action_masks,
    )
    assert torch.allclose(tensors.current + exact_delta, tensors.next_canvas)


def test_pixel_counterfactuals_are_unique_and_retrieve_true_outcome() -> None:
    examples = build_transition_split(8, 32, [0, 3], seed=29)
    pixel_tensors = build_pixel_tensors(examples, canvas_size=32)
    counterfactuals = build_pixel_counterfactual_tensors(examples, canvas_size=32)
    exact_delta = exact_compositor_delta(
        pixel_tensors.current,
        pixel_tensors.actions,
        pixel_tensors.action_masks,
    )
    retrieval = pixel_counterfactual_retrieval(
        (pixel_tensors.current + exact_delta).clamp(0.0, 1.0),
        counterfactuals.candidate_next,
        counterfactuals.union_masks,
    )
    assert counterfactuals.all_candidates_unique
    assert bool(retrieval["top1_correct"].all())
    assert torch.equal(
        retrieval["predicted_index"],
        torch.zeros(len(examples), dtype=torch.long),
    )
    assert bool((retrieval["true_margin"] > 0).all())
