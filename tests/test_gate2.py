import numpy as np
import torch

from latent_stroke_dynamics.gate2 import (
    ACTION_DIM,
    COUNTERFACTUAL_ORDER,
    LinearPatchDeltaPredictor,
    MLPPatchDeltaPredictor,
    balanced_patch_mse,
    build_transition_split,
    counterfactual_canvases,
    counterfactual_retrieval,
    counterfactual_strokes,
    stroke_action_vector,
    stroke_patch_coverage,
    transition_fingerprint,
)
from latent_stroke_dynamics.renderer import Stroke


def test_transition_generation_is_deterministic_and_split_specific() -> None:
    first = build_transition_split(4, 32, [0, 3], seed=101)
    repeated = build_transition_split(4, 32, [0, 3], seed=101)
    different = build_transition_split(4, 32, [0, 3], seed=102)

    first_fingerprints = [transition_fingerprint(item) for item in first]
    repeated_fingerprints = [transition_fingerprint(item) for item in repeated]
    different_fingerprints = [transition_fingerprint(item) for item in different]

    assert first_fingerprints == repeated_fingerprints
    assert set(first_fingerprints).isdisjoint(different_fingerprints)
    assert all(
        np.any(np.asarray(item.current) != np.asarray(item.next_canvas))
        for item in first
    )


def test_action_encoding_is_undirected_and_normalized() -> None:
    forward = Stroke(0.1, 0.2, 0.8, 0.9, width=3, value=64)
    backward = Stroke(0.8, 0.9, 0.1, 0.2, width=3, value=64)

    forward_vector = stroke_action_vector(forward)
    backward_vector = stroke_action_vector(backward)

    assert forward_vector.shape == (ACTION_DIM,)
    assert torch.allclose(forward_vector, backward_vector, atol=1e-6)
    assert 0.0 <= float(forward_vector[0]) <= 1.0
    assert 0.0 <= float(forward_vector[1]) <= 1.0
    assert 0.0 <= float(forward_vector[2]) <= 1.0
    assert 0.0 <= float(forward_vector[5]) <= 1.0
    assert 0.0 <= float(forward_vector[6]) <= 1.0


def test_action_mask_has_fractional_patch_support() -> None:
    stroke = Stroke(0.1, 0.2, 0.9, 0.8, width=2, value=128)
    coverage = stroke_patch_coverage(stroke, canvas_size=64, patch_grid=(4, 4))

    assert coverage.shape == (16,)
    assert torch.all(coverage >= 0)
    assert torch.all(coverage <= 1)
    assert bool((coverage > 0).any())
    assert bool((coverage == 0).any())


def test_predictors_return_one_delta_per_patch() -> None:
    current = torch.randn(2, 4, 8)
    actions = torch.rand(2, ACTION_DIM)
    masks = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]])

    linear = LinearPatchDeltaPredictor(feature_dim=8, patch_grid=(2, 2))
    nonlinear = MLPPatchDeltaPredictor(
        feature_dim=8,
        patch_grid=(2, 2),
        hidden_dim=16,
    )

    assert linear(current, actions, masks).shape == current.shape
    assert nonlinear(current, actions, masks).shape == current.shape


def test_balanced_loss_does_not_hide_action_region_error() -> None:
    true_delta = torch.zeros(1, 4, 3)
    true_delta[:, 0, :] = 1.0
    identity = torch.zeros_like(true_delta)
    perfect = true_delta.clone()
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    identity_loss = balanced_patch_mse(identity, true_delta, mask)
    perfect_loss = balanced_patch_mse(perfect, true_delta, mask)

    assert float(identity_loss) == 0.5
    assert float(perfect_loss) == 0.0


def test_counterfactuals_change_each_controlled_property() -> None:
    stroke = Stroke(0.2, 0.3, 0.7, 0.8, width=2, value=64)
    variants = counterfactual_strokes(stroke)

    assert len(variants) == len(COUNTERFACTUAL_ORDER)
    assert variants[0] == stroke
    assert variants[1] != stroke
    assert variants[2].width != stroke.width
    assert variants[3].value != stroke.value

    example = build_transition_split(1, 32, [0], seed=9)[0]
    canvases = counterfactual_canvases(example)
    assert len(canvases) == 4
    assert np.array_equal(np.asarray(canvases[0]), np.asarray(example.next_canvas))


def test_counterfactual_retrieval_recovers_exact_true_candidate() -> None:
    predicted = torch.randn(2, 4, 6)
    candidates = torch.randn(2, 4, 4, 6)
    candidates[:, 0] = predicted
    union_masks = torch.ones(2, 4)

    result = counterfactual_retrieval(predicted, candidates, union_masks)

    assert torch.equal(result["predicted_index"], torch.zeros(2, dtype=torch.long))
    assert bool(result["top1_correct"].all())
    assert bool((result["true_margin"] > 0).all())
