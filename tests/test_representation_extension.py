from copy import deepcopy
from pathlib import Path

import pytest
import torch

from latent_stroke_dynamics.extension_training import (
    build_patch_counterfactual_payload,
    build_patch_feature_payload,
    exact_target_oracle_retrieval,
    load_autoencoder_checkpoint,
    mean_image_baseline_mse,
    model_state_sha256,
    save_autoencoder_checkpoint,
    total_parameter_count,
)
from latent_stroke_dynamics.gate2 import build_transition_split
from latent_stroke_dynamics.representation_extension import (
    StrokeAutoencoder,
    deterministic_mae_noise,
    fit_latent_channel_statistics,
    freeze_module,
    load_extension_config,
    mean_latent_channel_std,
    reconstruction_metrics,
    restore_mae_patch_order,
    standardize_latent_tokens,
    validate_extension_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "representation-extension-2026-08-22.json"
)


def test_frozen_extension_config_is_valid_and_seed_disjoint() -> None:
    config = load_extension_config(CONFIG_PATH)
    assert config["historical_decisions_unchanged"] is True
    assert set(config["new_representations"]) == {
        "vit_mae",
        "task_autoencoder",
    }


def test_extension_config_rejects_seed_reuse() -> None:
    config = load_extension_config(CONFIG_PATH)
    broken = deepcopy(config)
    broken["primary_splits"]["test"]["seed"] = broken["development_smoke"][
        "test_seed"
    ]
    with pytest.raises(ValueError, match="seeds must be disjoint"):
        validate_extension_config(broken)


def test_mae_noise_is_deterministic_identity_order() -> None:
    first = deterministic_mae_noise(
        2,
        4,
        device="cpu",
        dtype=torch.float32,
    )
    second = deterministic_mae_noise(
        2,
        4,
        device="cpu",
        dtype=torch.float32,
    )
    assert torch.equal(first, second)
    assert torch.equal(first[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_mae_patch_order_is_restored_after_shuffle() -> None:
    shuffled_patch_values = torch.tensor([2.0, 0.0, 3.0, 1.0])
    hidden = torch.cat(
        (torch.tensor([-1.0]), shuffled_patch_values)
    ).reshape(1, 5, 1)
    ids_restore = torch.tensor([[1, 3, 0, 2]], dtype=torch.long)
    restored = restore_mae_patch_order(hidden, ids_restore)
    assert torch.equal(
        restored.flatten(),
        torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )


def test_autoencoder_has_frozen_spatial_shapes_and_output_range() -> None:
    model = StrokeAutoencoder()
    images = torch.rand(2, 1, 64, 64)
    latent = model.encode_map(images)
    reconstruction = model.decode_map(latent)
    assert latent.shape == (2, 32, 16, 16)
    assert reconstruction.shape == images.shape
    assert bool((reconstruction >= 0).all())
    assert bool((reconstruction <= 1).all())


def test_train_latent_statistics_standardize_channels() -> None:
    generator = torch.Generator().manual_seed(17)
    latent = torch.randn(5, 3, 4, 4, generator=generator)
    statistics = fit_latent_channel_statistics(latent)
    tokens = standardize_latent_tokens(latent, statistics)
    flattened = tokens.reshape(-1, 3)
    assert tokens.shape == (5, 16, 3)
    assert torch.allclose(flattened.mean(dim=0), torch.zeros(3), atol=1e-6)
    assert torch.allclose(
        flattened.std(dim=0, unbiased=False),
        torch.ones(3),
        atol=1e-6,
    )
    assert mean_latent_channel_std(statistics) > 0.0


def test_freeze_module_disables_all_autoencoder_gradients() -> None:
    model = freeze_module(StrokeAutoencoder())
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_reconstruction_metrics_are_per_example() -> None:
    target = torch.zeros(2, 1, 4, 4)
    reconstruction = target.clone()
    reconstruction[1] = 1.0
    metrics = reconstruction_metrics(reconstruction, target)
    assert torch.equal(metrics["mse"], torch.tensor([0.0, 1.0]))
    assert torch.equal(metrics["mae"], torch.tensor([0.0, 1.0]))


def test_mean_image_baseline_uses_only_training_mean() -> None:
    train = torch.cat(
        (torch.zeros(1, 1, 64, 64), torch.ones(1, 1, 64, 64)),
        dim=0,
    )
    evaluation = torch.zeros(2, 1, 64, 64)
    assert mean_image_baseline_mse(train, evaluation) == pytest.approx(0.25)


def test_autoencoder_checkpoint_round_trip_is_exact(tmp_path: Path) -> None:
    torch.manual_seed(91)
    model = StrokeAutoencoder().eval()
    inputs = torch.rand(2, 1, 64, 64)
    with torch.inference_mode():
        expected = model.encode_map(inputs)
    checkpoint = save_autoencoder_checkpoint(
        model,
        {"selected_seed": 101, "test_rows_used_for_selection": False},
        tmp_path / "autoencoder.pt",
    )
    loaded, metadata = load_autoencoder_checkpoint(checkpoint)
    with torch.inference_mode():
        actual = loaded.encode_map(inputs)
    assert metadata["selected_seed"] == 101
    assert metadata["test_rows_used_for_selection"] is False
    assert torch.equal(actual, expected)
    assert model_state_sha256(loaded) == model_state_sha256(model)
    assert total_parameter_count(loaded) == 49_569


def test_patch_payload_oracle_retrieves_true_candidate() -> None:
    examples = build_transition_split(3, 64, [0, 5], seed=808)
    generator = torch.Generator().manual_seed(17)
    current = torch.randn(3, 4, 5, generator=generator)
    next_features = current + 0.1 * torch.randn(3, 4, 5, generator=generator)
    payload = build_patch_feature_payload(
        examples,
        current,
        next_features,
        patch_grid=(2, 2),
    )
    candidates = torch.stack(
        (
            next_features,
            next_features + 0.25,
            next_features - 0.50,
            next_features + 0.75,
        ),
        dim=1,
    )
    counterfactuals = build_patch_counterfactual_payload(
        examples,
        candidates,
        patch_grid=(2, 2),
    )
    oracle = exact_target_oracle_retrieval(payload, counterfactuals)
    assert payload.size == 3
    assert counterfactuals.all_encoded_candidates_unique
    assert oracle["passed"] is True
    assert oracle["top1_accuracy"] == 1.0
    assert oracle["maximum_candidate_zero_difference"] == 0.0


def test_model_state_hash_changes_when_weights_change() -> None:
    torch.manual_seed(13)
    model = StrokeAutoencoder()
    before = model_state_sha256(model)
    with torch.no_grad():
        first_parameter = next(model.parameters())
        first_parameter.view(-1)[0] += 1.0
    after = model_state_sha256(model)
    assert before != after
