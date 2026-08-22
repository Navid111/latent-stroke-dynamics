from copy import deepcopy
from pathlib import Path

import pytest
import torch

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
