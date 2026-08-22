"""Frozen representation wrappers and integrity helpers for the extension."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTMAEConfig, ViTMAEModel


DEFAULT_EXTENSION_CONFIG = Path(
    "configs/representation-extension-2026-08-22.json"
)


@dataclass(frozen=True)
class SpatialEncodingBatch:
    """Spatial token representations held on CPU."""

    patch_features: torch.Tensor
    patch_grid: tuple[int, int]


@dataclass(frozen=True)
class LatentChannelStatistics:
    """Train-only channel statistics for task-latent standardization."""

    mean: torch.Tensor
    std: torch.Tensor


def deterministic_mae_noise(
    batch_size: int,
    patch_count: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return strictly ordered MAE noise so unmasked token order is stable."""

    if batch_size < 1 or patch_count < 1:
        raise ValueError("batch_size and patch_count must be positive.")
    order = torch.arange(patch_count, device=device, dtype=dtype)
    return order[None, :].expand(batch_size, -1).clone()


def restore_mae_patch_order(
    last_hidden_state: torch.Tensor,
    ids_restore: torch.Tensor,
) -> torch.Tensor:
    """Remove the class token and restore MAE patches to raster order.

    ViT-MAE shuffles patches even when its mask ratio is zero. With all patches
    retained, ``ids_restore`` maps the shuffled encoder output back to the
    original spatial order.
    """

    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have shape [batch, tokens, dim].")
    if ids_restore.ndim != 2:
        raise ValueError("ids_restore must have shape [batch, patches].")
    batch, tokens, feature_dim = last_hidden_state.shape
    if ids_restore.shape[0] != batch:
        raise ValueError("Hidden-state and restoration batch sizes must match.")
    patch_count = ids_restore.shape[1]
    if tokens != patch_count + 1:
        raise ValueError(
            "Unmasked MAE output must contain one class token and every patch."
        )
    if ids_restore.dtype != torch.long:
        ids_restore = ids_restore.long()
    expected = torch.arange(patch_count, device=ids_restore.device)[None, :]
    if not torch.equal(torch.sort(ids_restore, dim=1).values, expected.expand(batch, -1)):
        raise ValueError("ids_restore must contain one permutation per example.")

    shuffled_patches = last_hidden_state[:, 1:, :]
    gather_index = ids_restore[:, :, None].expand(-1, -1, feature_dim)
    return torch.gather(shuffled_patches, dim=1, index=gather_index)


class FrozenViTMAEEncoder:
    """Deterministic unmasked final spatial tokens from frozen ViT-MAE."""

    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("The frozen extension requires CPU encoding.")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        config = ViTMAEConfig.from_pretrained(model_name)
        config.mask_ratio = 0.0
        self.model = ViTMAEModel.from_pretrained(
            model_name,
            config=config,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 8,
    ) -> SpatialEncodingBatch:
        """Encode images as L2-normalized raster-ordered spatial tokens."""

        if not images:
            raise ValueError("At least one image is required.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        patch_batches: list[torch.Tensor] = []
        patch_grid: tuple[int, int] | None = None
        for start in tqdm(
            range(0, len(images), batch_size),
            desc="Encoding ViT-MAE canvases",
        ):
            batch = [image.convert("RGB") for image in images[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None or pixel_values.ndim != 4:
                raise RuntimeError("ViT-MAE processor did not return pixel_values.")

            patch_size = int(self.model.config.patch_size)
            height, width = pixel_values.shape[-2:]
            if height % patch_size or width % patch_size:
                raise RuntimeError("Processed image is not divisible by MAE patch size.")
            rows, columns = height // patch_size, width // patch_size
            patch_count = rows * columns
            noise = deterministic_mae_noise(
                pixel_values.shape[0],
                patch_count,
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
            outputs = self.model(**inputs, noise=noise)
            if outputs.mask is None or bool((outputs.mask != 0).any()):
                raise RuntimeError("ViT-MAE masking was not fully disabled.")
            if outputs.ids_restore is None:
                raise RuntimeError("ViT-MAE did not expose ids_restore.")

            patches = restore_mae_patch_order(
                outputs.last_hidden_state,
                outputs.ids_restore,
            )
            if patches.shape[1] != patch_count:
                raise RuntimeError("Unexpected ViT-MAE spatial token count.")
            patches = F.normalize(patches, dim=-1)
            patch_batches.append(patches.cpu())

            current_grid = (rows, columns)
            if patch_grid is not None and patch_grid != current_grid:
                raise RuntimeError("ViT-MAE patch grid changed between batches.")
            patch_grid = current_grid

        assert patch_grid is not None
        return SpatialEncodingBatch(
            patch_features=torch.cat(patch_batches, dim=0),
            patch_grid=patch_grid,
        )


class StrokeAutoencoder(nn.Module):
    """Small 64x64 grayscale autoencoder frozen by the extension protocol."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def encode_map(self, images: torch.Tensor) -> torch.Tensor:
        """Encode `[batch, 1, 64, 64]` canvases to `[batch, 32, 16, 16]`."""

        if images.ndim != 4 or images.shape[1:] != (1, 64, 64):
            raise ValueError("Autoencoder expects shape [batch, 1, 64, 64].")
        latent = self.encoder(images)
        if latent.shape[1:] != (32, 16, 16):
            raise RuntimeError("Unexpected autoencoder latent shape.")
        return latent

    def decode_map(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode `[batch, 32, 16, 16]` latent maps to grayscale canvases."""

        if latent.ndim != 4 or latent.shape[1:] != (32, 16, 16):
            raise ValueError("Decoder expects shape [batch, 32, 16, 16].")
        reconstruction = self.decoder(latent)
        if reconstruction.shape[1:] != (1, 64, 64):
            raise RuntimeError("Unexpected autoencoder reconstruction shape.")
        return reconstruction

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode_map(self.encode_map(images))


def images_to_grayscale_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    """Convert square 64x64 Pillow canvases to `[N, 1, 64, 64]` in `[0, 1]`."""

    if not images:
        raise ValueError("At least one image is required.")
    arrays: list[torch.Tensor] = []
    for image in images:
        grayscale = image.convert("L")
        if grayscale.size != (64, 64):
            raise ValueError("Task-autoencoder canvases must be 64x64.")
        array = torch.from_numpy(
            __import__("numpy").asarray(grayscale, dtype="float32").copy()
        )
        arrays.append(array / 255.0)
    return torch.stack(arrays)[:, None, :, :]


def fit_latent_channel_statistics(
    latent_maps: torch.Tensor,
) -> LatentChannelStatistics:
    """Fit channel mean/std over train examples and spatial positions only."""

    if latent_maps.ndim != 4:
        raise ValueError("latent_maps must have shape [batch, channels, rows, columns].")
    mean = latent_maps.mean(dim=(0, 2, 3))
    std = latent_maps.std(dim=(0, 2, 3), unbiased=False)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("Latent statistics must be finite.")
    return LatentChannelStatistics(mean=mean.detach(), std=std.detach())


def standardize_latent_tokens(
    latent_maps: torch.Tensor,
    statistics: LatentChannelStatistics,
    minimum_std: float = 1e-6,
) -> torch.Tensor:
    """Apply train-only channel statistics and flatten maps in raster order."""

    if latent_maps.ndim != 4:
        raise ValueError("latent_maps must have shape [batch, channels, rows, columns].")
    channels = latent_maps.shape[1]
    if statistics.mean.shape != (channels,) or statistics.std.shape != (channels,):
        raise ValueError("Latent statistics do not match the channel dimension.")
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive.")
    safe_std = statistics.std.clamp_min(minimum_std)
    standardized = (
        latent_maps - statistics.mean[None, :, None, None]
    ) / safe_std[None, :, None, None]
    return standardized.permute(0, 2, 3, 1).reshape(
        latent_maps.shape[0],
        latent_maps.shape[2] * latent_maps.shape[3],
        channels,
    )


def mean_latent_channel_std(statistics: LatentChannelStatistics) -> float:
    """Return the collapse-guard summary fixed by the protocol."""

    return float(statistics.std.mean().item())


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in evaluation mode and disable every gradient."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def reconstruction_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-example full-canvas reconstruction MSE and MAE."""

    if reconstruction.shape != target.shape or reconstruction.ndim != 4:
        raise ValueError("Reconstruction and target must share a 4D shape.")
    difference = reconstruction - target
    return {
        "mse": difference.square().flatten(start_dim=1).mean(dim=1),
        "mae": difference.abs().flatten(start_dim=1).mean(dim=1),
    }


def load_extension_config(
    path: str | Path = DEFAULT_EXTENSION_CONFIG,
) -> dict[str, Any]:
    """Load and validate the frozen representation-extension JSON config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_extension_config(config)
    return config


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def validate_extension_config(config: Mapping[str, Any]) -> None:
    """Reject configuration drift before any extension data are generated."""

    if config.get("experiment_id") != "representation-extension-2026-08-22":
        raise ValueError("Unexpected extension experiment_id.")
    if config.get("status") != "frozen_before_implementation":
        raise ValueError("Extension config is not frozen before implementation.")
    if config.get("historical_decisions_unchanged") is not True:
        raise ValueError("Historical decisions must remain unchanged.")
    if config.get("canvas_size") != 64 or config.get("device") != "cpu":
        raise ValueError("Canvas size and CPU device are frozen.")

    representations = _require_mapping(
        config.get("new_representations"),
        "new_representations",
    )
    if set(representations) != {"vit_mae", "task_autoencoder"}:
        raise ValueError("Exactly the two frozen new representations are required.")
    mae = _require_mapping(representations["vit_mae"], "vit_mae")
    if mae.get("model_id") != "facebook/vit-mae-base":
        raise ValueError("The frozen ViT-MAE model identifier changed.")
    if mae.get("mask_ratio") != 0.0 or mae.get("frozen") is not True:
        raise ValueError("ViT-MAE must be frozen and fully unmasked.")
    if mae.get("patch_grid") != [14, 14] or mae.get("feature_dim") != 768:
        raise ValueError("ViT-MAE spatial shape changed.")

    autoencoder = _require_mapping(
        representations["task_autoencoder"],
        "task_autoencoder",
    )
    if autoencoder.get("latent_grid") != [16, 16]:
        raise ValueError("Task-autoencoder latent grid changed.")
    if autoencoder.get("latent_dim") != 32:
        raise ValueError("Task-autoencoder latent dimension changed.")
    if autoencoder.get("model_seeds") != [101, 202, 303]:
        raise ValueError("Task-autoencoder seed set changed.")

    development = _require_mapping(config.get("development_smoke"), "development_smoke")
    primary = _require_mapping(config.get("primary_splits"), "primary_splits")
    stress = _require_mapping(config.get("stress_splits"), "stress_splits")
    data_seeds = {
        int(development["train_seed"]),
        int(development["validation_seed"]),
        int(development["test_seed"]),
    }
    expected_seed_count = len(data_seeds)
    for split in primary.values():
        split_mapping = _require_mapping(split, "primary split")
        data_seeds.add(int(split_mapping["seed"]))
        expected_seed_count += 1
    for split in stress.values():
        split_mapping = _require_mapping(split, "stress split")
        data_seeds.add(int(split_mapping["seed"]))
        expected_seed_count += 1
    if len(data_seeds) != expected_seed_count:
        raise ValueError("Development, primary, and stress data seeds must be disjoint.")

    dynamics = _require_mapping(config.get("dynamics"), "dynamics")
    if dynamics.get("families") != ["linear", "mlp"]:
        raise ValueError("Dynamics model families changed.")
    if dynamics.get("model_seeds") != [11, 22, 33]:
        raise ValueError("Dynamics model seeds changed.")
    if int(dynamics.get("maximum_parameters", 0)) > 1_000_000:
        raise ValueError("Dynamics parameter cap exceeds the frozen limit.")

    classification = _require_mapping(config.get("classification"), "classification")
    if float(classification.get("minimum_retrieval", -1)) != 0.5:
        raise ValueError("Action-usable retrieval threshold changed.")
    if float(classification.get("required_oracle_retrieval", -1)) != 1.0:
        raise ValueError("Oracle retrieval requirement changed.")
    if config.get("output_dir") != "outputs/representation-extension-2026-08-22":
        raise ValueError("Frozen output directory changed.")
