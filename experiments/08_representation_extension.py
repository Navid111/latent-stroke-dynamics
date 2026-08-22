#!/usr/bin/env python3
"""Validation and encoder smoke checks for the frozen representation extension."""

from __future__ import annotations

import argparse
import json

import torch

from latent_stroke_dynamics.gate2 import parameter_count
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke
from latent_stroke_dynamics.representation_extension import (
    FrozenViTMAEEncoder,
    StrokeAutoencoder,
    load_extension_config,
)


def validate_foundation(config_path: str) -> dict[str, object]:
    config = load_extension_config(config_path)
    autoencoder = StrokeAutoencoder()
    with torch.inference_mode():
        latent = autoencoder.encode_map(torch.ones(1, 1, 64, 64))
        reconstruction = autoencoder.decode_map(latent)
    return {
        "status": "foundation_valid",
        "experiment_id": config["experiment_id"],
        "historical_decisions_unchanged": config[
            "historical_decisions_unchanged"
        ],
        "new_representations": sorted(config["new_representations"]),
        "autoencoder_parameters": parameter_count(autoencoder),
        "autoencoder_latent_shape": list(latent.shape),
        "autoencoder_output_shape": list(reconstruction.shape),
        "extension_data_generated": False,
    }


def run_mae_smoke(config_path: str) -> dict[str, object]:
    config = load_extension_config(config_path)
    mae_config = config["new_representations"]["vit_mae"]
    encoder = FrozenViTMAEEncoder(model_name=mae_config["model_id"], device="cpu")
    blank = blank_canvas(64)
    stroked = render_stroke(
        blank,
        Stroke(0.15, 0.25, 0.85, 0.70, width=3, value=32),
    )
    first = encoder.encode([blank, stroked], batch_size=2)
    second = encoder.encode([blank, stroked], batch_size=2)
    maximum_repeat_difference = float(
        (first.patch_features - second.patch_features).abs().max().item()
    )
    expected_grid = tuple(mae_config["patch_grid"])
    expected_dim = int(mae_config["feature_dim"])
    tolerance = float(mae_config["determinism_atol"])
    if first.patch_grid != expected_grid:
        raise RuntimeError(
            f"Expected ViT-MAE grid {expected_grid}, received {first.patch_grid}."
        )
    if first.patch_features.shape != (2, expected_grid[0] * expected_grid[1], expected_dim):
        raise RuntimeError("ViT-MAE feature tensor has an unexpected shape.")
    if maximum_repeat_difference > tolerance:
        raise RuntimeError(
            "ViT-MAE repeatability exceeded the frozen absolute tolerance."
        )
    if any(parameter.requires_grad for parameter in encoder.model.parameters()):
        raise RuntimeError("ViT-MAE parameters were not fully frozen.")
    stroke_signal = float(
        (first.patch_features[0] - first.patch_features[1])
        .square()
        .mean()
        .item()
    )
    if not torch.isfinite(torch.tensor(stroke_signal)) or stroke_signal <= 0.0:
        raise RuntimeError("ViT-MAE smoke did not detect a finite stroke signal.")
    return {
        "status": "mae_encoder_smoke_passed",
        "model_id": mae_config["model_id"],
        "patch_grid": list(first.patch_grid),
        "feature_shape": list(first.patch_features.shape),
        "maximum_repeat_difference": maximum_repeat_difference,
        "determinism_atol": tolerance,
        "mean_squared_stroke_signal": stroke_signal,
        "model_frozen": True,
        "extension_data_generated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/representation-extension-2026-08-22.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mae-smoke", action="store_true")
    args = parser.parse_args()
    if int(args.validate_only) + int(args.mae_smoke) != 1:
        parser.error("Choose exactly one of --validate-only or --mae-smoke.")
    return args


def main() -> None:
    args = parse_args()
    if args.validate_only:
        result = validate_foundation(args.config)
    else:
        result = run_mae_smoke(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
