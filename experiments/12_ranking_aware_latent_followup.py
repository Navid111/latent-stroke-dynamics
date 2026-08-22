#!/usr/bin/env python3
"""Validate the frozen ranking-aware latent follow-up without generating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from latent_stroke_dynamics.extension_training import (
    create_patch_predictor,
    load_autoencoder_checkpoint,
    model_state_sha256,
    total_parameter_count,
)
from latent_stroke_dynamics.gate2 import parameter_count
from latent_stroke_dynamics.ranking_latent import (
    file_sha256,
    load_latent_channel_statistics,
    load_ranking_config,
    ranking_aware_objective,
)


CONFIG = Path("configs/ranking-aware-latent-2026-08-22.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        parser.error("Only --validate-only is implemented and authorized.")
    return args


def synthetic_objective_check() -> dict[str, float | bool]:
    generator = torch.Generator().manual_seed(8128)
    current = torch.randn(2, 4, 3, generator=generator)
    true_delta = 0.05 * torch.randn(2, 4, 3, generator=generator)
    predicted_delta = true_delta.clone().requires_grad_(True)
    true_next = current + true_delta
    candidates = torch.stack(
        (
            true_next,
            true_next + 0.25,
            true_next - 0.50,
            true_next + 0.75,
        ),
        dim=1,
    )
    masks = torch.ones(2, 4)
    losses = ranking_aware_objective(
        current,
        predicted_delta,
        true_delta,
        masks,
        candidates,
        masks,
        ranking_weight=0.3,
        temperature=0.1,
    )
    losses["total"].backward()
    gradient_finite = bool(
        predicted_delta.grad is not None
        and torch.isfinite(predicted_delta.grad).all()
    )
    return {
        "total": float(losses["total"].detach().item()),
        "balanced_mse": float(losses["balanced_mse"].detach().item()),
        "ranking_cross_entropy": float(
            losses["ranking_cross_entropy"].detach().item()
        ),
        "gradient_finite": gradient_finite,
    }


def main() -> None:
    parse_args()
    config = load_ranking_config(CONFIG)
    representation = config["frozen_representation"]
    checkpoint_path = Path(representation["checkpoint_path"])
    statistics_path = Path(representation["latent_statistics_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing frozen checkpoint: {checkpoint_path}")
    if not statistics_path.is_file():
        raise FileNotFoundError(f"Missing frozen latent statistics: {statistics_path}")

    model, metadata = load_autoencoder_checkpoint(checkpoint_path)
    state_hash = model_state_sha256(model)
    if state_hash != representation["checkpoint_sha256"]:
        raise RuntimeError("Frozen task-autoencoder state hash does not match.")
    if total_parameter_count(model) != representation["total_parameter_count"]:
        raise RuntimeError("Frozen task-autoencoder parameter count changed.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen task-autoencoder has trainable parameters.")

    statistics = load_latent_channel_statistics(statistics_path)
    statistics_file_hash = file_sha256(statistics_path)
    frozen_statistics_hash = representation["latent_statistics_sha256"]
    statistics_hash_matches = bool(
        frozen_statistics_hash is not None
        and frozen_statistics_hash == statistics_file_hash
    )

    predictor = create_patch_predictor("mlp", 32, (16, 16), 256)
    predictor_parameters = parameter_count(predictor)
    if predictor_parameters != config["predictor"]["parameter_count"]:
        raise RuntimeError("Ranking follow-up predictor parameter count changed.")

    development_dir = Path(config["development_output_dir"])
    formal_dir = Path(config["formal_output_dir"])
    synthetic = synthetic_objective_check()
    if not synthetic["gradient_finite"]:
        raise RuntimeError("Synthetic ranking objective gradient is non-finite.")

    status = (
        "ranking_latent_foundation_valid"
        if statistics_hash_matches
        else "ranking_latent_foundation_valid_hash_freeze_required"
    )
    result = {
        "status": status,
        "tests_are_still_required": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_state_sha256": state_hash,
        "checkpoint_metadata_selected_seed": metadata.get("selected_seed"),
        "checkpoint_metadata_best_epoch": metadata.get("best_epoch"),
        "autoencoder_total_parameters": total_parameter_count(model),
        "autoencoder_all_parameters_frozen": True,
        "latent_statistics_path": str(statistics_path),
        "latent_statistics_file_sha256": statistics_file_hash,
        "latent_statistics_hash_frozen_in_config": frozen_statistics_hash,
        "latent_statistics_hash_matches": statistics_hash_matches,
        "latent_statistics_channels": int(statistics.mean.numel()),
        "latent_statistics_mean_channel_std": float(statistics.std.mean().item()),
        "predictor_parameter_count": predictor_parameters,
        "ranking_grid": config["ranking_grid"],
        "synthetic_objective_check": synthetic,
        "development_authorized": config["development"]["authorized"],
        "formal_authorized": config["formal_reserved"]["authorized"],
        "development_output_dir_available": not development_dir.exists(),
        "formal_output_dir_available": not formal_dir.exists(),
        "followup_data_generated": False,
        "models_trained": False,
        "historical_decisions_unchanged": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
