#!/usr/bin/env python3
"""Validate frozen latent-planner resources without generating planner data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from latent_stroke_dynamics.latent_planner import (
    DEFAULT_LATENT_PLANNER_CONFIG,
    encode_task_latents,
    latent_candidate_scores,
    load_latent_planner_config,
    load_latent_predictor_ensembles,
    load_task_latent_resources,
)
from latent_stroke_dynamics.learned_pixel_planner import (
    load_pixel_checkpoint,
    state_dict_sha256,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_LATENT_PLANNER_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        parser.error("Only --validate-only is permitted in the foundation phase.")
    return args


def main() -> None:
    args = parse_args()
    config = load_latent_planner_config(args.config)
    autoencoder, statistics = load_task_latent_resources(config)
    ensembles = load_latent_predictor_ensembles(config)

    pixel_path = Path(config["pixel_predictor"]["path"])
    pixel_model, pixel_metadata = load_pixel_checkpoint(pixel_path, device="cpu")
    pixel_digest = state_dict_sha256(pixel_model)
    if pixel_digest != config["pixel_predictor"]["state_sha256"]:
        raise RuntimeError("Frozen pixel predictor SHA-256 mismatch.")
    pixel_model.eval()
    for parameter in pixel_model.parameters():
        parameter.requires_grad_(False)

    current = blank_canvas(64)
    target = render_stroke(
        current,
        Stroke(0.10, 0.20, 0.90, 0.80, width=3, value=32),
    )
    candidates = (
        Stroke(0.10, 0.20, 0.90, 0.80, width=3, value=32),
        Stroke(0.10, 0.80, 0.90, 0.20, width=1, value=128),
        Stroke(0.15, 0.25, 0.85, 0.75, width=4, value=64),
    )
    first_encoding = encode_task_latents(
        autoencoder, statistics, (current, target), batch_size=2
    )
    second_encoding = encode_task_latents(
        autoencoder, statistics, (current, target), batch_size=2
    )
    encoding_repeat_difference = float(
        (first_encoding - second_encoding).abs().max().item()
    )
    current_tokens = first_encoding[0:1]
    target_tokens = first_encoding[1:2]

    scoring_checks: dict[str, object] = {}
    for method in ("mse_only", "ranking_aware"):
        models = [loaded.model for loaded in ensembles[method]]
        first_scores, first_per_model = latent_candidate_scores(
            models,
            current_tokens,
            target_tokens,
            candidates,
            batch_size=2,
        )
        second_scores, second_per_model = latent_candidate_scores(
            models,
            current_tokens,
            target_tokens,
            candidates,
            batch_size=2,
        )
        scoring_checks[method] = {
            "aggregate_shape": list(first_scores.shape),
            "per_model_shape": list(first_per_model.shape),
            "all_scores_finite": bool(
                np.isfinite(first_scores).all() and np.isfinite(first_per_model).all()
            ),
            "maximum_repeat_difference": float(
                max(
                    np.max(np.abs(first_scores - second_scores)),
                    np.max(np.abs(first_per_model - second_per_model)),
                )
            ),
            "selected_index": int(np.argmin(first_scores)),
            "score_range": float(np.ptp(first_scores)),
        }

    latent_hashes = {
        method: {
            str(loaded.seed): loaded.state_sha256
            for loaded in ensembles[method]
        }
        for method in ("mse_only", "ranking_aware")
    }
    all_predictors_frozen = all(
        not any(parameter.requires_grad for parameter in loaded.model.parameters())
        for group in ensembles.values()
        for loaded in group
    )
    smoke_dir = Path(config["smoke"]["output_dir"])
    controlled_dir = Path(config["controlled"]["output_dir"])
    result = {
        "status": "latent_planner_foundation_valid_hash_freeze_required",
        "config_status": config["status"],
        "autoencoder_all_parameters_frozen": True,
        "autoencoder_state_sha256": config["representation"][
            "autoencoder_state_sha256"
        ],
        "latent_statistics_sha256": config["representation"][
            "latent_statistics_sha256"
        ],
        "latent_predictor_state_sha256": latent_hashes,
        "latent_predictor_ensemble_sizes": {
            method: len(group) for method, group in ensembles.items()
        },
        "all_latent_predictors_frozen": all_predictors_frozen,
        "pixel_predictor_state_sha256": pixel_digest,
        "pixel_predictor_model_seed": pixel_metadata.model_seed,
        "encoding_shape": list(first_encoding.shape),
        "encoding_maximum_repeat_difference": encoding_repeat_difference,
        "synthetic_scoring_checks": scoring_checks,
        "smoke_authorized": config["smoke"]["authorized"],
        "controlled_authorized": config["controlled"]["authorized"],
        "smoke_output_dir_available": not smoke_dir.exists(),
        "smoke_incomplete_dir_available": not smoke_dir.with_name(
            smoke_dir.name + ".incomplete"
        ).exists(),
        "controlled_output_dir_available": not controlled_dir.exists(),
        "controlled_incomplete_dir_available": not controlled_dir.with_name(
            controlled_dir.name + ".incomplete"
        ).exists(),
        "planner_data_generated": False,
        "models_trained_or_finetuned": False,
        "historical_results_unchanged": True,
    }
    if not all_predictors_frozen:
        raise RuntimeError("A formal latent predictor is not frozen.")
    if encoding_repeat_difference != 0.0:
        raise RuntimeError("Task-latent encoding is not deterministic.")
    if not all(
        check["all_scores_finite"]
        and check["maximum_repeat_difference"] == 0.0
        and check["aggregate_shape"] == [3]
        and check["per_model_shape"] == [3, 3]
        for check in scoring_checks.values()
    ):
        raise RuntimeError("Latent candidate scoring foundation check failed.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
