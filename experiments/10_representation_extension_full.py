#!/usr/bin/env python3
"""Run the single frozen full post-core representation extension."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from latent_stroke_dynamics.extension_training import (
    AutoencoderFitResult,
    PatchCounterfactualPayload,
    PatchFeaturePayload,
    baseline_models,
    build_patch_counterfactual_payload,
    build_patch_feature_payload,
    create_patch_predictor,
    encode_autoencoder_maps,
    evaluate_patch_model,
    exact_target_oracle_retrieval,
    load_autoencoder_checkpoint,
    mean_image_baseline_mse,
    model_state_sha256,
    run_patch_overfit_check,
    save_autoencoder_checkpoint,
    total_parameter_count,
    train_patch_predictor,
    train_stroke_autoencoder,
)
from latent_stroke_dynamics.gate2 import (
    COUNTERFACTUAL_ORDER,
    TransitionExample,
    build_counterfactual_set,
    build_transition_split,
    parameter_count,
    transition_fingerprint,
)
from latent_stroke_dynamics.representation_extension import (
    FrozenViTMAEEncoder,
    LatentChannelStatistics,
    StrokeAutoencoder,
    fit_latent_channel_statistics,
    images_to_grayscale_tensor,
    load_extension_config,
    mean_latent_channel_std,
    reconstruction_metrics,
    standardize_latent_tokens,
)
from latent_stroke_dynamics.retrieval_diagnostics import (
    summarize_retrieval,
    summarize_retrieval_families,
)


SCIENTIFIC_CONFIG = Path("configs/representation-extension-2026-08-22.json")
COMMAND_CONFIG = Path(
    "configs/representation-extension-full-command-2026-08-22.json"
)
METRIC_COLUMNS = [
    "full_patch_mse",
    "action_region_mse",
    "outside_region_mse",
    "action_region_next_cosine_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-frozen-extension", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if int(args.validate_only) + int(args.run_frozen_extension) != 1:
        parser.error(
            "Choose exactly one of --validate-only or --run-frozen-extension."
        )
    if args.threads < 0:
        parser.error("--threads cannot be negative.")
    return args


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_command_config() -> dict[str, Any]:
    with COMMAND_CONFIG.open("r", encoding="utf-8") as handle:
        command = json.load(handle)
    expected = {
        "status": "frozen_after_development_review_before_primary_data",
        "script": "experiments/10_representation_extension_full.py",
        "validation_command": (
            "python experiments/10_representation_extension_full.py --validate-only"
        ),
        "run_command": (
            "python experiments/10_representation_extension_full.py "
            "--run-frozen-extension"
        ),
        "scientific_config": str(SCIENTIFIC_CONFIG),
        "output_dir": "outputs/representation-extension-2026-08-22",
        "primary_seeds": [20261024, 20261025, 20261026],
        "stress_seeds": [20261027, 20261028, 20261029, 20261030],
        "development_metrics_used_to_change_settings": False,
        "reporting_only_repairs_after_smoke": [
            "count all autoencoder parameters after freezing"
        ],
        "single_authorized_run": True,
        "historical_decisions_unchanged": True,
    }
    if command != expected:
        raise RuntimeError("Full-command configuration drifted from the frozen file.")
    return command


def validate_full_command() -> dict[str, Any]:
    scientific = load_extension_config(SCIENTIFIC_CONFIG)
    command = load_command_config()
    final_dir = Path(command["output_dir"])
    incomplete_dir = final_dir.with_name(final_dir.name + ".incomplete")
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(
            "A completed or incomplete full-extension directory already exists."
        )

    autoencoder = StrokeAutoencoder()
    parameter_counts: dict[str, dict[str, int]] = {}
    for name, settings in scientific["new_representations"].items():
        feature_dim = int(settings["feature_dim"] if name == "vit_mae" else 32)
        grid = tuple(settings["patch_grid"] if name == "vit_mae" else [16, 16])
        parameter_counts[name] = {
            family: parameter_count(
                create_patch_predictor(
                    family,
                    feature_dim,
                    grid,
                    int(scientific["dynamics"]["hidden_dim"]),
                )
            )
            for family in scientific["dynamics"]["families"]
        }
    cap = int(scientific["dynamics"]["maximum_parameters"])
    if any(
        count > cap
        for family_counts in parameter_counts.values()
        for count in family_counts.values()
    ):
        raise RuntimeError("A frozen dynamics model exceeds the parameter cap.")

    return {
        "status": "full_command_valid",
        "scientific_config_status": scientific["status"],
        "command_config_status": command["status"],
        "autoencoder_total_parameters": total_parameter_count(autoencoder),
        "dynamics_parameter_counts": parameter_counts,
        "primary_splits": scientific["primary_splits"],
        "stress_splits": scientific["stress_splits"],
        "output_dir_available": True,
        "development_metrics_changed_settings": False,
        "primary_or_stress_data_generated": False,
        "authorized_run_has_started": False,
        "historical_decisions_unchanged": True,
    }


def build_full_examples(
    config: dict[str, Any],
) -> dict[str, list[TransitionExample]]:
    distribution = config["transition_distribution"]
    primary_crowding = distribution["primary_crowding"]
    primary_widths = distribution["widths"]
    primary_values = distribution["values"]
    specs: dict[str, dict[str, Any]] = {}
    for name, split in config["primary_splits"].items():
        specs[name] = {
            "samples": split["samples"],
            "seed": split["seed"],
            "crowding": primary_crowding,
            "widths": primary_widths,
            "values": primary_values,
        }
    stress = config["stress_splits"]
    specs["unseen_width_5"] = {
        "samples": stress["unseen_width_5"]["samples"],
        "seed": stress["unseen_width_5"]["seed"],
        "crowding": primary_crowding,
        "widths": [5],
        "values": primary_values,
    }
    specs["unseen_intensities"] = {
        "samples": stress["unseen_intensities"]["samples"],
        "seed": stress["unseen_intensities"]["seed"],
        "crowding": primary_crowding,
        "widths": primary_widths,
        "values": stress["unseen_intensities"]["values"],
    }
    specs["crowding_30"] = {
        "samples": stress["crowding_30"]["samples"],
        "seed": stress["crowding_30"]["seed"],
        "crowding": [30],
        "widths": primary_widths,
        "values": primary_values,
    }
    specs["crowding_60"] = {
        "samples": stress["crowding_60"]["samples"],
        "seed": stress["crowding_60"]["seed"],
        "crowding": [60],
        "widths": primary_widths,
        "values": primary_values,
    }

    examples = {
        name: build_transition_split(
            samples=int(spec["samples"]),
            canvas_size=int(config["canvas_size"]),
            crowding_levels=spec["crowding"],
            seed=int(spec["seed"]),
            width_choices=spec["widths"],
            value_choices=spec["values"],
            min_length=float(distribution["minimum_length"]),
        )
        for name, spec in specs.items()
    }
    fingerprints = {
        name: {transition_fingerprint(item) for item in items}
        for name, items in examples.items()
    }
    names = list(fingerprints)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = fingerprints[left].intersection(fingerprints[right])
            if overlap:
                raise RuntimeError(
                    f"Frozen splits {left} and {right} overlap by "
                    f"{len(overlap)} transitions."
                )
    return examples


def split_metadata(
    examples: dict[str, list[TransitionExample]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, items in examples.items():
        for sample_id, item in enumerate(items):
            rows.append(
                {
                    "split": split_name,
                    "sample_id": sample_id,
                    "fingerprint": transition_fingerprint(item),
                    "crowding": item.crowding,
                    "stroke_width": item.stroke.width,
                    "stroke_value": item.stroke.value,
                    "stroke_length": float(
                        np.hypot(
                            item.stroke.x1 - item.stroke.x0,
                            item.stroke.y1 - item.stroke.y0,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def transition_images(
    examples: Sequence[TransitionExample],
) -> list[Image.Image]:
    return [image for item in examples for image in (item.current, item.next_canvas)]


def save_reconstruction_preview(
    model: StrokeAutoencoder,
    images: torch.Tensor,
    path: Path,
) -> None:
    count = min(6, len(images))
    with torch.inference_mode():
        reconstructed = model(images[:count]).cpu()
    montage = Image.new("L", (128, 64 * count), color=255)
    for row in range(count):
        target = Image.fromarray(
            (images[row, 0].clamp(0, 1).numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        prediction = Image.fromarray(
            (reconstructed[row, 0].clamp(0, 1).numpy() * 255.0)
            .round()
            .astype(np.uint8)
        )
        montage.paste(target, (0, row * 64))
        montage.paste(prediction, (64, row * 64))
    montage.save(path)


def train_full_autoencoder(
    examples: dict[str, list[TransitionExample]],
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[StrokeAutoencoder, LatentChannelStatistics, dict[str, Any]]:
    settings = config["new_representations"]["task_autoencoder"]
    train_images = images_to_grayscale_tensor(transition_images(examples["train"]))
    validation_images = images_to_grayscale_tensor(
        transition_images(examples["validation"])
    )
    test_images = images_to_grayscale_tensor(transition_images(examples["test"]))

    fits: list[AutoencoderFitResult] = []
    history_rows: list[dict[str, int | float]] = []
    selection_rows: list[dict[str, int | float]] = []
    for seed in settings["model_seeds"]:
        print(f"Training full task autoencoder seed {seed}...")
        fit = train_stroke_autoencoder(
            train_images,
            validation_images,
            seed=int(seed),
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            batch_size=int(settings["batch_size"]),
            max_epochs=int(settings["max_epochs"]),
            patience=int(settings["patience"]),
        )
        fits.append(fit)
        history_rows.extend(fit.history)
        selection_rows.append(
            {
                "seed": fit.seed,
                "best_epoch": fit.best_epoch,
                "best_validation_reconstruction_mse": fit.best_validation_mse,
                "epochs_completed": len(fit.history),
            }
        )
        print(
            f"  selected epoch {fit.best_epoch}, "
            f"validation MSE={fit.best_validation_mse:.8g}"
        )

    selected = min(fits, key=lambda item: item.best_validation_mse)
    baseline = mean_image_baseline_mse(train_images, validation_images)
    improvement = 1.0 - selected.best_validation_mse / max(baseline, 1e-12)
    metadata = {
        "run_mode": "full_extension",
        "selected_seed": selected.seed,
        "best_epoch": selected.best_epoch,
        "best_validation_reconstruction_mse": selected.best_validation_mse,
        "train_seed": config["primary_splits"]["train"]["seed"],
        "validation_seed": config["primary_splits"]["validation"]["seed"],
        "test_rows_used_for_training_or_selection": False,
        "historical_decisions_unchanged": True,
    }
    checkpoint = save_autoencoder_checkpoint(
        selected.model,
        metadata,
        output_dir / "checkpoints" / "task_autoencoder.pt",
    )
    loaded, loaded_metadata = load_autoencoder_checkpoint(checkpoint)
    if loaded_metadata != metadata:
        raise RuntimeError("Full autoencoder checkpoint metadata changed on reload.")
    preview = validation_images[:4]
    before = encode_autoencoder_maps(selected.model, preview, batch_size=4)
    after = encode_autoencoder_maps(loaded, preview, batch_size=4)
    reload_difference = float((before - after).abs().max().item())

    train_maps = encode_autoencoder_maps(
        loaded,
        train_images,
        batch_size=int(settings["batch_size"]),
    )
    statistics = fit_latent_channel_statistics(train_maps)
    mean_channel_std = mean_latent_channel_std(statistics)
    with torch.inference_mode():
        test_reconstruction = loaded(test_images)
    test_values = reconstruction_metrics(test_reconstruction, test_images)
    all_finite = bool(
        all(
            np.isfinite(row["best_validation_reconstruction_mse"])
            for row in selection_rows
        )
        and torch.isfinite(test_values["mse"]).all()
        and torch.isfinite(test_values["mae"]).all()
    )
    reconstruction_passed = bool(
        improvement
        >= float(settings["validation_improvement_vs_mean_image_minimum"])
    )
    latent_noncollapsed = bool(
        mean_channel_std >= float(settings["minimum_mean_channel_std"])
    )
    reload_passed = bool(reload_difference <= float(settings["reload_atol"]))
    implementation_integrity = bool(
        all_finite and latent_noncollapsed and reload_passed
    )
    protocol_eligibility = bool(
        implementation_integrity and reconstruction_passed
    )

    pd.DataFrame(history_rows).to_csv(
        output_dir / "autoencoder_training_history.csv",
        index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        output_dir / "autoencoder_seed_selection.csv",
        index=False,
    )
    write_json(
        output_dir / "latent_channel_statistics.json",
        {
            "source": "full_train_current_and_next_canvases_only",
            "mean": statistics.mean.tolist(),
            "std": statistics.std.tolist(),
            "mean_channel_std": mean_channel_std,
        },
    )
    save_reconstruction_preview(
        loaded,
        test_images,
        output_dir / "autoencoder_reconstruction_preview.png",
    )
    summary = {
        "selected_seed": selected.seed,
        "best_epoch": selected.best_epoch,
        "best_validation_reconstruction_mse": selected.best_validation_mse,
        "validation_mean_image_baseline_mse": baseline,
        "validation_improvement_vs_mean_image": improvement,
        "reconstruction_threshold_met": reconstruction_passed,
        "test_reconstruction_mse": float(test_values["mse"].mean().item()),
        "test_reconstruction_mae": float(test_values["mae"].mean().item()),
        "mean_train_latent_channel_std": mean_channel_std,
        "latent_noncollapsed": latent_noncollapsed,
        "checkpoint_reload_maximum_difference": reload_difference,
        "checkpoint_reload_passed": reload_passed,
        "state_dict_sha256": model_state_sha256(loaded),
        "total_parameter_count": total_parameter_count(loaded),
        "all_losses_finite": all_finite,
        "implementation_integrity_passed": implementation_integrity,
        "protocol_eligibility_passed": protocol_eligibility,
        "test_rows_used_for_training_or_selection": False,
    }
    write_json(output_dir / "autoencoder_summary.json", summary)
    return loaded, statistics, summary


def encode_task_payloads(
    model: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    examples: dict[str, list[TransitionExample]],
    candidate_sets: Sequence[Any],
    batch_size: int,
) -> tuple[dict[str, PatchFeaturePayload], PatchCounterfactualPayload]:
    payloads: dict[str, PatchFeaturePayload] = {}
    for split_name, items in examples.items():
        print(f"Encoding task-autoencoder {split_name} transitions...")
        current = images_to_grayscale_tensor([item.current for item in items])
        next_images = images_to_grayscale_tensor([item.next_canvas for item in items])
        current_tokens = standardize_latent_tokens(
            encode_autoencoder_maps(model, current, batch_size),
            statistics,
        )
        next_tokens = standardize_latent_tokens(
            encode_autoencoder_maps(model, next_images, batch_size),
            statistics,
        )
        payloads[split_name] = build_patch_feature_payload(
            items,
            current_tokens,
            next_tokens,
            patch_grid=(16, 16),
        )
    candidate_images = images_to_grayscale_tensor(
        [canvas for candidate_set in candidate_sets for canvas in candidate_set.canvases]
    )
    candidate_tokens = standardize_latent_tokens(
        encode_autoencoder_maps(model, candidate_images, batch_size),
        statistics,
    ).reshape(len(candidate_sets), 4, 256, 32)
    counterfactuals = build_patch_counterfactual_payload(
        examples["test"],
        candidate_tokens,
        patch_grid=(16, 16),
    )
    return payloads, counterfactuals


def encode_mae_payloads(
    examples: dict[str, list[TransitionExample]],
    candidate_sets: Sequence[Any],
    config: dict[str, Any],
) -> tuple[dict[str, PatchFeaturePayload], PatchCounterfactualPayload]:
    settings = config["new_representations"]["vit_mae"]
    encoder = FrozenViTMAEEncoder(settings["model_id"], device="cpu")
    batch_size = int(settings["encoding_batch_size"])
    expected_grid = tuple(settings["patch_grid"])
    payloads: dict[str, PatchFeaturePayload] = {}
    for split_name, items in examples.items():
        print(f"Encoding ViT-MAE {split_name} transitions...")
        encoded = encoder.encode(transition_images(items), batch_size=batch_size)
        if encoded.patch_grid != expected_grid:
            raise RuntimeError("ViT-MAE grid differs from the frozen full grid.")
        payloads[split_name] = build_patch_feature_payload(
            items,
            encoded.patch_features[0::2].to(torch.float16),
            encoded.patch_features[1::2].to(torch.float16),
            patch_grid=encoded.patch_grid,
        )
    print("Encoding ViT-MAE test counterfactuals...")
    candidate_images = [
        canvas for candidate_set in candidate_sets for canvas in candidate_set.canvases
    ]
    encoded_candidates = encoder.encode(candidate_images, batch_size=batch_size)
    candidate_features = encoded_candidates.patch_features.reshape(
        len(candidate_sets),
        4,
        expected_grid[0] * expected_grid[1],
        int(settings["feature_dim"]),
    ).to(torch.float16)
    counterfactuals = build_patch_counterfactual_payload(
        examples["test"],
        candidate_features,
        patch_grid=expected_grid,
    )
    del encoder
    gc.collect()
    return payloads, counterfactuals


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def aggregate_metrics(
    metrics: pd.DataFrame,
    by_crowding: bool = False,
) -> pd.DataFrame:
    groups = ["model", "seed", "split"]
    if by_crowding:
        groups.append("crowding")
    return flatten_columns(
        metrics.groupby(groups, sort=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )


def save_spatial_figure(
    model: torch.nn.Module,
    payload: PatchFeaturePayload,
    path: Path,
) -> None:
    current = payload.current[:1].float()
    next_features = payload.next_features[:1].float()
    actions = payload.actions[:1].float()
    masks = payload.action_masks[:1].float()
    model.eval()
    with torch.inference_mode():
        prediction = model(current, actions, masks)
    true_delta = next_features - current
    true_magnitude = torch.linalg.vector_norm(true_delta[0], dim=-1)
    predicted_magnitude = torch.linalg.vector_norm(prediction[0], dim=-1)
    error = torch.linalg.vector_norm(prediction[0] - true_delta[0], dim=-1)
    grid = payload.patch_grid

    figure, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(true_magnitude.reshape(*grid), cmap="magma")
    axes[0].set_title("True residual")
    axes[1].imshow(predicted_magnitude.reshape(*grid), cmap="magma")
    axes[1].set_title("Predicted residual")
    axes[2].imshow(error.reshape(*grid), cmap="viridis")
    axes[2].set_title("Prediction error")
    axes[3].imshow(masks[0].reshape(*grid), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Action mask")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def classify_representation(
    *,
    improvement_identity: float,
    improvement_mean: float,
    positive_every_crowding: bool,
    retrieval: float,
    stable_seeds: bool,
    implementation_integrity: bool,
    representation_eligible: bool,
    config: dict[str, Any],
) -> tuple[str, list[str]]:
    criteria = config["classification"]
    reasons: list[str] = []
    if not implementation_integrity:
        reasons.append("implementation_integrity_failed")
    if not representation_eligible:
        reasons.append("representation_eligibility_failed")
    average_pass = bool(
        improvement_identity
        >= float(criteria["minimum_improvement_vs_identity"])
        and improvement_mean
        >= float(criteria["minimum_improvement_vs_mean_delta"])
    )
    if not average_pass:
        reasons.append("average_error_threshold_failed")
    if not positive_every_crowding:
        reasons.append("primary_crowding_condition_failed")
    if retrieval < float(criteria["minimum_retrieval"]):
        reasons.append("retrieval_threshold_failed")
    if not stable_seeds:
        reasons.append("seed_stability_failed")

    if (
        implementation_integrity
        and representation_eligible
        and average_pass
        and positive_every_crowding
        and retrieval >= float(criteria["minimum_retrieval"])
        and stable_seeds
    ):
        return "action_usable", reasons
    if implementation_integrity and representation_eligible and average_pass:
        return "average_predictable_but_not_action_usable", reasons
    return "not_predictively_usable", reasons


def run_full_dynamics(
    representation: str,
    payloads: dict[str, PatchFeaturePayload],
    counterfactuals: PatchCounterfactualPayload,
    config: dict[str, Any],
    output_dir: Path,
    representation_eligible: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = config["dynamics"]
    train_payload = payloads["train"]
    validation_payload = payloads["validation"]
    test_payload = payloads["test"]
    overfit = run_patch_overfit_check(
        train_payload,
        hidden_dim=int(settings["hidden_dim"]),
    )
    write_json(output_dir / "overfit_check.json", overfit)

    identity, mean_delta = baseline_models(
        train_payload,
        int(settings["batch_size"]),
    )
    models: list[tuple[str, int, torch.nn.Module]] = [
        ("identity", -1, identity),
        ("mean_delta", -1, mean_delta),
    ]
    trained: dict[tuple[str, int], torch.nn.Module] = {}
    history_rows: list[dict[str, int | float | str]] = []
    parameter_counts = {"identity": 0, "mean_delta": 0}
    for family in settings["families"]:
        for seed in settings["model_seeds"]:
            print(f"Training full {representation} {family}/{seed}...")
            fit = train_patch_predictor(
                family,
                int(seed),
                train_payload,
                validation_payload,
                hidden_dim=int(settings["hidden_dim"]),
                learning_rate=float(settings["learning_rate"]),
                weight_decay=float(settings["weight_decay"]),
                batch_size=int(settings["batch_size"]),
                max_epochs=int(settings["max_epochs"]),
                patience=int(settings["patience"]),
            )
            models.append((family, int(seed), fit.model))
            trained[(family, int(seed))] = fit.model
            history_rows.extend(fit.history)
            parameter_counts[family] = parameter_count(fit.model)
            print(
                f"  selected epoch {fit.best_epoch}, "
                f"validation loss={fit.best_validation_loss:.8g}"
            )

    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    evaluation_splits = [name for name in payloads if name != "train"]
    for model_name, seed, model in models:
        for split_name in evaluation_splits:
            rows, retrieval_rows_part = evaluate_patch_model(
                model_name,
                seed,
                model,
                split_name,
                payloads[split_name],
                batch_size=int(settings["batch_size"]),
                counterfactuals=(
                    counterfactuals if split_name == "test" else None
                ),
            )
            metric_rows.extend(rows)
            retrieval_rows.extend(retrieval_rows_part)

    metrics = pd.DataFrame(metric_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    history = pd.DataFrame(history_rows)
    validation = metrics.loc[
        (metrics["split"] == "validation")
        & metrics["model"].isin(settings["families"])
    ]
    selected_family = str(
        validation.groupby("model")["action_region_mse"].mean().idxmin()
    )
    test = metrics.loc[metrics["split"] == "test"]
    identity_error = float(
        test.loc[test["model"] == "identity", "action_region_mse"].mean()
    )
    mean_error = float(
        test.loc[test["model"] == "mean_delta", "action_region_mse"].mean()
    )
    selected_test = test.loc[test["model"] == selected_family]
    selected_error = float(selected_test["action_region_mse"].mean())
    improvement_identity = 1.0 - selected_error / max(identity_error, 1e-12)
    improvement_mean = 1.0 - selected_error / max(mean_error, 1e-12)
    selected_retrieval = retrieval.loc[retrieval["model"] == selected_family]
    top1 = float(selected_retrieval["top1_correct"].mean())
    seed_errors = selected_test.groupby("seed")["action_region_mse"].mean()
    stable_seeds = bool((seed_errors < identity_error).all())

    crowding_improvements: dict[str, float] = {}
    positive_every_crowding = True
    for crowding in config["transition_distribution"]["primary_crowding"]:
        identity_value = float(
            test.loc[
                (test["model"] == "identity")
                & (test["crowding"] == crowding),
                "action_region_mse",
            ].mean()
        )
        selected_value = float(
            selected_test.loc[
                selected_test["crowding"] == crowding,
                "action_region_mse",
            ].mean()
        )
        improvement = 1.0 - selected_value / max(identity_value, 1e-12)
        crowding_improvements[f"crowding_{crowding}_improvement"] = improvement
        positive_every_crowding = bool(
            positive_every_crowding and improvement > 0.0
        )

    retrieval_by_seed = summarize_retrieval(retrieval)
    retrieval_by_family = summarize_retrieval_families(retrieval_by_seed)
    selected_retrieval_summary = retrieval_by_family.loc[
        retrieval_by_family["model"] == selected_family
    ].iloc[0]
    oracle = exact_target_oracle_retrieval(test_payload, counterfactuals)
    metrics_finite = bool(
        np.isfinite(metrics[METRIC_COLUMNS].to_numpy(dtype=float)).all()
    )
    retrieval_finite = bool(
        np.isfinite(
            retrieval[
                ["true_margin"]
                + [f"score_{name}" for name in COUNTERFACTUAL_ORDER]
            ].to_numpy(dtype=float)
        ).all()
    )
    within_cap = bool(
        all(
            count <= int(settings["maximum_parameters"])
            for count in parameter_counts.values()
        )
    )
    integrity = bool(
        overfit["loss_decreased"]
        and counterfactuals.all_encoded_candidates_unique
        and oracle["passed"]
        and metrics_finite
        and retrieval_finite
        and within_cap
    )
    classification, classification_reasons = classify_representation(
        improvement_identity=improvement_identity,
        improvement_mean=improvement_mean,
        positive_every_crowding=positive_every_crowding,
        retrieval=top1,
        stable_seeds=stable_seeds,
        implementation_integrity=integrity,
        representation_eligible=representation_eligible,
        config=config,
    )

    stress_rows: list[dict[str, Any]] = []
    for split_name in evaluation_splits:
        if split_name in {"validation", "test"}:
            continue
        split_frame = metrics.loc[metrics["split"] == split_name]
        identity_value = float(
            split_frame.loc[
                split_frame["model"] == "identity",
                "action_region_mse",
            ].mean()
        )
        selected_value = float(
            split_frame.loc[
                split_frame["model"] == selected_family,
                "action_region_mse",
            ].mean()
        )
        stress_rows.append(
            {
                "split": split_name,
                "identity_action_region_mse": identity_value,
                "selected_action_region_mse": selected_value,
                "improvement_vs_identity": (
                    1.0 - selected_value / max(identity_value, 1e-12)
                ),
            }
        )

    metrics.to_csv(output_dir / "prediction_metrics.csv", index=False)
    aggregate_metrics(metrics).to_csv(
        output_dir / "aggregate_metrics.csv",
        index=False,
    )
    aggregate_metrics(metrics, by_crowding=True).to_csv(
        output_dir / "aggregate_metrics_by_crowding.csv",
        index=False,
    )
    history.to_csv(output_dir / "training_history.csv", index=False)
    retrieval.to_csv(output_dir / "counterfactual_retrieval.csv", index=False)
    retrieval_by_seed.to_csv(
        output_dir / "retrieval_summary_by_seed.csv",
        index=False,
    )
    retrieval_by_family.to_csv(
        output_dir / "retrieval_summary_by_family.csv",
        index=False,
    )
    pd.DataFrame(stress_rows).to_csv(
        output_dir / "stress_summary.csv",
        index=False,
    )
    example_seed = min(int(seed) for seed in settings["model_seeds"])
    save_spatial_figure(
        trained[(selected_family, example_seed)],
        test_payload,
        output_dir / "example_spatial_prediction.png",
    )

    summary = {
        "representation": representation,
        "classification": classification,
        "classification_reasons": classification_reasons,
        "representation_eligibility_passed": representation_eligible,
        "patch_grid": list(train_payload.patch_grid),
        "feature_dim": int(train_payload.current.shape[-1]),
        "selected_model_family_by_validation": selected_family,
        "model_parameter_counts": parameter_counts,
        "identity_action_region_mse": identity_error,
        "mean_delta_action_region_mse": mean_error,
        "selected_action_region_mse": selected_error,
        "improvement_vs_identity": improvement_identity,
        "improvement_vs_mean_delta": improvement_mean,
        "counterfactual_top1_accuracy": top1,
        "counterfactual_top1_seed_std": float(
            selected_retrieval_summary["top1_accuracy_seed_std"]
        ),
        "true_beats_shift_position_rate": float(
            selected_retrieval_summary["true_beats_shift_position_rate"]
        ),
        "true_beats_change_width_rate": float(
            selected_retrieval_summary["true_beats_change_width_rate"]
        ),
        "true_beats_change_intensity_rate": float(
            selected_retrieval_summary["true_beats_change_intensity_rate"]
        ),
        "positive_improvement_every_crowding": positive_every_crowding,
        "all_model_seeds_beat_identity": stable_seeds,
        "all_encoded_counterfactuals_unique": (
            counterfactuals.all_encoded_candidates_unique
        ),
        "exact_target_oracle": oracle,
        "overfit_check": overfit,
        "all_metrics_finite": bool(metrics_finite and retrieval_finite),
        "parameters_within_cap": within_cap,
        "implementation_integrity_passed": integrity,
        "stress_results": stress_rows,
        "elapsed_seconds": time.perf_counter() - started,
        **crowding_improvements,
    }
    write_json(output_dir / "representation_summary.json", summary)
    return summary


def run_full_extension(
    scientific: dict[str, Any],
    command: dict[str, Any],
    threads: int,
) -> dict[str, Any]:
    if threads > 0:
        torch.set_num_threads(threads)
    final_dir = Path(command["output_dir"])
    incomplete_dir = final_dir.with_name(final_dir.name + ".incomplete")
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite a completed or incomplete full extension."
        )
    incomplete_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    started = time.perf_counter()
    write_json(
        incomplete_dir / "run_config.json",
        {
            "scientific_config": scientific,
            "command_config": command,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "threads_requested": threads,
            },
            "development_metrics_changed_settings": False,
            "historical_decisions_unchanged": True,
        },
    )

    print("Generating the single frozen primary and stress splits...")
    examples = build_full_examples(scientific)
    split_metadata(examples).to_csv(
        incomplete_dir / "split_metadata.csv",
        index=False,
    )
    candidate_sets = [build_counterfactual_set(item) for item in examples["test"]]

    auto_dir = incomplete_dir / "task_autoencoder"
    auto_dir.mkdir()
    autoencoder, statistics, auto_summary = train_full_autoencoder(
        examples,
        scientific,
        auto_dir,
    )
    task_payloads, task_counterfactuals = encode_task_payloads(
        autoencoder,
        statistics,
        examples,
        candidate_sets,
        int(
            scientific["new_representations"]["task_autoencoder"][
                "batch_size"
            ]
        ),
    )
    task_summary = run_full_dynamics(
        "task_autoencoder",
        task_payloads,
        task_counterfactuals,
        scientific,
        incomplete_dir / "task_autoencoder_dynamics",
        representation_eligible=bool(auto_summary["protocol_eligibility_passed"]),
    )
    del autoencoder, task_payloads, task_counterfactuals
    gc.collect()

    mae_payloads, mae_counterfactuals = encode_mae_payloads(
        examples,
        candidate_sets,
        scientific,
    )
    mae_summary = run_full_dynamics(
        "vit_mae",
        mae_payloads,
        mae_counterfactuals,
        scientific,
        incomplete_dir / "vit_mae_dynamics",
        representation_eligible=True,
    )

    integrity = bool(
        auto_summary["implementation_integrity_passed"]
        and task_summary["implementation_integrity_passed"]
        and mae_summary["implementation_integrity_passed"]
    )
    summary = {
        "status": (
            "full_extension_complete"
            if integrity
            else "full_extension_completed_with_failed_integrity_checks"
        ),
        "single_authorized_run": True,
        "implementation_integrity_passed": integrity,
        "autoencoder": auto_summary,
        "representations": {
            "task_autoencoder": task_summary,
            "vit_mae": mae_summary,
        },
        "primary_seeds": command["primary_seeds"],
        "stress_seeds": command["stress_seeds"],
        "development_metrics_changed_settings": False,
        "historical_decisions_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
        "do_not_rerun_or_retune": True,
    }
    write_json(incomplete_dir / "extension_summary.json", summary)
    os.replace(incomplete_dir, final_dir)
    return summary


def main() -> None:
    args = parse_args()
    scientific = load_extension_config(SCIENTIFIC_CONFIG)
    command = load_command_config()
    if args.validate_only:
        print(json.dumps(validate_full_command(), indent=2, sort_keys=True))
        return

    summary = run_full_extension(scientific, command, args.threads)
    print("\nFrozen representation extension complete\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved outputs to: {Path(command['output_dir']).resolve()}")
    print("Do not rerun or retune this extension.")
    print("All previous Gate 2, pixel-control, and Stage 3 decisions remain unchanged.")


if __name__ == "__main__":
    main()
