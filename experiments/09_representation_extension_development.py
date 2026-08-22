#!/usr/bin/env python3
"""Run the frozen development-only smoke for both new representations."""

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

import numpy as np
import pandas as pd
import torch
from PIL import Image

from latent_stroke_dynamics.extension_training import (
    AutoencoderFitResult,
    PatchCounterfactualPayload,
    PatchFeaturePayload,
    autoencoder_mean_loss,
    baseline_models,
    build_patch_counterfactual_payload,
    build_patch_feature_payload,
    encode_autoencoder_maps,
    evaluate_patch_model,
    exact_target_oracle_retrieval,
    load_autoencoder_checkpoint,
    mean_image_baseline_mse,
    model_state_sha256,
    run_patch_overfit_check,
    save_autoencoder_checkpoint,
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


METRIC_COLUMNS = [
    "full_patch_mse",
    "action_region_mse",
    "outside_region_mse",
    "action_region_next_cosine_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/representation-extension-2026-08-22.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/representation-extension-development-smoke"),
    )
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_development_examples(
    config: dict[str, Any],
) -> dict[str, list[TransitionExample]]:
    smoke = config["development_smoke"]
    distribution = config["transition_distribution"]
    specs = {
        "train": (smoke["train_samples"], smoke["train_seed"]),
        "validation": (
            smoke["validation_samples"],
            smoke["validation_seed"],
        ),
        "test": (smoke["test_samples"], smoke["test_seed"]),
    }
    examples = {
        name: build_transition_split(
            samples=int(samples),
            canvas_size=int(config["canvas_size"]),
            crowding_levels=distribution["primary_crowding"],
            seed=int(seed),
            width_choices=distribution["widths"],
            value_choices=distribution["values"],
            min_length=float(distribution["minimum_length"]),
        )
        for name, (samples, seed) in specs.items()
    }
    fingerprint_sets = {
        name: {transition_fingerprint(item) for item in rows}
        for name, rows in examples.items()
    }
    names = list(fingerprint_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = fingerprint_sets[left].intersection(fingerprint_sets[right])
            if overlap:
                raise RuntimeError(
                    f"Development splits {left} and {right} overlap by "
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
    count = min(4, len(images))
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


def train_and_select_autoencoder(
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
    seed_rows: list[dict[str, int | float]] = []
    for seed in settings["model_seeds"]:
        print(f"Training task autoencoder seed {seed}...")
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
        seed_rows.append(
            {
                "seed": fit.seed,
                "best_epoch": fit.best_epoch,
                "best_validation_reconstruction_mse": fit.best_validation_mse,
                "epochs_completed": len(fit.history),
            }
        )
        print(
            f"  selected epoch {fit.best_epoch}, "
            f"validation MSE={fit.best_validation_mse:.8f}"
        )

    selected = min(fits, key=lambda item: item.best_validation_mse)
    validation_baseline = mean_image_baseline_mse(train_images, validation_images)
    validation_improvement = 1.0 - selected.best_validation_mse / max(
        validation_baseline,
        1e-12,
    )
    checkpoint_metadata = {
        "run_mode": "development_smoke",
        "selected_seed": selected.seed,
        "best_epoch": selected.best_epoch,
        "best_validation_reconstruction_mse": selected.best_validation_mse,
        "train_seed": config["development_smoke"]["train_seed"],
        "validation_seed": config["development_smoke"]["validation_seed"],
        "test_rows_used_for_training_or_selection": False,
        "historical_decisions_unchanged": True,
    }
    checkpoint_path = save_autoencoder_checkpoint(
        selected.model,
        checkpoint_metadata,
        output_dir / "checkpoints" / "task_autoencoder.pt",
    )
    loaded, loaded_metadata = load_autoencoder_checkpoint(checkpoint_path)
    preview = validation_images[:4]
    original_maps = encode_autoencoder_maps(selected.model, preview, batch_size=4)
    reloaded_maps = encode_autoencoder_maps(loaded, preview, batch_size=4)
    reload_maximum_difference = float(
        (original_maps - reloaded_maps).abs().max().item()
    )
    if loaded_metadata != checkpoint_metadata:
        raise RuntimeError("Autoencoder checkpoint metadata changed on reload.")

    train_latent_maps = encode_autoencoder_maps(
        loaded,
        train_images,
        batch_size=int(settings["batch_size"]),
    )
    statistics = fit_latent_channel_statistics(train_latent_maps)
    channel_std_mean = mean_latent_channel_std(statistics)
    with torch.inference_mode():
        test_reconstruction = loaded(test_images)
    test_metrics = reconstruction_metrics(test_reconstruction, test_images)
    finite_losses = bool(
        all(
            np.isfinite(row["best_validation_reconstruction_mse"])
            for row in seed_rows
        )
        and torch.isfinite(test_metrics["mse"]).all()
        and torch.isfinite(test_metrics["mae"]).all()
    )
    reconstruction_threshold_met = bool(
        validation_improvement
        >= float(settings["validation_improvement_vs_mean_image_minimum"])
    )
    latent_noncollapsed = bool(
        channel_std_mean >= float(settings["minimum_mean_channel_std"])
    )
    reload_passed = bool(
        reload_maximum_difference <= float(settings["reload_atol"])
    )
    implementation_integrity = bool(
        finite_losses and latent_noncollapsed and reload_passed
    )

    pd.DataFrame(history_rows).to_csv(
        output_dir / "autoencoder_training_history.csv",
        index=False,
    )
    pd.DataFrame(seed_rows).to_csv(
        output_dir / "autoencoder_seed_selection.csv",
        index=False,
    )
    write_json(
        output_dir / "latent_channel_statistics.json",
        {
            "source": "development_train_current_and_next_canvases_only",
            "mean": statistics.mean.tolist(),
            "std": statistics.std.tolist(),
            "mean_channel_std": channel_std_mean,
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
        "validation_mean_image_baseline_mse": validation_baseline,
        "validation_improvement_vs_mean_image": validation_improvement,
        "development_reconstruction_threshold_met": reconstruction_threshold_met,
        "test_reconstruction_mse": float(test_metrics["mse"].mean().item()),
        "test_reconstruction_mae": float(test_metrics["mae"].mean().item()),
        "mean_train_latent_channel_std": channel_std_mean,
        "latent_noncollapsed": latent_noncollapsed,
        "checkpoint_reload_maximum_difference": reload_maximum_difference,
        "checkpoint_reload_passed": reload_passed,
        "state_dict_sha256": model_state_sha256(loaded),
        "parameter_count": parameter_count(loaded),
        "all_losses_finite": finite_losses,
        "implementation_integrity_passed": implementation_integrity,
        "test_rows_used_for_training_or_selection": False,
        "development_only": True,
    }
    write_json(output_dir / "autoencoder_summary.json", summary)
    return loaded, statistics, summary


def encode_task_autoencoder_payloads(
    model: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    examples: dict[str, list[TransitionExample]],
    candidate_sets: Sequence[Any],
    batch_size: int,
) -> tuple[dict[str, PatchFeaturePayload], PatchCounterfactualPayload]:
    payloads: dict[str, PatchFeaturePayload] = {}
    for split_name, items in examples.items():
        current_images = images_to_grayscale_tensor([item.current for item in items])
        next_images = images_to_grayscale_tensor([item.next_canvas for item in items])
        current_maps = encode_autoencoder_maps(model, current_images, batch_size)
        next_maps = encode_autoencoder_maps(model, next_images, batch_size)
        current_tokens = standardize_latent_tokens(current_maps, statistics)
        next_tokens = standardize_latent_tokens(next_maps, statistics)
        payloads[split_name] = build_patch_feature_payload(
            items,
            current_tokens,
            next_tokens,
            patch_grid=(16, 16),
        )

    candidate_images = images_to_grayscale_tensor(
        [canvas for candidate_set in candidate_sets for canvas in candidate_set.canvases]
    )
    candidate_maps = encode_autoencoder_maps(model, candidate_images, batch_size)
    candidate_tokens = standardize_latent_tokens(candidate_maps, statistics).reshape(
        len(candidate_sets),
        len(COUNTERFACTUAL_ORDER),
        16 * 16,
        32,
    )
    counterfactuals = build_patch_counterfactual_payload(
        examples["test"],
        candidate_tokens,
        patch_grid=(16, 16),
    )
    return payloads, counterfactuals


def encode_vit_mae_payloads(
    examples: dict[str, list[TransitionExample]],
    candidate_sets: Sequence[Any],
    config: dict[str, Any],
) -> tuple[dict[str, PatchFeaturePayload], PatchCounterfactualPayload]:
    settings = config["new_representations"]["vit_mae"]
    encoder = FrozenViTMAEEncoder(settings["model_id"], device="cpu")
    batch_size = int(settings["encoding_batch_size"])
    payloads: dict[str, PatchFeaturePayload] = {}
    for split_name, items in examples.items():
        print(f"Encoding ViT-MAE {split_name} transitions...")
        encoded = encoder.encode(transition_images(items), batch_size=batch_size)
        expected_grid = tuple(settings["patch_grid"])
        if encoded.patch_grid != expected_grid:
            raise RuntimeError("ViT-MAE development grid differs from the frozen grid.")
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
    candidate_encoding = encoder.encode(candidate_images, batch_size=batch_size)
    grid = candidate_encoding.patch_grid
    candidate_features = candidate_encoding.patch_features.reshape(
        len(candidate_sets),
        len(COUNTERFACTUAL_ORDER),
        grid[0] * grid[1],
        int(settings["feature_dim"]),
    ).to(torch.float16)
    counterfactuals = build_patch_counterfactual_payload(
        examples["test"],
        candidate_features,
        patch_grid=grid,
    )
    del encoder
    gc.collect()
    return payloads, counterfactuals


def flatten_aggregate_columns(frame: pd.DataFrame) -> pd.DataFrame:
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
    result = (
        metrics.groupby(groups, sort=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    return flatten_aggregate_columns(result)


def run_representation_dynamics(
    representation: str,
    payloads: dict[str, PatchFeaturePayload],
    counterfactuals: PatchCounterfactualPayload,
    config: dict[str, Any],
    output_dir: Path,
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
        batch_size=int(settings["batch_size"]),
    )
    models: list[tuple[str, int, torch.nn.Module]] = [
        ("identity", -1, identity),
        ("mean_delta", -1, mean_delta),
    ]
    history_rows: list[dict[str, int | float | str]] = []
    parameter_counts = {"identity": 0, "mean_delta": 0}
    for family in settings["families"]:
        for seed in settings["model_seeds"]:
            print(f"Training {representation} {family}/{seed}...")
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
            history_rows.extend(fit.history)
            parameter_counts[family] = parameter_count(fit.model)
            print(
                f"  selected epoch {fit.best_epoch}, "
                f"validation loss={fit.best_validation_loss:.8g}"
            )

    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for model_name, seed, model in models:
        for split_name in ("validation", "test"):
            rows, retrieval = evaluate_patch_model(
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
            retrieval_rows.extend(retrieval)

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
    all_seeds_beat_identity = bool((seed_errors < identity_error).all())
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

    retrieval_summary = summarize_retrieval(retrieval)
    retrieval_family_summary = summarize_retrieval_families(retrieval_summary)
    selected_retrieval_summary = retrieval_family_summary.loc[
        retrieval_family_summary["model"] == selected_family
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
    parameters_within_cap = bool(
        all(
            count <= int(settings["maximum_parameters"])
            for count in parameter_counts.values()
        )
    )
    implementation_integrity = bool(
        overfit["loss_decreased"]
        and counterfactuals.all_encoded_candidates_unique
        and oracle["passed"]
        and metrics_finite
        and retrieval_finite
        and parameters_within_cap
    )
    classification = config["classification"]
    would_meet_action_usable_thresholds = bool(
        improvement_identity
        >= float(classification["minimum_improvement_vs_identity"])
        and improvement_mean
        >= float(classification["minimum_improvement_vs_mean_delta"])
        and positive_every_crowding
        and top1 >= float(classification["minimum_retrieval"])
        and implementation_integrity
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
    retrieval_summary.to_csv(
        output_dir / "retrieval_summary_by_seed.csv",
        index=False,
    )
    retrieval_family_summary.to_csv(
        output_dir / "retrieval_summary_by_family.csv",
        index=False,
    )

    summary = {
        "representation": representation,
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
        "all_model_seeds_beat_identity": all_seeds_beat_identity,
        "all_encoded_counterfactuals_unique": (
            counterfactuals.all_encoded_candidates_unique
        ),
        "exact_target_oracle": oracle,
        "overfit_check": overfit,
        "all_metrics_finite": bool(metrics_finite and retrieval_finite),
        "parameters_within_cap": parameters_within_cap,
        "implementation_integrity_passed": implementation_integrity,
        "would_meet_action_usable_thresholds_on_smoke_only": (
            would_meet_action_usable_thresholds
        ),
        "classification": "development_diagnostic_only",
        "decision_making": False,
        "elapsed_seconds": time.perf_counter() - started,
        **crowding_improvements,
    }
    write_json(output_dir / "dynamics_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.threads < 0:
        raise ValueError("--threads cannot be negative.")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    config = load_extension_config(args.config)
    if config["development_smoke"]["decision_making"] is not False:
        raise RuntimeError("Frozen development smoke must remain non-decision-making.")

    final_dir = args.output_dir
    incomplete_dir = final_dir.with_name(final_dir.name + ".incomplete")
    if final_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed development smoke: {final_dir}"
        )
    if incomplete_dir.exists():
        raise FileExistsError(
            f"Preserved incomplete smoke exists: {incomplete_dir}. Review it first."
        )
    incomplete_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    started = time.perf_counter()

    run_metadata = {
        "run_mode": "development_smoke",
        "decision_making": False,
        "frozen_config": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads_requested": args.threads,
        },
        "formal_primary_or_stress_data_generated": False,
        "historical_decisions_unchanged": True,
    }
    write_json(incomplete_dir / "run_config.json", run_metadata)

    print("Generating development-only transitions...")
    examples = build_development_examples(config)
    split_metadata(examples).to_csv(
        incomplete_dir / "split_metadata.csv",
        index=False,
    )
    candidate_sets = [build_counterfactual_set(item) for item in examples["test"]]

    autoencoder_dir = incomplete_dir / "task_autoencoder"
    autoencoder_dir.mkdir()
    autoencoder, statistics, autoencoder_summary = train_and_select_autoencoder(
        examples,
        config,
        autoencoder_dir,
    )
    print("Encoding task-autoencoder transitions and counterfactuals...")
    task_payloads, task_counterfactuals = encode_task_autoencoder_payloads(
        autoencoder,
        statistics,
        examples,
        candidate_sets,
        batch_size=int(
            config["new_representations"]["task_autoencoder"]["batch_size"]
        ),
    )
    task_summary = run_representation_dynamics(
        "task_autoencoder",
        task_payloads,
        task_counterfactuals,
        config,
        incomplete_dir / "task_autoencoder_dynamics",
    )
    del task_payloads, task_counterfactuals, autoencoder
    gc.collect()

    print("Encoding frozen ViT-MAE development data...")
    mae_payloads, mae_counterfactuals = encode_vit_mae_payloads(
        examples,
        candidate_sets,
        config,
    )
    mae_summary = run_representation_dynamics(
        "vit_mae",
        mae_payloads,
        mae_counterfactuals,
        config,
        incomplete_dir / "vit_mae_dynamics",
    )

    implementation_integrity = bool(
        autoencoder_summary["implementation_integrity_passed"]
        and task_summary["implementation_integrity_passed"]
        and mae_summary["implementation_integrity_passed"]
    )
    final_summary = {
        "status": (
            "development_smoke_complete"
            if implementation_integrity
            else "development_smoke_completed_with_failed_integrity_checks"
        ),
        "decision_making": False,
        "implementation_integrity_passed": implementation_integrity,
        "autoencoder": autoencoder_summary,
        "representations": {
            "task_autoencoder": task_summary,
            "vit_mae": mae_summary,
        },
        "development_seeds": {
            "train": config["development_smoke"]["train_seed"],
            "validation": config["development_smoke"]["validation_seed"],
            "test": config["development_smoke"]["test_seed"],
        },
        "formal_primary_or_stress_data_generated": False,
        "historical_decisions_unchanged": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(incomplete_dir / "smoke_summary.json", final_summary)
    os.replace(incomplete_dir, final_dir)

    print("\nRepresentation extension development smoke complete\n")
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    print(f"\nSaved actual JSON/CSV artifacts to: {final_dir.resolve()}")
    print("This development smoke cannot classify either representation.")
    print("No frozen primary or stress split was generated.")


if __name__ == "__main__":
    main()
