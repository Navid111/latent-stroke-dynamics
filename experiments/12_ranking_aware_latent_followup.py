#!/usr/bin/env python3
"""Validate or run the guarded ranking-aware latent development grid."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from latent_stroke_dynamics.extension_training import (
    PatchCounterfactualPayload,
    PatchFeaturePayload,
    baseline_models,
    build_patch_counterfactual_payload,
    build_patch_feature_payload,
    encode_autoencoder_maps,
    evaluate_patch_model,
    load_autoencoder_checkpoint,
    model_state_sha256,
    total_parameter_count,
    train_patch_predictor,
)
from latent_stroke_dynamics.gate2 import (
    COUNTERFACTUAL_ORDER,
    TransitionExample,
    build_counterfactual_set,
    build_transition_split,
    parameter_count,
    transition_fingerprint,
)
from latent_stroke_dynamics.ranking_latent import (
    file_sha256,
    load_latent_channel_statistics,
    load_ranking_config,
    ranking_aware_objective,
)
from latent_stroke_dynamics.ranking_training import (
    protocol_oracle_retrieval,
    run_ranking_overfit_check,
    select_ranking_setting,
    train_ranking_predictor,
)
from latent_stroke_dynamics.representation_extension import (
    LatentChannelStatistics,
    StrokeAutoencoder,
    images_to_grayscale_tensor,
    standardize_latent_tokens,
)


CONFIG = Path("configs/ranking-aware-latent-2026-08-22.json")
COMMAND_CONFIG = Path(
    "configs/ranking-aware-latent-development-command-2026-08-22.json"
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
    parser.add_argument("--run-development-grid", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if int(args.validate_only) + int(args.run_development_grid) != 1:
        parser.error("Choose exactly one validation or development mode.")
    if args.threads < 0:
        parser.error("--threads cannot be negative.")
    return args


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_command_config() -> dict[str, Any]:
    with COMMAND_CONFIG.open("r", encoding="utf-8") as handle:
        command = json.load(handle)
    expected = {
        "script": "experiments/12_ranking_aware_latent_followup.py",
        "validation_command": (
            "python experiments/12_ranking_aware_latent_followup.py --validate-only"
        ),
        "run_command": (
            "python experiments/12_ranking_aware_latent_followup.py "
            "--run-development-grid"
        ),
        "scientific_config": str(CONFIG),
        "development_output_dir": (
            "outputs/ranking-aware-latent-development-2026-08-22"
        ),
        "development_seeds": [20261101, 20261102, 20261103],
        "formal_seeds_reserved": [
            20261104,
            20261105,
            20261106,
            20261107,
            20261108,
            20261109,
            20261110,
        ],
        "encoding_batch_size": 32,
        "atomic_incomplete_output": True,
        "overwrite_allowed": False,
        "formal_data_generation_allowed": False,
        "historical_decisions_unchanged": True,
    }
    authorized = command.get("authorized")
    expected_status = {
        False: "implemented_before_development_authorization",
        True: "authorized_after_local_validation",
    }.get(authorized)
    if expected_status is None or command.get("status") != expected_status:
        raise RuntimeError("Development command authorization state is invalid.")
    remainder = {
        key: value
        for key, value in command.items()
        if key not in {"status", "authorized"}
    }
    if remainder != expected:
        raise RuntimeError("Development command configuration drifted.")
    return command


def validate_frozen_inputs(
    config: Mapping[str, Any],
    command: Mapping[str, Any],
) -> tuple[StrokeAutoencoder, LatentChannelStatistics, dict[str, Any]]:
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
        raise RuntimeError("Frozen autoencoder state hash does not match.")
    if total_parameter_count(model) != representation["total_parameter_count"]:
        raise RuntimeError("Frozen autoencoder parameter count changed.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen autoencoder has a trainable parameter.")
    statistics_hash = file_sha256(statistics_path)
    if statistics_hash != representation["latent_statistics_sha256"]:
        raise RuntimeError("Frozen latent-statistics file hash does not match.")
    statistics = load_latent_channel_statistics(statistics_path)
    predictor = __import__(
        "latent_stroke_dynamics.extension_training",
        fromlist=["create_patch_predictor"],
    ).create_patch_predictor("mlp", 32, (16, 16), 256)
    if parameter_count(predictor) != config["predictor"]["parameter_count"]:
        raise RuntimeError("Frozen predictor parameter count changed.")
    development_dir = Path(command["development_output_dir"])
    incomplete_dir = development_dir.with_name(development_dir.name + ".incomplete")
    formal_dir = Path(config["formal_output_dir"])
    validation = {
        "checkpoint_state_sha256": state_hash,
        "latent_statistics_file_sha256": statistics_hash,
        "autoencoder_total_parameters": total_parameter_count(model),
        "autoencoder_all_parameters_frozen": True,
        "latent_statistics_channels": int(statistics.mean.numel()),
        "latent_statistics_mean_channel_std": float(statistics.std.mean().item()),
        "predictor_parameter_count": parameter_count(predictor),
        "development_output_dir_available": not development_dir.exists(),
        "development_incomplete_dir_available": not incomplete_dir.exists(),
        "formal_output_dir_available": not formal_dir.exists(),
        "development_authorized_in_scientific_config": config["development"][
            "authorized"
        ],
        "development_authorized_in_command_config": command["authorized"],
        "formal_authorized": config["formal_reserved"]["authorized"],
        "followup_data_generated": False,
        "models_trained": False,
        "historical_decisions_unchanged": True,
        "checkpoint_metadata_selected_seed": metadata.get("selected_seed"),
        "checkpoint_metadata_best_epoch": metadata.get("best_epoch"),
    }
    return model, statistics, validation


def synthetic_objective_check() -> dict[str, float | bool]:
    generator = torch.Generator().manual_seed(8128)
    current = torch.randn(2, 4, 3, generator=generator)
    true_delta = 0.05 * torch.randn(2, 4, 3, generator=generator)
    predicted_delta = true_delta.clone().requires_grad_(True)
    true_next = current + true_delta
    candidates = torch.stack(
        (true_next, true_next + 0.25, true_next - 0.50, true_next + 0.75),
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


def build_development_examples(
    config: Mapping[str, Any],
) -> dict[str, list[TransitionExample]]:
    distribution = config["transition_distribution"]
    development = config["development"]
    examples: dict[str, list[TransitionExample]] = {}
    for name in ("train", "validation", "diagnostic_test"):
        split = development[name]
        examples[name] = build_transition_split(
            samples=int(split["samples"]),
            canvas_size=int(config["canvas_size"]),
            crowding_levels=distribution["primary_crowding"],
            seed=int(split["seed"]),
            width_choices=distribution["widths"],
            value_choices=distribution["values"],
            min_length=float(distribution["minimum_length"]),
        )
    fingerprint_sets = {
        name: {transition_fingerprint(item) for item in rows}
        for name, rows in examples.items()
    }
    names = list(fingerprint_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if fingerprint_sets[left].intersection(fingerprint_sets[right]):
                raise RuntimeError("Development split fingerprints overlap.")
    return examples


def split_metadata(examples: Mapping[str, Sequence[TransitionExample]]) -> pd.DataFrame:
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


def encode_development_payloads(
    model: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    examples: Mapping[str, Sequence[TransitionExample]],
    *,
    batch_size: int,
) -> tuple[
    dict[str, PatchFeaturePayload],
    dict[str, PatchCounterfactualPayload],
]:
    payloads: dict[str, PatchFeaturePayload] = {}
    counterfactuals: dict[str, PatchCounterfactualPayload] = {}
    for split_name, items in examples.items():
        print(f"Encoding frozen task latents for {split_name}...")
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
        candidate_sets = [build_counterfactual_set(item) for item in items]
        candidate_images = images_to_grayscale_tensor(
            [canvas for group in candidate_sets for canvas in group.canvases]
        )
        candidate_tokens = standardize_latent_tokens(
            encode_autoencoder_maps(model, candidate_images, batch_size),
            statistics,
        ).reshape(len(items), len(COUNTERFACTUAL_ORDER), 16 * 16, 32)
        counterfactuals[split_name] = build_patch_counterfactual_payload(
            items,
            candidate_tokens,
            patch_grid=(16, 16),
        )
    return payloads, counterfactuals


def setting_name(weight: float, temperature: float) -> str:
    def token(value: float) -> str:
        return format(value, "g").replace(".", "p")

    return f"ranking_l{token(weight)}_t{token(temperature)}"


def append_evaluation(
    *,
    model_name: str,
    seed: int,
    model: torch.nn.Module,
    payloads: Mapping[str, PatchFeaturePayload],
    counterfactuals: Mapping[str, PatchCounterfactualPayload],
    batch_size: int,
    ranking_weight: float | None,
    temperature: float | None,
    metric_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> None:
    for split_name in ("validation", "diagnostic_test"):
        metrics, retrieval = evaluate_patch_model(
            model_name,
            seed,
            model,
            split_name,
            payloads[split_name],
            batch_size=batch_size,
            counterfactuals=counterfactuals[split_name],
        )
        for row in metrics:
            row["ranking_weight"] = ranking_weight
            row["temperature"] = temperature
        for row in retrieval:
            row["split"] = split_name
            row["ranking_weight"] = ranking_weight
            row["temperature"] = temperature
        metric_rows.extend(metrics)
        retrieval_rows.extend(retrieval)


def evaluation_summary(
    model_name: str,
    split: str,
    metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> dict[str, Any]:
    metric_frame = metrics.loc[
        (metrics["model"] == model_name) & (metrics["split"] == split)
    ]
    retrieval_frame = retrieval.loc[
        (retrieval["model"] == model_name) & (retrieval["split"] == split)
    ]
    seed_top1 = retrieval_frame.groupby("seed")["top1_correct"].mean()
    frequencies = {
        label: int((retrieval_frame["predicted_label"] == label).sum())
        for label in COUNTERFACTUAL_ORDER
    }
    true_scores = retrieval_frame["score_true"]
    return {
        "model": model_name,
        "split": split,
        "action_region_mse": float(metric_frame["action_region_mse"].mean()),
        "top1_accuracy": float(retrieval_frame["top1_correct"].mean()),
        "top1_seed_std": float(seed_top1.std(ddof=0)),
        "mean_true_margin": float(retrieval_frame["true_margin"].mean()),
        "true_beats_shift_position_rate": float(
            (true_scores < retrieval_frame["score_shift_position"]).mean()
        ),
        "true_beats_change_width_rate": float(
            (true_scores < retrieval_frame["score_change_width"]).mean()
        ),
        "true_beats_change_intensity_rate": float(
            (true_scores < retrieval_frame["score_change_intensity"]).mean()
        ),
        "candidate_selection_counts": frequencies,
    }


def run_development_grid(
    config: dict[str, Any],
    command: dict[str, Any],
    model: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    validation: dict[str, Any],
    threads: int,
) -> dict[str, Any]:
    if config["development"]["authorized"] is not True:
        raise RuntimeError("Scientific config has not authorized development.")
    if command["authorized"] is not True:
        raise RuntimeError("Command config has not authorized development.")
    if threads > 0:
        torch.set_num_threads(threads)
    final_dir = Path(command["development_output_dir"])
    incomplete_dir = final_dir.with_name(final_dir.name + ".incomplete")
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError("Refusing to overwrite development output.")
    if not validation["formal_output_dir_available"]:
        raise FileExistsError("Formal output directory already exists.")
    incomplete_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    started = time.perf_counter()
    write_json(
        incomplete_dir / "run_config.json",
        {
            "scientific_config": config,
            "command_config": command,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "threads_requested": threads,
            },
            "validation": validation,
            "formal_data_generated": False,
            "historical_decisions_unchanged": True,
        },
    )

    print("Generating development-only transitions and counterfactuals...")
    examples = build_development_examples(config)
    split_metadata(examples).to_csv(incomplete_dir / "split_metadata.csv", index=False)
    payloads, counterfactuals = encode_development_payloads(
        model,
        statistics,
        examples,
        batch_size=int(command["encoding_batch_size"]),
    )
    oracles = {
        name: protocol_oracle_retrieval(payloads[name], counterfactuals[name])
        for name in payloads
    }
    if not all(value["passed"] for value in oracles.values()):
        raise RuntimeError("A development protocol oracle failed.")

    settings = config["predictor"]
    train_payload = payloads["train"]
    validation_payload = payloads["validation"]
    batch_size = int(settings["batch_size"])
    models: list[tuple[str, int, torch.nn.Module, float | None, float | None]] = []
    history_rows: list[dict[str, Any]] = []

    identity, mean_delta = baseline_models(train_payload, batch_size)
    models.extend(
        [
            ("identity", -1, identity, None, None),
            ("mean_delta", -1, mean_delta, None, None),
        ]
    )
    for seed in settings["model_seeds"]:
        print(f"Training matched MSE-only baseline seed {seed}...")
        fit = train_patch_predictor(
            "mlp",
            int(seed),
            train_payload,
            validation_payload,
            hidden_dim=int(settings["hidden_dim"]),
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            batch_size=batch_size,
            max_epochs=int(settings["max_epochs"]),
            patience=int(settings["patience"]),
        )
        models.append(("mse_only", int(seed), fit.model, None, None))
        for row in fit.history:
            item = dict(row)
            item["method"] = "mse_only"
            item["ranking_weight"] = None
            item["temperature"] = None
            history_rows.append(item)

    for weight in config["ranking_grid"]["lambda"]:
        for temperature in config["ranking_grid"]["temperature"]:
            name = setting_name(float(weight), float(temperature))
            for seed in settings["model_seeds"]:
                print(f"Training {name} seed {seed}...")
                fit = train_ranking_predictor(
                    int(seed),
                    train_payload,
                    counterfactuals["train"],
                    validation_payload,
                    counterfactuals["validation"],
                    hidden_dim=int(settings["hidden_dim"]),
                    learning_rate=float(settings["learning_rate"]),
                    weight_decay=float(settings["weight_decay"]),
                    batch_size=batch_size,
                    max_epochs=int(settings["max_epochs"]),
                    patience=int(settings["patience"]),
                    ranking_weight=float(weight),
                    temperature=float(temperature),
                )
                models.append(
                    (
                        name,
                        int(seed),
                        fit.model,
                        float(weight),
                        float(temperature),
                    )
                )
                history_rows.extend(fit.history)

    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for name, seed, fitted_model, weight, temperature in models:
        append_evaluation(
            model_name=name,
            seed=seed,
            model=fitted_model,
            payloads=payloads,
            counterfactuals=counterfactuals,
            batch_size=batch_size,
            ranking_weight=weight,
            temperature=temperature,
            metric_rows=metric_rows,
            retrieval_rows=retrieval_rows,
        )

    metrics = pd.DataFrame(metric_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    history = pd.DataFrame(history_rows)
    selection_rows: list[dict[str, Any]] = []
    for weight in config["ranking_grid"]["lambda"]:
        for temperature in config["ranking_grid"]["temperature"]:
            name = setting_name(float(weight), float(temperature))
            metric_frame = metrics.loc[
                (metrics["model"] == name) & (metrics["split"] == "validation")
            ]
            retrieval_frame = retrieval.loc[
                (retrieval["model"] == name)
                & (retrieval["split"] == "validation")
            ]
            selection_rows.append(
                {
                    "model": name,
                    "ranking_weight": float(weight),
                    "temperature": float(temperature),
                    "mean_validation_top1": float(
                        retrieval_frame["top1_correct"].mean()
                    ),
                    "mean_validation_true_margin": float(
                        retrieval_frame["true_margin"].mean()
                    ),
                    "mean_validation_action_region_mse": float(
                        metric_frame["action_region_mse"].mean()
                    ),
                }
            )
    selected = select_ranking_setting(selection_rows)
    selected_name = str(selected["model"])
    overfit = run_ranking_overfit_check(
        train_payload,
        counterfactuals["train"],
        hidden_dim=int(settings["hidden_dim"]),
    )

    mse_validation = evaluation_summary(
        "mse_only", "validation", metrics, retrieval
    )
    selected_validation = evaluation_summary(
        selected_name, "validation", metrics, retrieval
    )
    mse_diagnostic = evaluation_summary(
        "mse_only", "diagnostic_test", metrics, retrieval
    )
    selected_diagnostic = evaluation_summary(
        selected_name, "diagnostic_test", metrics, retrieval
    )
    metric_finite = bool(
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
    history_numeric = history.select_dtypes(include=[np.number])
    history_finite = bool(np.isfinite(history_numeric.to_numpy(dtype=float)).all())
    parameter_counts = {
        name: parameter_count(fitted_model)
        for name, _, fitted_model, _, _ in models
        if name not in {"identity", "mean_delta"}
    }
    parameters_valid = bool(
        parameter_counts
        and all(
            count == int(settings["parameter_count"])
            for count in parameter_counts.values()
        )
    )
    candidates_unique = bool(
        all(value.all_encoded_candidates_unique for value in counterfactuals.values())
    )
    implementation_integrity = bool(
        all(value["passed"] for value in oracles.values())
        and candidates_unique
        and metric_finite
        and retrieval_finite
        and history_finite
        and parameters_valid
        and overfit["loss_decreased"]
    )

    metrics.to_csv(incomplete_dir / "prediction_metrics.csv", index=False)
    retrieval.to_csv(incomplete_dir / "counterfactual_retrieval.csv", index=False)
    history.to_csv(incomplete_dir / "training_history.csv", index=False)
    pd.DataFrame(selection_rows).sort_values(
        [
            "mean_validation_top1",
            "mean_validation_true_margin",
            "mean_validation_action_region_mse",
            "ranking_weight",
            "temperature",
        ],
        ascending=[False, False, True, True, False],
    ).to_csv(incomplete_dir / "validation_selection.csv", index=False)
    write_json(incomplete_dir / "protocol_oracles.json", oracles)
    write_json(incomplete_dir / "overfit_check.json", overfit)

    summary = {
        "status": (
            "ranking_aware_development_complete"
            if implementation_integrity
            else "ranking_aware_development_completed_with_failed_integrity"
        ),
        "development_only": True,
        "selection_used_validation_only": True,
        "diagnostic_test_used_for_selection": False,
        "formal_data_generated": False,
        "formal_authorized": False,
        "selected_ranking_setting": selected,
        "mse_only_validation": mse_validation,
        "selected_ranking_validation": selected_validation,
        "mse_only_diagnostic_test": mse_diagnostic,
        "selected_ranking_diagnostic_test": selected_diagnostic,
        "diagnostic_absolute_top1_gain": (
            selected_diagnostic["top1_accuracy"]
            - mse_diagnostic["top1_accuracy"]
        ),
        "protocol_oracles": oracles,
        "all_encoded_candidates_unique": candidates_unique,
        "all_metrics_and_histories_finite": bool(
            metric_finite and retrieval_finite and history_finite
        ),
        "all_predictor_parameter_counts_valid": parameters_valid,
        "ranking_overfit_check": overfit,
        "implementation_integrity_passed": implementation_integrity,
        "elapsed_seconds": time.perf_counter() - started,
        "do_not_rerun_development": True,
        "historical_decisions_unchanged": True,
    }
    write_json(incomplete_dir / "development_summary.json", summary)
    os.replace(incomplete_dir, final_dir)
    return summary


def main() -> None:
    args = parse_args()
    config = load_ranking_config(CONFIG)
    command = load_command_config()
    model, statistics, validation = validate_frozen_inputs(config, command)
    synthetic = synthetic_objective_check()
    if not synthetic["gradient_finite"]:
        raise RuntimeError("Synthetic ranking objective gradient is non-finite.")

    if args.validate_only:
        result = {
            "status": (
                "ranking_latent_development_runner_valid_authorized"
                if config["development"]["authorized"] is True
                and command["authorized"] is True
                else "ranking_latent_development_runner_valid_unauthorized"
            ),
            **validation,
            "ranking_grid": config["ranking_grid"],
            "development_seeds": command["development_seeds"],
            "formal_seeds_reserved": command["formal_seeds_reserved"],
            "synthetic_objective_check": synthetic,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    summary = run_development_grid(
        config,
        command,
        model,
        statistics,
        validation,
        args.threads,
    )
    print("\nRanking-aware latent development grid complete\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved outputs to: {Path(command['development_output_dir']).resolve()}")
    print("Do not rerun this development grid.")
    print("Formal data remain untouched and unauthorized.")


if __name__ == "__main__":
    main()
