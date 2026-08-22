#!/usr/bin/env python3
"""Validate or run the frozen formal ranking-aware latent comparison."""

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
    create_patch_predictor,
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
from latent_stroke_dynamics.ranking_development_adjudication import (
    METRIC_COLUMNS,
    RETRIEVAL_COLUMNS,
    adjudicate_history_finiteness,
    finite_columns,
)
from latent_stroke_dynamics.ranking_formal import classify_formal_ranking_result
from latent_stroke_dynamics.ranking_latent import (
    file_sha256,
    load_latent_channel_statistics,
    load_ranking_config,
    ranking_aware_objective,
)
from latent_stroke_dynamics.ranking_training import (
    protocol_oracle_retrieval,
    run_ranking_overfit_check,
    train_ranking_predictor,
)
from latent_stroke_dynamics.representation_extension import (
    LatentChannelStatistics,
    StrokeAutoencoder,
    images_to_grayscale_tensor,
    standardize_latent_tokens,
)


SCIENTIFIC_CONFIG = Path("configs/ranking-aware-latent-2026-08-22.json")
SELECTED_SETTING = Path(
    "configs/ranking-aware-latent-selected-setting-2026-08-22.json"
)
COMMAND_CONFIG = Path(
    "configs/ranking-aware-latent-formal-command-2026-08-22.json"
)
DEVELOPMENT_ADJUDICATION = Path(
    "results/ranking-aware-latent/development-protocol-adjudication.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-formal-comparison", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    if int(args.validate_only) + int(args.run_formal_comparison) != 1:
        parser.error("Choose exactly one validation or formal-run mode.")
    if args.threads < 0:
        parser.error("--threads cannot be negative.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_selected_setting() -> dict[str, Any]:
    selected = read_json(SELECTED_SETTING)
    expected = {
        "status": "frozen_after_validated_development_before_formal_data",
        "source": str(DEVELOPMENT_ADJUDICATION),
        "selection_used_validation_only": True,
        "diagnostic_test_used_for_selection": False,
        "ranking_weight": 1.0,
        "temperature": 0.05,
        "model": "ranking_l1_t0p05",
        "mean_validation_top1": 0.7083333333333334,
        "mean_validation_true_margin": 0.0014645709306932986,
        "mean_validation_action_region_mse": 0.6157332132570446,
        "development_diagnostic_top1": 0.7604166666666666,
        "development_diagnostic_absolute_gain_over_mse": 0.4895833333333333,
        "source_artifact_sha256": {
            "history": "33e52a8ed1e7aeeb82903ea624ad4ab9a56c5855d5aa4440e9715b0d19809089",
            "metrics": "4dd5cf65402cc24a5e58875810a68d7c45323e9001aff026bf33b345f8b2ec1c",
            "retrieval": "d69e217a538a49154ebbbda0ba57817304284d3a98e0cb24d5f6c128446b073c",
            "summary": "abc6e4c9b3ffcedbaa9d0f6a824e05557c7e128c498cbfa783453984a25d8b1d",
        },
        "formal_authorized": False,
        "historical_decisions_unchanged": True,
    }
    if selected != expected:
        raise RuntimeError("Selected ranking setting drifted from its frozen file.")
    return selected


def load_command_config() -> dict[str, Any]:
    command = read_json(COMMAND_CONFIG)
    expected_remainder = {
        "script": "experiments/14_ranking_aware_latent_formal.py",
        "validation_command": (
            "python experiments/14_ranking_aware_latent_formal.py --validate-only"
        ),
        "run_command": (
            "python experiments/14_ranking_aware_latent_formal.py "
            "--run-formal-comparison"
        ),
        "scientific_config": str(SCIENTIFIC_CONFIG),
        "selected_setting": str(SELECTED_SETTING),
        "development_adjudication": str(DEVELOPMENT_ADJUDICATION),
        "development_split_metadata": (
            "outputs/ranking-aware-latent-development-2026-08-22/"
            "split_metadata.csv"
        ),
        "formal_output_dir": "outputs/ranking-aware-latent-formal-2026-08-22",
        "formal_seeds": [
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
        "single_formal_run": True,
        "historical_decisions_unchanged": True,
    }
    authorized = command.get("authorized")
    expected_status = {
        False: "implemented_before_formal_authorization",
        True: "authorized_after_local_validation",
    }.get(authorized)
    if expected_status is None or command.get("status") != expected_status:
        raise RuntimeError("Formal command authorization state is invalid.")
    remainder = {
        key: value
        for key, value in command.items()
        if key not in {"status", "authorized"}
    }
    if remainder != expected_remainder:
        raise RuntimeError("Formal command configuration drifted.")
    return command


def validate_frozen_inputs(
    config: Mapping[str, Any],
    command: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> tuple[StrokeAutoencoder, LatentChannelStatistics, dict[str, Any]]:
    development = read_json(DEVELOPMENT_ADJUDICATION)
    if development.get("written_protocol_implementation_integrity_passed") is not True:
        raise RuntimeError("Archived development integrity did not pass.")
    if development.get("formal_data_generated") is not False:
        raise RuntimeError("Archived development reports formal data generation.")
    archived_selected = development.get("selected_ranking_setting")
    if not isinstance(archived_selected, Mapping):
        raise RuntimeError("Archived development lacks selected ranking setting.")
    if float(archived_selected.get("ranking_weight", -1)) != float(
        selected["ranking_weight"]
    ) or float(archived_selected.get("temperature", -1)) != float(
        selected["temperature"]
    ):
        raise RuntimeError("Selected setting differs from development adjudication.")

    representation = config["frozen_representation"]
    checkpoint_path = Path(representation["checkpoint_path"])
    statistics_path = Path(representation["latent_statistics_path"])
    development_metadata = Path(command["development_split_metadata"])
    for path in (checkpoint_path, statistics_path, development_metadata):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen prerequisite: {path}")
    model, metadata = load_autoencoder_checkpoint(checkpoint_path)
    state_hash = model_state_sha256(model)
    if state_hash != representation["checkpoint_sha256"]:
        raise RuntimeError("Frozen autoencoder state hash changed.")
    if total_parameter_count(model) != representation["total_parameter_count"]:
        raise RuntimeError("Frozen autoencoder parameter count changed.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen autoencoder has a trainable parameter.")
    statistics_hash = file_sha256(statistics_path)
    if statistics_hash != representation["latent_statistics_sha256"]:
        raise RuntimeError("Frozen latent-statistics hash changed.")
    statistics = load_latent_channel_statistics(statistics_path)
    predictor = create_patch_predictor("mlp", 32, (16, 16), 256)
    if parameter_count(predictor) != config["predictor"]["parameter_count"]:
        raise RuntimeError("Frozen predictor parameter count changed.")

    final_dir = Path(command["formal_output_dir"])
    incomplete_dir = final_dir.with_name(final_dir.name + ".incomplete")
    validation = {
        "checkpoint_state_sha256": state_hash,
        "latent_statistics_file_sha256": statistics_hash,
        "development_adjudication_sha256": file_sha256(DEVELOPMENT_ADJUDICATION),
        "development_split_metadata_sha256": file_sha256(development_metadata),
        "selected_ranking_weight": float(selected["ranking_weight"]),
        "selected_temperature": float(selected["temperature"]),
        "autoencoder_total_parameters": total_parameter_count(model),
        "autoencoder_all_parameters_frozen": True,
        "predictor_parameter_count": parameter_count(predictor),
        "formal_authorized_in_scientific_config": config["formal_reserved"][
            "authorized"
        ],
        "formal_authorized_in_command_config": command["authorized"],
        "formal_output_dir_available": not final_dir.exists(),
        "formal_incomplete_dir_available": not incomplete_dir.exists(),
        "formal_data_generated": False,
        "models_trained": False,
        "historical_decisions_unchanged": True,
        "checkpoint_metadata_selected_seed": metadata.get("selected_seed"),
        "checkpoint_metadata_best_epoch": metadata.get("best_epoch"),
    }
    return model, statistics, validation


def synthetic_objective_check(weight: float, temperature: float) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(1414)
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
        ranking_weight=weight,
        temperature=temperature,
    )
    losses["total"].backward()
    return {
        "total": float(losses["total"].detach().item()),
        "balanced_mse": float(losses["balanced_mse"].detach().item()),
        "ranking_cross_entropy": float(
            losses["ranking_cross_entropy"].detach().item()
        ),
        "gradient_finite": bool(
            predicted_delta.grad is not None
            and torch.isfinite(predicted_delta.grad).all()
        ),
    }


def build_formal_examples(config: Mapping[str, Any]) -> dict[str, list[TransitionExample]]:
    distribution = config["transition_distribution"]
    formal = config["formal_reserved"]
    specifications: dict[str, dict[str, Any]] = {
        "train": {
            **formal["train"],
            "crowding": distribution["primary_crowding"],
            "widths": distribution["widths"],
            "values": distribution["values"],
        },
        "validation": {
            **formal["validation"],
            "crowding": distribution["primary_crowding"],
            "widths": distribution["widths"],
            "values": distribution["values"],
        },
        "test": {
            **formal["test"],
            "crowding": distribution["primary_crowding"],
            "widths": distribution["widths"],
            "values": distribution["values"],
        },
        "unseen_width_5": {
            **formal["unseen_width_5"],
            "crowding": distribution["primary_crowding"],
            "widths": [5],
            "values": distribution["values"],
        },
        "unseen_intensities": {
            **formal["unseen_intensities"],
            "crowding": distribution["primary_crowding"],
            "widths": distribution["widths"],
        },
        "crowding_30": {
            **formal["crowding_30"],
            "crowding": [30],
            "widths": distribution["widths"],
            "values": distribution["values"],
        },
        "crowding_60": {
            **formal["crowding_60"],
            "crowding": [60],
            "widths": distribution["widths"],
            "values": distribution["values"],
        },
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
        for name, spec in specifications.items()
    }
    sets = {
        name: {transition_fingerprint(item) for item in rows}
        for name, rows in examples.items()
    }
    names = list(sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if sets[left].intersection(sets[right]):
                raise RuntimeError("Formal split fingerprints overlap.")
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


def encode_payloads(
    model: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    examples: Mapping[str, Sequence[TransitionExample]],
    *,
    batch_size: int,
) -> tuple[dict[str, PatchFeaturePayload], dict[str, PatchCounterfactualPayload]]:
    payloads: dict[str, PatchFeaturePayload] = {}
    counterfactuals: dict[str, PatchCounterfactualPayload] = {}
    for split_name, items in examples.items():
        print(f"Encoding frozen task latents for formal {split_name}...")
        current = images_to_grayscale_tensor([item.current for item in items])
        next_images = images_to_grayscale_tensor([item.next_canvas for item in items])
        current_tokens = standardize_latent_tokens(
            encode_autoencoder_maps(model, current, batch_size), statistics
        )
        next_tokens = standardize_latent_tokens(
            encode_autoencoder_maps(model, next_images, batch_size), statistics
        )
        payloads[split_name] = build_patch_feature_payload(
            items, current_tokens, next_tokens, patch_grid=(16, 16)
        )
        candidate_sets = [build_counterfactual_set(item) for item in items]
        candidate_images = images_to_grayscale_tensor(
            [canvas for group in candidate_sets for canvas in group.canvases]
        )
        candidate_tokens = standardize_latent_tokens(
            encode_autoencoder_maps(model, candidate_images, batch_size), statistics
        ).reshape(len(items), len(COUNTERFACTUAL_ORDER), 16 * 16, 32)
        counterfactuals[split_name] = build_patch_counterfactual_payload(
            items, candidate_tokens, patch_grid=(16, 16)
        )
    return payloads, counterfactuals


def append_evaluation(
    model_name: str,
    seed: int,
    model: torch.nn.Module,
    splits: Sequence[str],
    payloads: Mapping[str, PatchFeaturePayload],
    counterfactuals: Mapping[str, PatchCounterfactualPayload],
    batch_size: int,
    metric_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> None:
    for split in splits:
        metrics, retrieval = evaluate_patch_model(
            model_name,
            seed,
            model,
            split,
            payloads[split],
            batch_size=batch_size,
            counterfactuals=counterfactuals[split],
        )
        for row in retrieval:
            row["split"] = split
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
        "candidate_selection_counts": {
            label: int((retrieval_frame["predicted_label"] == label).sum())
            for label in COUNTERFACTUAL_ORDER
        },
    }


def run_formal(
    config: dict[str, Any],
    command: dict[str, Any],
    selected: dict[str, Any],
    model: StrokeAutoencoder,
    statistics: LatentChannelStatistics,
    validation: dict[str, Any],
    threads: int,
) -> dict[str, Any]:
    if config["formal_reserved"]["authorized"] is not True:
        raise RuntimeError("Scientific config has not authorized formal data.")
    if command["authorized"] is not True:
        raise RuntimeError("Formal command has not been authorized.")
    if threads > 0:
        torch.set_num_threads(threads)
    final_dir = Path(command["formal_output_dir"])
    incomplete_dir = final_dir.with_name(final_dir.name + ".incomplete")
    if final_dir.exists() or incomplete_dir.exists():
        raise FileExistsError("Refusing to overwrite formal output.")
    incomplete_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_dir.mkdir()
    started = time.perf_counter()
    write_json(
        incomplete_dir / "run_config.json",
        {
            "scientific_config": config,
            "selected_setting": selected,
            "command_config": command,
            "validation": validation,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "threads_requested": threads,
            },
            "historical_decisions_unchanged": True,
        },
    )

    print("Generating the single untouched formal and stress splits...")
    examples = build_formal_examples(config)
    metadata = split_metadata(examples)
    development_metadata = pd.read_csv(command["development_split_metadata"])
    if set(metadata["fingerprint"]).intersection(
        set(development_metadata["fingerprint"])
    ):
        raise RuntimeError("Formal fingerprints overlap development fingerprints.")
    metadata.to_csv(incomplete_dir / "split_metadata.csv", index=False)
    payloads, counterfactuals = encode_payloads(
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
        raise RuntimeError("A formal protocol oracle failed.")

    settings = config["predictor"]
    batch_size = int(settings["batch_size"])
    train_payload = payloads["train"]
    validation_payload = payloads["validation"]
    identity, mean_delta = baseline_models(train_payload, batch_size)
    models: list[tuple[str, int, torch.nn.Module]] = [
        ("identity", -1, identity),
        ("mean_delta", -1, mean_delta),
    ]
    history_rows: list[dict[str, Any]] = []
    checkpoints = incomplete_dir / "checkpoints"
    checkpoints.mkdir()

    for seed in settings["model_seeds"]:
        print(f"Training formal MSE-only seed {seed}...")
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
        models.append(("mse_only", int(seed), fit.model))
        for row in fit.history:
            item = dict(row)
            item["method"] = "mse_only"
            item["ranking_weight"] = None
            item["temperature"] = None
            history_rows.append(item)
        torch.save(
            {"method": "mse_only", "seed": int(seed), "state_dict": fit.model.state_dict()},
            checkpoints / f"mse_only_seed{seed}.pt",
        )

    for seed in settings["model_seeds"]:
        print(f"Training formal ranking-aware seed {seed}...")
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
            ranking_weight=float(selected["ranking_weight"]),
            temperature=float(selected["temperature"]),
        )
        models.append(("ranking_aware", int(seed), fit.model))
        history_rows.extend(fit.history)
        torch.save(
            {
                "method": "ranking_aware",
                "seed": int(seed),
                "ranking_weight": float(selected["ranking_weight"]),
                "temperature": float(selected["temperature"]),
                "state_dict": fit.model.state_dict(),
            },
            checkpoints / f"ranking_aware_seed{seed}.pt",
        )

    evaluation_splits = [name for name in payloads if name != "train"]
    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    for name, seed, fitted_model in models:
        append_evaluation(
            name,
            seed,
            fitted_model,
            evaluation_splits,
            payloads,
            counterfactuals,
            batch_size,
            metric_rows,
            retrieval_rows,
        )
    metrics = pd.DataFrame(metric_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    history = pd.DataFrame(history_rows)
    history_check = adjudicate_history_finiteness(history)
    overfit = run_ranking_overfit_check(
        train_payload,
        counterfactuals["train"],
        hidden_dim=int(settings["hidden_dim"]),
        ranking_weight=float(selected["ranking_weight"]),
        temperature=float(selected["temperature"]),
    )

    test_summaries = {
        name: evaluation_summary(name, "test", metrics, retrieval)
        for name in ("identity", "mean_delta", "mse_only", "ranking_aware")
    }
    ranking_error = test_summaries["ranking_aware"]["action_region_mse"]
    identity_error = test_summaries["identity"]["action_region_mse"]
    mean_error = test_summaries["mean_delta"]["action_region_mse"]
    improvement_identity = 1.0 - ranking_error / max(identity_error, 1e-12)
    improvement_mean = 1.0 - ranking_error / max(mean_error, 1e-12)

    crowding_improvements: dict[str, float] = {}
    positive_every_crowding = True
    test_metrics = metrics.loc[metrics["split"] == "test"]
    for crowding in config["transition_distribution"]["primary_crowding"]:
        identity_value = float(
            test_metrics.loc[
                (test_metrics["model"] == "identity")
                & (test_metrics["crowding"] == crowding),
                "action_region_mse",
            ].mean()
        )
        ranking_value = float(
            test_metrics.loc[
                (test_metrics["model"] == "ranking_aware")
                & (test_metrics["crowding"] == crowding),
                "action_region_mse",
            ].mean()
        )
        improvement = 1.0 - ranking_value / max(identity_value, 1e-12)
        crowding_improvements[f"crowding_{crowding}_improvement"] = improvement
        positive_every_crowding = bool(positive_every_crowding and improvement > 0)

    ranking_seed_errors = test_metrics.loc[
        test_metrics["model"] == "ranking_aware"
    ].groupby("seed")["action_region_mse"].mean()
    all_ranking_seeds_beat_identity = bool((ranking_seed_errors < identity_error).all())
    metrics_finite = finite_columns(metrics, METRIC_COLUMNS)
    retrieval_finite = finite_columns(retrieval, RETRIEVAL_COLUMNS)
    candidates_unique = bool(
        all(value.all_encoded_candidates_unique for value in counterfactuals.values())
    )
    parameter_counts_valid = bool(
        all(
            parameter_count(fitted_model) == int(settings["parameter_count"])
            for name, _, fitted_model in models
            if name in {"mse_only", "ranking_aware"}
        )
    )
    implementation_integrity = bool(
        metrics_finite
        and retrieval_finite
        and history_check.get("passed") is True
        and candidates_unique
        and parameter_counts_valid
        and overfit["loss_decreased"]
        and all(value["passed"] for value in oracles.values())
        and all_ranking_seeds_beat_identity
    )
    decision = classify_formal_ranking_result(
        ranking_retrieval=test_summaries["ranking_aware"]["top1_accuracy"],
        mse_retrieval=test_summaries["mse_only"]["top1_accuracy"],
        improvement_vs_identity=improvement_identity,
        improvement_vs_mean_delta=improvement_mean,
        positive_every_primary_crowding=positive_every_crowding,
        all_ranking_seeds_beat_identity=all_ranking_seeds_beat_identity,
        oracle_retrieval=oracles["test"]["top1_accuracy"],
        implementation_integrity=implementation_integrity,
        thresholds=config["classification"],
    )
    stress_summaries = {
        split: {
            method: evaluation_summary(method, split, metrics, retrieval)
            for method in ("mse_only", "ranking_aware")
        }
        for split in evaluation_splits
        if split not in {"validation", "test"}
    }

    metrics.to_csv(incomplete_dir / "prediction_metrics.csv", index=False)
    retrieval.to_csv(incomplete_dir / "counterfactual_retrieval.csv", index=False)
    history.to_csv(incomplete_dir / "training_history.csv", index=False)
    write_json(incomplete_dir / "protocol_oracles.json", oracles)
    write_json(incomplete_dir / "ranking_overfit_check.json", overfit)
    write_json(incomplete_dir / "stress_summaries.json", stress_summaries)

    summary = {
        "status": "formal_ranking_comparison_complete",
        "formal_decision": decision,
        "test_summaries": test_summaries,
        "stress_summaries": stress_summaries,
        "selected_ranking_setting": {
            "ranking_weight": float(selected["ranking_weight"]),
            "temperature": float(selected["temperature"]),
        },
        "primary_crowding_improvements": crowding_improvements,
        "positive_improvement_every_primary_crowding": positive_every_crowding,
        "all_ranking_seeds_beat_identity": all_ranking_seeds_beat_identity,
        "protocol_oracles": oracles,
        "all_encoded_candidates_unique": candidates_unique,
        "all_prediction_and_retrieval_metrics_finite": bool(
            metrics_finite and retrieval_finite
        ),
        "method_applicable_histories_finite": history_check,
        "all_predictor_parameter_counts_valid": parameter_counts_valid,
        "ranking_overfit_check": overfit,
        "implementation_integrity_passed": implementation_integrity,
        "test_used_for_training_early_stopping_or_selection": False,
        "stress_used_for_primary_decision": False,
        "formal_seeds": command["formal_seeds"],
        "single_formal_run": True,
        "elapsed_seconds": time.perf_counter() - started,
        "do_not_rerun_or_retune": True,
        "historical_decisions_unchanged": True,
    }
    write_json(incomplete_dir / "formal_summary.json", summary)
    os.replace(incomplete_dir, final_dir)
    return summary


def main() -> None:
    args = parse_args()
    config = load_ranking_config(SCIENTIFIC_CONFIG)
    selected = load_selected_setting()
    command = load_command_config()
    model, statistics, validation = validate_frozen_inputs(config, command, selected)
    synthetic = synthetic_objective_check(
        float(selected["ranking_weight"]),
        float(selected["temperature"]),
    )
    if not synthetic["gradient_finite"]:
        raise RuntimeError("Formal synthetic ranking gradient is non-finite.")

    if args.validate_only:
        result = {
            "status": (
                "ranking_latent_formal_runner_valid_authorized"
                if config["formal_reserved"]["authorized"] is True
                and command["authorized"] is True
                else "ranking_latent_formal_runner_valid_unauthorized"
            ),
            **validation,
            "formal_seeds": command["formal_seeds"],
            "synthetic_objective_check": synthetic,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    summary = run_formal(
        config,
        command,
        selected,
        model,
        statistics,
        validation,
        args.threads,
    )
    print("\nFrozen formal ranking-aware comparison complete\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved outputs to: {Path(command['formal_output_dir']).resolve()}")
    print("Do not rerun or retune this formal comparison.")


if __name__ == "__main__":
    main()
