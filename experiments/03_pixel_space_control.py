"""Paired action-conditioned pixel-space control for the Gate 2 result."""

from __future__ import annotations

import argparse
import copy
import json
import random
from math import hypot
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from latent_stroke_dynamics.gate2 import (
    COUNTERFACTUAL_ORDER,
    PRIMARY_CROWDING,
    PRIMARY_VALUES,
    PRIMARY_WIDTHS,
    TransitionExample,
    build_transition_split,
    parameter_count,
    transition_fingerprint,
)
from latent_stroke_dynamics.pixel_control import (
    PIXEL_INPUT_DIM,
    ExactCompositorPixelDeltaPredictor,
    IdentityPixelDeltaPredictor,
    LinearPixelDeltaPredictor,
    MLPPixelDeltaPredictor,
    MeanPixelDeltaPredictor,
    balanced_pixel_mse,
    build_pixel_counterfactual_tensors,
    build_pixel_tensors,
    pixel_counterfactual_retrieval,
    pixel_error_metrics,
)
from latent_stroke_dynamics.retrieval_diagnostics import (
    CANDIDATE_NAMES,
    summarize_retrieval,
    summarize_retrieval_by,
    summarize_retrieval_families,
)


FORMAL_CANVAS_SIZE = 64
FORMAL_SPLIT_SIZES = {"train": 1000, "validation": 200, "test": 300}
FORMAL_DATA_SEEDS = {"train": 20260824, "validation": 20260825, "test": 20260826}
FORMAL_STRESS_SEED = 20260827
FORMAL_MODEL_SEEDS = {11, 22, 33}
DEVELOPMENT_DATA_SEEDS = {
    "train": 20260830,
    "validation": 20260831,
    "test": 20260832,
    "stress": 20260833,
}
METRIC_COLUMNS = [
    "full_pixel_mse",
    "action_region_mse",
    "outside_region_mse",
    "action_region_mae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canvas-size", type=int, default=FORMAL_CANVAS_SIZE)
    parser.add_argument("--crowding", nargs="+", type=int, default=list(PRIMARY_CROWDING))
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--stress-samples", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["train"])
    parser.add_argument("--val-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["validation"])
    parser.add_argument("--test-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["test"])
    parser.add_argument("--stress-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["stress"])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[11])
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--overfit-examples", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=30)
    parser.add_argument("--overfit-learning-rate", type=float, default=5e-3)
    parser.add_argument(
        "--train-device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--paired-control-run",
        action="store_true",
        help="Use only for the single frozen paired control command.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pixel-control-smoke-1"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "hidden_dim": args.hidden_dim,
        "overfit_examples": args.overfit_examples,
        "overfit_steps": args.overfit_steps,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.stress_samples < 0:
        raise ValueError("--stress-samples cannot be negative.")
    if any(level < 0 for level in args.crowding):
        raise ValueError("Crowding levels cannot be negative.")
    if args.learning_rate <= 0 or args.overfit_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")
    if args.weight_decay < 0:
        raise ValueError("Weight decay cannot be negative.")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.train_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if args.train_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")


def _split_spec(
    name: str,
    samples: int,
    seed: int,
    crowding: Sequence[int],
    widths: Sequence[int],
    values: Sequence[int],
    canvas_size: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "samples": int(samples),
        "seed": int(seed),
        "crowding": [int(value) for value in crowding],
        "widths": [int(value) for value in widths],
        "values": [int(value) for value in values],
        "canvas_size": int(canvas_size),
        "min_length": 0.20,
    }


def _build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [
        _split_spec(
            "train",
            args.train_samples,
            args.train_seed,
            args.crowding,
            PRIMARY_WIDTHS,
            PRIMARY_VALUES,
            args.canvas_size,
        ),
        _split_spec(
            "validation",
            args.val_samples,
            args.val_seed,
            args.crowding,
            PRIMARY_WIDTHS,
            PRIMARY_VALUES,
            args.canvas_size,
        ),
        _split_spec(
            "test",
            args.test_samples,
            args.test_seed,
            args.crowding,
            PRIMARY_WIDTHS,
            PRIMARY_VALUES,
            args.canvas_size,
        ),
    ]
    if args.stress_samples:
        specs.extend(
            [
                _split_spec(
                    "stress_width",
                    args.stress_samples,
                    args.stress_seed,
                    args.crowding,
                    (5,),
                    PRIMARY_VALUES,
                    args.canvas_size,
                ),
                _split_spec(
                    "stress_intensity",
                    args.stress_samples,
                    args.stress_seed + 1,
                    args.crowding,
                    PRIMARY_WIDTHS,
                    (16, 80, 176),
                    args.canvas_size,
                ),
                _split_spec(
                    "stress_crowding",
                    args.stress_samples,
                    args.stress_seed + 2,
                    (10,),
                    PRIMARY_WIDTHS,
                    PRIMARY_VALUES,
                    args.canvas_size,
                ),
            ]
        )
    return specs


def _generate_examples(spec: dict[str, Any]) -> list[TransitionExample]:
    return build_transition_split(
        samples=spec["samples"],
        canvas_size=spec["canvas_size"],
        crowding_levels=spec["crowding"],
        seed=spec["seed"],
        width_choices=spec["widths"],
        value_choices=spec["values"],
        min_length=spec["min_length"],
    )


def _validate_split_separation(
    examples_by_name: dict[str, list[TransitionExample]],
) -> None:
    fingerprints = {
        name: {transition_fingerprint(example) for example in examples}
        for name, examples in examples_by_name.items()
    }
    names = list(fingerprints)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = fingerprints[left].intersection(fingerprints[right])
            if overlap:
                raise RuntimeError(
                    f"Detected {len(overlap)} duplicated transitions between "
                    f"{left} and {right}."
                )


def _prepare_payloads(
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    specs = _build_specs(args)
    examples_by_name = {spec["name"]: _generate_examples(spec) for spec in specs}
    _validate_split_separation(examples_by_name)
    payloads: dict[str, dict[str, Any]] = {}
    for spec in specs:
        examples = examples_by_name[spec["name"]]
        tensors = build_pixel_tensors(examples, args.canvas_size)
        payloads[spec["name"]] = {
            "spec": spec,
            "current": tensors.current,
            "next": tensors.next_canvas,
            "actions": tensors.actions,
            "action_masks": tensors.action_masks,
            "crowding": torch.tensor(
                [example.crowding for example in examples], dtype=torch.int64
            ),
            "width": torch.tensor(
                [example.stroke.width for example in examples], dtype=torch.int64
            ),
            "value": torch.tensor(
                [example.stroke.value for example in examples], dtype=torch.int64
            ),
            "length": torch.tensor(
                [
                    hypot(
                        example.stroke.x1 - example.stroke.x0,
                        example.stroke.y1 - example.stroke.y0,
                    )
                    for example in examples
                ],
                dtype=torch.float32,
            ),
            "fingerprints": [transition_fingerprint(example) for example in examples],
        }

    test_counterfactuals = build_pixel_counterfactual_tensors(
        examples_by_name["test"],
        args.canvas_size,
    )
    counter_payload = {
        "candidate_next": test_counterfactuals.candidate_next,
        "union_masks": test_counterfactuals.union_masks,
        "all_candidates_unique": test_counterfactuals.all_candidates_unique,
        "fingerprints": list(payloads["test"]["fingerprints"]),
    }
    return payloads, counter_payload, specs


def _device(name: str) -> torch.device:
    return torch.device(name)


def _batch_tensors(
    payload: dict[str, Any],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = payload["current"][indices].to(device=device, dtype=torch.float32)
    next_canvas = payload["next"][indices].to(device=device, dtype=torch.float32)
    actions = payload["actions"][indices].to(device=device, dtype=torch.float32)
    masks = payload["action_masks"][indices].to(device=device, dtype=torch.float32)
    return current, next_canvas, actions, masks


def _training_mean_delta(payload: dict[str, Any]) -> torch.Tensor:
    return (payload["next"].float() - payload["current"].float()).mean(dim=0)


def _create_model(family: str, hidden_dim: int) -> nn.Module:
    if family == "linear":
        return LinearPixelDeltaPredictor()
    if family == "mlp":
        return MLPPixelDeltaPredictor(hidden_dim)
    raise ValueError(f"Unknown model family: {family}")


def _mean_training_loss(
    model: nn.Module,
    payload: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(payload["current"]), batch_size):
            stop = min(start + batch_size, len(payload["current"]))
            indices = torch.arange(start, stop)
            current, next_canvas, actions, masks = _batch_tensors(
                payload, indices, device
            )
            true_delta = next_canvas - current
            per_example = balanced_pixel_mse(
                model(current, actions, masks),
                true_delta,
                masks,
                reduction="none",
            )
            total += float(per_example.sum())
            count += len(indices)
    return total / count


def _train_model(
    family: str,
    seed: int,
    train_payload: dict[str, Any],
    validation_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[nn.Module, list[dict[str, float | int | str]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(args.train_device)
    model = _create_model(family, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    best_validation = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, float | int | str]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_payload["current"]), generator=generator)
        train_total = 0.0
        train_count = 0
        for start in range(0, len(permutation), args.train_batch_size):
            indices = permutation[start : start + args.train_batch_size]
            current, next_canvas, actions, masks = _batch_tensors(
                train_payload, indices, device
            )
            true_delta = next_canvas - current
            loss = balanced_pixel_mse(
                model(current, actions, masks), true_delta, masks
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(indices)
            train_count += len(indices)

        validation_loss = _mean_training_loss(
            model,
            validation_payload,
            args.train_batch_size,
            device,
        )
        train_loss = train_total / train_count
        history.append(
            {
                "model": family,
                "seed": seed,
                "epoch": epoch,
                "train_balanced_mse": train_loss,
                "validation_balanced_mse": validation_loss,
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  {family}/{seed} epoch {epoch}: "
                f"train={train_loss:.6g}, validation={validation_loss:.6g}"
            )
        if validation_loss < best_validation - 1e-12:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"  {family}/{seed} early-stopped at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model.cpu(), history


def _run_overfit_check(
    train_payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float | int | bool]:
    torch.manual_seed(31415)
    device = _device(args.train_device)
    count = min(args.overfit_examples, len(train_payload["current"]))
    indices = torch.arange(count)
    current, next_canvas, actions, masks = _batch_tensors(
        train_payload, indices, device
    )
    true_delta = next_canvas - current
    model = MLPPixelDeltaPredictor(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.overfit_learning_rate)
    with torch.no_grad():
        initial = float(
            balanced_pixel_mse(model(current, actions, masks), true_delta, masks)
        )
    for _ in range(args.overfit_steps):
        prediction = model(current, actions, masks)
        loss = balanced_pixel_mse(prediction, true_delta, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(
            balanced_pixel_mse(model(current, actions, masks), true_delta, masks)
        )
    return {
        "examples": count,
        "steps": args.overfit_steps,
        "initial_balanced_mse": initial,
        "final_balanced_mse": final,
        "relative_reduction": 1.0 - final / max(initial, 1e-12),
        "loss_decreased": final < initial,
    }


def _evaluate_model(
    model_name: str,
    seed: int,
    model: nn.Module,
    split_name: str,
    payload: dict[str, Any],
    batch_size: int,
    device: torch.device,
    counter_payload: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = model.to(device).eval()
    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(payload["current"]), batch_size):
            stop = min(start + batch_size, len(payload["current"]))
            indices = torch.arange(start, stop)
            current, next_canvas, actions, masks = _batch_tensors(
                payload, indices, device
            )
            true_delta = next_canvas - current
            predicted_delta = model(current, actions, masks)
            metrics = pixel_error_metrics(
                current, predicted_delta, true_delta, masks
            )
            metrics_cpu = {name: values.cpu() for name, values in metrics.items()}
            for offset, sample_index in enumerate(range(start, stop)):
                row: dict[str, Any] = {
                    "model": model_name,
                    "seed": seed,
                    "split": split_name,
                    "sample_id": sample_index,
                    "fingerprint": payload["fingerprints"][sample_index],
                    "crowding": int(payload["crowding"][sample_index]),
                    "stroke_width": int(payload["width"][sample_index]),
                    "stroke_value": int(payload["value"][sample_index]),
                    "stroke_length": float(payload["length"][sample_index]),
                }
                row.update(
                    {
                        metric_name: float(metric_values[offset])
                        for metric_name, metric_values in metrics_cpu.items()
                    }
                )
                metric_rows.append(row)

            if counter_payload is not None:
                predicted_next = (current + predicted_delta).clamp(0.0, 1.0)
                candidates = counter_payload["candidate_next"][start:stop].to(
                    device=device,
                    dtype=torch.float32,
                )
                union_masks = counter_payload["union_masks"][start:stop].to(
                    device=device,
                    dtype=torch.float32,
                )
                retrieval = pixel_counterfactual_retrieval(
                    predicted_next, candidates, union_masks
                )
                scores = retrieval["scores"].cpu()
                predicted_index = retrieval["predicted_index"].cpu()
                top1 = retrieval["top1_correct"].cpu()
                margins = retrieval["true_margin"].cpu()
                for offset, sample_index in enumerate(range(start, stop)):
                    row = {
                        "model": model_name,
                        "seed": seed,
                        "sample_id": sample_index,
                        "fingerprint": payload["fingerprints"][sample_index],
                        "crowding": int(payload["crowding"][sample_index]),
                        "stroke_width": int(payload["width"][sample_index]),
                        "stroke_value": int(payload["value"][sample_index]),
                        "stroke_length": float(payload["length"][sample_index]),
                        "predicted_index": int(predicted_index[offset]),
                        "predicted_label": COUNTERFACTUAL_ORDER[
                            int(predicted_index[offset])
                        ],
                        "top1_correct": bool(top1[offset]),
                        "true_margin": float(margins[offset]),
                    }
                    for candidate_index, candidate_name in enumerate(
                        COUNTERFACTUAL_ORDER
                    ):
                        row[f"score_{candidate_name}"] = float(
                            scores[offset, candidate_index]
                        )
                    retrieval_rows.append(row)
    return metric_rows, retrieval_rows


def _flatten_aggregate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in frame.columns
    ]
    return frame


def _aggregate_metrics(metrics: pd.DataFrame, by_crowding: bool = False) -> pd.DataFrame:
    group_columns = ["model", "seed", "split"]
    if by_crowding:
        group_columns.append("crowding")
    aggregate = (
        metrics.groupby(group_columns, sort=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )
    return _flatten_aggregate_columns(aggregate)


def _select_model_family(metrics: pd.DataFrame) -> str:
    validation = metrics.loc[
        (metrics["split"] == "validation")
        & (metrics["model"].isin(["linear", "mlp"]))
    ]
    if validation.empty:
        raise RuntimeError("No validation rows are available for model selection.")
    scores = validation.groupby("model")["action_region_mse"].mean()
    return str(scores.idxmin())


def _paired_control_eligible(args: argparse.Namespace, candidates_unique: bool) -> bool:
    return bool(
        args.paired_control_run
        and args.canvas_size == FORMAL_CANVAS_SIZE
        and tuple(args.crowding) == PRIMARY_CROWDING
        and args.train_samples == FORMAL_SPLIT_SIZES["train"]
        and args.val_samples == FORMAL_SPLIT_SIZES["validation"]
        and args.test_samples == FORMAL_SPLIT_SIZES["test"]
        and args.stress_samples == 100
        and args.train_seed == FORMAL_DATA_SEEDS["train"]
        and args.val_seed == FORMAL_DATA_SEEDS["validation"]
        and args.test_seed == FORMAL_DATA_SEEDS["test"]
        and args.stress_seed == FORMAL_STRESS_SEED
        and len(args.model_seeds) == len(FORMAL_MODEL_SEEDS)
        and set(args.model_seeds) == FORMAL_MODEL_SEEDS
        and args.epochs == 30
        and args.patience == 6
        and np.isclose(args.learning_rate, 0.001)
        and np.isclose(args.weight_decay, 0.0001)
        and args.hidden_dim == 64
        and args.train_batch_size == 16
        and args.overfit_examples == 4
        and args.overfit_steps == 30
        and np.isclose(args.overfit_learning_rate, 0.005)
        and args.train_device == "cpu"
        and candidates_unique
    )


def _build_control_diagnostics(
    metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
    selected_family: str,
    args: argparse.Namespace,
    candidates_unique: bool,
    overfit: dict[str, float | int | bool],
) -> pd.DataFrame:
    test = metrics.loc[metrics["split"] == "test"]
    identity = float(
        test.loc[test["model"] == "identity", "action_region_mse"].mean()
    )
    mean_delta = float(
        test.loc[test["model"] == "mean_delta", "action_region_mse"].mean()
    )
    selected = test.loc[test["model"] == selected_family]
    selected_error = float(selected["action_region_mse"].mean())
    improvement_identity = 1.0 - selected_error / max(identity, 1e-12)
    improvement_mean = 1.0 - selected_error / max(mean_delta, 1e-12)

    crowding_improvements: dict[str, float] = {}
    positive_every_crowding = True
    for crowding in sorted(set(args.crowding)):
        identity_error = float(
            test.loc[
                (test["model"] == "identity") & (test["crowding"] == crowding),
                "action_region_mse",
            ].mean()
        )
        selected_error_at_crowding = float(
            selected.loc[
                selected["crowding"] == crowding,
                "action_region_mse",
            ].mean()
        )
        improvement = 1.0 - selected_error_at_crowding / max(identity_error, 1e-12)
        crowding_improvements[
            f"crowding_{crowding}_improvement_vs_identity"
        ] = improvement
        positive_every_crowding = positive_every_crowding and improvement > 0

    seed_errors = selected.groupby("seed")["action_region_mse"].mean()
    stable_seeds = bool((seed_errors < identity).all())
    selected_retrieval = retrieval.loc[retrieval["model"] == selected_family]
    retrieval_accuracy = float(selected_retrieval["top1_correct"].mean())
    oracle_retrieval = retrieval.loc[retrieval["model"] == "exact_oracle"]
    oracle_accuracy = float(oracle_retrieval["top1_correct"].mean())
    oracle_action_mse = float(
        test.loc[test["model"] == "exact_oracle", "action_region_mse"].max()
    )

    metrics_finite = bool(
        np.isfinite(metrics[METRIC_COLUMNS].to_numpy(dtype=float)).all()
    )
    retrieval_finite = bool(
        np.isfinite(
            retrieval[
                ["true_margin"]
                + [f"score_{candidate}" for candidate in COUNTERFACTUAL_ORDER]
            ].to_numpy(dtype=float)
        ).all()
    )
    oracle_exact = bool(oracle_accuracy == 1.0 and oracle_action_mse <= 1e-12)
    implementation_sanity = bool(
        overfit["loss_decreased"]
        and metrics_finite
        and retrieval_finite
        and candidates_unique
        and oracle_exact
    )
    eligible = _paired_control_eligible(args, candidates_unique)
    minimum_improvement = min(improvement_identity, improvement_mean)
    if not eligible:
        status = "diagnostic_only"
    elif not implementation_sanity:
        status = "invalid"
    elif (
        minimum_improvement >= 0.30
        and positive_every_crowding
        and retrieval_accuracy >= 0.50
        and stable_seeds
    ):
        status = "success"
    elif retrieval_accuracy >= 0.35:
        status = "partial"
    else:
        status = "failure"

    if retrieval_accuracy >= 0.50:
        interpretation = "exact_action_information_recoverable_in_pixel_formulation"
    elif retrieval_accuracy <= 0.35 and minimum_improvement >= 0.30:
        interpretation = "average_error_success_but_action_failure_reproduced"
    else:
        interpretation = "inconclusive_or_general_predictor_limitation"

    return pd.DataFrame(
        [
            {
                "paired_control_eligible": eligible,
                "control_status": status,
                "interpretation_label": interpretation,
                "selected_model_family": selected_family,
                "identity_action_region_mse": identity,
                "mean_delta_action_region_mse": mean_delta,
                "selected_action_region_mse": selected_error,
                "improvement_vs_identity": improvement_identity,
                "improvement_vs_mean_delta": improvement_mean,
                "counterfactual_top1_accuracy": retrieval_accuracy,
                "positive_improvement_every_crowding": positive_every_crowding,
                "all_model_seeds_beat_identity": stable_seeds,
                "overfit_loss_decreased": bool(overfit["loss_decreased"]),
                "all_metrics_finite": metrics_finite and retrieval_finite,
                "all_counterfactual_candidates_unique": candidates_unique,
                "exact_oracle_top1_accuracy": oracle_accuracy,
                "exact_oracle_max_action_region_mse": oracle_action_mse,
                "implementation_sanity_passed": implementation_sanity,
                **crowding_improvements,
            }
        ]
    )


def _save_baseline_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    test = metrics.loc[metrics["split"] == "test"]
    summary = test.groupby("model", sort=False)["action_region_mse"].mean()
    identity = float(summary["identity"])
    improvement = 1.0 - summary / max(identity, 1e-12)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(improvement.index, improvement.values)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(0.30, color="tab:green", linestyle="--", label="30% reference")
    axis.set_ylabel("Improvement over identity")
    axis.set_title("Pixel-control held-out action-region prediction")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_crowding_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    test = metrics.loc[metrics["split"] == "test"]
    levels = sorted(test["crowding"].unique())
    identity = (
        test.loc[test["model"] == "identity"]
        .groupby("crowding")["action_region_mse"]
        .mean()
    )
    models = ["mean_delta", "linear", "mlp"]
    x = np.arange(len(levels), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for index, model in enumerate(models):
        model_error = (
            test.loc[test["model"] == model]
            .groupby("crowding")["action_region_mse"]
            .mean()
        )
        improvement = [
            1.0 - float(model_error.loc[level]) / max(float(identity.loc[level]), 1e-12)
            for level in levels
        ]
        axis.bar(x + (index - 1) * width, improvement, width=width, label=model)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, [str(level) for level in levels])
    axis.set_xlabel("Prior-stroke crowding")
    axis.set_ylabel("Improvement over identity")
    axis.set_title("Pixel prediction by crowding")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_training_plot(history: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for (model, seed), group in history.groupby(["model", "seed"], sort=False):
        axis.plot(
            group["epoch"],
            group["validation_balanced_mse"],
            marker="o",
            markersize=3,
            label=f"{model}/{seed}",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation balanced residual MSE")
    axis.set_title("Pixel-control validation curves")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_retrieval_plot(retrieval: pd.DataFrame, output_path: Path) -> None:
    summary = retrieval.groupby("model", sort=False)["top1_correct"].mean()
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(summary.index, summary.values)
    axis.axhline(0.25, color="black", linestyle=":", label="25% chance")
    axis.axhline(0.50, color="tab:green", linestyle="--", label="50% threshold")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Counterfactual top-1 accuracy")
    axis.set_title("Pixel-control counterfactual retrieval")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _model_order(frame: pd.DataFrame) -> list[str]:
    preferred = ["identity", "mean_delta", "linear", "mlp", "exact_oracle"]
    observed = list(frame["model"].unique())
    return [name for name in preferred if name in observed] + [
        name for name in observed if name not in preferred
    ]


def _save_candidate_plot(summary: pd.DataFrame, output_path: Path) -> None:
    order = _model_order(summary)
    indexed = summary.set_index("model")
    bottom = np.zeros(len(order), dtype=float)
    figure, axis = plt.subplots(figsize=(9, 5))
    for candidate in CANDIDATE_NAMES:
        values = indexed.loc[order, f"predicted_{candidate}_rate"].to_numpy()
        axis.bar(order, values, bottom=bottom, label=candidate)
        bottom += values
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean fraction selected across model seeds")
    axis.set_title("Pixel candidate-selection distribution")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_pairwise_plot(summary: pd.DataFrame, output_path: Path) -> None:
    order = _model_order(summary)
    indexed = summary.set_index("model")
    alternatives = list(CANDIDATE_NAMES[1:])
    x = np.arange(len(order), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, alternative in enumerate(alternatives):
        values = indexed.loc[
            order, f"true_beats_{alternative}_rate"
        ].to_numpy()
        axis.bar(x + (index - 1) * width, values, width=width, label=alternative)
    axis.axhline(0.5, color="black", linestyle=":", label="50% pairwise")
    axis.set_xticks(x, order)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean true-outcome win rate across seeds")
    axis.set_title("Pixel true outcome versus each counterfactual class")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_example_plot(
    model: nn.Module,
    payload: dict[str, Any],
    device: torch.device,
    output_path: Path,
) -> None:
    indices = torch.tensor([0])
    current, next_canvas, actions, masks = _batch_tensors(payload, indices, device)
    model = model.to(device).eval()
    with torch.no_grad():
        predicted = (current + model(current, actions, masks)).clamp(0.0, 1.0)
    error = (predicted - next_canvas).abs()
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    panels = [
        (current[0].cpu(), "Current canvas", "gray", 0.0, 1.0),
        (next_canvas[0].cpu(), "True next canvas", "gray", 0.0, 1.0),
        (predicted[0].cpu(), "Predicted next canvas", "gray", 0.0, 1.0),
        (error[0].cpu(), "Absolute error", "magma", 0.0, None),
        (masks[0].cpu(), "Proposed-stroke mask", "gray", 0.0, 1.0),
    ]
    for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels, strict=True):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _split_metadata(payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, payload in payloads.items():
        for index, fingerprint in enumerate(payload["fingerprints"]):
            rows.append(
                {
                    "split": split_name,
                    "sample_id": index,
                    "fingerprint": fingerprint,
                    "crowding": int(payload["crowding"][index]),
                    "stroke_width": int(payload["width"][index]),
                    "stroke_value": int(payload["value"][index]),
                    "stroke_length": float(payload["length"][index]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads, counter_payload, specs = _prepare_payloads(args)
    train_payload = payloads["train"]
    validation_payload = payloads["validation"]
    test_payload = payloads["test"]
    train_device = _device(args.train_device)

    overfit = _run_overfit_check(train_payload, args)
    (args.output_dir / "overfit_check.json").write_text(
        json.dumps(overfit, indent=2), encoding="utf-8"
    )

    identity = IdentityPixelDeltaPredictor()
    mean_delta = MeanPixelDeltaPredictor(_training_mean_delta(train_payload))
    oracle = ExactCompositorPixelDeltaPredictor()
    trained_models: dict[tuple[str, int], nn.Module] = {}
    histories: list[dict[str, float | int | str]] = []
    parameter_counts = {"identity": 0, "mean_delta": 0, "exact_oracle": 0}
    for family in ("linear", "mlp"):
        for seed in args.model_seeds:
            print(f"Training {family} pixel predictor with seed {seed}...")
            model, history = _train_model(
                family,
                seed,
                train_payload,
                validation_payload,
                args,
            )
            trained_models[(family, seed)] = model
            histories.extend(history)
            parameter_counts[family] = parameter_count(model)

    models: list[tuple[str, int, nn.Module]] = [
        ("identity", -1, identity),
        ("mean_delta", -1, mean_delta),
    ]
    models.extend(
        (family, seed, model)
        for (family, seed), model in trained_models.items()
    )
    models.append(("exact_oracle", -1, oracle))

    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    evaluation_splits = [name for name in payloads if name != "train"]
    for model_name, seed, model in models:
        for split_name in evaluation_splits:
            rows, retrieval_for_model = _evaluate_model(
                model_name,
                seed,
                model,
                split_name,
                payloads[split_name],
                args.train_batch_size,
                train_device,
                counter_payload if split_name == "test" else None,
            )
            metric_rows.extend(rows)
            retrieval_rows.extend(retrieval_for_model)

    metrics = pd.DataFrame(metric_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    history = pd.DataFrame(histories)
    aggregate = _aggregate_metrics(metrics)
    aggregate_by_crowding = _aggregate_metrics(metrics, by_crowding=True)
    selected_family = _select_model_family(metrics)
    diagnostics = _build_control_diagnostics(
        metrics,
        retrieval,
        selected_family,
        args,
        bool(counter_payload["all_candidates_unique"]),
        overfit,
    )

    retrieval_summary = summarize_retrieval(retrieval)
    retrieval_family_summary = summarize_retrieval_families(retrieval_summary)
    retrieval_by_crowding = summarize_retrieval_by(retrieval, "crowding")
    retrieval_by_width = summarize_retrieval_by(retrieval, "stroke_width")
    retrieval_by_value = summarize_retrieval_by(retrieval, "stroke_value")
    retrieval_with_length = retrieval.copy()
    retrieval_with_length["stroke_length_bin"] = pd.cut(
        retrieval_with_length["stroke_length"],
        bins=[0.0, 0.35, 0.55, 0.75, float("inf")],
        labels=["short", "medium", "long", "very_long"],
        include_lowest=True,
    )
    retrieval_by_length = summarize_retrieval_by(
        retrieval_with_length, "stroke_length_bin"
    )

    metrics.to_csv(args.output_dir / "prediction_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    aggregate_by_crowding.to_csv(
        args.output_dir / "aggregate_metrics_by_crowding.csv", index=False
    )
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    _split_metadata(payloads).to_csv(
        args.output_dir / "split_metadata.csv", index=False
    )
    diagnostics.to_csv(args.output_dir / "control_diagnostics.csv", index=False)
    retrieval.to_csv(args.output_dir / "counterfactual_retrieval.csv", index=False)
    retrieval_summary.to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    retrieval_family_summary.to_csv(
        args.output_dir / "retrieval_family_summary.csv", index=False
    )
    retrieval_by_crowding.to_csv(
        args.output_dir / "retrieval_by_crowding.csv", index=False
    )
    retrieval_by_width.to_csv(
        args.output_dir / "retrieval_by_stroke_width.csv", index=False
    )
    retrieval_by_value.to_csv(
        args.output_dir / "retrieval_by_stroke_value.csv", index=False
    )
    retrieval_by_length.to_csv(
        args.output_dir / "retrieval_by_stroke_length.csv", index=False
    )

    _save_baseline_plot(metrics, args.output_dir / "baseline_improvement.png")
    _save_crowding_plot(metrics, args.output_dir / "crowding_improvement.png")
    _save_training_plot(history, args.output_dir / "training_curves.png")
    _save_retrieval_plot(retrieval, args.output_dir / "counterfactual_retrieval.png")
    _save_candidate_plot(
        retrieval_family_summary,
        args.output_dir / "candidate_selection_distribution.png",
    )
    _save_pairwise_plot(
        retrieval_family_summary,
        args.output_dir / "pairwise_true_win_rates.png",
    )
    example_seed = sorted(args.model_seeds)[0]
    _save_example_plot(
        trained_models[(selected_family, example_seed)],
        test_payload,
        train_device,
        args.output_dir / "example_pixel_prediction.png",
    )

    config = {
        "analysis_type": "paired_pixel_space_control",
        "canvas_size": args.canvas_size,
        "pixel_input_dim": PIXEL_INPUT_DIM,
        "split_specs": specs,
        "model_seeds": args.model_seeds,
        "model_parameter_counts": parameter_counts,
        "selected_model_family_by_validation": selected_family,
        "train_device": args.train_device,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "overfit_examples": args.overfit_examples,
        "overfit_steps": args.overfit_steps,
        "overfit_learning_rate": args.overfit_learning_rate,
        "paired_control_run_requested": args.paired_control_run,
        "paired_control_eligible": bool(
            diagnostics.iloc[0]["paired_control_eligible"]
        ),
        "control_status": str(diagnostics.iloc[0]["control_status"]),
        "all_counterfactual_candidates_unique": bool(
            counter_payload["all_candidates_unique"]
        ),
        "objective": "balanced inside/outside normalized-pixel residual MSE",
        "evaluation_target": "clamped normalized next-canvas pixels",
        "formal_gate2_decision_unchanged": "fail",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nOverfit implementation check:\n")
    print(json.dumps(overfit, indent=2))
    print("\nPixel-control diagnostics:\n")
    print(diagnostics.to_string(index=False))
    print(f"\nSaved pixel-control outputs to: {args.output_dir.resolve()}")
    if not bool(diagnostics.iloc[0]["paired_control_eligible"]):
        print(
            "This run is diagnostic only. It cannot declare the paired control "
            "result because it does not match the frozen configuration."
        )


if __name__ == "__main__":
    main()
