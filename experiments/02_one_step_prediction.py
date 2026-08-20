"""Gate 2: deterministic one-step prediction of frozen spatial canvas features."""

from __future__ import annotations

import argparse
import copy
import gc
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
from tqdm.auto import tqdm

from latent_stroke_dynamics.encoder import FrozenVisionEncoder
from latent_stroke_dynamics.gate2 import (
    COUNTERFACTUAL_ORDER,
    PRIMARY_CROWDING,
    PRIMARY_VALUES,
    PRIMARY_WIDTHS,
    IdentityPatchDeltaPredictor,
    LinearPatchDeltaPredictor,
    MLPPatchDeltaPredictor,
    MeanPatchDeltaPredictor,
    TransitionExample,
    balanced_patch_mse,
    build_action_tensors,
    build_counterfactual_set,
    build_transition_split,
    counterfactual_retrieval,
    counterfactual_union_mask,
    parameter_count,
    residual_error_metrics,
    transition_fingerprint,
)


FORMAL_MODEL_NAME = "facebook/dinov2-small"
FORMAL_CANVAS_SIZE = 64
FORMAL_SPLIT_SIZES = {"train": 1000, "validation": 200, "test": 300}
FORMAL_DATA_SEEDS = {"train": 20260824, "validation": 20260825, "test": 20260826}
FORMAL_STRESS_SEED = 20260827
FORMAL_MODEL_SEEDS = {11, 22, 33}
DEVELOPMENT_DATA_SEEDS = {
    "train": 20260820,
    "validation": 20260821,
    "test": 20260822,
    "stress": 20260823,
}
COUNTERFACTUAL_CACHE_VERSION = 2
METRIC_COLUMNS = [
    "full_patch_mse",
    "action_region_mse",
    "outside_region_mse",
    "action_region_next_cosine_distance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=FORMAL_MODEL_NAME)
    parser.add_argument("--canvas-size", type=int, default=FORMAL_CANVAS_SIZE)
    parser.add_argument("--crowding", nargs="+", type=int, default=list(PRIMARY_CROWDING))

    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--test-samples", type=int, default=32)
    parser.add_argument("--stress-samples", type=int, default=0)

    parser.add_argument("--train-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["train"])
    parser.add_argument("--val-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["validation"])
    parser.add_argument("--test-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["test"])
    parser.add_argument("--stress-seed", type=int, default=DEVELOPMENT_DATA_SEEDS["stress"])
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[11])

    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--encode-chunk-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--overfit-examples", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=30)
    parser.add_argument("--overfit-learning-rate", type=float, default=5e-3)

    parser.add_argument(
        "--encoder-device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument(
        "--train-device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--skip-counterfactuals", action="store_true")
    parser.add_argument(
        "--formal-run",
        action="store_true",
        help="Required for a formal decision; do not use until the command is frozen.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate2-smoke"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "encode_batch_size": args.encode_batch_size,
        "encode_chunk_size": args.encode_chunk_size,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "hidden_dim": args.hidden_dim,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.stress_samples < 0:
        raise ValueError("--stress-samples cannot be negative.")
    if args.overfit_examples < 1 or args.overfit_steps < 1:
        raise ValueError("The overfit check requires positive examples and steps.")
    if any(level < 0 for level in args.crowding):
        raise ValueError("Crowding levels cannot be negative.")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.encoder_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for encoding but is unavailable.")
    if args.train_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for training but is unavailable.")
    if args.encoder_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested for encoding but is unavailable.")
    if args.train_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested for training but is unavailable.")


def _split_spec(
    name: str,
    samples: int,
    seed: int,
    crowding: Sequence[int],
    widths: Sequence[int],
    values: Sequence[int],
    canvas_size: int,
    model: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "samples": int(samples),
        "seed": int(seed),
        "crowding": [int(item) for item in crowding],
        "widths": [int(item) for item in widths],
        "values": [int(item) for item in values],
        "canvas_size": int(canvas_size),
        "min_length": 0.20,
        "model": model,
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
            args.model,
        ),
        _split_spec(
            "validation",
            args.val_samples,
            args.val_seed,
            args.crowding,
            PRIMARY_WIDTHS,
            PRIMARY_VALUES,
            args.canvas_size,
            args.model,
        ),
        _split_spec(
            "test",
            args.test_samples,
            args.test_seed,
            args.crowding,
            PRIMARY_WIDTHS,
            PRIMARY_VALUES,
            args.canvas_size,
            args.model,
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
                    args.model,
                ),
                _split_spec(
                    "stress_intensity",
                    args.stress_samples,
                    args.stress_seed + 1,
                    args.crowding,
                    PRIMARY_WIDTHS,
                    (16, 80, 176),
                    args.canvas_size,
                    args.model,
                ),
                _split_spec(
                    "stress_crowding",
                    args.stress_samples,
                    args.stress_seed + 2,
                    (10,),
                    PRIMARY_WIDTHS,
                    PRIMARY_VALUES,
                    args.canvas_size,
                    args.model,
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
        name: {transition_fingerprint(item) for item in examples}
        for name, examples in examples_by_name.items()
    }
    names = list(fingerprints)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = fingerprints[left_name].intersection(fingerprints[right_name])
            if overlap:
                raise RuntimeError(
                    f"Detected {len(overlap)} duplicated transitions between "
                    f"{left_name} and {right_name}."
                )


def _load_cache(path: Path, expected_spec: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("spec") != expected_spec:
        raise ValueError(
            f"Cache metadata mismatch for {path}. Remove it or omit --reuse-cache."
        )
    return payload


def _encode_examples(
    encoder: FrozenVisionEncoder,
    examples: Sequence[TransitionExample],
    spec: dict[str, Any],
    encode_batch_size: int,
    encode_chunk_size: int,
) -> dict[str, Any]:
    current_parts: list[torch.Tensor] = []
    next_parts: list[torch.Tensor] = []
    patch_grid: tuple[int, int] | None = None

    for start in tqdm(
        range(0, len(examples), encode_chunk_size),
        desc=f"Preparing {spec['name']} features",
    ):
        chunk = examples[start : start + encode_chunk_size]
        images = [image for item in chunk for image in (item.current, item.next_canvas)]
        encoded = encoder.encode(images, batch_size=encode_batch_size)
        if patch_grid is None:
            patch_grid = encoded.patch_grid
        elif patch_grid != encoded.patch_grid:
            raise RuntimeError("Encoder patch grid changed within one split.")
        current_parts.append(encoded.patch_features[0::2].to(torch.float16))
        next_parts.append(encoded.patch_features[1::2].to(torch.float16))

    assert patch_grid is not None
    actions, masks = build_action_tensors(
        examples,
        canvas_size=spec["canvas_size"],
        patch_grid=patch_grid,
    )
    return {
        "spec": spec,
        "patch_grid": tuple(patch_grid),
        "current": torch.cat(current_parts, dim=0),
        "next": torch.cat(next_parts, dim=0),
        "actions": actions,
        "action_masks": masks,
        "crowding": torch.tensor([item.crowding for item in examples], dtype=torch.int64),
        "width": torch.tensor([item.stroke.width for item in examples], dtype=torch.int64),
        "value": torch.tensor([item.stroke.value for item in examples], dtype=torch.int64),
        "length": torch.tensor(
            [
                hypot(
                    item.stroke.x1 - item.stroke.x0,
                    item.stroke.y1 - item.stroke.y0,
                )
                for item in examples
            ],
            dtype=torch.float32,
        ),
        "fingerprints": [transition_fingerprint(item) for item in examples],
    }


def _encoded_candidates_are_unique(candidates: torch.Tensor) -> bool:
    for sample in candidates:
        for left in range(sample.shape[0]):
            for right in range(left + 1, sample.shape[0]):
                if torch.equal(sample[left], sample[right]):
                    return False
    return True


def _encode_counterfactuals(
    encoder: FrozenVisionEncoder,
    examples: Sequence[TransitionExample],
    test_payload: dict[str, Any],
    encode_batch_size: int,
    encode_chunk_size: int,
) -> dict[str, Any]:
    candidate_parts: list[torch.Tensor] = []
    patch_grid = tuple(test_payload["patch_grid"])
    candidate_sets = [build_counterfactual_set(item) for item in examples]

    for start in tqdm(
        range(0, len(examples), encode_chunk_size),
        desc="Preparing test counterfactuals",
    ):
        chunk = candidate_sets[start : start + encode_chunk_size]
        images = [canvas for item in chunk for canvas in item.canvases]
        encoded = encoder.encode(images, batch_size=encode_batch_size)
        if encoded.patch_grid != patch_grid:
            raise RuntimeError("Counterfactual patch grid does not match test features.")
        candidate_parts.append(
            encoded.patch_features.reshape(
                len(chunk),
                len(COUNTERFACTUAL_ORDER),
                encoded.patch_features.shape[1],
                encoded.patch_features.shape[2],
            ).to(torch.float16)
        )

    candidates = torch.cat(candidate_parts, dim=0)
    if not _encoded_candidates_are_unique(candidates):
        raise RuntimeError(
            "At least one pixel-distinct counterfactual pair became identical after "
            "encoding. Retrieval chance would not be 25%."
        )

    union_masks = torch.stack(
        [
            counterfactual_union_mask(
                item,
                canvas_size=test_payload["spec"]["canvas_size"],
                patch_grid=patch_grid,
            )
            for item in examples
        ]
    )
    return {
        "spec": {
            "cache_version": COUNTERFACTUAL_CACHE_VERSION,
            "test_spec": test_payload["spec"],
            "candidate_order": list(COUNTERFACTUAL_ORDER),
        },
        "candidate_next": candidates,
        "union_masks": union_masks,
        "fingerprints": list(test_payload["fingerprints"]),
        "all_rendered_candidates_unique": True,
        "all_encoded_candidates_unique": True,
    }


def _prepare_features(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    cache_dir = args.cache_dir or (args.output_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(args)
    examples_by_name = {spec["name"]: _generate_examples(spec) for spec in specs}
    _validate_split_separation(examples_by_name)

    payloads: dict[str, dict[str, Any]] = {}
    missing_specs: list[dict[str, Any]] = []
    for spec in specs:
        path = cache_dir / f"{spec['name']}.pt"
        if args.reuse_cache and path.exists():
            payloads[spec["name"]] = _load_cache(path, spec)
        else:
            missing_specs.append(spec)

    counter_path = cache_dir / "test_counterfactuals.pt"
    expected_counter_spec = {
        "cache_version": COUNTERFACTUAL_CACHE_VERSION,
        "test_spec": next(spec for spec in specs if spec["name"] == "test"),
        "candidate_order": list(COUNTERFACTUAL_ORDER),
    }
    counter_payload: dict[str, Any] | None = None
    counter_missing = not args.skip_counterfactuals
    if not args.skip_counterfactuals and args.reuse_cache and counter_path.exists():
        candidate = torch.load(counter_path, map_location="cpu", weights_only=False)
        if candidate.get("spec") == expected_counter_spec:
            counter_payload = candidate
            counter_missing = False
        else:
            print("Ignoring stale counterfactual cache and rebuilding its candidates.")

    encoder: FrozenVisionEncoder | None = None
    if missing_specs or counter_missing:
        encoder = FrozenVisionEncoder(model_name=args.model, device=args.encoder_device)

    if encoder is not None:
        for spec in missing_specs:
            payload = _encode_examples(
                encoder,
                examples_by_name[spec["name"]],
                spec,
                encode_batch_size=args.encode_batch_size,
                encode_chunk_size=args.encode_chunk_size,
            )
            payloads[spec["name"]] = payload
            torch.save(payload, cache_dir / f"{spec['name']}.pt")

        if counter_missing:
            test_payload = payloads["test"]
            counter_payload = _encode_counterfactuals(
                encoder,
                examples_by_name["test"],
                test_payload,
                encode_batch_size=args.encode_batch_size,
                encode_chunk_size=args.encode_chunk_size,
            )
            torch.save(counter_payload, counter_path)

        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grids = {tuple(payload["patch_grid"]) for payload in payloads.values()}
    if len(grids) != 1:
        raise RuntimeError(f"All splits must use one patch grid; observed {grids}.")
    return payloads, counter_payload, specs


def _model_device(name: str) -> torch.device:
    return torch.device(name)


def _batch_tensors(
    payload: dict[str, Any],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = payload["current"][indices].to(device=device, dtype=torch.float32)
    next_features = payload["next"][indices].to(device=device, dtype=torch.float32)
    actions = payload["actions"][indices].to(device=device, dtype=torch.float32)
    masks = payload["action_masks"][indices].to(device=device, dtype=torch.float32)
    return current, next_features, actions, masks


def _training_mean_delta(payload: dict[str, Any], batch_size: int) -> torch.Tensor:
    patches = payload["current"].shape[1]
    features = payload["current"].shape[2]
    total = torch.zeros(patches, features, dtype=torch.float32)
    count = 0
    for start in range(0, len(payload["current"]), batch_size):
        stop = min(start + batch_size, len(payload["current"]))
        current = payload["current"][start:stop].float()
        next_features = payload["next"][start:stop].float()
        total += (next_features - current).sum(dim=0)
        count += stop - start
    return total / count


def _mean_loss(
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
            indices = torch.arange(start, min(start + batch_size, len(payload["current"])))
            current, next_features, actions, masks = _batch_tensors(
                payload,
                indices,
                device,
            )
            true_delta = next_features - current
            predicted = model(current, actions, masks)
            per_example = balanced_patch_mse(
                predicted,
                true_delta,
                masks,
                reduction="none",
            )
            total += float(per_example.sum())
            count += len(indices)
    return total / count


def _create_model(
    family: str,
    feature_dim: int,
    patch_grid: tuple[int, int],
    hidden_dim: int,
) -> nn.Module:
    if family == "linear":
        return LinearPatchDeltaPredictor(feature_dim, patch_grid)
    if family == "mlp":
        return MLPPatchDeltaPredictor(feature_dim, patch_grid, hidden_dim)
    raise ValueError(f"Unknown model family: {family}")


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

    device = _model_device(args.train_device)
    patch_grid = tuple(train_payload["patch_grid"])
    feature_dim = int(train_payload["current"].shape[-1])
    model = _create_model(family, feature_dim, patch_grid, args.hidden_dim).to(device)
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
            current, next_features, actions, masks = _batch_tensors(
                train_payload,
                indices,
                device,
            )
            true_delta = next_features - current
            predicted = model(current, actions, masks)
            loss = balanced_patch_mse(predicted, true_delta, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_total += float(loss.detach()) * len(indices)
            train_count += len(indices)

        validation_loss = _mean_loss(
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
    seed = 31415
    torch.manual_seed(seed)
    device = _model_device(args.train_device)
    count = min(args.overfit_examples, len(train_payload["current"]))
    indices = torch.arange(count)
    current, next_features, actions, masks = _batch_tensors(
        train_payload,
        indices,
        device,
    )
    true_delta = next_features - current

    model = MLPPatchDeltaPredictor(
        feature_dim=current.shape[-1],
        patch_grid=tuple(train_payload["patch_grid"]),
        hidden_dim=min(args.hidden_dim, 128),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.overfit_learning_rate)

    with torch.no_grad():
        initial = float(balanced_patch_mse(model(current, actions, masks), true_delta, masks))
    for _ in range(args.overfit_steps):
        prediction = model(current, actions, masks)
        loss = balanced_patch_mse(prediction, true_delta, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = float(balanced_patch_mse(model(current, actions, masks), true_delta, masks))

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
            current, next_features, actions, masks = _batch_tensors(
                payload,
                indices,
                device,
            )
            true_delta = next_features - current
            predicted_delta = model(current, actions, masks)
            metrics = residual_error_metrics(
                current,
                predicted_delta,
                true_delta,
                masks,
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
                candidates = counter_payload["candidate_next"][start:stop].to(
                    device=device,
                    dtype=torch.float32,
                )
                union_masks = counter_payload["union_masks"][start:stop].to(
                    device=device,
                    dtype=torch.float32,
                )
                retrieval = counterfactual_retrieval(
                    current + predicted_delta,
                    candidates,
                    union_masks,
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


def _formal_eligible(
    args: argparse.Namespace,
    counter_payload: dict[str, Any] | None,
) -> bool:
    return (
        args.formal_run
        and args.model == FORMAL_MODEL_NAME
        and args.canvas_size == FORMAL_CANVAS_SIZE
        and args.train_samples == FORMAL_SPLIT_SIZES["train"]
        and args.val_samples == FORMAL_SPLIT_SIZES["validation"]
        and args.test_samples == FORMAL_SPLIT_SIZES["test"]
        and args.train_seed == FORMAL_DATA_SEEDS["train"]
        and args.val_seed == FORMAL_DATA_SEEDS["validation"]
        and args.test_seed == FORMAL_DATA_SEEDS["test"]
        and args.stress_seed == FORMAL_STRESS_SEED
        and len(args.model_seeds) == len(FORMAL_MODEL_SEEDS)
        and set(args.model_seeds) == FORMAL_MODEL_SEEDS
        and args.stress_samples == 100
        and counter_payload is not None
        and bool(counter_payload.get("all_rendered_candidates_unique"))
        and bool(counter_payload.get("all_encoded_candidates_unique"))
        and tuple(args.crowding) == PRIMARY_CROWDING
    )


def _build_gate_diagnostics(
    metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
    selected_family: str,
    args: argparse.Namespace,
    counter_payload: dict[str, Any] | None,
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
        selected_crowding_error = float(
            selected.loc[
                selected["crowding"] == crowding,
                "action_region_mse",
            ].mean()
        )
        improvement = 1.0 - selected_crowding_error / max(identity_error, 1e-12)
        crowding_improvements[
            f"crowding_{crowding}_improvement_vs_identity"
        ] = improvement
        positive_every_crowding = positive_every_crowding and improvement > 0.0

    per_seed = selected.groupby("seed")["action_region_mse"].mean()
    stable_seeds = bool((per_seed < identity).all())

    selected_retrieval = retrieval.loc[retrieval["model"] == selected_family]
    retrieval_accuracy = (
        float(selected_retrieval["top1_correct"].mean())
        if not selected_retrieval.empty
        else float("nan")
    )

    metric_values_finite = bool(
        np.isfinite(metrics[METRIC_COLUMNS].to_numpy(dtype=float)).all()
    )
    retrieval_values_finite = bool(
        not selected_retrieval.empty
        and np.isfinite(
            selected_retrieval[
                ["true_margin"]
                + [f"score_{candidate}" for candidate in COUNTERFACTUAL_ORDER]
            ].to_numpy(dtype=float)
        ).all()
    )
    candidates_unique = bool(
        counter_payload is not None
        and counter_payload.get("all_rendered_candidates_unique")
        and counter_payload.get("all_encoded_candidates_unique")
    )
    implementation_sanity = bool(
        overfit["loss_decreased"]
        and metric_values_finite
        and retrieval_values_finite
        and candidates_unique
    )

    formal = _formal_eligible(args, counter_payload)
    minimum_improvement = min(improvement_identity, improvement_mean)
    if not formal:
        status = "diagnostic_only"
    elif not implementation_sanity:
        status = "fail"
    elif (
        minimum_improvement >= 0.30
        and positive_every_crowding
        and retrieval_accuracy >= 0.50
        and stable_seeds
    ):
        status = "pass"
    elif minimum_improvement >= 0.10 and retrieval_accuracy >= 0.35:
        status = "borderline"
    else:
        status = "fail"

    row: dict[str, Any] = {
        "formal_eligible": formal,
        "gate_status": status,
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
        "all_metrics_finite": metric_values_finite and retrieval_values_finite,
        "all_counterfactual_candidates_unique": candidates_unique,
        "implementation_sanity_passed": implementation_sanity,
        **crowding_improvements,
    }
    return pd.DataFrame([row])


def _save_error_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    test = metrics.loc[metrics["split"] == "test"]
    summary = test.groupby("model", sort=False)["action_region_mse"].mean()
    identity = float(summary["identity"])
    improvements = 1.0 - summary / max(identity, 1e-12)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(improvements.index, improvements.values)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(0.30, color="tab:green", linestyle="--", label="30% gate margin")
    axis.set_ylabel("Improvement over identity")
    axis.set_title("Gate 2 held-out action-region prediction")
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
        improvements = [
            1.0 - float(model_error.loc[level]) / max(float(identity.loc[level]), 1e-12)
            for level in levels
        ]
        axis.bar(
            x + (index - 1) * width,
            improvements,
            width=width,
            label=model,
        )

    axis.axhline(0.0, color="black", linewidth=1)
    axis.axhline(0.30, color="tab:green", linestyle="--", label="30% gate margin")
    axis.set_xticks(x, [str(level) for level in levels])
    axis.set_xlabel("Prior-stroke crowding")
    axis.set_ylabel("Improvement over identity")
    axis.set_title("Held-out action-region prediction by crowding")
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
    axis.set_ylabel("Validation balanced MSE")
    axis.set_title("Gate 2 validation curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_retrieval_plot(retrieval: pd.DataFrame, output_path: Path) -> None:
    summary = retrieval.groupby("model", sort=False)["top1_correct"].mean()
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(summary.index, summary.values)
    axis.axhline(0.25, color="black", linestyle=":", label="25% chance")
    axis.axhline(0.50, color="tab:green", linestyle="--", label="50% gate threshold")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Counterfactual top-1 accuracy")
    axis.set_title("Gate 2 counterfactual retrieval")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_example_heatmap(
    model: nn.Module,
    payload: dict[str, Any],
    device: torch.device,
    output_path: Path,
) -> None:
    indices = torch.tensor([0])
    current, next_features, actions, masks = _batch_tensors(payload, indices, device)
    model = model.to(device).eval()
    with torch.no_grad():
        predicted = model(current, actions, masks)
    true_delta = next_features - current
    true_magnitude = torch.linalg.vector_norm(true_delta[0], dim=-1).cpu()
    predicted_magnitude = torch.linalg.vector_norm(predicted[0], dim=-1).cpu()
    error = torch.linalg.vector_norm(predicted[0] - true_delta[0], dim=-1).cpu()
    action_mask = masks[0].cpu()
    patch_grid = tuple(payload["patch_grid"])
    shared_min = float(torch.minimum(true_magnitude.min(), predicted_magnitude.min()))
    shared_max = float(torch.maximum(true_magnitude.max(), predicted_magnitude.max()))

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    first = axes[0].imshow(
        true_magnitude.reshape(*patch_grid),
        cmap="magma",
        interpolation="nearest",
        vmin=shared_min,
        vmax=shared_max,
    )
    axes[0].set_title("True residual magnitude")
    axes[1].imshow(
        predicted_magnitude.reshape(*patch_grid),
        cmap="magma",
        interpolation="nearest",
        vmin=shared_min,
        vmax=shared_max,
    )
    axes[1].set_title("Predicted residual magnitude")
    second = axes[2].imshow(
        error.reshape(*patch_grid),
        cmap="viridis",
        interpolation="nearest",
    )
    axes[2].set_title("Prediction error")
    third = axes[3].imshow(
        action_mask.reshape(*patch_grid),
        cmap="gray",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    axes[3].set_title("Proposed-stroke mask")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(first, ax=axes[:2], fraction=0.025, pad=0.02)
    figure.colorbar(second, ax=axes[2], fraction=0.046, pad=0.04)
    figure.colorbar(third, ax=axes[3], fraction=0.046, pad=0.04)
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

    payloads, counter_payload, specs = _prepare_features(args)
    train_payload = payloads["train"]
    validation_payload = payloads["validation"]
    test_payload = payloads["test"]
    patch_grid = tuple(train_payload["patch_grid"])
    feature_dim = int(train_payload["current"].shape[-1])
    train_device = _model_device(args.train_device)

    overfit = _run_overfit_check(train_payload, args)
    (args.output_dir / "overfit_check.json").write_text(
        json.dumps(overfit, indent=2),
        encoding="utf-8",
    )

    mean_delta = _training_mean_delta(train_payload, args.train_batch_size)
    identity_model = IdentityPatchDeltaPredictor()
    mean_model = MeanPatchDeltaPredictor(mean_delta)

    trained_models: dict[tuple[str, int], nn.Module] = {}
    histories: list[dict[str, float | int | str]] = []
    model_parameter_counts: dict[str, int] = {
        "identity": 0,
        "mean_delta": 0,
    }

    for family in ("linear", "mlp"):
        for seed in args.model_seeds:
            print(f"Training {family} predictor with seed {seed}...")
            model, history = _train_model(
                family,
                seed,
                train_payload,
                validation_payload,
                args,
            )
            trained_models[(family, seed)] = model
            histories.extend(history)
            model_parameter_counts[family] = parameter_count(model)

    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    evaluation_splits = [name for name in payloads if name != "train"]

    models_to_evaluate: list[tuple[str, int, nn.Module]] = [
        ("identity", -1, identity_model),
        ("mean_delta", -1, mean_model),
    ]
    models_to_evaluate.extend(
        (family, seed, model)
        for (family, seed), model in trained_models.items()
    )

    for model_name, seed, model in models_to_evaluate:
        for split_name in evaluation_splits:
            split_counter = (
                counter_payload
                if split_name == "test" and counter_payload is not None
                else None
            )
            rows, retrieval_rows_for_model = _evaluate_model(
                model_name,
                seed,
                model,
                split_name,
                payloads[split_name],
                batch_size=args.train_batch_size,
                device=train_device,
                counter_payload=split_counter,
            )
            metric_rows.extend(rows)
            retrieval_rows.extend(retrieval_rows_for_model)

    metrics = pd.DataFrame(metric_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    history_frame = pd.DataFrame(histories)
    aggregate = _aggregate_metrics(metrics)
    aggregate_by_crowding = _aggregate_metrics(metrics, by_crowding=True)
    selected_family = _select_model_family(metrics)
    diagnostics = _build_gate_diagnostics(
        metrics,
        retrieval,
        selected_family,
        args,
        counter_payload,
        overfit,
    )

    metrics.to_csv(args.output_dir / "prediction_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    aggregate_by_crowding.to_csv(
        args.output_dir / "aggregate_metrics_by_crowding.csv",
        index=False,
    )
    history_frame.to_csv(args.output_dir / "training_history.csv", index=False)
    _split_metadata(payloads).to_csv(
        args.output_dir / "split_metadata.csv",
        index=False,
    )
    diagnostics.to_csv(args.output_dir / "gate_diagnostics.csv", index=False)
    if not retrieval.empty:
        retrieval.to_csv(
            args.output_dir / "counterfactual_retrieval.csv",
            index=False,
        )

    _save_error_plot(metrics, args.output_dir / "baseline_improvement.png")
    _save_crowding_plot(metrics, args.output_dir / "crowding_improvement.png")
    _save_training_plot(history_frame, args.output_dir / "training_curves.png")
    if not retrieval.empty:
        _save_retrieval_plot(
            retrieval,
            args.output_dir / "counterfactual_retrieval.png",
        )
    example_seed = sorted(args.model_seeds)[0]
    _save_example_heatmap(
        trained_models[(selected_family, example_seed)],
        test_payload,
        train_device,
        args.output_dir / "example_residual_prediction.png",
    )

    config = {
        "model": args.model,
        "canvas_size": args.canvas_size,
        "patch_grid": list(patch_grid),
        "feature_dim": feature_dim,
        "split_specs": specs,
        "model_seeds": args.model_seeds,
        "model_parameter_counts": model_parameter_counts,
        "selected_model_family_by_validation": selected_family,
        "encoder_device": args.encoder_device,
        "train_device": args.train_device,
        "encode_batch_size": args.encode_batch_size,
        "encode_chunk_size": args.encode_chunk_size,
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "counterfactual_cache_version": COUNTERFACTUAL_CACHE_VERSION,
        "counterfactuals_enabled": counter_payload is not None,
        "all_counterfactual_candidates_unique": bool(
            counter_payload is not None
            and counter_payload.get("all_rendered_candidates_unique")
            and counter_payload.get("all_encoded_candidates_unique")
        ),
        "formal_run_requested": args.formal_run,
        "formal_eligible": bool(diagnostics.iloc[0]["formal_eligible"]),
        "gate_status": str(diagnostics.iloc[0]["gate_status"]),
        "objective": "balanced action-region and outside-region residual MSE",
        "target": "DINOv2 final-layer patch-token residual",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nOverfit implementation check:\n")
    print(json.dumps(overfit, indent=2))
    print("\nGate diagnostics:\n")
    print(diagnostics.to_string(index=False))
    print(f"\nSaved Gate 2 outputs to: {args.output_dir.resolve()}")
    if not bool(diagnostics.iloc[0]["formal_eligible"]):
        print(
            "This run is diagnostic only. It cannot declare Gate 2 pass/fail because "
            "it does not match the frozen formal configuration."
        )


if __name__ == "__main__":
    main()
