"""Training and evaluation utilities for the frozen representation extension."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from hashlib import sha256
from math import hypot
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .gate2 import (
    COUNTERFACTUAL_ORDER,
    IdentityPatchDeltaPredictor,
    LinearPatchDeltaPredictor,
    MLPPatchDeltaPredictor,
    MeanPatchDeltaPredictor,
    TransitionExample,
    balanced_patch_mse,
    build_action_tensors,
    counterfactual_retrieval,
    counterfactual_union_mask,
    residual_error_metrics,
    transition_fingerprint,
)
from .representation_extension import StrokeAutoencoder, reconstruction_metrics


@dataclass
class AutoencoderFitResult:
    """Selected state and validation history for one autoencoder seed."""

    model: StrokeAutoencoder
    seed: int
    best_epoch: int
    best_validation_mse: float
    history: list[dict[str, int | float]]


@dataclass
class PatchPredictorFitResult:
    """Selected state and validation history for one dynamics model."""

    model: nn.Module
    family: str
    seed: int
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, int | float | str]]


@dataclass(frozen=True)
class PatchFeaturePayload:
    """One split of spatial states, actions, and transition metadata."""

    current: torch.Tensor
    next_features: torch.Tensor
    actions: torch.Tensor
    action_masks: torch.Tensor
    patch_grid: tuple[int, int]
    crowding: torch.Tensor
    width: torch.Tensor
    value: torch.Tensor
    length: torch.Tensor
    fingerprints: tuple[str, ...]

    @property
    def size(self) -> int:
        return int(self.current.shape[0])


@dataclass(frozen=True)
class PatchCounterfactualPayload:
    """Encoded four-way candidates and their shared spatial scoring masks."""

    candidate_next: torch.Tensor
    union_masks: torch.Tensor
    all_encoded_candidates_unique: bool


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic CPU training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def total_parameter_count(model: nn.Module) -> int:
    """Count all parameters, including parameters frozen for evaluation."""

    return sum(parameter.numel() for parameter in model.parameters())


def model_state_sha256(model: nn.Module) -> str:
    """Hash model tensor names, shapes, dtypes, and values deterministically."""

    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mean_image_baseline_mse(
    train_images: torch.Tensor,
    evaluation_images: torch.Tensor,
) -> float:
    """Evaluate the train-set mean image on a separate image tensor."""

    _validate_image_tensor(train_images, "train_images")
    _validate_image_tensor(evaluation_images, "evaluation_images")
    mean_image = train_images.mean(dim=0, keepdim=True)
    return float((evaluation_images - mean_image).square().mean().item())


def _validate_image_tensor(images: torch.Tensor, name: str) -> None:
    if images.ndim != 4 or images.shape[1:] != (1, 64, 64):
        raise ValueError(f"{name} must have shape [batch, 1, 64, 64].")
    if images.shape[0] < 1:
        raise ValueError(f"{name} cannot be empty.")
    if not bool(torch.isfinite(images).all()):
        raise ValueError(f"{name} must contain only finite values.")


def autoencoder_mean_loss(
    model: StrokeAutoencoder,
    images: torch.Tensor,
    batch_size: int,
) -> float:
    """Return full-canvas reconstruction MSE over an image tensor."""

    _validate_image_tensor(images, "images")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            values = reconstruction_metrics(model(batch), batch)["mse"]
            total += float(values.sum().item())
            count += len(batch)
    return total / count


def train_stroke_autoencoder(
    train_images: torch.Tensor,
    validation_images: torch.Tensor,
    *,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> AutoencoderFitResult:
    """Train one frozen-architecture autoencoder on CPU with early stopping."""

    _validate_image_tensor(train_images, "train_images")
    _validate_image_tensor(validation_images, "validation_images")
    if min(batch_size, max_epochs, patience) < 1:
        raise ValueError("Batch size, epochs, and patience must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid autoencoder optimizer settings.")

    seed_everything(seed)
    model = StrokeAutoencoder().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    best_validation = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, int | float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_images), generator=generator)
        train_total = 0.0
        train_count = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            batch = train_images[indices]
            loss = (model(batch) - batch).square().mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Autoencoder training produced a non-finite loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_total += float(loss.detach().item()) * len(indices)
            train_count += len(indices)

        train_loss = train_total / train_count
        validation_loss = autoencoder_mean_loss(
            model,
            validation_images,
            batch_size,
        )
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise RuntimeError("Autoencoder history contains a non-finite loss.")
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_reconstruction_mse": train_loss,
                "validation_reconstruction_mse": validation_loss,
            }
        )
        if validation_loss < best_validation - 1e-12:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    selected_loss = autoencoder_mean_loss(model, validation_images, batch_size)
    if not np.isclose(selected_loss, best_validation, rtol=0.0, atol=1e-12):
        raise RuntimeError("Selected autoencoder state did not reproduce its best loss.")
    return AutoencoderFitResult(
        model=model.eval(),
        seed=seed,
        best_epoch=best_epoch,
        best_validation_mse=best_validation,
        history=history,
    )


@torch.inference_mode()
def encode_autoencoder_maps(
    model: StrokeAutoencoder,
    images: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Encode image tensors in bounded batches on CPU."""

    _validate_image_tensor(images, "images")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model.eval()
    parts = [
        model.encode_map(images[start : start + batch_size]).cpu()
        for start in range(0, len(images), batch_size)
    ]
    return torch.cat(parts, dim=0)


def save_autoencoder_checkpoint(
    model: StrokeAutoencoder,
    metadata: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Save a local extension checkpoint under an ignored output directory."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "architecture": "StrokeAutoencoder",
        "metadata": dict(metadata),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_autoencoder_checkpoint(
    path: str | Path,
) -> tuple[StrokeAutoencoder, dict[str, Any]]:
    """Load the exact frozen extension autoencoder architecture."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported autoencoder checkpoint format.")
    if payload.get("architecture") != "StrokeAutoencoder":
        raise ValueError("Unexpected autoencoder checkpoint architecture.")
    state = payload.get("state_dict")
    metadata = payload.get("metadata")
    if not isinstance(state, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Malformed autoencoder checkpoint.")
    model = StrokeAutoencoder()
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(metadata)


def build_patch_feature_payload(
    examples: Sequence[TransitionExample],
    current: torch.Tensor,
    next_features: torch.Tensor,
    patch_grid: tuple[int, int],
    canvas_size: int = 64,
) -> PatchFeaturePayload:
    """Combine externally encoded states with the existing action tensors."""

    if not examples:
        raise ValueError("At least one transition is required.")
    if current.shape != next_features.shape or current.ndim != 3:
        raise ValueError("Current and next features must share [N, patches, dim].")
    expected = (len(examples), patch_grid[0] * patch_grid[1])
    if current.shape[:2] != expected:
        raise ValueError("Encoded feature shape does not match examples and grid.")
    if not bool(torch.isfinite(current).all() and torch.isfinite(next_features).all()):
        raise ValueError("Encoded state features must be finite.")

    actions, action_masks = build_action_tensors(
        examples,
        canvas_size=canvas_size,
        patch_grid=patch_grid,
    )
    return PatchFeaturePayload(
        current=current.cpu(),
        next_features=next_features.cpu(),
        actions=actions.cpu(),
        action_masks=action_masks.cpu(),
        patch_grid=patch_grid,
        crowding=torch.tensor([item.crowding for item in examples], dtype=torch.int64),
        width=torch.tensor([item.stroke.width for item in examples], dtype=torch.int64),
        value=torch.tensor([item.stroke.value for item in examples], dtype=torch.int64),
        length=torch.tensor(
            [
                hypot(
                    item.stroke.x1 - item.stroke.x0,
                    item.stroke.y1 - item.stroke.y0,
                )
                for item in examples
            ],
            dtype=torch.float32,
        ),
        fingerprints=tuple(transition_fingerprint(item) for item in examples),
    )


def encoded_candidates_are_unique(candidate_next: torch.Tensor) -> bool:
    """Check exact encoded equality within every candidate set."""

    if candidate_next.ndim != 4 or candidate_next.shape[1] != len(
        COUNTERFACTUAL_ORDER
    ):
        raise ValueError("Candidates must have shape [N, 4, patches, features].")
    for sample in candidate_next:
        for left in range(sample.shape[0]):
            for right in range(left + 1, sample.shape[0]):
                if torch.equal(sample[left], sample[right]):
                    return False
    return True


def build_patch_counterfactual_payload(
    examples: Sequence[TransitionExample],
    candidate_next: torch.Tensor,
    patch_grid: tuple[int, int],
    canvas_size: int = 64,
) -> PatchCounterfactualPayload:
    """Attach union masks to externally encoded counterfactual candidates."""

    if candidate_next.ndim != 4:
        raise ValueError("candidate_next must have four dimensions.")
    if candidate_next.shape[:3] != (
        len(examples),
        len(COUNTERFACTUAL_ORDER),
        patch_grid[0] * patch_grid[1],
    ):
        raise ValueError("Counterfactual features do not match examples and grid.")
    if not bool(torch.isfinite(candidate_next).all()):
        raise ValueError("Counterfactual features must be finite.")
    union_masks = torch.stack(
        [
            counterfactual_union_mask(
                item,
                canvas_size=canvas_size,
                patch_grid=patch_grid,
            )
            for item in examples
        ]
    )
    return PatchCounterfactualPayload(
        candidate_next=candidate_next.cpu(),
        union_masks=union_masks.cpu(),
        all_encoded_candidates_unique=encoded_candidates_are_unique(candidate_next),
    )


def _patch_batch(
    payload: PatchFeaturePayload,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        payload.current[indices].float(),
        payload.next_features[indices].float(),
        payload.actions[indices].float(),
        payload.action_masks[indices].float(),
    )


def training_mean_delta(
    payload: PatchFeaturePayload,
    batch_size: int,
) -> torch.Tensor:
    """Compute a residual mean using the training payload only."""

    total = torch.zeros(
        payload.current.shape[1],
        payload.current.shape[2],
        dtype=torch.float32,
    )
    count = 0
    for start in range(0, payload.size, batch_size):
        stop = min(start + batch_size, payload.size)
        total += (
            payload.next_features[start:stop].float()
            - payload.current[start:stop].float()
        ).sum(dim=0)
        count += stop - start
    return total / count


def create_patch_predictor(
    family: str,
    feature_dim: int,
    patch_grid: tuple[int, int],
    hidden_dim: int,
) -> nn.Module:
    """Create one of the two frozen dynamics families."""

    if family == "linear":
        return LinearPatchDeltaPredictor(feature_dim, patch_grid)
    if family == "mlp":
        return MLPPatchDeltaPredictor(feature_dim, patch_grid, hidden_dim)
    raise ValueError(f"Unknown patch predictor family: {family}")


def patch_mean_loss(
    model: nn.Module,
    payload: PatchFeaturePayload,
    batch_size: int,
) -> float:
    """Evaluate balanced patch residual loss on CPU."""

    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, payload.size, batch_size):
            indices = torch.arange(start, min(start + batch_size, payload.size))
            current, next_features, actions, masks = _patch_batch(payload, indices)
            values = balanced_patch_mse(
                model(current, actions, masks),
                next_features - current,
                masks,
                reduction="none",
            )
            total += float(values.sum().item())
            count += len(indices)
    return total / count


def train_patch_predictor(
    family: str,
    seed: int,
    train_payload: PatchFeaturePayload,
    validation_payload: PatchFeaturePayload,
    *,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> PatchPredictorFitResult:
    """Train one frozen-family action-conditioned patch predictor on CPU."""

    if train_payload.patch_grid != validation_payload.patch_grid:
        raise ValueError("Train and validation patch grids must match.")
    if train_payload.current.shape[-1] != validation_payload.current.shape[-1]:
        raise ValueError("Train and validation feature dimensions must match.")
    seed_everything(seed)
    model = create_patch_predictor(
        family,
        int(train_payload.current.shape[-1]),
        train_payload.patch_grid,
        hidden_dim,
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    best_validation = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: list[dict[str, int | float | str]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_payload.size, generator=generator)
        train_total = 0.0
        train_count = 0
        for start in range(0, train_payload.size, batch_size):
            indices = permutation[start : start + batch_size]
            current, next_features, actions, masks = _patch_batch(
                train_payload,
                indices,
            )
            loss = balanced_patch_mse(
                model(current, actions, masks),
                next_features - current,
                masks,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Patch dynamics training produced a non-finite loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_total += float(loss.detach().item()) * len(indices)
            train_count += len(indices)

        train_loss = train_total / train_count
        validation_loss = patch_mean_loss(model, validation_payload, batch_size)
        history.append(
            {
                "family": family,
                "seed": seed,
                "epoch": epoch,
                "train_balanced_mse": train_loss,
                "validation_balanced_mse": validation_loss,
            }
        )
        if validation_loss < best_validation - 1e-12:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    selected_loss = patch_mean_loss(model, validation_payload, batch_size)
    if not np.isclose(selected_loss, best_validation, rtol=0.0, atol=1e-12):
        raise RuntimeError("Selected dynamics state did not reproduce its best loss.")
    return PatchPredictorFitResult(
        model=model.eval(),
        family=family,
        seed=seed,
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        history=history,
    )


def run_patch_overfit_check(
    payload: PatchFeaturePayload,
    *,
    hidden_dim: int,
    examples: int = 4,
    steps: int = 30,
    learning_rate: float = 0.005,
) -> dict[str, int | float | bool]:
    """Check that the nonlinear dynamics implementation can reduce tiny-set loss."""

    seed_everything(31415)
    count = min(examples, payload.size)
    indices = torch.arange(count)
    current, next_features, actions, masks = _patch_batch(payload, indices)
    true_delta = next_features - current
    model = MLPPatchDeltaPredictor(
        feature_dim=current.shape[-1],
        patch_grid=payload.patch_grid,
        hidden_dim=min(hidden_dim, 128),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    with torch.inference_mode():
        initial = float(
            balanced_patch_mse(model(current, actions, masks), true_delta, masks).item()
        )
    for _ in range(steps):
        loss = balanced_patch_mse(model(current, actions, masks), true_delta, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        final = float(
            balanced_patch_mse(model(current, actions, masks), true_delta, masks).item()
        )
    return {
        "examples": count,
        "steps": steps,
        "initial_balanced_mse": initial,
        "final_balanced_mse": final,
        "relative_reduction": 1.0 - final / max(initial, 1e-12),
        "loss_decreased": bool(final < initial),
    }


def evaluate_patch_model(
    model_name: str,
    seed: int,
    model: nn.Module,
    split_name: str,
    payload: PatchFeaturePayload,
    batch_size: int,
    counterfactuals: PatchCounterfactualPayload | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate residual errors and optional four-way retrieval on CPU."""

    model.eval()
    metric_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, payload.size, batch_size):
            stop = min(start + batch_size, payload.size)
            indices = torch.arange(start, stop)
            current, next_features, actions, masks = _patch_batch(payload, indices)
            true_delta = next_features - current
            predicted_delta = model(current, actions, masks)
            metrics = residual_error_metrics(
                current,
                predicted_delta,
                true_delta,
                masks,
            )
            for offset, sample_index in enumerate(range(start, stop)):
                row: dict[str, Any] = {
                    "model": model_name,
                    "seed": seed,
                    "split": split_name,
                    "sample_id": sample_index,
                    "fingerprint": payload.fingerprints[sample_index],
                    "crowding": int(payload.crowding[sample_index]),
                    "stroke_width": int(payload.width[sample_index]),
                    "stroke_value": int(payload.value[sample_index]),
                    "stroke_length": float(payload.length[sample_index]),
                }
                row.update(
                    {
                        metric_name: float(metric_values[offset].cpu().item())
                        for metric_name, metric_values in metrics.items()
                    }
                )
                metric_rows.append(row)

            if counterfactuals is not None:
                candidates = counterfactuals.candidate_next[start:stop].float()
                union_masks = counterfactuals.union_masks[start:stop].float()
                result = counterfactual_retrieval(
                    current + predicted_delta,
                    candidates,
                    union_masks,
                )
                scores = result["scores"].cpu()
                predicted_index = result["predicted_index"].cpu()
                correct = result["top1_correct"].cpu()
                margins = result["true_margin"].cpu()
                for offset, sample_index in enumerate(range(start, stop)):
                    index = int(predicted_index[offset])
                    row = {
                        "model": model_name,
                        "seed": seed,
                        "sample_id": sample_index,
                        "fingerprint": payload.fingerprints[sample_index],
                        "predicted_index": index,
                        "predicted_label": COUNTERFACTUAL_ORDER[index],
                        "top1_correct": bool(correct[offset]),
                        "true_margin": float(margins[offset].item()),
                    }
                    for candidate_index, candidate_name in enumerate(
                        COUNTERFACTUAL_ORDER
                    ):
                        row[f"score_{candidate_name}"] = float(
                            scores[offset, candidate_index].item()
                        )
                    retrieval_rows.append(row)
    return metric_rows, retrieval_rows


def exact_target_oracle_retrieval(
    test_payload: PatchFeaturePayload,
    counterfactuals: PatchCounterfactualPayload,
) -> dict[str, float | bool]:
    """Verify that the true encoded next state retrieves candidate zero."""

    maximum_candidate_zero_difference = float(
        (
            test_payload.next_features.float()
            - counterfactuals.candidate_next[:, 0].float()
        )
        .abs()
        .max()
        .item()
    )
    result = counterfactual_retrieval(
        test_payload.next_features.float(),
        counterfactuals.candidate_next.float(),
        counterfactuals.union_masks.float(),
    )
    accuracy = float(result["top1_correct"].float().mean().item())
    return {
        "top1_accuracy": accuracy,
        "maximum_candidate_zero_difference": maximum_candidate_zero_difference,
        "passed": bool(
            accuracy == 1.0
            and maximum_candidate_zero_difference == 0.0
            and counterfactuals.all_encoded_candidates_unique
        ),
    }


def baseline_models(
    train_payload: PatchFeaturePayload,
    batch_size: int,
) -> tuple[IdentityPatchDeltaPredictor, MeanPatchDeltaPredictor]:
    """Build identity and train-only mean-delta baselines."""

    return (
        IdentityPatchDeltaPredictor(),
        MeanPatchDeltaPredictor(training_mean_delta(train_payload, batch_size)),
    )
