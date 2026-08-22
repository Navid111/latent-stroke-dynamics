"""Training utilities for the ranking-aware latent development comparison."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .extension_training import (
    PatchCounterfactualPayload,
    PatchFeaturePayload,
    create_patch_predictor,
    seed_everything,
)
from .gate2 import counterfactual_retrieval
from .ranking_latent import ranking_aware_objective


@dataclass
class RankingPredictorFitResult:
    """Selected model state and history for one ranking-aware fit."""

    model: nn.Module
    seed: int
    ranking_weight: float
    temperature: float
    best_epoch: int
    best_validation_total: float
    history: list[dict[str, int | float | str]]


def _validate_alignment(
    payload: PatchFeaturePayload,
    counterfactuals: PatchCounterfactualPayload,
) -> None:
    if payload.size != int(counterfactuals.candidate_next.shape[0]):
        raise ValueError("Feature and counterfactual payload sizes differ.")
    if payload.patch_grid[0] * payload.patch_grid[1] != int(
        counterfactuals.candidate_next.shape[2]
    ):
        raise ValueError("Feature and counterfactual patch grids differ.")
    if payload.current.shape[-1] != counterfactuals.candidate_next.shape[-1]:
        raise ValueError("Feature and counterfactual dimensions differ.")
    if counterfactuals.union_masks.shape != payload.action_masks.shape:
        raise ValueError("Counterfactual union masks have an unexpected shape.")


def _batch(
    payload: PatchFeaturePayload,
    counterfactuals: PatchCounterfactualPayload,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        payload.current[indices].float(),
        payload.next_features[indices].float(),
        payload.actions[indices].float(),
        payload.action_masks[indices].float(),
        counterfactuals.candidate_next[indices].float(),
        counterfactuals.union_masks[indices].float(),
    )


def mean_ranking_objective(
    model: nn.Module,
    payload: PatchFeaturePayload,
    counterfactuals: PatchCounterfactualPayload,
    *,
    batch_size: int,
    ranking_weight: float,
    temperature: float,
) -> dict[str, float]:
    """Evaluate each component of the frozen ranking-aware objective."""

    _validate_alignment(payload, counterfactuals)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    totals = {"total": 0.0, "balanced_mse": 0.0, "ranking_cross_entropy": 0.0}
    count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, payload.size, batch_size):
            indices = torch.arange(start, min(start + batch_size, payload.size))
            current, next_features, actions, masks, candidates, union = _batch(
                payload,
                counterfactuals,
                indices,
            )
            predicted_delta = model(current, actions, masks)
            losses = ranking_aware_objective(
                current,
                predicted_delta,
                next_features - current,
                masks,
                candidates,
                union,
                ranking_weight=ranking_weight,
                temperature=temperature,
            )
            for name in totals:
                totals[name] += float(losses[name].item()) * len(indices)
            count += len(indices)
    return {name: value / count for name, value in totals.items()}


def train_ranking_predictor(
    seed: int,
    train_payload: PatchFeaturePayload,
    train_counterfactuals: PatchCounterfactualPayload,
    validation_payload: PatchFeaturePayload,
    validation_counterfactuals: PatchCounterfactualPayload,
    *,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    ranking_weight: float,
    temperature: float,
) -> RankingPredictorFitResult:
    """Fit the frozen MLP with balanced MSE plus counterfactual cross-entropy."""

    _validate_alignment(train_payload, train_counterfactuals)
    _validate_alignment(validation_payload, validation_counterfactuals)
    if train_payload.patch_grid != validation_payload.patch_grid:
        raise ValueError("Train and validation patch grids differ.")
    if train_payload.current.shape[-1] != validation_payload.current.shape[-1]:
        raise ValueError("Train and validation feature dimensions differ.")
    if min(batch_size, max_epochs, patience) < 1:
        raise ValueError("Batch size, epochs, and patience must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer settings.")

    seed_everything(seed)
    model = create_patch_predictor(
        "mlp",
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
        train_sums = {
            "total": 0.0,
            "balanced_mse": 0.0,
            "ranking_cross_entropy": 0.0,
        }
        train_count = 0
        for start in range(0, train_payload.size, batch_size):
            indices = permutation[start : start + batch_size]
            current, next_features, actions, masks, candidates, union = _batch(
                train_payload,
                train_counterfactuals,
                indices,
            )
            predicted_delta = model(current, actions, masks)
            losses = ranking_aware_objective(
                current,
                predicted_delta,
                next_features - current,
                masks,
                candidates,
                union,
                ranking_weight=ranking_weight,
                temperature=temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for name in train_sums:
                train_sums[name] += float(losses[name].detach().item()) * len(indices)
            train_count += len(indices)

        validation = mean_ranking_objective(
            model,
            validation_payload,
            validation_counterfactuals,
            batch_size=batch_size,
            ranking_weight=ranking_weight,
            temperature=temperature,
        )
        train_means = {
            name: value / train_count for name, value in train_sums.items()
        }
        values = list(train_means.values()) + list(validation.values())
        if not all(np.isfinite(value) for value in values):
            raise RuntimeError("Ranking-aware history contains a non-finite loss.")
        history.append(
            {
                "method": "ranking_aware",
                "seed": seed,
                "ranking_weight": ranking_weight,
                "temperature": temperature,
                "epoch": epoch,
                "train_total": train_means["total"],
                "train_balanced_mse": train_means["balanced_mse"],
                "train_ranking_cross_entropy": train_means[
                    "ranking_cross_entropy"
                ],
                "validation_total": validation["total"],
                "validation_balanced_mse": validation["balanced_mse"],
                "validation_ranking_cross_entropy": validation[
                    "ranking_cross_entropy"
                ],
            }
        )
        if validation["total"] < best_validation - 1e-12:
            best_validation = validation["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    selected = mean_ranking_objective(
        model,
        validation_payload,
        validation_counterfactuals,
        batch_size=batch_size,
        ranking_weight=ranking_weight,
        temperature=temperature,
    )["total"]
    if not np.isclose(selected, best_validation, rtol=0.0, atol=1e-12):
        raise RuntimeError("Selected ranking-aware state did not reproduce its loss.")
    return RankingPredictorFitResult(
        model=model.eval(),
        seed=seed,
        ranking_weight=ranking_weight,
        temperature=temperature,
        best_epoch=best_epoch,
        best_validation_total=best_validation,
        history=history,
    )


def protocol_oracle_retrieval(
    payload: PatchFeaturePayload,
    counterfactuals: PatchCounterfactualPayload,
) -> dict[str, float | bool]:
    """Apply the written oracle rule: 100% retrieval plus unique candidates."""

    _validate_alignment(payload, counterfactuals)
    next_features = payload.next_features.float()
    candidate_next = counterfactuals.candidate_next.float()
    result = counterfactual_retrieval(
        next_features,
        candidate_next,
        counterfactuals.union_masks.float(),
    )
    accuracy = float(result["top1_correct"].float().mean().item())
    difference = float(
        (next_features - candidate_next[:, 0]).abs().max().item()
    )
    passed = bool(
        accuracy == 1.0 and counterfactuals.all_encoded_candidates_unique
    )
    return {
        "top1_accuracy": accuracy,
        "maximum_candidate_zero_difference": difference,
        "all_encoded_candidates_unique": (
            counterfactuals.all_encoded_candidates_unique
        ),
        "candidate_zero_bit_equality_required": False,
        "passed": passed,
    }


def select_ranking_setting(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen validation-only ranking-setting tie-break order."""

    if not rows:
        raise ValueError("At least one ranking setting is required.")
    required = {
        "model",
        "ranking_weight",
        "temperature",
        "mean_validation_top1",
        "mean_validation_true_margin",
        "mean_validation_action_region_mse",
    }
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for row in rows:
        if not required.issubset(row):
            raise ValueError("A ranking-selection row is missing required fields.")
        item = dict(row)
        numeric = [
            float(item["ranking_weight"]),
            float(item["temperature"]),
            float(item["mean_validation_top1"]),
            float(item["mean_validation_true_margin"]),
            float(item["mean_validation_action_region_mse"]),
        ]
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("Ranking-selection values must be finite.")
        key = (numeric[0], numeric[1])
        if key in seen:
            raise ValueError("Ranking-selection settings must be unique.")
        seen.add(key)
        normalized.append(item)
    selected = sorted(
        normalized,
        key=lambda item: (
            -float(item["mean_validation_top1"]),
            -float(item["mean_validation_true_margin"]),
            float(item["mean_validation_action_region_mse"]),
            float(item["ranking_weight"]),
            -float(item["temperature"]),
        ),
    )[0]
    return dict(selected)


def run_ranking_overfit_check(
    payload: PatchFeaturePayload,
    counterfactuals: PatchCounterfactualPayload,
    *,
    hidden_dim: int,
    ranking_weight: float = 0.3,
    temperature: float = 0.1,
    examples: int = 4,
    steps: int = 30,
    learning_rate: float = 0.005,
) -> dict[str, int | float | bool]:
    """Verify that the combined implementation reduces a tiny-set objective."""

    _validate_alignment(payload, counterfactuals)
    seed_everything(27182)
    count = min(examples, payload.size)
    indices = torch.arange(count)
    current, next_features, actions, masks, candidates, union = _batch(
        payload,
        counterfactuals,
        indices,
    )
    model = create_patch_predictor(
        "mlp",
        int(current.shape[-1]),
        payload.patch_grid,
        min(hidden_dim, 128),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def objective() -> torch.Tensor:
        predicted_delta = model(current, actions, masks)
        return ranking_aware_objective(
            current,
            predicted_delta,
            next_features - current,
            masks,
            candidates,
            union,
            ranking_weight=ranking_weight,
            temperature=temperature,
        )["total"]

    with torch.inference_mode():
        initial = float(objective().item())
    for _ in range(steps):
        loss = objective()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        final = float(objective().item())
    return {
        "examples": count,
        "steps": steps,
        "initial_total_objective": initial,
        "final_total_objective": final,
        "relative_reduction": 1.0 - final / max(initial, 1e-12),
        "loss_decreased": bool(final < initial),
    }
